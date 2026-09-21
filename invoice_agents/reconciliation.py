"""Deterministic arithmetic over an extracted invoice.

No LLM touches this. The model transcribes what the document claims; this recomputes what
the document *should* say and compares. Keeping the two apart is what makes the comparison
mean anything -- a model that both transcribes and calculates is checking itself, and
agrees with itself every time.

It earns its place three ways:

*It repairs OCR damage.* INV-1012 prints a line total of "$3,500.O0" with a letter O. The
transcription of that column is garbage; 7 x 500 is not.

*It catches genuine vendor errors.* INV-1009 states a subtotal of 1000.00 against line
items summing to -250.00.

*It feeds the retry edge.* A mismatch is ambiguous between a misread and a bad invoice, so
it routes back to the extractor once. If the numbers reconcile on re-read it was a misread;
if they do not, the invoice really is inconsistent.

Money is `Decimal` end to end, built from `str(value)` so a float that arrived as
1472.8000000000002 becomes exactly 1472.80. Comparisons still allow one cent, because
vendors round tax by their own conventions and that is not an error worth blocking a
payment over.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from time import perf_counter
from typing import Any

from .inventory.naming import normalize_item
from .schemas import (
    Finding,
    InvoiceData,
    LineItem,
    LineRecomputation,
    ReconciliationResult,
    Severity,
)
from .state import InvoiceState, log

NODE = "reconciler"

CENTS = Decimal("0.01")

#: Vendors round tax differently; a cent of drift is not a discrepancy.
MONEY_TOLERANCE = Decimal("0.01")

#: Total extractor passes allowed. One initial read plus one re-read after a mismatch --
#: enough to tell a misread from an inconsistent document, and no more. INV-1009 can
#: never reconcile, so an unbounded loop would spin on it forever.
MAX_EXTRACTION_ATTEMPTS = 2

#: Sales tax above this is not a rate anyone charges on industrial parts. Set well clear
#: of the highest legitimate rate in the sample set (10%) so ordinary invoices never trip
#: it, and low enough to catch a padded tax line dressed up as a statutory charge.
MAX_PLAUSIBLE_TAX_RATE = Decimal("0.25")

#: Freight above this share of the goods is worth a look. Shipping is the classic place to
#: hide an inflated charge, because nothing downstream validates it against a catalog the
#: way a line item gets validated against stock.
MAX_SHIPPING_RATIO = Decimal("0.25")


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------


def _dec(value: float | None) -> Decimal | None:
    """Exact Decimal from a float, via str to avoid binary float artifacts."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _money(value: float | None) -> Decimal | None:
    amount = _dec(value)
    return None if amount is None else amount.quantize(CENTS, rounding=ROUND_HALF_UP)


def _matches(left: Decimal | None, right: Decimal | None) -> bool:
    """True when the two agree, or when one is absent so there is nothing to contradict."""
    if left is None or right is None:
        return True
    return abs(left - right) <= MONEY_TOLERANCE


def _f(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _fmt(value: Decimal | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


# ---------------------------------------------------------------------------
# Line-level work
# ---------------------------------------------------------------------------


def _recompute_line(line: LineItem) -> LineRecomputation:
    quantity = _dec(line.quantity)
    unit_price = _money(line.unit_price)
    stated = _money(line.amount)

    computed: Decimal | None = None
    if quantity is not None and unit_price is not None:
        computed = (quantity * unit_price).quantize(CENTS, rounding=ROUND_HALF_UP)

    delta: Decimal | None = None
    matches = True
    if computed is not None and stated is not None:
        delta = (stated - computed).quantize(CENTS, rounding=ROUND_HALF_UP)
        matches = abs(delta) <= MONEY_TOLERANCE

    return LineRecomputation(
        item=line.item,
        quantity=line.quantity,
        unit_price=_f(unit_price),
        stated_amount=_f(stated),
        computed_amount=_f(computed),
        delta=_f(delta),
        matches=matches,
    )


def _line_integrity(index: int, line: LineItem) -> list[Finding]:
    """Problems a re-read cannot fix, so they never reach `discrepancies`."""
    findings: list[Finding] = []
    position = f"line_items[{index}]"
    quantity = _dec(line.quantity)

    if not line.item or not line.item.strip():
        findings.append(
            Finding(
                code="data.missing_item_name",
                severity=Severity.CRITICAL,
                message=f"Line {index + 1} has no item name.",
                field=position,
            )
        )

    if quantity is not None and quantity < 0:
        findings.append(
            Finding(
                code="data.negative_quantity",
                severity=Severity.CRITICAL,
                message=(
                    f"Line {index + 1} ({line.item}) bills a negative quantity "
                    f"({line.quantity:g}). An invoice is not a credit note."
                ),
                field=position,
                evidence={"item": line.item, "quantity": line.quantity},
            )
        )
    elif quantity is not None and quantity == 0:
        findings.append(
            Finding(
                code="data.zero_quantity",
                severity=Severity.WARNING,
                message=f"Line {index + 1} ({line.item}) bills a quantity of zero.",
                field=position,
            )
        )
    elif quantity is not None and quantity != quantity.to_integral_value():
        findings.append(
            Finding(
                code="data.fractional_quantity",
                severity=Severity.WARNING,
                message=(
                    f"Line {index + 1} ({line.item}) bills a fractional quantity "
                    f"({line.quantity:g}) of a discrete part."
                ),
                field=position,
            )
        )

    unit_price = _dec(line.unit_price)
    if unit_price is None:
        findings.append(
            Finding(
                code="data.missing_unit_price",
                severity=Severity.WARNING,
                message=(
                    f"Line {index + 1} ({line.item}) has no unit price, so its amount "
                    "cannot be independently verified."
                ),
                field=position,
            )
        )
    elif unit_price < 0:
        findings.append(
            Finding(
                code="data.negative_unit_price",
                severity=Severity.CRITICAL,
                message=f"Line {index + 1} ({line.item}) has a negative unit price.",
                field=position,
                evidence={"item": line.item, "unit_price": line.unit_price},
            )
        )

    return findings


def _tax_integrity(invoice: InvoiceData, subtotal: Decimal | None) -> list[Finding]:
    """Sanity-check the tax rate, stated or implied.

    An invoice can be arithmetically perfect and still charge a tax nobody levies. The
    rate is checked whether or not the document states one: when only an amount is given
    the rate is derived from it, so INV-1010-style invoices that print "Sales Tax: $335"
    with no percentage are covered the same way.
    """
    rate = _dec(invoice.tax_rate)
    stated_tax = _money(invoice.tax_amount)
    derived = False

    if rate is None and stated_tax is not None and subtotal is not None and subtotal > 0:
        rate = stated_tax / subtotal
        derived = True

    if rate is None:
        return []

    source = (
        f"implied by a tax of {stated_tax:,.2f} on a subtotal of {subtotal:,.2f}"
        if derived
        else "stated on the invoice"
    )

    if rate < 0:
        return [
            Finding(
                code="tax.negative_rate",
                severity=Severity.CRITICAL,
                message=(
                    f"Tax rate of {rate * 100:g}% is negative ({source}). Tax is not a "
                    f"discount; a credit belongs on a credit note, not buried in the tax "
                    f"line of an invoice."
                ),
                field="tax_rate",
                evidence={"rate": float(rate), "derived": derived},
            )
        ]

    if rate > MAX_PLAUSIBLE_TAX_RATE:
        return [
            Finding(
                code="tax.implausible_rate",
                severity=Severity.WARNING,
                message=(
                    f"Tax rate of {rate * 100:g}% ({source}) exceeds any plausible sales "
                    f"tax on industrial goods. Confirm the charge before paying it."
                ),
                field="tax_rate",
                evidence={"rate": float(rate), "derived": derived},
            )
        ]

    return []


def _duplicate_lines(invoice: InvoiceData) -> list[Finding]:
    """Flag the same goods billed twice inside one document.

    Matched on normalized name, quantity and unit price. Repeating a SKU is ordinary --
    INV-1013 bills WidgetA three times at different prices with different notes, which is
    how volume discounts look. Repeating it at the *same* quantity and the *same* price is
    either a split shipment or a line that got pasted twice, and only a human can say
    which, so this warns rather than blocking.
    """
    findings: list[Finding] = []
    seen: dict[tuple[str, Decimal, Decimal | None], int] = {}

    for index, line in enumerate(invoice.line_items):
        key_name = normalize_item(line.item or "")
        quantity = _dec(line.quantity)
        if not key_name or quantity is None or quantity <= 0:
            continue

        key = (key_name, quantity, _money(line.unit_price))
        first = seen.get(key)
        if first is None:
            seen[key] = index
            continue

        price = f"{line.unit_price:,.2f}" if line.unit_price is not None else "no price"
        findings.append(
            Finding(
                code="data.duplicate_line",
                severity=Severity.WARNING,
                message=(
                    f"Lines {first + 1} and {index + 1} bill the same goods twice: "
                    f"{line.item} x{line.quantity:g} at {price}. Confirm this is a split "
                    f"shipment and not a double charge."
                ),
                field=f"line_items[{index}]",
                evidence={
                    "item": line.item,
                    "quantity": line.quantity,
                    "unit_price": line.unit_price,
                    "first_line": first + 1,
                    "duplicate_line": index + 1,
                },
            )
        )

    return findings


def _charge_integrity(invoice: InvoiceData, subtotal: Decimal | None) -> list[Finding]:
    """Check the charges that sit outside the line items.

    Shipping is validated by nothing. A line item is checked against a catalog price and
    a stock level; freight is whatever the invoice says it is, which makes it the easiest
    place to inflate a bill without tripping anything.
    """
    shipping = _money(invoice.shipping)
    if shipping is None or shipping <= 0 or subtotal is None or subtotal <= 0:
        return []

    ratio = shipping / subtotal
    if ratio <= MAX_SHIPPING_RATIO:
        return []

    return [
        Finding(
            code="charges.disproportionate",
            severity=Severity.WARNING,
            message=(
                f"Shipping of {shipping:,.2f} is {ratio:.0%} of the {subtotal:,.2f} of "
                f"goods on this invoice. Freight is not validated against a catalog, so "
                f"an outsized charge here is worth confirming."
            ),
            field="shipping",
            evidence={"shipping": float(shipping), "subtotal": float(subtotal)},
        )
    ]


def _total_integrity(stated_total: Decimal | None) -> list[Finding]:
    """A grand total at or below zero is not an invoice."""
    if stated_total is None or stated_total >= 0:
        return []

    return [
        Finding(
            code="data.negative_total",
            severity=Severity.CRITICAL,
            message=(
                f"Grand total of {stated_total:,.2f} is negative. An invoice requests "
                f"payment; a negative balance is a credit note and does not belong in "
                f"this workflow."
            ),
            field="total",
            evidence={"total": float(stated_total)},
        )
    ]


def aggregate_quantities(lines: list[LineItem]) -> dict[str, float]:
    """Total quantity per SKU, summed across every line naming the same product.

    This is the check that matters. INV-1013 bills WidgetA on three separate lines
    (15 + 5 + 2 = 22 against 15 in stock) and INV-1010 splits it across a standard line
    and a rush line (8 + 4 = 12). Every individual line passes a stock check; only the
    aggregate fails. Grouping is by normalized name, so "WidgetA", "Widget A" and
    "WidgetA (rush order)" land in the same bucket, while the key keeps the first spelling
    seen so downstream messages quote something a human recognizes.
    """
    totals: dict[str, Decimal] = {}
    display: dict[str, str] = {}

    for line in lines:
        key = normalize_item(line.item or "")
        if not key:
            continue
        quantity = _dec(line.quantity)
        if quantity is None:
            continue
        display.setdefault(key, line.item)
        totals[key] = totals.get(key, Decimal(0)) + quantity

    return {display[key]: float(total) for key, total in totals.items()}


# ---------------------------------------------------------------------------
# Whole-invoice reconciliation
# ---------------------------------------------------------------------------


def reconcile(invoice: InvoiceData) -> ReconciliationResult:
    """Recompute an invoice's arithmetic. Pure function, no state, no I/O."""
    line_results = [_recompute_line(line) for line in invoice.line_items]
    discrepancies: list[str] = []
    integrity: list[Finding] = []

    for index, line in enumerate(invoice.line_items):
        integrity.extend(_line_integrity(index, line))

    for index, result in enumerate(line_results):
        if not result.matches:
            discrepancies.append(
                f"line {index + 1} ({result.item}): stated {_fmt(_dec(result.stated_amount))}, "
                f"computed {result.quantity:g} x {_fmt(_dec(result.unit_price))} = "
                f"{_fmt(_dec(result.computed_amount))}"
            )

    # Prefer each line's recomputation; fall back to what the document stated when a
    # line cannot be recomputed, so one missing unit price does not void the subtotal.
    computed_subtotal: Decimal | None = Decimal(0)
    for result in line_results:
        contribution = _dec(result.computed_amount)
        if contribution is None:
            contribution = _dec(result.stated_amount)
        if contribution is None:
            continue
        computed_subtotal += contribution
    computed_subtotal = computed_subtotal.quantize(CENTS, rounding=ROUND_HALF_UP)

    stated_subtotal = _money(invoice.subtotal)
    if not _matches(computed_subtotal, stated_subtotal):
        difference = abs(computed_subtotal - stated_subtotal)
        discrepancies.append(
            f"subtotal: stated {_fmt(stated_subtotal)}, computed {_fmt(computed_subtotal)} "
            f"(difference {_fmt(difference)})"
        )

    stated_tax = _money(invoice.tax_amount)
    tax_rate = _dec(invoice.tax_rate)
    computed_tax: Decimal | None = None
    if tax_rate is not None:
        computed_tax = (computed_subtotal * tax_rate).quantize(CENTS, rounding=ROUND_HALF_UP)
        if not _matches(computed_tax, stated_tax):
            discrepancies.append(
                f"tax: stated {_fmt(stated_tax)}, computed {tax_rate * 100:g}% of "
                f"{_fmt(computed_subtotal)} = {_fmt(computed_tax)}"
            )

    shipping = _money(invoice.shipping)

    # Build the total from the STATED tax when there is one. Otherwise a single wrong tax
    # figure would be reported twice: once as a tax discrepancy and again as a total one.
    tax_for_total = stated_tax if stated_tax is not None else (computed_tax or Decimal(0))
    computed_total = (computed_subtotal + tax_for_total + (shipping or Decimal(0))).quantize(
        CENTS, rounding=ROUND_HALF_UP
    )

    stated_total = _money(invoice.total)
    if not _matches(computed_total, stated_total):
        parts = f"{_fmt(computed_subtotal)} + {_fmt(tax_for_total)} tax"
        if shipping is not None:
            parts += f" + {_fmt(shipping)} shipping"
        discrepancies.append(
            f"total: stated {_fmt(stated_total)}, computed {parts} = {_fmt(computed_total)}"
        )

    # Plausibility, as distinct from arithmetic. Every one of these can sit inside an
    # invoice that adds up perfectly, and no amount of re-reading will change them, so
    # they join the integrity findings rather than the discrepancies.
    integrity.extend(_tax_integrity(invoice, computed_subtotal))
    integrity.extend(_duplicate_lines(invoice))
    integrity.extend(_charge_integrity(invoice, computed_subtotal))
    integrity.extend(_total_integrity(stated_total))

    return ReconciliationResult(
        lines=line_results,
        computed_subtotal=_f(computed_subtotal),
        stated_subtotal=_f(stated_subtotal),
        computed_tax=_f(computed_tax),
        stated_tax=_f(stated_tax),
        shipping=_f(shipping),
        computed_total=_f(computed_total),
        stated_total=_f(stated_total),
        aggregated_quantities=aggregate_quantities(invoice.line_items),
        is_consistent=not discrepancies,
        discrepancies=discrepancies,
        integrity_findings=integrity,
    )


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def reconcile_node(state: InvoiceState) -> dict[str, Any]:
    """LangGraph node. Reads invoice, writes reconciliation."""
    started = perf_counter()
    invoice = state.get("invoice")

    if invoice is None:
        return {
            "reconciliation": None,
            "audit_log": [
                log(
                    NODE,
                    "skipped",
                    detail="extraction produced no invoice",
                    severity=Severity.WARNING,
                )
            ],
        }

    result = reconcile(invoice)
    elapsed = (perf_counter() - started) * 1000

    if result.is_consistent:
        detail = (
            f"arithmetic verified: subtotal {result.computed_subtotal:,.2f}, "
            f"total {result.computed_total:,.2f}"
        )
        severity = Severity.INFO
    else:
        detail = f"{len(result.discrepancies)} discrepancy(ies): " + "; ".join(
            result.discrepancies
        )
        severity = Severity.WARNING

    entries = [log(NODE, "reconciled", detail=detail, severity=severity, elapsed_ms=elapsed)]

    if result.integrity_findings:
        entries.append(
            log(
                NODE,
                "integrity_issues",
                detail="; ".join(f.code for f in result.integrity_findings),
                severity=Severity.WARNING,
            )
        )

    return {"reconciliation": result, "audit_log": entries}


def route_after_reconciliation(state: InvoiceState) -> str:
    """Conditional edge: re-read, carry on, or give up.

    A mismatch sends the invoice back to the extractor exactly once. If the re-read
    reconciles, the first pass misread the document. If it does not, the document itself
    is inconsistent -- a finding, not a parsing problem -- and it goes forward to be
    judged on its merits rather than being retried into the ground.
    """
    attempts = state.get("extraction_attempts", 0)

    if state.get("invoice") is None:
        return "extract" if attempts < MAX_EXTRACTION_ATTEMPTS else "abort"

    reconciliation = state.get("reconciliation")
    if reconciliation is None or reconciliation.is_consistent:
        return "validate"

    return "extract" if attempts < MAX_EXTRACTION_ATTEMPTS else "validate"


def integrity_findings(state: InvoiceState) -> list[Finding]:
    """Reconciliation's findings, for the next node to fold into state exactly once."""
    reconciliation = state.get("reconciliation")
    return list(reconciliation.integrity_findings) if reconciliation else []
