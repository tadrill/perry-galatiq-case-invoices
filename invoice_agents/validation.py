"""Deterministic validation: everything whose answer is mechanical.

This runs before the validator agent and settles the questions a model should never be
asked. Whether 22 exceeds 15 is arithmetic. Whether a vendor appears on the approved list
is a lookup. Spending a model call on either would be slower, costlier and less reliable
than the `>` operator.

What it deliberately does NOT settle is any name the catalog could not resolve outright.
Those are collected into `unresolved_items` and `unresolved_vendor` and handed to the
agent, because deciding whether "WidgetC" is a typo for WidgetA or a genuinely unknown
product is a judgment, and the wrong call either invents a phantom order or blocks a
legitimate one.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from .inventory import (
    ItemRecord,
    LedgerEntry,
    Lookup,
    StockCheck,
    VendorRecord,
    check_stock,
    compute_line_hash,
    find_prior_invoices,
    lookup_item,
    lookup_vendor,
)
from .schemas import Finding, InvoiceData, ReconciliationResult, Severity

#: Unit prices may drift this far from catalog before it is worth a human's attention.
#: INV-1013's volume discounts sit at -4%; INV-1010's rush line at +20%.
PRICE_TOLERANCE = 0.10

#: "Net 30" rarely means exactly 30 days to the vendor's billing system.
DUE_DATE_TOLERANCE_DAYS = 5

_NET_TERMS = re.compile(r"net\s*(\d+)", re.IGNORECASE)


@dataclass
class Prevalidation:
    """Deterministic results, plus whatever still needs a judgment call."""

    findings: list[Finding] = field(default_factory=list)
    unresolved_items: list[Lookup] = field(default_factory=list)
    unresolved_vendor: Lookup | None = None
    resolved_items: dict[str, ItemRecord] = field(default_factory=dict)
    stock_checks: list[StockCheck] = field(default_factory=list)
    prior_invoices: list[LedgerEntry] = field(default_factory=list)
    vendor: VendorRecord | None = None

    @property
    def needs_adjudication(self) -> bool:
        return bool(self.unresolved_items or self.unresolved_vendor)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_stock_levels(
    conn: sqlite3.Connection, aggregated: dict[str, float], result: Prevalidation
) -> None:
    """Stock, against the aggregate per SKU rather than per line."""
    for name, quantity in aggregated.items():
        found = lookup_item(conn, name)

        if not isinstance(found.exact, ItemRecord):
            result.unresolved_items.append(found)
            continue

        result.resolved_items[name] = found.exact

        if quantity <= 0:
            continue  # negative quantities are reconciliation's finding, not a stockout

        stock = check_stock(conn, found.exact.item, int(quantity))
        result.stock_checks.append(stock)

        if stock.status == "ok":
            continue

        severity = (
            Severity.CRITICAL if stock.status == "discontinued" else Severity.WARNING
        )
        messages = {
            "insufficient": (
                f"{stock.item}: invoice bills {quantity:g} against {stock.available} "
                f"in stock (short {stock.shortfall})."
            ),
            "out_of_stock": (
                f"{stock.item}: invoice bills {quantity:g} but the item is out of stock."
            ),
            "discontinued": (
                f"{stock.item}: invoice bills {quantity:g} of a DISCONTINUED item. "
                f"An inbound invoice for a delisted SKU is a fraud signal, not a "
                f"supply problem."
            ),
        }
        result.findings.append(
            Finding(
                code=f"stock.{stock.status}",
                severity=severity,
                message=messages.get(stock.status, f"{stock.item}: {stock.status}"),
                field="line_items",
                evidence={**stock.to_dict(), "note": stock.note},
            )
        )


def _check_vendor(
    conn: sqlite3.Connection, invoice: InvoiceData, result: Prevalidation
) -> None:
    if not invoice.vendor_name:
        result.findings.append(
            Finding(
                code="vendor.missing",
                severity=Severity.CRITICAL,
                message="Invoice carries no vendor name, so payment cannot be directed.",
                field="vendor_name",
            )
        )
        return

    found = lookup_vendor(conn, invoice.vendor_name)

    if not isinstance(found.exact, VendorRecord):
        result.unresolved_vendor = found
        return

    result.vendor = found.exact

    if found.exact.status == "blocked":
        result.findings.append(
            Finding(
                code="vendor.blocked",
                severity=Severity.CRITICAL,
                message=f"{found.exact.name} is blocked for payment.",
                field="vendor_name",
                evidence={"notes": found.exact.notes},
            )
        )
    elif found.exact.status == "unverified":
        result.findings.append(
            Finding(
                code="vendor.unverified",
                severity=Severity.WARNING,
                message=(
                    f"{found.exact.name} is on the vendor list but unverified. "
                    f"{found.exact.notes or ''}".strip()
                ),
                field="vendor_name",
                evidence={"status": found.exact.status},
            )
        )


def _check_duplicates(
    conn: sqlite3.Connection, invoice: InvoiceData, result: Prevalidation
) -> None:
    if not invoice.invoice_number:
        return

    digest = compute_line_hash(
        [(line.item, line.quantity, line.unit_price or 0.0) for line in invoice.line_items]
    )
    prior = find_prior_invoices(conn, invoice.invoice_number, line_hash=digest)
    result.prior_invoices = prior

    for entry in prior:
        same_number = entry.invoice_number == invoice.invoice_number
        same_content = entry.line_hash == digest

        if same_number and same_content:
            code, severity = "duplicate.resubmission", Severity.CRITICAL
            message = (
                f"{invoice.invoice_number} was already processed on "
                f"{entry.processed_at} with an identical line set "
                f"(decision: {entry.decision})."
            )
        elif same_number:
            code, severity = "duplicate.revised", Severity.WARNING
            message = (
                f"{invoice.invoice_number} was already processed on "
                f"{entry.processed_at} for {entry.total}, but this submission bills "
                f"{invoice.total}. Confirm this supersedes rather than duplicates it."
            )
        else:
            code, severity = "duplicate.same_content", Severity.CRITICAL
            message = (
                f"Different invoice number, identical line items to "
                f"{entry.invoice_number} processed on {entry.processed_at}."
            )

        result.findings.append(
            Finding(
                code=code,
                severity=severity,
                message=message,
                field="invoice_number",
                evidence=entry.to_dict(),
            )
        )


def _check_prices(invoice: InvoiceData, result: Prevalidation) -> None:
    """Unit prices against the catalog, where the comparison is meaningful."""
    for index, line in enumerate(invoice.line_items):
        record = result.resolved_items.get(line.item)
        if record is None or record.unit_price is None or line.unit_price is None:
            continue
        if record.currency != invoice.currency:
            continue  # cross-currency comparison needs an FX rate we do not have
        if record.unit_price == 0:
            continue

        variance = (line.unit_price - record.unit_price) / record.unit_price
        if abs(variance) <= PRICE_TOLERANCE:
            continue

        result.findings.append(
            Finding(
                code="price.variance",
                severity=Severity.WARNING,
                message=(
                    f"Line {index + 1} ({line.item}) bills {line.unit_price:,.2f} against "
                    f"a catalog price of {record.unit_price:,.2f} "
                    f"({variance:+.0%})."
                    + (f" Line note: {line.note}." if line.note else "")
                ),
                field=f"line_items[{index}]",
                evidence={
                    "invoiced": line.unit_price,
                    "catalog": record.unit_price,
                    "variance": round(variance, 4),
                    "note": line.note,
                },
            )
        )


def _check_dates(invoice: InvoiceData, result: Prevalidation, today: date) -> None:
    if invoice.due_date is None:
        result.findings.append(
            Finding(
                code="date.due_unresolvable",
                severity=Severity.WARNING,
                message=(
                    "Due date could not be resolved to a calendar date"
                    + (
                        f" (printed as {invoice.due_date_raw!r})."
                        if invoice.due_date_raw
                        else "."
                    )
                ),
                field="due_date",
                evidence={"raw": invoice.due_date_raw},
            )
        )
        return

    if invoice.invoice_date and invoice.due_date < invoice.invoice_date:
        result.findings.append(
            Finding(
                code="date.due_before_issue",
                severity=Severity.CRITICAL,
                message=(
                    f"Due date {invoice.due_date} precedes the invoice date "
                    f"{invoice.invoice_date}."
                ),
                field="due_date",
            )
        )

    if invoice.due_date < today:
        result.findings.append(
            Finding(
                code="date.already_overdue",
                severity=Severity.INFO,
                message=f"Due date {invoice.due_date} has already passed.",
                field="due_date",
            )
        )

    terms = _NET_TERMS.search(invoice.payment_terms or "")
    if terms and invoice.invoice_date:
        expected = invoice.invoice_date + timedelta(days=int(terms.group(1)))
        drift = abs((invoice.due_date - expected).days)
        if drift > DUE_DATE_TOLERANCE_DAYS:
            result.findings.append(
                Finding(
                    code="date.terms_mismatch",
                    severity=Severity.WARNING,
                    message=(
                        f"Payment terms say {invoice.payment_terms!r}, which implies a due "
                        f"date near {expected}, but the invoice states {invoice.due_date} "
                        f"({drift} days out)."
                    ),
                    field="due_date",
                    evidence={"terms": invoice.payment_terms, "expected": str(expected)},
                )
            )


def _check_currency(invoice: InvoiceData, result: Prevalidation) -> None:
    if result.vendor and result.vendor.currency != invoice.currency:
        result.findings.append(
            Finding(
                code="currency.unexpected",
                severity=Severity.WARNING,
                message=(
                    f"Invoice is denominated in {invoice.currency} but "
                    f"{result.vendor.name} is set up to bill in "
                    f"{result.vendor.currency}."
                ),
                field="currency",
            )
        )


def _fold_reconciliation(
    reconciliation: ReconciliationResult | None, result: Prevalidation
) -> None:
    """Bring reconciliation's output into the findings stream, exactly once."""
    if reconciliation is None:
        return

    result.findings.extend(reconciliation.integrity_findings)

    for discrepancy in reconciliation.discrepancies:
        result.findings.append(
            Finding(
                code="arithmetic.mismatch",
                severity=Severity.WARNING,
                message=(
                    f"Arithmetic does not reconcile after re-reading the document, so "
                    f"the invoice itself is inconsistent: {discrepancy}"
                ),
                field="total",
                evidence={"discrepancy": discrepancy},
            )
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def prevalidate(
    conn: sqlite3.Connection,
    invoice: InvoiceData,
    reconciliation: ReconciliationResult | None = None,
    *,
    today: date | None = None,
) -> Prevalidation:
    """Run every deterministic check and collect what still needs judgment."""
    result = Prevalidation()
    today = today or datetime.now(UTC).date()

    aggregated = (
        reconciliation.aggregated_quantities
        if reconciliation
        else {line.item: line.quantity for line in invoice.line_items}
    )

    _check_stock_levels(conn, aggregated, result)
    _check_vendor(conn, invoice, result)
    _check_duplicates(conn, invoice, result)
    _check_prices(invoice, result)
    _check_dates(invoice, result, today)
    _check_currency(invoice, result)
    _fold_reconciliation(reconciliation, result)

    return result
