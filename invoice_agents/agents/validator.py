"""Validator agent: a tool-using adjudicator over the legacy inventory database.

The deterministic checks in `invoice_agents.validation` have already run by the time this
node executes. Whether 22 exceeds 15 is not a question for a language model, and asking it
would be slower, costlier and less reliable than `>`.

What reaches the agent is what those checks could not settle: an item name the catalog
failed to resolve, a vendor name that missed the approved list, and the soft signals no
rule can enumerate ahead of time. The agent investigates with tools and rules on them.

The case that justifies the whole design is INV-1016's "WidgetC". It scores 0.857 against
WidgetA *and* 0.857 against WidgetB -- similar enough that any auto-resolving matcher
would book it against whichever sorted first, inventing an order the company never placed.
The correct answer is that a SKU differing in its trailing character is a different
product, and that is a judgment, not a threshold.

The agent returns judgments, not findings. Python converts them into `Finding` objects
with stable codes so the vocabulary downstream stays fixed.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import ValidationError

from ..config import Settings, get_settings
from ..inventory import VendorRecord, check_stock, connect, lookup_vendor
from ..llm import Purpose, get_llm, register_mock_handler
from ..policy import SCRUTINY_THRESHOLD
from ..schemas import (
    Finding,
    InvoiceData,
    ReconciliationResult,
    Severity,
    ValidationVerdict,
)
from ..state import InvoiceState, log
from ..tools import TOOLS_BY_NAME, VALIDATOR_TOOLS
from ..validation import Prevalidation, prevalidate
from .mock_validation import mock_validate

NODE = "validator"

from ..thresholds import MAX_TOOL_ITERATIONS

SYSTEM_PROMPT = """\
You are an accounts-payable validation specialist at a manufacturing firm.

Deterministic checks have ALREADY run against this invoice: stock levels, approved-vendor \
membership, duplicate detection, price variance, date consistency and arithmetic. Their \
results are given to you. Do not redo them and do not restate them.

You are here to settle what they could not.

1. ITEM NAMES THE CATALOG COULD NOT RESOLVE
   Call lookup_item to see the near misses and their similarity scores, then rule:
   - same_product -- the invoice misspelled a catalog SKU. Spacing, capitalization or \
OCR damage: "Widget A" for WidgetA, "Gadget X" for GadgetX.
   - different_product -- it resembles a catalog SKU but is a distinct item. SKUs that \
differ in a trailing letter or digit are almost always different products. WidgetC is \
NOT WidgetA, no matter how high the similarity score.
   - unknown -- nothing in the catalog resembles it.
   A similarity score is evidence, not a verdict. Resolving a real product onto the wrong \
SKU books goods the company never ordered, which is worse than flagging it for a human.
   When you rule same_product, call check_stock on the resolved SKU to confirm it can \
actually be fulfilled.

2. VENDOR NAMES THAT MISSED THE APPROVED LIST
   Call lookup_vendor. A near miss on spelling ("Distributers" for "Distributors") is \
usually the same company; a name that resembles nothing on the list is not approved for \
payment, and paying an unapproved vendor is how invoice fraud succeeds. Check the status \
field -- 'unverified' means something changed recently and has not been re-confirmed.

3. RISK SIGNALS NO RULE ANTICIPATED
   Read the notes, the dates, the round numbers and the total. Report what would make an \
experienced clerk hesitate: urgency or pressure language, a demand for an unusual payment \
method, a total parked just below an approval threshold, a vendor that recently changed \
its banking details, a brand-new vendor billing an unusually large amount.
   Report only what the deterministic findings missed.

Use the tools rather than guessing at catalog contents. Be concise and specific: every \
ruling is read by a human deciding whether to release money."""


VERDICT_REQUEST = """\
Now return your structured verdict.

Include one item_adjudication for every unresolved item name, a vendor_adjudication if a \
vendor name was unresolved, and any risk signals you identified. The summary should tell a \
reviewer in two or three sentences what is wrong with this invoice and what you had to \
judge."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_invoice(invoice: InvoiceData) -> str:
    lines = [
        "INVOICE",
        f"  number:  {invoice.invoice_number or 'MISSING'}"
        + (f" (revision {invoice.revision})" if invoice.revision else ""),
        f"  vendor:  {invoice.vendor_name or 'MISSING'}",
        (
            f"  dates:   issued {invoice.invoice_date or invoice.invoice_date_raw or '?'}, "
            f"due {invoice.due_date or invoice.due_date_raw or '?'}"
        ),
        f"  terms:   {invoice.payment_terms or 'not stated'}",
        f"  total:   {invoice.total:,.2f} {invoice.currency}"
        if invoice.total is not None
        else "  total:   MISSING",
        "  line items:",
    ]
    for index, line in enumerate(invoice.line_items):
        price = f"{line.unit_price:,.2f}" if line.unit_price is not None else "?"
        note = f"  [{line.note}]" if line.note else ""
        lines.append(f"    {index + 1}. {line.item!r} qty {line.quantity:g} @ {price}{note}")
    if invoice.notes:
        lines.append(f"  notes:   {invoice.notes}")
    if invoice.unreadable_fields:
        lines.append(f"  unreadable: {', '.join(invoice.unreadable_fields)}")
    return "\n".join(lines)


def _format_arithmetic(reconciliation: ReconciliationResult | None) -> str:
    if reconciliation is None:
        return "ARITHMETIC\n  not available"

    status = "reconciles" if reconciliation.is_consistent else "DOES NOT RECONCILE"
    lines = [
        "ARITHMETIC (recomputed in Python, not by a model)",
        f"  status:   {status}",
        (
            f"  subtotal: computed {reconciliation.computed_subtotal}, "
            f"stated {reconciliation.stated_subtotal}"
        ),
        (
            f"  total:    computed {reconciliation.computed_total}, "
            f"stated {reconciliation.stated_total}"
        ),
    ]
    if reconciliation.aggregated_quantities:
        totals = ", ".join(
            f"{name} x{qty:g}" for name, qty in reconciliation.aggregated_quantities.items()
        )
        lines.append(f"  per-SKU totals across all lines: {totals}")
    return "\n".join(lines)


def _format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "DETERMINISTIC FINDINGS\n  none"
    lines = ["DETERMINISTIC FINDINGS (already recorded -- do not restate these)"]
    lines.extend(f"  [{f.severity.value}] {f.code}: {f.message}" for f in findings)
    return "\n".join(lines)


def _format_unresolved(prevalidation: Prevalidation) -> str:
    if not prevalidation.needs_adjudication:
        return "NEEDS YOUR RULING\n  No unresolved names. Report risk signals only."

    lines = ["NEEDS YOUR RULING"]
    for lookup in prevalidation.unresolved_items:
        nearest = (
            ", ".join(f"{c.item} {c.score:.3f}" for c in lookup.candidates) or "nothing similar"
        )
        lines.append(f"  UNRESOLVED ITEM: {lookup.query}  (nearest: {nearest})")
    if prevalidation.unresolved_vendor:
        lookup = prevalidation.unresolved_vendor
        nearest = (
            ", ".join(f"{c.name} {c.score:.3f}" for c in lookup.candidates)
            or "nothing similar"
        )
        lines.append(f"  UNRESOLVED VENDOR: {lookup.query}  (nearest: {nearest})")
    return "\n".join(lines)


def build_messages(
    invoice: InvoiceData,
    reconciliation: ReconciliationResult | None,
    prevalidation: Prevalidation,
) -> list[BaseMessage]:
    """Assemble the validator's opening context."""
    context = "\n\n".join(
        [
            _format_invoice(invoice),
            _format_arithmetic(reconciliation),
            _format_findings(prevalidation.findings),
            _format_unresolved(prevalidation),
            (
                "POLICY CONTEXT\n  Invoices at or above "
                f"{SCRUTINY_THRESHOLD:,.0f} require additional scrutiny."
            ),
        ]
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------


def run_tool_loop(
    llm: BaseChatModel, messages: list[BaseMessage], *, max_iterations: int = MAX_TOOL_ITERATIONS
) -> tuple[list[BaseMessage], list[str]]:
    """Drive tool calls until the agent stops asking or the budget runs out.

    A failing tool returns its error to the agent rather than raising: a bad argument is
    something the model can correct on the next turn, and killing the run over it would
    lose the work already done.
    """
    bound = llm.bind_tools(VALIDATOR_TOOLS)
    conversation = list(messages)
    performed: list[str] = []

    for _ in range(max_iterations):
        response = bound.invoke(conversation)
        conversation.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {})
            selected = TOOLS_BY_NAME.get(name)

            if selected is None:
                content = json.dumps({"error": f"No such tool: {name}"})
            else:
                try:
                    content = str(selected.invoke(args))
                except Exception as exc:  # noqa: BLE001 - hand the error back to the agent
                    content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})

            performed.append(f"{name}({json.dumps(args, default=str)})")
            conversation.append(
                ToolMessage(content=content, tool_call_id=call.get("id", ""), name=name)
            )

    return conversation, performed


# ---------------------------------------------------------------------------
# Verdict -> findings
# ---------------------------------------------------------------------------


def _adjudication_findings(
    verdict: ValidationVerdict,
    aggregated: dict[str, float],
    conn: Any,
) -> list[Finding]:
    findings: list[Finding] = []

    for ruling in verdict.item_adjudications:
        evidence = {
            "invoice_name": ruling.invoice_name,
            "verdict": ruling.verdict,
            "resolved_sku": ruling.resolved_sku,
            "reasoning": ruling.reasoning,
        }

        if ruling.verdict == "same_product" and ruling.resolved_sku:
            findings.append(
                Finding(
                    code="catalog.name_corrected",
                    severity=Severity.INFO,
                    message=(
                        f"{ruling.invoice_name!r} resolved to catalog SKU "
                        f"{ruling.resolved_sku}. {ruling.reasoning}"
                    ),
                    field="line_items",
                    evidence=evidence,
                )
            )
            # A corrected name still has to clear the shelf.
            quantity = aggregated.get(ruling.invoice_name)
            if quantity and quantity > 0:
                stock = check_stock(conn, ruling.resolved_sku, int(quantity))
                if stock.status != "ok":
                    findings.append(
                        Finding(
                            code=f"stock.{stock.status}",
                            severity=(
                                Severity.CRITICAL
                                if stock.status == "discontinued"
                                else Severity.WARNING
                            ),
                            message=(
                                f"{stock.item} (billed as {ruling.invoice_name!r}): "
                                f"{quantity:g} requested, {stock.available} available "
                                f"({stock.status})."
                            ),
                            field="line_items",
                            evidence=stock.to_dict(),
                        )
                    )
        else:
            findings.append(
                Finding(
                    code="catalog.unknown_item",
                    severity=Severity.CRITICAL,
                    message=(
                        f"{ruling.invoice_name!r} is not a catalog product "
                        f"({ruling.verdict}). {ruling.reasoning}"
                    ),
                    field="line_items",
                    evidence=evidence,
                )
            )

    return findings


def _vendor_findings(verdict: ValidationVerdict, conn: Any) -> list[Finding]:
    ruling = verdict.vendor_adjudication
    if ruling is None:
        return []

    evidence = {
        "invoice_name": ruling.invoice_name,
        "verdict": ruling.verdict,
        "resolved_vendor": ruling.resolved_vendor,
        "reasoning": ruling.reasoning,
    }

    if ruling.verdict != "same_vendor" or not ruling.resolved_vendor:
        return [
            Finding(
                code="vendor.unapproved",
                severity=Severity.CRITICAL,
                message=(
                    f"{ruling.invoice_name!r} is not an approved vendor "
                    f"({ruling.verdict}). {ruling.reasoning}"
                ),
                field="vendor_name",
                evidence=evidence,
            )
        ]

    findings = [
        Finding(
            code="vendor.name_corrected",
            severity=Severity.WARNING,
            message=(
                f"Vendor name {ruling.invoice_name!r} resolved to approved vendor "
                f"{ruling.resolved_vendor}. {ruling.reasoning}"
            ),
            field="vendor_name",
            evidence=evidence,
        )
    ]

    resolved = lookup_vendor(conn, ruling.resolved_vendor).exact
    if isinstance(resolved, VendorRecord) and resolved.status != "approved":
        findings.append(
            Finding(
                code=f"vendor.{resolved.status}",
                severity=(
                    Severity.CRITICAL if resolved.status == "blocked" else Severity.WARNING
                ),
                message=(
                    f"{resolved.name} is {resolved.status}. {resolved.notes or ''}".strip()
                ),
                field="vendor_name",
                evidence={"status": resolved.status},
            )
        )
    return findings


def verdict_to_findings(
    verdict: ValidationVerdict, aggregated: dict[str, float], conn: Any
) -> list[Finding]:
    """Convert the agent's judgments into findings with stable codes."""
    findings = _adjudication_findings(verdict, aggregated, conn)
    findings.extend(_vendor_findings(verdict, conn))
    findings.extend(
        Finding(
            code="risk.signal",
            severity=signal.severity,
            message=f"{signal.signal}: {signal.evidence}",
            evidence={"signal": signal.signal, "evidence": signal.evidence},
        )
        for signal in verdict.risk_signals
    )
    return findings


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def build_validator(settings: Settings | None = None, llm: BaseChatModel | None = None):
    return llm or get_llm(Purpose.VALIDATOR, settings)


def validate_node(state: InvoiceState) -> dict[str, Any]:
    """LangGraph node. Reads invoice and reconciliation, appends findings."""
    started = perf_counter()
    invoice = state.get("invoice")

    if invoice is None:
        return {
            "audit_log": [
                log(
                    NODE,
                    "skipped",
                    detail="no invoice to validate",
                    severity=Severity.WARNING,
                )
            ]
        }

    reconciliation = state.get("reconciliation")
    settings = get_settings()

    with connect(settings.resolved_db_path) as conn:
        prevalidation = prevalidate(conn, invoice, reconciliation)
        deterministic = list(prevalidation.findings)

        aggregated = (
            reconciliation.aggregated_quantities
            if reconciliation
            else {line.item: line.quantity for line in invoice.line_items}
        )

        entries = [
            log(
                NODE,
                "deterministic_checks",
                detail=(
                    f"{len(deterministic)} finding(s); "
                    f"{len(prevalidation.unresolved_items)} item name(s) and "
                    f"{1 if prevalidation.unresolved_vendor else 0} vendor name(s) "
                    f"need adjudication"
                ),
                severity=Severity.INFO,
            )
        ]

        try:
            llm = build_validator(settings)
            messages = build_messages(invoice, reconciliation, prevalidation)
            conversation, performed = run_tool_loop(llm, messages)

            verdict: ValidationVerdict = llm.with_structured_output(
                ValidationVerdict
            ).invoke([*conversation, HumanMessage(content=VERDICT_REQUEST)])

            judged = verdict_to_findings(verdict, aggregated, conn)
        except (ValidationError, Exception) as exc:  # noqa: BLE001
            elapsed = (perf_counter() - started) * 1000
            entries.append(
                log(
                    NODE,
                    "adjudication_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    severity=Severity.CRITICAL,
                    elapsed_ms=elapsed,
                )
            )
            # The deterministic findings are still sound and must not be lost.
            return {
                "findings": deterministic,
                "errors": [f"validator adjudication failed: {type(exc).__name__}: {exc}"],
                "audit_log": entries,
            }

    elapsed = (perf_counter() - started) * 1000

    if performed:
        entries.append(
            log(NODE, "tools_called", detail="; ".join(performed), severity=Severity.INFO)
        )
    entries.append(
        log(
            NODE,
            "adjudicated",
            detail=f"{len(judged)} finding(s) from judgment. {verdict.summary}",
            severity=Severity.INFO,
            elapsed_ms=elapsed,
        )
    )

    return {"findings": deterministic + judged, "audit_log": entries}


register_mock_handler(Purpose.VALIDATOR, mock_validate)
