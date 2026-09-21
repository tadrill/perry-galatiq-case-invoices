"""Graph state: the only channel through which the agents communicate.

LangGraph nodes do not call each other. Each receives the whole state and returns a
partial update that LangGraph merges in; the edges decide who runs next. So the extractor
writes `invoice`, and the validator reads it.

Two details carry weight:

*Reducers.* The default merge overwrites, which is right for single-author fields like
`invoice`. Fields several nodes contribute to are annotated with `operator.add` so they
append instead -- without that, the approver's log update would silently discard
everything the validator wrote.

*Counters.* `extraction_attempts` and `critique_rounds` are what make the self-correction
loops terminate. An invoice whose arithmetic genuinely cannot reconcile (INV-1009 states a
subtotal of 1000 against line items summing to -250) would otherwise cycle forever.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .schemas import (
    ApprovalDecision,
    Critique,
    Finding,
    InvoiceData,
    LogEntry,
    PaymentResult,
    ReconciliationResult,
    Severity,
)


class InvoiceState(TypedDict, total=False):
    """State for one invoice's trip through the graph.

    `total=False` so every node can return only the keys it owns.
    """

    # --- Inputs, set once before the graph runs -------------------------------
    source_path: str
    source_format: str
    raw_text: str

    # --- Stage outputs, each written by one node ------------------------------
    invoice: InvoiceData | None
    reconciliation: ReconciliationResult | None
    decision: ApprovalDecision | None
    critique: Critique | None
    payment: PaymentResult | None

    # --- Accumulated across nodes ---------------------------------------------
    findings: Annotated[list[Finding], operator.add]
    errors: Annotated[list[str], operator.add]
    audit_log: Annotated[list[LogEntry], operator.add]

    # --- Control flow ----------------------------------------------------------
    extraction_attempts: int
    critique_rounds: int


def initial_state(*, source_path: str, source_format: str, raw_text: str) -> InvoiceState:
    """Build the starting state for one invoice."""
    return InvoiceState(
        source_path=source_path,
        source_format=source_format,
        raw_text=raw_text,
        invoice=None,
        reconciliation=None,
        decision=None,
        critique=None,
        payment=None,
        findings=[],
        errors=[],
        audit_log=[],
        extraction_attempts=0,
        critique_rounds=0,
    )


def log(
    node: str,
    event: str,
    *,
    detail: str | None = None,
    severity: Severity = Severity.INFO,
    elapsed_ms: float | None = None,
) -> LogEntry:
    """Build one audit-trail entry.

    Nodes return these in a list so the `operator.add` reducer appends them, leaving a
    complete record of what happened and why -- which is the observability story the
    case asks for, obtained as a side effect of the architecture rather than bolted on.
    """
    return LogEntry(
        node=node, event=event, detail=detail, severity=severity, elapsed_ms=elapsed_ms
    )


def summarize(state: InvoiceState) -> dict[str, Any]:
    """Flatten a finished run into a compact dict for logging or the CLI."""
    invoice = state.get("invoice")
    decision = state.get("decision")
    findings = state.get("findings") or []

    return {
        "source": state.get("source_path"),
        "format": state.get("source_format"),
        "invoice_number": invoice.invoice_number if invoice else None,
        "vendor": invoice.vendor_name if invoice else None,
        "total": invoice.total if invoice else None,
        "currency": invoice.currency if invoice else "USD",
        "decision": decision.decision.value if decision else None,
        "findings": len(findings),
        "critical": sum(1 for f in findings if f.severity is Severity.CRITICAL),
        "extraction_attempts": state.get("extraction_attempts", 0),
        "errors": list(state.get("errors") or []),
    }
