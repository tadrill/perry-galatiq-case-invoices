"""Mock payment API and the terminal node of the graph.

`mock_payment` keeps the signature the assignment specified so it stays recognizable,
with a reference number added because a payment you cannot later identify is not much of
a record.

The node does the other half of the job: whatever the decision was, the run is written to
the ledger. That is what makes duplicate detection work at all -- INV-1004 can only be
recognized as a resubmission because the first pass recorded it. Rejections are recorded
too, so a vendor who is refused and resubmits under a new number is caught by the content
hash rather than starting from a clean slate.
"""

from __future__ import annotations

import uuid
from time import perf_counter
from typing import Any

from .config import get_settings
from .inventory import compute_line_hash, connect, record_invoice
from .schemas import Decision, PaymentResult, Severity
from .state import InvoiceState, log

NODE = "payment"

from .thresholds import MAX_REASON_CHARS


def mock_payment(vendor: str, amount: float, currency: str = "USD") -> dict[str, Any]:
    """Stand-in for the banking API.

    Returns:
        A status envelope with a payment reference for reconciliation.
    """
    return {
        "status": "success",
        "vendor": vendor,
        "amount": amount,
        "currency": currency,
        "reference": f"PAY-{uuid.uuid4().hex[:12].upper()}",
    }


def _ledger_entry(state: InvoiceState, decision_value: str, reason: str) -> None:
    """Record the run, whether or not it got as far as producing an invoice.

    A run that fails before extraction has no invoice number, so it writes a row with a
    null one and whatever went wrong. A failure must not be less visible than a success in
    the one table whose job is to say what happened.
    """
    invoice = state.get("invoice")
    errors = list(state.get("errors") or [])
    backend = get_settings().llm_mode

    if invoice is None:
        detail = " | ".join(errors) or reason
        with connect(get_settings().resolved_db_path) as conn:
            record_invoice(
                conn,
                invoice_number=None,
                source_path=state.get("source_path"),
                decision="error",
                reason=detail[:MAX_REASON_CHARS],
                llm_mode=backend,
            )
        return

    digest = compute_line_hash(
        [(line.item, line.quantity, line.unit_price or 0.0) for line in invoice.line_items]
    )
    # Errors that did not stop the run still belong on the record.
    detail = reason if not errors else f"{reason} | errors: {' | '.join(errors)}"

    with connect(get_settings().resolved_db_path) as conn:
        record_invoice(
            conn,
            invoice_number=invoice.invoice_number,
            revision=invoice.revision,
            vendor_name=invoice.vendor_name,
            total=invoice.total,
            currency=invoice.currency,
            line_hash=digest,
            source_path=state.get("source_path"),
            decision=decision_value,
            reason=detail[:MAX_REASON_CHARS],
            llm_mode=backend,
        )


def finalize_node(state: InvoiceState) -> dict[str, Any]:
    """LangGraph node. Pays if approved, records the outcome either way."""
    started = perf_counter()
    invoice = state.get("invoice")
    decision = state.get("decision")

    if decision is None:
        reason = "Pipeline ended without a decision."
        _ledger_entry(state, "error", reason)
        return {
            "payment": PaymentResult(status="not_attempted", detail=reason),
            "audit_log": [
                log(NODE, "no_decision", detail=reason, severity=Severity.CRITICAL)
            ],
        }

    if decision.decision is not Decision.APPROVED:
        _ledger_entry(state, decision.decision.value, decision.rationale)
        elapsed = (perf_counter() - started) * 1000
        return {
            "payment": PaymentResult(
                status="withheld",
                vendor=invoice.vendor_name if invoice else None,
                amount=invoice.total if invoice else None,
                currency=invoice.currency if invoice else "USD",
                detail=decision.rationale,
            ),
            "audit_log": [
                log(
                    NODE,
                    "payment_withheld",
                    detail=(
                        f"{decision.decision.value}: {decision.rationale.splitlines()[0]}"
                    ),
                    severity=Severity.WARNING,
                    elapsed_ms=elapsed,
                )
            ],
        }

    # Approved. Policy has already guaranteed a vendor and a total exist.
    vendor = (invoice.vendor_name if invoice else None) or "UNKNOWN"
    amount = (invoice.total if invoice else None) or 0.0
    currency = invoice.currency if invoice else "USD"

    receipt = mock_payment(vendor, amount, currency)
    _ledger_entry(state, "approved", decision.rationale)
    elapsed = (perf_counter() - started) * 1000

    return {
        "payment": PaymentResult(
            status=receipt["status"],
            vendor=vendor,
            amount=amount,
            currency=currency,
            reference=receipt["reference"],
            detail=f"Paid {amount:,.2f} {currency} to {vendor}",
        ),
        "audit_log": [
            log(
                NODE,
                "payment_sent",
                detail=f"{receipt['reference']}: {amount:,.2f} {currency} to {vendor}",
                severity=Severity.INFO,
                elapsed_ms=elapsed,
            )
        ],
    }
