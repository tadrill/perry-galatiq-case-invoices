"""Approval policy: the rules that bind regardless of what any agent concludes.

The approver is a language model reasoning about an invoice, and its judgment is the
useful part. But some outcomes must not be reachable by reasoning at all. An invoice
billing a discontinued SKU from an unlisted vendor with "pay immediately, wire transfer
preferred" in the notes must not be approvable, however persuasive a case could be made
for it. Prompt text is guidance; this module is enforcement.

The floor only ever makes an outcome more conservative. It can downgrade an approval to
a review, and it never upgrades anything -- a model that decides to reject is not
overruled into paying.
"""

from __future__ import annotations

from .schemas import ApprovalDecision, Decision, Finding, InvoiceData, Severity

#: Above this, an invoice needs a human's attention rather than an automatic release.
SCRUTINY_THRESHOLD = 10_000.0


def critical_findings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity is Severity.CRITICAL]


def warning_findings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity is Severity.WARNING]


def exceeds_threshold(invoice: InvoiceData | None) -> bool:
    return bool(invoice and invoice.total is not None and invoice.total >= SCRUTINY_THRESHOLD)


def policy_flags(invoice: InvoiceData | None, findings: list[Finding]) -> list[str]:
    """Standing facts about this invoice that the approver must weigh."""
    flags: list[str] = []

    if exceeds_threshold(invoice) and invoice and invoice.total is not None:
        flags.append(
            f"total {invoice.total:,.2f} {invoice.currency} is at or above the "
            f"{SCRUTINY_THRESHOLD:,.0f} scrutiny threshold"
        )
    if critical := critical_findings(findings):
        flags.append(f"{len(critical)} critical finding(s)")
    if warnings := warning_findings(findings):
        flags.append(f"{len(warnings)} warning(s)")
    if invoice is None:
        flags.append("extraction produced no invoice")
    elif invoice.total is None:
        flags.append("invoice states no total")

    return flags


def verify_citations(
    decision: ApprovalDecision, findings: list[Finding]
) -> tuple[ApprovalDecision, list[str]]:
    """Check that the decision rests on findings that actually exist.

    This exists because of a real failure. On a live run the approver refused a $22,562.80
    invoice as a "duplicate resubmission of an invoice already processed on 2026-09-20" --
    against an empty ledger, with no duplicate finding anywhere in its input. The rationale
    was fluent, specific, and about an event that never happened. A human reading it would
    have had no way to tell.

    Prose can't be validated, but a citation can. The approver names the codes it relied
    on; anything it names that was not in its input is a fabrication, and a decision built
    on one is not a decision. It goes to a human instead -- including a rejection, because
    refusing a supplier over an invented duplicate is its own kind of harm.

    Returns:
        The decision (possibly redirected) and the list of unsupported codes.
    """
    available = {finding.code for finding in findings}
    unsupported = [code for code in decision.driving_findings if code not in available]

    if not unsupported:
        return decision, []

    redirected = decision.model_copy(
        update={
            "decision": Decision.NEEDS_REVIEW,
            "rationale": (
                f"{decision.rationale}\n\nUNSUPPORTED CITATION: this decision cited "
                f"{', '.join(unsupported)}, which was not among the findings for this "
                f"invoice. The stated grounds cannot be verified, so it is held for a "
                f"human rather than acted on."
            ),
            "policy_flags": [*decision.policy_flags, "unsupported_citation"],
        }
    )
    return redirected, unsupported


def apply_floor(
    decision: ApprovalDecision, invoice: InvoiceData | None, findings: list[Finding]
) -> ApprovalDecision:
    """Force a decision down to the most permissive outcome policy actually allows.

    Downgrades only. A rejection stands; an approval that policy forbids becomes a
    review rather than being silently converted into a rejection, because "a human must
    look at this" is the honest outcome and rejecting outright would be a judgment the
    policy has not earned.
    """
    if decision.decision is not Decision.APPROVED:
        return decision

    blocked: str | None = None

    if invoice is None:
        blocked = "No invoice was extracted, so there is nothing to approve."
    elif invoice.total is None:
        blocked = "The invoice states no total, so no amount can be released."
    elif critical := critical_findings(findings):
        blocked = (
            f"Policy forbids approving an invoice with critical findings "
            f"({', '.join(f.code for f in critical)})."
        )
    elif exceeds_threshold(invoice) and warning_findings(findings):
        blocked = (
            f"Policy forbids auto-approving an invoice at or above "
            f"{SCRUTINY_THRESHOLD:,.0f} while warnings remain open."
        )

    if blocked is None:
        return decision

    return decision.model_copy(
        update={
            "decision": Decision.NEEDS_REVIEW,
            "rationale": f"{decision.rationale}\n\nPOLICY OVERRIDE: {blocked}",
            "policy_flags": [*decision.policy_flags, "policy_floor_applied"],
        }
    )
