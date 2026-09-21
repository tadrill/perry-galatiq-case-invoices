"""Approver agent: VP-level review, with a critique pass that can overturn it.

Two nodes and an edge between them, rather than one node that reflects internally. A
model asked to reconsider its own answer in the same breath tends to restate it; asked
to argue against a draft as a separate task, with the draft in front of it, it finds
what it glossed over. Making that a real graph edge also puts the revision in the audit
trail, where a reviewer can see the decision change and why.

Whatever the agent concludes, `invoice_agents.policy` gets the last word. The prompt
tells the model what the rules are; the policy floor enforces them, because an invoice
from an unlisted vendor for a discontinued part must not be approvable no matter how
persuasive an argument could be constructed for it.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..config import Settings
from ..llm import Purpose, get_llm, register_mock_handler
from ..policy import SCRUTINY_THRESHOLD, apply_floor, policy_flags, verify_citations
from ..schemas import (
    ApprovalDecision,
    Critique,
    Decision,
    Finding,
    InvoiceData,
    ReconciliationResult,
    Severity,
)
from ..state import InvoiceState, log
from .mock_approval import mock_approve

NODE = "approver"
CRITIQUE_NODE = "critic"

#: Critique passes allowed. Two means the draft gets reviewed, may be revised once, and
#: the revision gets reviewed -- after which the loop ends whether or not the critic is
#: still unhappy. A critic is always able to find one more thing to say.
MAX_CRITIQUE_ROUNDS = 2


SYSTEM_PROMPT = f"""\
You are a VP of Finance at a manufacturing firm, deciding whether an invoice is paid.

Everything factual has been established before it reaches you: the arithmetic was \
recomputed in Python, stock and vendor records were checked against the database, and a \
validation specialist ruled on anything ambiguous. You are not re-investigating. You are \
weighing what was found and deciding.

YOUR OPTIONS
  approved      Release payment. Only when nothing material is outstanding.
  rejected      Do not pay, and the reason is conclusive. Reserve this for findings that \
CORROBORATE each other: an unapproved vendor billing for a discontinued part with pressure \
language in the notes, or a confirmed duplicate of an invoice already paid. One anomaly on \
its own is almost never conclusive -- an unknown SKU from an approved vendor whose \
arithmetic is clean is more likely a new product the catalog has not caught up with than \
an attempt at fraud, and refusing a good supplier over it costs the business a relationship.
  needs_review  A human must look before money moves. This is the right answer far more \
often than rejection: a stock shortfall, a pricing anomaly, an unfamiliar item, a revised \
invoice number, a tax line that needs confirming.

STANDING RULES
  - Invoices at or above {SCRUTINY_THRESHOLD:,.0f} get additional scrutiny. Size alone is \
not a reason to refuse, but it removes any benefit of the doubt.
  - A critical finding cannot be approved. Choose between rejected and needs_review.
  - Distinguish a supply problem from a fraud signal. Billing more of a real part than is \
in stock is a purchasing conversation. Billing a discontinued part, from a vendor who is \
not on the approved list, with urgency language in the notes, is not.
  - An invoice whose own arithmetic does not reconcile has a real error in it. It was \
re-read to rule out a misreading before reaching you.

CITE YOUR GROUNDS
  List in `driving_findings` the exact `code` of every finding you relied on, copied from \
the FINDINGS block above. Cite only what is written there. Do not infer a finding that is \
not listed, do not describe a prior submission unless a duplicate finding says so, and do \
not attribute anything to the database that you were not shown. Your citations are checked \
against the input, and a decision resting on a code that was not supplied is discarded and \
sent to a human.

Write the rationale for the person who acts on it. Name the specific findings that drove \
you. Be direct, and do not hedge across all three outcomes."""


CRITIQUE_PROMPT = """\
You are reviewing a colleague's draft decision on an invoice, before it takes effect.

Your job is to argue with it, not to ratify it. Look for:
  - a finding the draft under-weighted or passed over
  - reasoning that does not follow from the evidence
  - a standing rule that was overlooked
  - an outcome that is too harsh for what was actually found, as well as too lenient

Recommend revision ONLY if a concern would change the decision itself. A rationale you \
would have phrased differently is not grounds for revision. If the draft is right, say \
so plainly -- agreeing is a legitimate outcome and manufacturing a concern to look \
thorough is worse than saying nothing."""


REVISION_PROMPT = """\
A reviewer raised concerns about your draft decision.

DRAFT
  decision:  {decision}
  rationale: {rationale}

REVIEWER'S CONCERNS
{concerns}

  reviewer's reasoning: {reasoning}

Decide again, taking these seriously. You are not obliged to agree -- if the reviewer is \
wrong, keep your decision and say why in the rationale. Set `revised` to true only if you \
actually changed the decision."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "FINDINGS\n  None. Every check passed."

    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    ranked = sorted(findings, key=lambda f: order[f.severity])
    lines = [f"FINDINGS ({len(findings)})"]
    lines.extend(f"  [{f.severity.value.upper()}] {f.code}: {f.message}" for f in ranked)
    return "\n".join(lines)


def build_context(
    invoice: InvoiceData | None,
    reconciliation: ReconciliationResult | None,
    findings: list[Finding],
) -> str:
    """The evidence packet the approver decides from."""
    if invoice is None:
        return (
            "INVOICE\n  Extraction failed; no invoice data is available.\n\n"
            + _format_findings(findings)
        )

    total = (
        f"{invoice.total:,.2f} {invoice.currency}" if invoice.total is not None else "MISSING"
    )
    header = "\n".join(
        [
            "INVOICE",
            f"  number: {invoice.invoice_number or 'MISSING'}"
            + (f" (revision {invoice.revision})" if invoice.revision else ""),
            f"  vendor: {invoice.vendor_name or 'MISSING'}",
            f"  total:  {total}",
            f"  due:    {invoice.due_date or invoice.due_date_raw or 'MISSING'}",
            f"  terms:  {invoice.payment_terms or 'not stated'}",
            f"  items:  {len(invoice.line_items)} line(s)",
        ]
    )
    if invoice.notes:
        header += f"\n  notes:  {invoice.notes}"

    arithmetic = "ARITHMETIC\n  not available"
    if reconciliation is not None:
        state = "reconciles" if reconciliation.is_consistent else "DOES NOT RECONCILE"
        arithmetic = f"ARITHMETIC\n  {state}"
        for discrepancy in reconciliation.discrepancies:
            arithmetic += f"\n  - {discrepancy}"

    flags = policy_flags(invoice, findings)
    policy = "POLICY\n" + (
        "\n".join(f"  - {flag}" for flag in flags) or "  Nothing exceptional."
    )

    return "\n\n".join([header, arithmetic, _format_findings(findings), policy])


def build_messages(state: InvoiceState) -> list[BaseMessage]:
    context = build_context(
        state.get("invoice"), state.get("reconciliation"), list(state.get("findings") or [])
    )

    critique = state.get("critique")
    draft = state.get("decision")
    if critique is not None and draft is not None and critique.recommend_revision:
        concerns = "\n".join(f"  - {c}" for c in critique.concerns) or "  (none stated)"
        context += "\n\n" + REVISION_PROMPT.format(
            decision=draft.decision.value,
            rationale=draft.rationale,
            concerns=concerns,
            reasoning=critique.reasoning,
        )
    else:
        context += "\n\nDecide."

    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]


def build_critique_messages(state: InvoiceState) -> list[BaseMessage]:
    draft = state.get("decision")
    context = build_context(
        state.get("invoice"), state.get("reconciliation"), list(state.get("findings") or [])
    )
    draft_block = (
        "DRAFT DECISION\n"
        f"  decision:  {draft.decision.value if draft else 'MISSING'}\n"
        f"  rationale: {draft.rationale if draft else 'MISSING'}"
    )
    return [
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=f"{context}\n\n{draft_block}\n\nReview this draft."),
    ]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def build_approver(settings: Settings | None = None, llm: BaseChatModel | None = None):
    return llm or get_llm(Purpose.APPROVER, settings)


def approve_node(state: InvoiceState) -> dict[str, Any]:
    """LangGraph node. Decides, then lets policy have the last word."""
    started = perf_counter()
    invoice = state.get("invoice")
    findings = list(state.get("findings") or [])
    critique = state.get("critique")
    revising = critique is not None and critique.recommend_revision

    try:
        llm = build_approver()
        decision: ApprovalDecision = llm.with_structured_output(ApprovalDecision).invoke(
            build_messages(state)
        )
    except Exception as exc:  # noqa: BLE001 - never leave an invoice in limbo
        elapsed = (perf_counter() - started) * 1000
        return {
            "decision": ApprovalDecision(
                decision=Decision.NEEDS_REVIEW,
                rationale=(
                    f"The approval agent could not be reached "
                    f"({type(exc).__name__}: {exc}), so this invoice is held for a human "
                    f"rather than defaulted either way."
                ),
                policy_flags=["approver_unavailable"],
            ),
            "errors": [f"approver failed: {type(exc).__name__}: {exc}"],
            "audit_log": [
                log(
                    NODE,
                    "approval_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    severity=Severity.CRITICAL,
                    elapsed_ms=elapsed,
                )
            ],
        }

    raw = decision.decision
    decision = decision.model_copy(
        update={"policy_flags": [*decision.policy_flags, *policy_flags(invoice, findings)]}
    )
    # Grounds before policy: a decision citing something it was never shown is not a
    # decision to apply rules to.
    decision, unsupported = verify_citations(decision, findings)
    decision = apply_floor(decision, invoice, findings)
    elapsed = (perf_counter() - started) * 1000

    errors: list[str] = []
    if unsupported:
        errors.append(
            f"approver cited findings that were not supplied: {', '.join(unsupported)}"
        )

    entries = [
        log(
            NODE,
            "revised" if revising else "decided",
            detail=f"{decision.decision.value}: {decision.rationale.splitlines()[0]}",
            severity=(
                Severity.INFO
                if decision.decision is Decision.APPROVED
                else Severity.WARNING
            ),
            elapsed_ms=elapsed,
        )
    ]
    if unsupported:
        entries.append(
            log(
                NODE,
                "unsupported_citation",
                detail=(
                    f"cited {', '.join(unsupported)}, not among this invoice's findings; "
                    f"decision redirected to a human"
                ),
                severity=Severity.CRITICAL,
            )
        )
    elif decision.decision is not raw:
        entries.append(
            log(
                NODE,
                "policy_floor_applied",
                detail=f"agent said {raw.value}; policy forced {decision.decision.value}",
                severity=Severity.WARNING,
            )
        )

    update: dict[str, Any] = {"decision": decision, "audit_log": entries}
    if errors:
        update["errors"] = errors
    return update


def critique_node(state: InvoiceState) -> dict[str, Any]:
    """LangGraph node. Reviews the draft decision and may send it back."""
    started = perf_counter()
    rounds = state.get("critique_rounds", 0) + 1

    if state.get("decision") is None:
        return {
            "critique": None,
            "critique_rounds": rounds,
            "audit_log": [log(CRITIQUE_NODE, "skipped", detail="no draft to review")],
        }

    try:
        llm = build_approver()
        critique: Critique = llm.with_structured_output(Critique).invoke(
            build_critique_messages(state)
        )
    except Exception as exc:  # noqa: BLE001 - a failed critique must not block the run
        return {
            "critique": None,
            "critique_rounds": rounds,
            "audit_log": [
                log(
                    CRITIQUE_NODE,
                    "critique_failed",
                    detail=f"{type(exc).__name__}: {exc}; keeping the draft decision",
                    severity=Severity.WARNING,
                )
            ],
        }

    elapsed = (perf_counter() - started) * 1000
    verdict = "revision recommended" if critique.recommend_revision else "draft upheld"

    return {
        "critique": critique,
        "critique_rounds": rounds,
        "audit_log": [
            log(
                CRITIQUE_NODE,
                "critiqued",
                detail=f"round {rounds}: {verdict}. {critique.reasoning}",
                severity=(
                    Severity.WARNING if critique.recommend_revision else Severity.INFO
                ),
                elapsed_ms=elapsed,
            )
        ],
    }


def route_after_critique(state: InvoiceState) -> str:
    """Conditional edge: revise the decision, or take it as final."""
    critique = state.get("critique")
    if critique is None or not critique.recommend_revision:
        return "finalize"
    if state.get("critique_rounds", 0) < MAX_CRITIQUE_ROUNDS:
        return "approve"
    return "finalize"


register_mock_handler(Purpose.APPROVER, mock_approve)
