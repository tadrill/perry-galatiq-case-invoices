"""Offline stand-in for the approver and its critic.

Three call shapes, distinguished by the schema requested and by what is in the prompt:
a first-pass decision, a critique of that draft, and a revision responding to it.

The draft and the critique apply genuinely different reasoning, which is the point. The
draft counts severities and defaults to caution: anything critical is held for a human.
The critic asks a different question -- whether the indicators are *conclusive* rather
than merely serious -- and an invoice for a discontinued part, from a vendor nobody has
approved, with "wire transfer preferred" in the notes, is not an ambiguous case someone
needs to think about. It is a rejection.

So the loop can genuinely overturn a draft here rather than theatrically agreeing with
it. It is still a scoring table standing in for judgment, and the real model weighs these
against each other rather than adding up points.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage

from ..policy import SCRUTINY_THRESHOLD
from ..schemas import ApprovalDecision, Critique, Decision, Severity
from ..thresholds import MOCK_CONCLUSIVE_SCORE as CONCLUSIVE_SCORE
from ..thresholds import MOCK_FRAUD_WEIGHTS as FRAUD_WEIGHTS

_FINDING_RE = re.compile(r"^\s*\[(CRITICAL|WARNING|INFO)\]\s+(\S+?):", re.MULTILINE)
_TOTAL_RE = re.compile(r"^\s*total:\s+([\d,]+\.\d{2})\s+([A-Z]{3})", re.MULTILINE)
_DRAFT_RE = re.compile(r"^\s*decision:\s+(\w+)", re.MULTILINE)


def _prompt_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


def _findings(prompt: str) -> list[tuple[Severity, str]]:
    return [
        (Severity(severity.lower()), code) for severity, code in _FINDING_RE.findall(prompt)
    ]


def _total(prompt: str) -> float | None:
    match = _TOTAL_RE.search(prompt)
    return float(match.group(1).replace(",", "")) if match else None


def _fraud_score(findings: list[tuple[Severity, str]]) -> int:
    score = sum(FRAUD_WEIGHTS.get(code, 0) for _, code in findings)
    score += 2 * sum(
        1 for severity, code in findings if code == "risk.signal" and severity is Severity.CRITICAL
    )
    return score


def _decide(prompt: str) -> ApprovalDecision:
    findings = _findings(prompt)
    criticals = [code for severity, code in findings if severity is Severity.CRITICAL]
    warnings = [code for severity, code in findings if severity is Severity.WARNING]
    total = _total(prompt)
    revising = "REVIEWER'S CONCERNS" in prompt
    score = _fraud_score(findings)

    flags: list[str] = []
    if total is not None and total >= SCRUTINY_THRESHOLD:
        flags.append("above_scrutiny_threshold")

    if revising and score >= CONCLUSIVE_SCORE:
        return ApprovalDecision(
            decision=Decision.REJECTED,
            rationale=(
                f"On review the reviewer is right. {', '.join(sorted(set(criticals)))} do "
                f"not point at a clerical problem someone needs to untangle; together they "
                f"describe an invoice that should never be paid. Rejecting outright and "
                f"logging the reasoning."
            ),
            driving_findings=sorted(set(criticals)),
            policy_flags=flags,
            revised=True,
        )

    if criticals:
        return ApprovalDecision(
            decision=Decision.NEEDS_REVIEW,
            rationale=(
                f"{len(criticals)} critical finding(s) -- "
                f"{', '.join(sorted(set(criticals)))}. Policy forbids releasing payment "
                f"with any of these outstanding, so this is held for a human."
            ),
            driving_findings=sorted(set(criticals)),
            policy_flags=flags,
            revised=revising,
        )

    if warnings:
        return ApprovalDecision(
            decision=Decision.NEEDS_REVIEW,
            rationale=(
                f"No fraud indicators, but {len(warnings)} warning(s) remain open "
                f"({', '.join(sorted(set(warnings)))}). These look like supply and "
                f"purchasing questions rather than reasons to refuse, so they belong with "
                f"a buyer before payment goes out."
            ),
            driving_findings=sorted(set(warnings)),
            policy_flags=flags,
            revised=revising,
        )

    if total is not None and total >= SCRUTINY_THRESHOLD:
        return ApprovalDecision(
            decision=Decision.APPROVED,
            rationale=(
                f"Every check passed. The total of {total:,.2f} is above the "
                f"{SCRUTINY_THRESHOLD:,.0f} scrutiny threshold, so it was examined more "
                f"closely, but size alone with a clean record is not a reason to withhold "
                f"payment from an approved vendor."
            ),
            policy_flags=flags,
            revised=revising,
        )

    return ApprovalDecision(
        decision=Decision.APPROVED,
        rationale=(
            "Arithmetic reconciles, every item resolves to a catalog SKU within stock, "
            "the vendor is approved and nothing was flagged. Cleared for payment."
        ),
        policy_flags=flags,
        revised=revising,
    )


def _critique(prompt: str) -> Critique:
    findings = _findings(prompt)
    draft_match = _DRAFT_RE.search(prompt)
    draft = draft_match.group(1) if draft_match else ""
    criticals = [code for severity, code in findings if severity is Severity.CRITICAL]
    score = _fraud_score(findings)

    if draft == Decision.NEEDS_REVIEW.value and score >= CONCLUSIVE_SCORE:
        return Critique(
            concerns=[
                (
                    f"The draft treats this as ambiguous, but "
                    f"{', '.join(sorted(set(criticals)))} corroborate each other rather "
                    f"than standing alone."
                ),
                (
                    "Sending a conclusive case to a human queue delays the refusal and "
                    "wastes a reviewer's time on a decision the evidence has already made."
                ),
            ],
            recommend_revision=True,
            reasoning=(
                "The indicators here are mutually reinforcing, not independent question "
                "marks. This warrants outright rejection."
            ),
        )

    if draft == Decision.REJECTED.value and not criticals:
        return Critique(
            concerns=[
                "The draft rejects an invoice with no critical findings against it.",
                (
                    "Refusing a supplier over warnings alone damages a working "
                    "relationship for something a buyer could resolve in an email."
                ),
            ],
            recommend_revision=True,
            reasoning="Too harsh for what was actually found; this should go to review.",
        )

    if draft == Decision.APPROVED.value and criticals:
        return Critique(
            concerns=[f"Approves despite {len(criticals)} critical finding(s)."],
            recommend_revision=True,
            reasoning="Policy does not permit approval while these are outstanding.",
        )

    return Critique(
        concerns=[],
        recommend_revision=False,
        reasoning=(
            "The decision follows from the findings and the standing rules, and the "
            "rationale names the evidence that drove it. Nothing to add."
        ),
    )


def mock_approve(messages: list[BaseMessage], schema: Any = None, tools: Any = None) -> Any:
    """Offline approver and critic. Signature matches the MockHandler contract."""
    prompt = _prompt_text(messages)

    if schema is Critique:
        return _critique(prompt)
    return _decide(prompt)
