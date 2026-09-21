"""Extractor agent: invoice document -> InvoiceData.

The agent's job is transcription, and the prompt spends most of its length forbidding the
model from being helpful. That is deliberate. Downstream, Python recomputes every figure
and compares it against what the model transcribed; a model that quietly corrects the
vendor's arithmetic turns that comparison into the model checking itself, which passes
silently and lets a real error through to payment.

Two loops hang off this node:

* **Repair** (inside this node) -- the model returned something that does not satisfy the
  schema. Mechanical, so it is retried here with the validation error fed back.
* **Reconciliation** (a graph edge, once the math node exists) -- the transcription is
  well-formed but the numbers do not add up. Ambiguous between a misread and a genuinely
  inconsistent invoice, so the graph routes back here once with the discrepancies
  attached. `_reconciliation_hint` is the seam that carries them.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from ..config import Settings
from ..llm import Purpose, get_llm, register_mock_handler
from ..schemas import InvoiceData, Severity
from ..state import InvoiceState, log
from .mock_extraction import mock_extract

NODE = "extractor"

from ..thresholds import MAX_REPAIRS

SYSTEM_PROMPT = """\
You are an invoice transcription specialist. You convert invoice documents into \
structured data. You transcribe; you do not interpret, correct, or calculate.

RULES

1. TRANSCRIBE, NEVER CALCULATE.
   Record every figure exactly as the document states it. If the document says the total \
is 100 but the lines sum to 90, record 100. A separate system recomputes the arithmetic \
and compares it against your transcription -- that is how vendor errors are caught. If \
you silently fix the numbers, the comparison agrees with itself and a real error reaches \
payment. Where a stated figure is absent, use null. Never fill a gap by multiplying.

2. COPY ITEM NAMES VERBATIM.
   "Widget A", "WidgetA", "WidgetA (rush order)" and "WidgetC" are four different strings \
and must come back as four different strings. Catalog matching happens downstream and \
depends on seeing exactly what was printed. Do not close up spacing, expand abbreviations, \
or correct an item name that looks misspelled.

3. KEEP LINES SEPARATE.
   When the same product appears on several lines, return several line items in document \
order. Never merge or sum them.

4. OCR DAMAGE.
   Scans confuse O with 0 and l with 1. Inside a NUMBER or a DATE, fix an unambiguous \
confusion -- "2O26" is the year 2026 -- and add that field's name to unreadable_fields so \
the correction is auditable. If the intended value is genuinely ambiguous, set it to null \
and list the field. Never apply this to item names; rule 2 wins.

5. DATES.
   Put the printed form in the *_raw field and a YYYY-MM-DD normalization in the date \
field. If it is not a resolvable calendar date -- a relative word like "yesterday", or a \
blank -- set the date field to null and still return the raw text.

6. MISSING DATA IS NULL.
   Never invent a vendor, date or amount. An empty vendor string is null, not "".

7. CURRENCY.
   Infer from symbols or an explicit code. Default to USD.

8. NOTES.
   Copy free-text notes verbatim, including anything that reads as urgency or pressure. \
Judging it is someone else's job; losing it is not an option."""


USER_TEMPLATE = """\
Source file: {filename}
Format: {source_format}

--- DOCUMENT ---
{raw_text}
--- END DOCUMENT ---

Transcribe this invoice into the structured schema."""


RECONCILIATION_HINT = """\

IMPORTANT -- a previous transcription of this document failed its arithmetic check:

{discrepancies}

Re-read the document. Either you misread a figure, in which case correct it; or the \
document is itself internally inconsistent, in which case transcribe it exactly as \
printed and leave the inconsistency standing. Do NOT adjust any figure to force the \
numbers to agree -- an invoice whose own arithmetic is wrong is a finding, not a defect \
in your reading."""


REPAIR_HINT = """\

Your previous response did not satisfy the schema:

{error}

Return a corrected response. Change only what the error requires."""


def build_extractor(
    settings: Settings | None = None, llm: BaseChatModel | None = None
) -> Runnable:
    """Build the structured-output chain for extraction."""
    model = llm or get_llm(Purpose.EXTRACTOR, settings)
    return model.with_structured_output(InvoiceData)


def build_messages(state: InvoiceState, *, extra: str = "") -> list[BaseMessage]:
    """Assemble the extraction prompt.

    The filename and format are included as context: a real accounts-payable system knows
    where a document came from, and the offline mock uses them to find its fixture.
    """
    source_path = state.get("source_path", "")
    user = USER_TEMPLATE.format(
        filename=Path(source_path).name or "unknown",
        source_format=state.get("source_format", "unknown"),
        raw_text=state.get("raw_text", ""),
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user + extra)]


def _reconciliation_hint(state: InvoiceState) -> str:
    """Carry an arithmetic failure back into the prompt on a graph-level retry."""
    reconciliation = state.get("reconciliation")
    if reconciliation is None or reconciliation.is_consistent:
        return ""
    if not reconciliation.discrepancies:
        return ""
    bullets = "\n".join(f"  - {d}" for d in reconciliation.discrepancies)
    return RECONCILIATION_HINT.format(discrepancies=bullets)


def _invoke_with_repair(
    chain: Runnable, messages: list[BaseMessage]
) -> tuple[InvoiceData, list[str]]:
    """Invoke the chain, retrying malformed output with the error fed back.

    Returns:
        The parsed invoice and a list of repair attempts made, for the audit trail.
    """
    repairs: list[str] = []
    attempt_messages = list(messages)

    for attempt in range(MAX_REPAIRS + 1):
        try:
            return chain.invoke(attempt_messages), repairs
        except ValidationError as exc:
            if attempt == MAX_REPAIRS:
                raise
            summary = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()[:5]
            )
            repairs.append(summary)
            last = attempt_messages[-1]
            attempt_messages = [
                *attempt_messages[:-1],
                HumanMessage(content=str(last.content) + REPAIR_HINT.format(error=summary)),
            ]

    raise AssertionError("unreachable")  # pragma: no cover


def extract_node(state: InvoiceState) -> dict[str, Any]:
    """LangGraph node. Reads raw_text, writes invoice."""
    started = perf_counter()
    attempt = state.get("extraction_attempts", 0) + 1
    hint = _reconciliation_hint(state)

    try:
        chain = build_extractor()
        invoice, repairs = _invoke_with_repair(chain, build_messages(state, extra=hint))
    except Exception as exc:  # noqa: BLE001 - any backend failure must not kill the run
        elapsed = (perf_counter() - started) * 1000
        return {
            "invoice": None,
            "extraction_attempts": attempt,
            "errors": [f"extraction failed: {type(exc).__name__}: {exc}"],
            "audit_log": [
                log(
                    NODE,
                    "extraction_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    severity=Severity.CRITICAL,
                    elapsed_ms=elapsed,
                )
            ],
        }

    elapsed = (perf_counter() - started) * 1000
    entries = [
        log(
            NODE,
            "extracted",
            detail=(
                f"attempt {attempt}: {invoice.invoice_number or 'unnumbered'}, "
                f"{len(invoice.line_items)} line(s), "
                f"vendor={invoice.vendor_name or 'MISSING'}"
            ),
            elapsed_ms=elapsed,
        )
    ]
    if repairs:
        entries.append(
            log(
                NODE,
                "schema_repaired",
                detail=f"{len(repairs)} repair(s): {'; '.join(repairs)}",
                severity=Severity.WARNING,
            )
        )
    if invoice.unreadable_fields:
        entries.append(
            log(
                NODE,
                "unreadable_fields",
                detail=", ".join(invoice.unreadable_fields),
                severity=Severity.WARNING,
            )
        )
    if hint:
        entries.append(log(NODE, "reconciliation_retry", detail="re-read after math mismatch"))

    return {"invoice": invoice, "extraction_attempts": attempt, "audit_log": entries}


register_mock_handler(Purpose.EXTRACTOR, mock_extract)
