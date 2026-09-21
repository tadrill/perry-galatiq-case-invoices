"""Offline stand-in for the extractor's LLM call.

Resolution order, most faithful first:

1. **Recorded fixture** -- a real Grok extraction saved by `scripts/record_fixtures.py`.
   Replaying genuine model output is the only honest way to exercise the pipeline
   offline, because it reproduces the model's actual quirks rather than my guesses
   about them.
2. **Native JSON parse** -- for the JSON invoices the mapping is exact and mechanical,
   so a parser is strictly better than a guess and needs no recorded run.
3. **Refuse** -- raise, naming the fixture that would fix it. A plausible-looking stub
   here would be indistinguishable from a working pipeline inventing invoice data,
   which is the worst failure this system could have.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from ..config import PROJECT_ROOT
from ..schemas import InvoiceData

FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "extraction"

_SOURCE_RE = re.compile(r"^Source file:\s*(\S+)", re.MULTILINE)
_FORMAT_RE = re.compile(r"^Format:\s*(\w+)", re.MULTILINE)
_DOCUMENT_RE = re.compile(r"--- DOCUMENT ---\n(.*)\n--- END DOCUMENT ---", re.DOTALL)


class MockExtractionUnavailable(RuntimeError):
    """No fixture and no deterministic parse for this document."""


def _prompt_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


def _source_stem(prompt: str) -> str | None:
    match = _SOURCE_RE.search(prompt)
    return Path(match.group(1)).stem if match else None


def _fixture_keys(prompt: str) -> list[str]:
    """Fixture names to try, most specific first.

    invoice_1011 exists as both .txt and .pdf. They hold the same invoice, so one
    fixture normally serves both -- but PDF extraction is lossier, so a format-specific
    fixture (`invoice_1011.pdf.json`) takes precedence when one has been recorded.
    """
    stem = _source_stem(prompt)
    if not stem:
        return []
    fmt = _FORMAT_RE.search(prompt)
    return [f"{stem}.{fmt.group(1)}", stem] if fmt else [stem]


def _document_body(prompt: str) -> str:
    match = _DOCUMENT_RE.search(prompt)
    return match.group(1) if match else prompt


def _load_fixture(stem: str) -> dict[str, Any] | None:
    """Load a fixture, dropping its provenance metadata.

    Underscore-prefixed keys record where the fixture came from -- a recorded Grok run
    or a hand transcription. InvoiceData forbids extra fields, so they are stripped
    before validation rather than loosening the schema for bookkeeping.
    """
    path = FIXTURE_DIR / f"{stem}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def _parse_json_invoice(body: str) -> dict[str, Any] | None:
    """Map a JSON invoice onto InvoiceData. Exact, so no model needed."""
    try:
        raw = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, dict) or "line_items" not in raw:
        return None

    vendor = raw.get("vendor")
    if isinstance(vendor, dict):
        vendor_name, vendor_address = vendor.get("name"), vendor.get("address")
    else:
        vendor_name, vendor_address = vendor, None

    return {
        "invoice_number": raw.get("invoice_number"),
        "revision": raw.get("revision"),
        "vendor_name": vendor_name,
        "vendor_address": vendor_address,
        "invoice_date_raw": raw.get("date"),
        "invoice_date": raw.get("date"),
        "due_date_raw": raw.get("due_date"),
        "due_date": raw.get("due_date"),
        "line_items": [
            {
                "item": line.get("item"),
                "quantity": line.get("quantity"),
                "unit_price": line.get("unit_price"),
                "amount": line.get("amount"),
                "note": line.get("note"),
            }
            for line in raw.get("line_items") or []
        ],
        "subtotal": raw.get("subtotal"),
        "tax_rate": raw.get("tax_rate"),
        "tax_amount": raw.get("tax_amount"),
        "shipping": raw.get("shipping"),
        "total": raw.get("total"),
        "currency": raw.get("currency") or "USD",
        "payment_terms": raw.get("payment_terms"),
        "notes": raw.get("notes"),
        "unreadable_fields": [],
    }


def mock_extract(messages: list[BaseMessage], schema: Any = None, tools: Any = None) -> InvoiceData:
    """Produce an InvoiceData offline. Signature matches the MockHandler contract."""
    prompt = _prompt_text(messages)
    stem = _source_stem(prompt)

    for key in _fixture_keys(prompt):
        fixture = _load_fixture(key)
        if fixture is not None:
            return InvoiceData.model_validate(fixture)

    parsed = _parse_json_invoice(_document_body(prompt).strip())
    if parsed is not None:
        return InvoiceData.model_validate(parsed)

    raise MockExtractionUnavailable(
        f"No offline extraction available for {stem or 'this document'}. "
        f"Record one with: python scripts/record_fixtures.py --invoice {stem or '<stem>'} "
        f"(needs XAI_API_KEY), or run with LLM_MODE=grok. "
        f"Fixtures live in {FIXTURE_DIR.relative_to(PROJECT_ROOT)}/."
    )
