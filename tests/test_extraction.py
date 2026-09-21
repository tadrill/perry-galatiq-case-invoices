"""Tests for schemas, state, document loading, and the extractor node."""

from __future__ import annotations

import json
from datetime import date

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from invoice_agents.agents.extractor import (
    MAX_REPAIRS,
    _invoke_with_repair,
    _reconciliation_hint,
    build_messages,
    extract_node,
)
from invoice_agents.config import PROJECT_ROOT
from invoice_agents.ingestion import DocumentLoadError, load_document
from invoice_agents.schemas import (
    InvoiceData,
    LineItem,
    ReconciliationResult,
    Severity,
)
from invoice_agents.state import InvoiceState, initial_state, log, summarize

INVOICE_DIR = PROJECT_ROOT / "data" / "invoices"


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    """extract_node reads settings at call time; force the offline backend."""
    monkeypatch.setenv("LLM_MODE", "mock")
    from invoice_agents.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def state_for(filename: str) -> InvoiceState:
    document = load_document(INVOICE_DIR / filename)
    return initial_state(
        source_path=str(document.path),
        source_format=document.source_format,
        raw_text=document.raw_text,
    )


# ---------------------------------------------------------------------------
# Schema behaviour
# ---------------------------------------------------------------------------


def test_blank_vendor_becomes_none():
    """INV-1009 ships vendor.name as "". That is missing data, not a vendor named ''."""
    assert InvoiceData(vendor_name="").vendor_name is None
    assert InvoiceData(vendor_name="   ").vendor_name is None


def test_absent_currency_defaults_to_usd():
    assert InvoiceData().currency == "USD"
    assert InvoiceData(currency=None).currency == "USD"
    assert InvoiceData(currency="eur").currency == "EUR"


def test_negative_quantity_is_representable():
    """INV-1009 bills -5 units. The schema must carry it so validation can flag it;
    rejecting it here would turn a finding into a crash."""
    assert LineItem(item="WidgetA", quantity=-5, unit_price=250.0).quantity == -5


def test_stated_amount_is_optional_and_never_derived():
    """INV-1001 prints no line totals; INV-1012's is OCR-damaged. Both give null, and
    nothing in the schema fills the gap by multiplying."""
    line = LineItem(item="WidgetA", quantity=10, unit_price=250.0)
    assert line.amount is None


def test_unresolvable_due_date_is_null_but_raw_is_kept():
    """INV-1003 says 'yesterday' -- unusable as a date, but evidence of pressure."""
    invoice = InvoiceData(due_date_raw="yesterday", due_date=None)
    assert invoice.due_date is None
    assert invoice.due_date_raw == "yesterday"


def test_unparseable_date_string_fails_validation():
    """A model that puts 'yesterday' in the ISO field must be corrected, not accepted."""
    with pytest.raises(ValidationError):
        InvoiceData(due_date="yesterday")


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        InvoiceData(invented_field="nonsense")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_initial_state_is_complete():
    state = initial_state(source_path="a.txt", source_format="txt", raw_text="x")
    assert state["extraction_attempts"] == 0
    assert state["findings"] == []
    assert state["invoice"] is None


def test_summarize_survives_a_failed_extraction():
    """The CLI must render a run that never produced an invoice."""
    state = initial_state(source_path="a.txt", source_format="txt", raw_text="x")
    summary = summarize(state)
    assert summary["invoice_number"] is None
    assert summary["decision"] is None
    assert summary["findings"] == 0


def test_log_entries_carry_severity():
    entry = log("extractor", "extracted", detail="ok", severity=Severity.WARNING)
    assert entry.node == "extractor"
    assert entry.severity is Severity.WARNING


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [
        ("invoice_1001.txt", "txt"),
        ("invoice_1004.json", "json"),
        ("invoice_1006.csv", "csv"),
        ("invoice_1014.xml", "xml"),
        ("invoice_1011.pdf", "pdf"),
    ],
)
def test_every_supplied_format_loads(filename, expected_format):
    document = load_document(INVOICE_DIR / filename)
    assert document.source_format == expected_format
    assert not document.is_empty


def test_pdf_extraction_preserves_ocr_damage():
    """The pipeline must see the damage; a loader that 'helpfully' cleaned it would
    destroy the signal the validator needs."""
    document = load_document(INVOICE_DIR / "invoice_1012.pdf")
    assert "2O26" in document.raw_text  # letter O in the year
    assert "Distributers" in document.raw_text


def test_missing_file_is_actionable():
    with pytest.raises(DocumentLoadError, match="No such invoice file"):
        load_document(INVOICE_DIR / "invoice_9999.txt")


def test_unsupported_format_lists_what_is_supported(tmp_path):
    bad = tmp_path / "invoice.docx"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="Supported:"):
        load_document(bad)


def test_empty_document_is_rejected(tmp_path):
    empty = tmp_path / "invoice.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="no text"):
        load_document(empty)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_carries_source_and_document():
    state = state_for("invoice_1001.txt")
    messages = build_messages(state)
    body = str(messages[-1].content)

    assert "Source file: invoice_1001.txt" in body
    assert "Format: txt" in body
    assert "Widgets Inc." in body


def test_no_reconciliation_hint_when_arithmetic_is_fine():
    state = state_for("invoice_1001.txt")
    state["reconciliation"] = ReconciliationResult(is_consistent=True)
    assert _reconciliation_hint(state) == ""


def test_reconciliation_hint_forbids_forcing_agreement():
    """The retry must not become 'fudge until it balances' -- that would erase exactly
    the vendor errors the recomputation exists to surface."""
    state = state_for("invoice_1001.txt")
    state["reconciliation"] = ReconciliationResult(
        is_consistent=False, discrepancies=["subtotal: stated 1000.00, computed -250.00"]
    )

    hint = _reconciliation_hint(state)
    assert "stated 1000.00" in hint
    assert "Do NOT adjust" in hint
    assert "inconsistent" in hint


# ---------------------------------------------------------------------------
# Schema-repair loop
# ---------------------------------------------------------------------------


class FlakyChain:
    """Fails validation `failures` times, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(str(messages[-1].content))
        if len(self.calls) <= self.failures:
            InvoiceData.model_validate({"due_date": "yesterday"})
        return InvoiceData(invoice_number="INV-1001")


def test_malformed_output_is_repaired_with_the_error_fed_back():
    chain = FlakyChain(failures=1)
    invoice, repairs = _invoke_with_repair(chain, [HumanMessage(content="extract")])

    assert invoice.invoice_number == "INV-1001"
    assert len(repairs) == 1
    assert "due_date" in repairs[0]
    assert "did not satisfy the schema" in chain.calls[1]


def test_repair_loop_is_bounded():
    """Without a bound, a model that cannot satisfy the schema retries forever."""
    chain = FlakyChain(failures=MAX_REPAIRS + 1)
    with pytest.raises(ValidationError):
        _invoke_with_repair(chain, [HumanMessage(content="extract")])
    assert len(chain.calls) == MAX_REPAIRS + 1


def test_clean_output_needs_no_repair():
    chain = FlakyChain(failures=0)
    _, repairs = _invoke_with_repair(chain, [HumanMessage(content="extract")])
    assert repairs == []
    assert len(chain.calls) == 1


# ---------------------------------------------------------------------------
# The node, end to end, offline
# ---------------------------------------------------------------------------


def test_repeated_skus_stay_separate():
    """INV-1013 bills WidgetA on three lines. Merging them here would hide the stock
    breach, since no single line exceeds what is on the shelf."""
    update = extract_node(state_for("invoice_1013.json"))
    invoice = update["invoice"]

    assert len(invoice.line_items) == 8
    widget_a = [line for line in invoice.line_items if line.item == "WidgetA"]
    assert len(widget_a) == 3
    assert sum(line.quantity for line in widget_a) == 22


def test_negative_quantity_survives_extraction():
    update = extract_node(state_for("invoice_1009.json"))
    invoice = update["invoice"]

    assert invoice.vendor_name is None
    assert invoice.line_items[0].quantity == -5
    assert invoice.total == -250.0


def test_fixture_backed_text_invoice():
    update = extract_node(state_for("invoice_1003.txt"))
    invoice = update["invoice"]

    assert invoice.vendor_name == "Fraudster LLC"
    assert invoice.due_date_raw == "yesterday"
    assert invoice.due_date is None
    assert "URGENT" in invoice.notes


def test_invoice_number_transcribed_as_printed():
    """INV-1002 prints '1002' with no prefix. Normalizing it here would paper over a
    formatting anomaly the validator may want to see."""
    update = extract_node(state_for("invoice_1002.txt"))
    assert update["invoice"].invoice_number == "1002"


def test_node_increments_attempts_and_logs():
    update = extract_node(state_for("invoice_1004.json"))

    assert update["extraction_attempts"] == 1
    assert any(entry.event == "extracted" for entry in update["audit_log"])


def test_node_returns_an_error_instead_of_raising():
    """A backend failure must not kill the graph mid-run; it becomes state the CLI
    can report and the ledger can record."""
    state = initial_state(
        source_path="data/invoices/invoice_9999.csv", source_format="csv", raw_text="junk"
    )
    update = extract_node(state)

    assert update["invoice"] is None
    assert update["errors"]
    assert update["extraction_attempts"] == 1
    assert any(entry.severity is Severity.CRITICAL for entry in update["audit_log"])


def test_date_fields_parse_to_real_dates():
    update = extract_node(state_for("invoice_1004.json"))
    invoice = update["invoice"]
    assert invoice.invoice_date == date(2026, 1, 22)
    assert invoice.due_date == date(2026, 2, 22)


# ---------------------------------------------------------------------------
# Fixtures on disk
# ---------------------------------------------------------------------------


def test_shipped_fixtures_match_the_schema():
    """A fixture that no longer validates would fail every offline run; catch it here."""
    fixture_dir = PROJECT_ROOT / "data" / "fixtures" / "extraction"
    fixtures = list(fixture_dir.glob("*.json"))
    assert fixtures, "expected at least the hand-transcribed fixtures"

    for path in fixtures:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "_provenance" in payload, f"{path.name} must record where it came from"
        InvoiceData.model_validate({k: v for k, v in payload.items() if not k.startswith("_")})
