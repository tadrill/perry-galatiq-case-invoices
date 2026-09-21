"""Tests for deterministic validation, the tools, and the validator agent."""

from __future__ import annotations

import json
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from invoice_agents.agents.validator import (
    MAX_TOOL_ITERATIONS,
    build_messages,
    run_tool_loop,
    validate_node,
    verdict_to_findings,
)
from invoice_agents.config import get_settings
from invoice_agents.inventory import compute_line_hash, connect, initialize, record_invoice
from invoice_agents.schemas import (
    Finding,
    InvoiceData,
    ItemAdjudication,
    LineItem,
    ReconciliationResult,
    RiskSignal,
    Severity,
    ValidationVerdict,
    VendorAdjudication,
)
from invoice_agents.state import initial_state
from invoice_agents.tools import TOOLS_BY_NAME, VALIDATOR_TOOLS
from invoice_agents.validation import Prevalidation, prevalidate

TODAY = date(2026, 2, 1)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A fresh database, with settings pointed at it and the LLM offline."""
    path = tmp_path / "validation.db"
    initialize(path, reset=True)
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def conn(db):
    with connect(db) as connection:
        yield connection


def invoice(**overrides) -> InvoiceData:
    defaults = {
        "invoice_number": "INV-TEST",
        "vendor_name": "Widgets Inc.",
        "invoice_date": date(2026, 1, 15),
        "due_date": date(2026, 1, 30),
        "payment_terms": "Net 15",
        "line_items": [LineItem(item="WidgetA", quantity=10, unit_price=250.0)],
        "total": 2500.0,
    }
    return InvoiceData(**{**defaults, **overrides})


def codes(result: Prevalidation) -> list[str]:
    return [f.code for f in result.findings]


# ---------------------------------------------------------------------------
# Stock
# ---------------------------------------------------------------------------


def test_stock_is_checked_against_the_aggregate_not_each_line(conn):
    """INV-1013 bills WidgetA on three lines; each passes, the sum does not."""
    result = prevalidate(
        conn,
        invoice(line_items=[LineItem(item="WidgetA", quantity=22, unit_price=250.0)]),
        ReconciliationResult(aggregated_quantities={"WidgetA": 22.0}),
        today=TODAY,
    )
    assert "stock.insufficient" in codes(result)


def test_quantity_within_stock_raises_nothing(conn):
    result = prevalidate(
        conn, invoice(), ReconciliationResult(aggregated_quantities={"WidgetA": 10.0}), today=TODAY
    )
    assert "stock.insufficient" not in codes(result)


def test_discontinued_sku_is_critical_but_a_real_stockout_is_not(conn):
    """FakeItem is delisted -- on an inbound invoice that is fraud. CoolantPro is simply
    backordered. Collapsing the two would lose the distinction that matters."""
    fraud = prevalidate(
        conn,
        invoice(line_items=[LineItem(item="FakeItem", quantity=100, unit_price=1000.0)]),
        ReconciliationResult(aggregated_quantities={"FakeItem": 100.0}),
        today=TODAY,
    )
    stockout = prevalidate(
        conn,
        invoice(line_items=[LineItem(item="CoolantPro", quantity=5, unit_price=80.0)]),
        ReconciliationResult(aggregated_quantities={"CoolantPro": 5.0}),
        today=TODAY,
    )

    assert "stock.discontinued" in codes(fraud)
    assert next(f for f in fraud.findings if f.code == "stock.discontinued").severity is (
        Severity.CRITICAL
    )
    assert "stock.out_of_stock" in codes(stockout)
    assert next(f for f in stockout.findings if f.code == "stock.out_of_stock").severity is (
        Severity.WARNING
    )


def test_negative_quantity_is_not_reported_as_a_stockout(conn):
    """Reconciliation already flagged it; a second finding calling it a stock problem
    would misdescribe it."""
    result = prevalidate(
        conn,
        invoice(line_items=[LineItem(item="WidgetA", quantity=-5, unit_price=250.0)]),
        ReconciliationResult(aggregated_quantities={"WidgetA": -5.0}),
        today=TODAY,
    )
    assert not [c for c in codes(result) if c.startswith("stock.")]


def test_unresolvable_item_is_deferred_not_decided(conn):
    """WidgetC must reach the agent, not be resolved to WidgetA by a threshold."""
    result = prevalidate(
        conn,
        invoice(line_items=[LineItem(item="WidgetC", quantity=3, unit_price=350.0)]),
        ReconciliationResult(aggregated_quantities={"WidgetC": 3.0}),
        today=TODAY,
    )
    assert result.needs_adjudication
    assert [lookup.query for lookup in result.unresolved_items] == ["WidgetC"]
    assert not [c for c in codes(result) if c.startswith("catalog.")]


# ---------------------------------------------------------------------------
# Vendor
# ---------------------------------------------------------------------------


def test_approved_vendor_is_silent(conn):
    result = prevalidate(conn, invoice(), today=TODAY)
    assert not [c for c in codes(result) if c.startswith("vendor.")]


def test_missing_vendor_is_critical(conn):
    """INV-1009 has an empty vendor name; payment cannot be directed anywhere."""
    result = prevalidate(conn, invoice(vendor_name=None), today=TODAY)
    assert "vendor.missing" in codes(result)


def test_unverified_vendor_is_flagged(conn):
    """QuickShip rebranded and changed banking details; that is worth a human's eye."""
    result = prevalidate(conn, invoice(vendor_name="QuickShip Distributors"), today=TODAY)
    assert "vendor.unverified" in codes(result)


def test_unknown_vendor_is_deferred_to_the_agent(conn):
    result = prevalidate(conn, invoice(vendor_name="Fraudster LLC"), today=TODAY)
    assert result.unresolved_vendor is not None
    assert result.unresolved_vendor.query == "Fraudster LLC"


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def test_identical_resubmission_is_critical(conn):
    target = invoice(invoice_number="INV-1004", total=1890.0)
    digest = compute_line_hash(
        [(line.item, line.quantity, line.unit_price or 0.0) for line in target.line_items]
    )
    record_invoice(
        conn, invoice_number="INV-1004", total=1890.0, line_hash=digest, decision="approved"
    )

    result = prevalidate(conn, target, today=TODAY)
    assert "duplicate.resubmission" in codes(result)


def test_same_number_different_content_is_a_revision_not_a_duplicate(conn):
    """INV-1004_revised reuses the number with an extra line and a higher total. That is
    plausibly legitimate, so it warns rather than rejecting outright."""
    record_invoice(
        conn, invoice_number="INV-1004", total=1890.0, line_hash="other", decision="approved"
    )

    result = prevalidate(conn, invoice(invoice_number="INV-1004", total=5940.0), today=TODAY)
    found = next(f for f in result.findings if f.code == "duplicate.revised")
    assert found.severity is Severity.WARNING
    assert "5940" in found.message


def test_same_content_under_a_new_number_is_caught(conn):
    target = invoice(invoice_number="INV-9999")
    digest = compute_line_hash(
        [(line.item, line.quantity, line.unit_price or 0.0) for line in target.line_items]
    )
    record_invoice(
        conn, invoice_number="INV-1001", total=2500.0, line_hash=digest, decision="approved"
    )

    result = prevalidate(conn, target, today=TODAY)
    assert "duplicate.same_content" in codes(result)


def test_clean_ledger_raises_nothing(conn):
    result = prevalidate(conn, invoice(), today=TODAY)
    assert not [c for c in codes(result) if c.startswith("duplicate.")]


# ---------------------------------------------------------------------------
# Prices, dates, currency
# ---------------------------------------------------------------------------


def test_rush_pricing_is_flagged(conn):
    """INV-1010 bills a rush line at 300.00 against a 250.00 catalog price."""
    result = prevalidate(
        conn,
        invoice(
            line_items=[
                LineItem(item="WidgetA", quantity=4, unit_price=300.0, note="rush order")
            ]
        ),
        ReconciliationResult(aggregated_quantities={"WidgetA": 4.0}),
        today=TODAY,
    )
    found = next(f for f in result.findings if f.code == "price.variance")
    assert "+20%" in found.message
    assert "rush order" in found.message


def test_modest_volume_discount_is_not_flagged(conn):
    """INV-1013 discounts WidgetA to 240.00, which is -4% and ordinary commercial life."""
    result = prevalidate(
        conn,
        invoice(
            line_items=[LineItem(item="WidgetA", quantity=5, unit_price=240.0)]
        ),
        ReconciliationResult(aggregated_quantities={"WidgetA": 5.0}),
        today=TODAY,
    )
    assert "price.variance" not in codes(result)


def test_unresolvable_due_date_is_flagged_with_its_raw_text(conn):
    """INV-1003 prints 'yesterday'. The raw string is the evidence."""
    result = prevalidate(
        conn, invoice(due_date=None, due_date_raw="yesterday"), today=TODAY
    )
    found = next(f for f in result.findings if f.code == "date.due_unresolvable")
    assert "yesterday" in found.message


def test_due_date_contradicting_payment_terms_is_flagged(conn):
    """INV-1002 says Net 30 but dates the invoice and its due date the same day."""
    result = prevalidate(
        conn,
        invoice(
            invoice_date=date(2026, 1, 30),
            due_date=date(2026, 1, 30),
            payment_terms="Net 30",
        ),
        today=TODAY,
    )
    assert "date.terms_mismatch" in codes(result)


def test_a_few_days_of_drift_from_net_terms_is_tolerated(conn):
    """INV-1001 is Net 15 dated 2026-01-15 and due 2026-02-01 -- two days out, normal."""
    result = prevalidate(
        conn,
        invoice(
            invoice_date=date(2026, 1, 15),
            due_date=date(2026, 2, 1),
            payment_terms="Net 15",
        ),
        today=TODAY,
    )
    assert "date.terms_mismatch" not in codes(result)


def test_due_date_before_issue_date_is_critical(conn):
    result = prevalidate(
        conn,
        invoice(invoice_date=date(2026, 1, 20), due_date=date(2026, 1, 10)),
        today=TODAY,
    )
    found = next(f for f in result.findings if f.code == "date.due_before_issue")
    assert found.severity is Severity.CRITICAL


def test_unexpected_currency_is_flagged(conn):
    """TechParts bills in EUR; a USD invoice from them is worth a second look."""
    result = prevalidate(
        conn, invoice(vendor_name="TechParts International", currency="USD"), today=TODAY
    )
    assert "currency.unexpected" in codes(result)


# ---------------------------------------------------------------------------
# Folding reconciliation
# ---------------------------------------------------------------------------


def test_reconciliation_output_is_folded_in_once(conn):
    integrity = Finding(
        code="data.negative_quantity", severity=Severity.CRITICAL, message="negative"
    )
    result = prevalidate(
        conn,
        invoice(),
        ReconciliationResult(
            aggregated_quantities={"WidgetA": 10.0},
            is_consistent=False,
            discrepancies=["total: stated 22,562.80, computed 22,512.80"],
            integrity_findings=[integrity],
        ),
        today=TODAY,
    )

    assert codes(result).count("data.negative_quantity") == 1
    assert codes(result).count("arithmetic.mismatch") == 1


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_every_tool_is_registered():
    assert set(TOOLS_BY_NAME) == {
        "lookup_item",
        "lookup_vendor",
        "check_stock",
        "find_prior_invoices",
    }
    assert len(VALIDATOR_TOOLS) == 4


def test_lookup_item_tool_reports_an_exact_hit(db):
    payload = json.loads(TOOLS_BY_NAME["lookup_item"].invoke({"name": "Widget A"}))
    assert payload["exact_match"]["item"] == "WidgetA"
    assert payload["resolution_required"] is False


def test_lookup_item_tool_defers_a_near_miss(db):
    payload = json.loads(TOOLS_BY_NAME["lookup_item"].invoke({"name": "WidgetC"}))
    assert payload["exact_match"] is None
    assert payload["resolution_required"] is True
    assert {c["item"] for c in payload["candidates"]} >= {"WidgetA", "WidgetB"}


def test_check_stock_tool_returns_a_status(db):
    payload = json.loads(
        TOOLS_BY_NAME["check_stock"].invoke({"item": "WidgetA", "quantity": 22})
    )
    assert payload["status"] == "insufficient"
    assert payload["shortfall"] == 7


def test_find_prior_invoices_tool_is_empty_on_a_clean_ledger(db):
    payload = json.loads(
        TOOLS_BY_NAME["find_prior_invoices"].invoke({"invoice_number": "INV-1001"})
    )
    assert payload == []


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Replays a fixed sequence of assistant turns."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.invocations = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations += 1
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="done")


def call(name: str, args: dict, call_id: str = "1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_tool_loop_executes_calls_and_returns_results(db):
    llm = ScriptedLLM(call("lookup_item", {"name": "WidgetC"}), AIMessage(content="ruled"))

    conversation, performed = run_tool_loop(llm, [HumanMessage(content="check it")])

    assert performed == ['lookup_item({"name": "WidgetC"})']
    tool_replies = [m for m in conversation if m.__class__.__name__ == "ToolMessage"]
    assert "WidgetA" in tool_replies[0].content


def test_tool_loop_stops_when_the_agent_stops_calling(db):
    llm = ScriptedLLM(AIMessage(content="nothing to look up"))
    _, performed = run_tool_loop(llm, [HumanMessage(content="check it")])

    assert performed == []
    assert llm.invocations == 1


def test_tool_loop_is_bounded(db):
    """A model that calls tools forever must not hang the pipeline."""
    llm = ScriptedLLM(*[call("lookup_item", {"name": "WidgetA"}) for _ in range(20)])
    _, performed = run_tool_loop(llm, [HumanMessage(content="check it")])

    assert len(performed) == MAX_TOOL_ITERATIONS


def test_an_unknown_tool_is_reported_back_rather_than_raising(db):
    llm = ScriptedLLM(call("delete_everything", {}), AIMessage(content="ok"))
    conversation, _ = run_tool_loop(llm, [HumanMessage(content="go")])

    reply = next(m for m in conversation if m.__class__.__name__ == "ToolMessage")
    assert "No such tool" in reply.content


def test_a_failing_tool_hands_the_error_to_the_agent(db):
    """A bad argument is something the model can correct next turn. Raising would throw
    away the investigation already done."""
    llm = ScriptedLLM(call("check_stock", {"item": "WidgetA"}), AIMessage(content="ok"))
    conversation, _ = run_tool_loop(llm, [HumanMessage(content="go")])

    reply = next(m for m in conversation if m.__class__.__name__ == "ToolMessage")
    assert "error" in reply.content.lower()


# ---------------------------------------------------------------------------
# Verdict -> findings
# ---------------------------------------------------------------------------


def test_unknown_item_verdict_becomes_a_critical_finding(conn):
    verdict = ValidationVerdict(
        item_adjudications=[
            ItemAdjudication(
                invoice_name="WidgetC",
                verdict="different_product",
                reasoning="Trailing character differs.",
            )
        ],
        summary="x",
    )
    findings = verdict_to_findings(verdict, {"WidgetC": 3.0}, conn)

    assert findings[0].code == "catalog.unknown_item"
    assert findings[0].severity is Severity.CRITICAL


def test_a_corrected_name_still_gets_a_stock_check(conn):
    """Resolving the name is not the end of it -- the resolved SKU still has to be on
    the shelf in the quantity billed."""
    verdict = ValidationVerdict(
        item_adjudications=[
            ItemAdjudication(
                invoice_name="Widgt A",
                verdict="same_product",
                resolved_sku="WidgetA",
                reasoning="Transposition.",
            )
        ],
        summary="x",
    )
    findings = verdict_to_findings(verdict, {"Widgt A": 40.0}, conn)

    assert [f.code for f in findings] == ["catalog.name_corrected", "stock.insufficient"]


def test_unapproved_vendor_verdict_is_critical(conn):
    verdict = ValidationVerdict(
        vendor_adjudication=VendorAdjudication(
            invoice_name="Fraudster LLC", verdict="unknown", reasoning="Not on the list."
        ),
        summary="x",
    )
    findings = verdict_to_findings(verdict, {}, conn)

    assert findings[0].code == "vendor.unapproved"
    assert findings[0].severity is Severity.CRITICAL


def test_a_resolved_vendor_inherits_its_status(conn):
    """'QuickShip Distributers' resolves to an approved-list entry that is unverified,
    so correcting the spelling must not clear the underlying concern."""
    verdict = ValidationVerdict(
        vendor_adjudication=VendorAdjudication(
            invoice_name="QuickShip Distributers",
            verdict="same_vendor",
            resolved_vendor="QuickShip Distributors",
            reasoning="Misspelling.",
        ),
        summary="x",
    )
    findings = verdict_to_findings(verdict, {}, conn)

    assert [f.code for f in findings] == ["vendor.name_corrected", "vendor.unverified"]


def test_risk_signals_use_one_stable_code(conn):
    """The model supplies the label and severity; the code vocabulary stays fixed."""
    verdict = ValidationVerdict(
        risk_signals=[
            RiskSignal(signal="urgency", severity=Severity.CRITICAL, evidence="URGENT!!!"),
            RiskSignal(signal="round number", severity=Severity.INFO, evidence="100000"),
        ],
        summary="x",
    )
    findings = verdict_to_findings(verdict, {}, conn)

    assert [f.code for f in findings] == ["risk.signal", "risk.signal"]
    assert findings[0].severity is Severity.CRITICAL
    assert findings[1].severity is Severity.INFO


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_prompt_carries_the_aggregate_not_just_the_lines(conn):
    result = prevalidate(
        conn,
        invoice(line_items=[LineItem(item="WidgetC", quantity=3, unit_price=350.0)]),
        ReconciliationResult(aggregated_quantities={"WidgetC": 3.0}),
        today=TODAY,
    )
    messages = build_messages(
        invoice(), ReconciliationResult(aggregated_quantities={"WidgetA": 22.0}), result
    )
    body = str(messages[-1].content)

    assert "per-SKU totals across all lines: WidgetA x22" in body
    assert "UNRESOLVED ITEM: WidgetC" in body
    assert "do not restate" in body


def test_prompt_names_the_scrutiny_threshold(conn):
    messages = build_messages(invoice(), None, Prevalidation())
    assert "10,000" in str(messages[-1].content)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


def state_with(inv: InvoiceData, reconciliation: ReconciliationResult | None = None) -> dict:
    state = initial_state(source_path="x.json", source_format="json", raw_text="{}")
    state["invoice"] = inv
    state["reconciliation"] = reconciliation
    return state


def test_node_combines_deterministic_and_judged_findings(db):
    state = state_with(
        invoice(
            vendor_name="Fraudster LLC",
            line_items=[LineItem(item="FakeItem", quantity=100, unit_price=1000.0)],
            notes="URGENT - Pay immediately to avoid penalties!!! Wire transfer preferred.",
            total=100000.0,
        ),
        ReconciliationResult(aggregated_quantities={"FakeItem": 100.0}),
    )

    update = validate_node(state)
    found = [f.code for f in update["findings"]]

    assert "stock.discontinued" in found  # deterministic
    assert "vendor.unapproved" in found  # judged
    assert "risk.signal" in found  # judged


def test_node_skips_cleanly_without_an_invoice(db):
    update = validate_node(
        initial_state(source_path="x", source_format="json", raw_text="{}")
    )
    assert update.get("findings") is None
    assert any(entry.event == "skipped" for entry in update["audit_log"])


def test_deterministic_findings_survive_an_adjudication_failure(db, monkeypatch):
    """If the model call dies, the mechanical results are still sound and must not be
    thrown away with it."""
    import invoice_agents.agents.validator as module

    def explode(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(module, "build_validator", explode)

    state = state_with(
        invoice(line_items=[LineItem(item="FakeItem", quantity=100, unit_price=1000.0)]),
        ReconciliationResult(aggregated_quantities={"FakeItem": 100.0}),
    )
    update = validate_node(state)

    assert "stock.discontinued" in [f.code for f in update["findings"]]
    assert update["errors"]
    assert any(entry.event == "adjudication_failed" for entry in update["audit_log"])


# ---------------------------------------------------------------------------
# Offline adjudication stand-in
# ---------------------------------------------------------------------------


def test_offline_stand_in_refuses_to_resolve_widgetc(db):
    """The headline case: 0.857 against two different SKUs at once."""
    state = state_with(
        invoice(line_items=[LineItem(item="WidgetC", quantity=3, unit_price=350.0)]),
        ReconciliationResult(aggregated_quantities={"WidgetC": 3.0}),
    )
    update = validate_node(state)

    found = next(f for f in update["findings"] if f.code == "catalog.unknown_item")
    assert found.evidence["invoice_name"] == "WidgetC"
    assert found.evidence["resolved_sku"] is None


def test_offline_stand_in_accepts_a_vendor_misspelling(db):
    state = state_with(invoice(vendor_name="QuickShip Distributers"))
    update = validate_node(state)
    found = [f.code for f in update["findings"]]

    assert "vendor.name_corrected" in found
    assert "vendor.unverified" in found


def test_offline_stand_in_spots_a_total_below_the_threshold(db):
    """INV-1008 bills 9,900 and INV-1012 bills 9,975 -- both just under 10,000."""
    state = state_with(invoice(total=9900.0))
    update = validate_node(state)

    signals = [f for f in update["findings"] if f.code == "risk.signal"]
    assert any("below" in f.message for f in signals)
