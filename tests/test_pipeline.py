"""Tests for policy, the approver and its critique loop, payment, and the whole graph."""

from __future__ import annotations

import pytest

from invoice_agents.agents.approver import (
    MAX_CRITIQUE_ROUNDS,
    approve_node,
    build_context,
    critique_node,
    route_after_critique,
)
from invoice_agents.config import get_settings
from invoice_agents.graph import build_graph, process_invoice
from invoice_agents.ingestion import load_document
from invoice_agents.inventory import connect, find_prior_invoices, initialize
from invoice_agents.payment import finalize_node, mock_payment
from invoice_agents.policy import (
    SCRUTINY_THRESHOLD,
    apply_floor,
    policy_flags,
    verify_citations,
)
from invoice_agents.reconciliation import MAX_EXTRACTION_ATTEMPTS
from invoice_agents.schemas import (
    ApprovalDecision,
    Critique,
    Decision,
    Finding,
    InvoiceData,
    LineItem,
    Severity,
)
from invoice_agents.state import initial_state


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh database, offline LLM, settings pointed at both."""
    path = tmp_path / "pipeline.db"
    initialize(path, reset=True)
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


def invoice(**overrides) -> InvoiceData:
    defaults = {
        "invoice_number": "INV-TEST",
        "vendor_name": "Widgets Inc.",
        "line_items": [LineItem(item="WidgetA", quantity=10, unit_price=250.0)],
        "total": 2500.0,
    }
    return InvoiceData(**{**defaults, **overrides})


def finding(code: str, severity: Severity) -> Finding:
    return Finding(code=code, severity=severity, message=code)


def approved(**overrides) -> ApprovalDecision:
    return ApprovalDecision(
        **{"decision": Decision.APPROVED, "rationale": "Looks fine.", **overrides}
    )


def state_with(inv=None, findings=None, decision=None, critique=None, rounds=0) -> dict:
    state = initial_state(source_path="x.json", source_format="json", raw_text="{}")
    state["invoice"] = inv
    state["findings"] = findings or []
    state["decision"] = decision
    state["critique"] = critique
    state["critique_rounds"] = rounds
    return state


# ---------------------------------------------------------------------------
# Policy floor
# ---------------------------------------------------------------------------


def test_clean_approval_stands():
    decision = apply_floor(approved(), invoice(), [])
    assert decision.decision is Decision.APPROVED


def test_a_critical_finding_blocks_approval():
    """The single most important property: no argument gets past this."""
    decision = apply_floor(
        approved(), invoice(), [finding("vendor.unapproved", Severity.CRITICAL)]
    )
    assert decision.decision is Decision.NEEDS_REVIEW
    assert "POLICY OVERRIDE" in decision.rationale
    assert "policy_floor_applied" in decision.policy_flags


def test_a_large_invoice_with_open_warnings_cannot_auto_approve():
    decision = apply_floor(
        approved(),
        invoice(total=SCRUTINY_THRESHOLD + 1),
        [finding("stock.insufficient", Severity.WARNING)],
    )
    assert decision.decision is Decision.NEEDS_REVIEW


def test_a_large_clean_invoice_may_be_approved():
    """Size alone is not a reason to refuse an approved vendor with a clean record."""
    decision = apply_floor(approved(), invoice(total=SCRUTINY_THRESHOLD + 1), [])
    assert decision.decision is Decision.APPROVED


def test_a_small_invoice_with_warnings_may_be_approved():
    decision = apply_floor(
        approved(), invoice(total=100.0), [finding("date.already_overdue", Severity.WARNING)]
    )
    assert decision.decision is Decision.APPROVED


def test_an_invoice_with_no_total_cannot_be_approved():
    decision = apply_floor(approved(), invoice(total=None), [])
    assert decision.decision is Decision.NEEDS_REVIEW


def test_a_missing_invoice_cannot_be_approved():
    assert apply_floor(approved(), None, []).decision is Decision.NEEDS_REVIEW


def test_the_floor_never_upgrades_a_rejection():
    """It only ever makes an outcome more conservative. A model that decides to refuse
    is not overruled into paying."""
    rejected = ApprovalDecision(decision=Decision.REJECTED, rationale="Fraud.")
    assert apply_floor(rejected, invoice(), []).decision is Decision.REJECTED


def test_policy_flags_describe_the_situation():
    flags = policy_flags(
        invoice(total=25_000.0),
        [finding("a", Severity.CRITICAL), finding("b", Severity.WARNING)],
    )
    assert any("scrutiny threshold" in f for f in flags)
    assert any("1 critical" in f for f in flags)


# ---------------------------------------------------------------------------
# Approver
# ---------------------------------------------------------------------------


def test_clean_invoice_is_approved(env):
    update = approve_node(state_with(invoice()))
    assert update["decision"].decision is Decision.APPROVED


def test_critical_findings_prevent_approval(env):
    update = approve_node(
        state_with(invoice(), [finding("catalog.unknown_item", Severity.CRITICAL)])
    )
    assert update["decision"].decision is not Decision.APPROVED


def test_warnings_alone_go_to_review_not_rejection(env):
    """A stock shortfall is a purchasing conversation, not grounds to refuse a supplier."""
    update = approve_node(
        state_with(invoice(), [finding("stock.insufficient", Severity.WARNING)])
    )
    assert update["decision"].decision is Decision.NEEDS_REVIEW


def test_context_orders_findings_by_severity(env):
    context = build_context(
        invoice(),
        None,
        [finding("info.thing", Severity.INFO), finding("bad.thing", Severity.CRITICAL)],
    )
    assert context.index("bad.thing") < context.index("info.thing")


def test_context_survives_a_failed_extraction(env):
    assert "Extraction failed" in build_context(None, None, [])


def test_approver_failure_holds_rather_than_defaulting(env, monkeypatch):
    """Neither approving nor rejecting on a backend error -- both would be a decision
    the system did not actually make."""
    import invoice_agents.agents.approver as module

    monkeypatch.setattr(module, "build_approver", lambda *a, **k: 1 / 0)

    update = approve_node(state_with(invoice()))
    assert update["decision"].decision is Decision.NEEDS_REVIEW
    assert "approver_unavailable" in update["decision"].policy_flags
    assert update["errors"]


# ---------------------------------------------------------------------------
# Critique loop
# ---------------------------------------------------------------------------


def test_a_sound_decision_is_upheld(env):
    update = critique_node(state_with(invoice(), [], approved()))
    assert update["critique"].recommend_revision is False
    assert update["critique_rounds"] == 1


def test_a_conclusive_case_is_sent_back_for_rejection(env):
    """Draft says 'a human should look'; the critic says the evidence has already
    decided. This is the loop doing real work rather than ratifying."""
    findings = [
        finding("vendor.unapproved", Severity.CRITICAL),
        finding("stock.discontinued", Severity.CRITICAL),
    ]
    draft = ApprovalDecision(decision=Decision.NEEDS_REVIEW, rationale="Held for review.")

    update = critique_node(state_with(invoice(), findings, draft))
    assert update["critique"].recommend_revision is True
    assert update["critique"].concerns


def test_an_over_harsh_rejection_is_sent_back(env):
    """The critic looks in both directions, not only for leniency."""
    draft = ApprovalDecision(decision=Decision.REJECTED, rationale="No.")
    update = critique_node(
        state_with(invoice(), [finding("price.variance", Severity.WARNING)], draft)
    )
    assert update["critique"].recommend_revision is True


def test_critique_failure_keeps_the_draft(env, monkeypatch):
    import invoice_agents.agents.approver as module

    monkeypatch.setattr(module, "build_approver", lambda *a, **k: 1 / 0)

    update = critique_node(state_with(invoice(), [], approved()))
    assert update["critique"] is None
    assert update["critique_rounds"] == 1


def test_upheld_draft_proceeds():
    state = state_with(critique=Critique(recommend_revision=False, reasoning="fine"))
    assert route_after_critique(state) == "finalize"


def test_recommended_revision_returns_to_the_approver():
    state = state_with(
        critique=Critique(recommend_revision=True, reasoning="no"), rounds=1
    )
    assert route_after_critique(state) == "approve"


def test_the_critique_loop_is_bounded():
    """A critic can always find one more thing to say."""
    state = state_with(
        critique=Critique(recommend_revision=True, reasoning="still no"),
        rounds=MAX_CRITIQUE_ROUNDS,
    )
    assert route_after_critique(state) == "finalize"


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------


def test_mock_payment_returns_a_traceable_receipt():
    receipt = mock_payment("Widgets Inc.", 5000.0)
    assert receipt["status"] == "success"
    assert receipt["reference"].startswith("PAY-")


def test_approved_invoice_is_paid(env):
    state = state_with(invoice(), [], approved())
    update = finalize_node(state)

    assert update["payment"].status == "success"
    assert update["payment"].amount == 2500.0
    assert update["payment"].reference


def test_rejected_invoice_is_not_paid(env):
    state = state_with(
        invoice(), [], ApprovalDecision(decision=Decision.REJECTED, rationale="Fraud.")
    )
    update = finalize_node(state)

    assert update["payment"].status == "withheld"
    assert update["payment"].reference is None


def test_every_outcome_reaches_the_ledger(env):
    """Rejections are recorded too, so a refused vendor resubmitting under a new number
    is still caught by the content hash."""
    for number, decision in [
        ("INV-PAID", Decision.APPROVED),
        ("INV-REFUSED", Decision.REJECTED),
    ]:
        finalize_node(
            state_with(
                invoice(invoice_number=number),
                [],
                ApprovalDecision(decision=decision, rationale="x"),
            )
        )

    with connect(env) as conn:
        assert find_prior_invoices(conn, "INV-PAID")[0].decision == "approved"
        assert find_prior_invoices(conn, "INV-REFUSED")[0].decision == "rejected"


def test_a_missing_decision_is_recorded_as_an_error(env):
    update = finalize_node(state_with(invoice()))
    assert update["payment"].status == "not_attempted"

    with connect(env) as conn:
        assert find_prior_invoices(conn, "INV-TEST")[0].decision == "error"


# ---------------------------------------------------------------------------
# The whole graph
# ---------------------------------------------------------------------------


def run(filename: str, graph=None):
    document = load_document(get_settings().resolved_invoice_dir / filename)
    return process_invoice(
        source_path=str(document.path),
        source_format=document.source_format,
        raw_text=document.raw_text,
        graph=graph,
    )


def test_graph_compiles(env):
    assert build_graph() is not None


def test_a_clean_invoice_runs_through_to_payment(env):
    state = run("invoice_1001.txt")

    assert state["decision"].decision is Decision.APPROVED
    assert state["payment"].status == "success"
    assert state["extraction_attempts"] == 1


def test_the_fraud_invoice_is_rejected_after_the_critique_overturns_it(env):
    """INV-1003: the draft holds it for a human, the critic argues the indicators are
    conclusive, and the revised decision refuses it outright."""
    state = run("invoice_1003.txt")

    assert state["decision"].decision is Decision.REJECTED
    assert state["decision"].revised is True
    assert state["payment"].status == "withheld"

    events = [f"{e.node}.{e.event}" for e in state["audit_log"]]
    assert "approver.decided" in events
    assert "critic.critiqued" in events
    assert "approver.revised" in events


def test_an_unreconcilable_invoice_is_re_read_once_then_moves_on(env):
    """INV-1013's total is genuinely $50 wrong, so no re-read will fix it."""
    state = run("invoice_1013.json")

    assert state["extraction_attempts"] == 2
    assert state["reconciliation"].is_consistent is False
    assert state["decision"] is not None
    assert "arithmetic.mismatch" in [f.code for f in state["findings"]]


def test_an_unknown_sku_blocks_payment(env):
    state = run("invoice_1016.json")

    assert state["decision"].decision is not Decision.APPROVED
    assert "catalog.unknown_item" in [f.code for f in state["findings"]]


def test_findings_accumulate_across_the_real_graph(env):
    """The reducers, exercised by the actual pipeline rather than a toy graph."""
    state = run("invoice_1013.json")

    assert len(state["findings"]) >= 4
    assert len(state["audit_log"]) >= 6
    assert len({f.code for f in state["findings"]}) > 1


def test_a_resubmission_is_caught_on_the_second_run(env):
    """Duplicate detection only works because the first run wrote to the ledger."""
    graph = build_graph()
    first = run("invoice_1004.json", graph)
    assert "duplicate.resubmission" not in [f.code for f in first["findings"]]

    second = run("invoice_1004.json", graph)
    assert "duplicate.resubmission" in [f.code for f in second["findings"]]
    assert second["decision"].decision is not Decision.APPROVED


def test_a_revision_is_distinguished_from_a_duplicate(env):
    """INV-1004_revised reuses the number with an extra line and a higher total."""
    graph = build_graph()
    run("invoice_1004.json", graph)
    revised = run("invoice_1004_revised.json", graph)

    codes = [f.code for f in revised["findings"]]
    assert "duplicate.revised" in codes
    assert "duplicate.resubmission" not in codes


def test_an_extraction_failure_does_not_crash_the_graph(env):
    """When extraction produces nothing, the graph must still reach a decision, write a
    ledger entry and return cleanly rather than raising.

    The document is supplied inline rather than by filename so the test asserts the
    failure path itself, instead of relying on some file happening to lack a fixture.
    """
    state = process_invoice(
        source_path="data/invoices/invoice_0000_unreadable.txt",
        source_format="txt",
        raw_text="Dear Acme, thanks for lunch last week. Best, Jim",
    )

    assert state["invoice"] is None
    assert state["errors"]
    assert state["payment"].status == "not_attempted"
    assert state["extraction_attempts"] == MAX_EXTRACTION_ATTEMPTS


def test_the_final_state_is_a_complete_record(env):
    state = run("invoice_1003.txt")

    for key in ("invoice", "reconciliation", "findings", "decision", "payment", "audit_log"):
        assert key in state, key
    assert all(entry.at for entry in state["audit_log"])


# ---------------------------------------------------------------------------
# Fabricated grounds
# ---------------------------------------------------------------------------


def cited(*codes, decision=Decision.REJECTED) -> ApprovalDecision:
    return ApprovalDecision(
        decision=decision, rationale="Because of the above.", driving_findings=list(codes)
    )


def test_a_decision_citing_a_finding_it_was_not_given_is_redirected():
    """Reproduces a live failure: the approver refused a $22,562.80 invoice as a
    "duplicate resubmission of an invoice already processed on 2026-09-20" against an
    empty ledger. The prose was fluent and the event never happened."""
    supplied = [finding("arithmetic.mismatch", Severity.WARNING)]

    decision, unsupported = verify_citations(
        cited("duplicate.resubmission", "arithmetic.mismatch"), supplied
    )

    assert unsupported == ["duplicate.resubmission"]
    assert decision.decision is Decision.NEEDS_REVIEW
    assert "unsupported_citation" in decision.policy_flags
    assert "UNSUPPORTED CITATION" in decision.rationale


def test_honest_citations_pass_through_untouched():
    supplied = [finding("stock.insufficient", Severity.WARNING)]
    decision, unsupported = verify_citations(cited("stock.insufficient"), supplied)

    assert unsupported == []
    assert decision.decision is Decision.REJECTED
    assert "UNSUPPORTED" not in decision.rationale


def test_citing_nothing_is_allowed():
    """A clean invoice has no findings to cite, and that is not a fabrication."""
    decision, unsupported = verify_citations(cited(decision=Decision.APPROVED), [])
    assert unsupported == []
    assert decision.decision is Decision.APPROVED


def test_a_fabricated_rejection_is_not_left_standing():
    """Refusing a supplier over an invented duplicate is its own harm, so the redirect
    applies to rejections as well as approvals."""
    decision, _ = verify_citations(cited("duplicate.resubmission"), [])
    assert decision.decision is Decision.NEEDS_REVIEW


def test_the_node_surfaces_an_unsupported_citation_as_an_error(env, monkeypatch):
    import invoice_agents.agents.approver as module

    monkeypatch.setattr(
        module,
        "build_approver",
        lambda *a, **k: _FixedApprover(cited("duplicate.resubmission")),
    )

    update = approve_node(state_with(invoice(), [finding("price.variance", Severity.WARNING)]))

    assert update["decision"].decision is Decision.NEEDS_REVIEW
    assert any("duplicate.resubmission" in e for e in update["errors"])
    assert any(entry.event == "unsupported_citation" for entry in update["audit_log"])


class _FixedApprover:
    """Returns one canned decision, whatever it is asked."""

    def __init__(self, decision):
        self.decision = decision

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.decision


# ---------------------------------------------------------------------------
# Failures leave a record
# ---------------------------------------------------------------------------


def test_a_failed_extraction_still_writes_a_ledger_row(env):
    """Twelve documents once vanished from a batch without a trace, because a run that
    produced no invoice wrote nothing at all. A failure must not be less visible than a
    success in the table whose job is traceability."""
    process_invoice(
        source_path="data/invoices/invoice_0000_unreadable.txt",
        source_format="txt",
        raw_text="Dear Acme, thanks for lunch last week. Best, Jim",
    )

    with connect(env) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM invoice_ledger")]

    assert len(rows) == 1
    assert rows[0]["invoice_number"] is None
    assert rows[0]["decision"] == "error"
    assert rows[0]["source_path"].endswith("invoice_0000_unreadable.txt")
    assert rows[0]["reason"]


def test_a_failed_run_does_not_collide_with_real_invoices(env):
    """Null-numbered error rows must never match a duplicate lookup."""
    process_invoice(
        source_path="data/invoices/broken.txt", source_format="txt", raw_text="nonsense"
    )

    with connect(env) as conn:
        assert find_prior_invoices(conn, None) == []
        assert find_prior_invoices(conn, "INV-1001") == []


def test_errors_during_a_successful_run_reach_the_ledger(env):
    """An invoice can be decided while something still went wrong along the way."""
    state = state_with(invoice(), [], approved())
    state["errors"] = ["validator adjudication failed: TimeoutError"]
    finalize_node(state)

    with connect(env) as conn:
        row = find_prior_invoices(conn, "INV-TEST")[0]

    assert "TimeoutError" in row.reason


# ---------------------------------------------------------------------------
# Archived runs
# ---------------------------------------------------------------------------


def test_an_archived_run_replays_into_the_ledger(env, tmp_path):
    """Replaying a recorded live run puts the model's own reasoning in front of a reader
    without spending an API call."""
    import json

    from invoice_agents.archive import ledger_provenance, restore

    archive = tmp_path / "run.json"
    archive.write_text(
        json.dumps(
            {
                "model": "grok-4.6",
                "ledger": [
                    {
                        "invoice_number": "INV-1003",
                        "vendor_name": "Fraudster LLC",
                        "total": 100000.0,
                        "currency": "USD",
                        "decision": "rejected",
                        "reason": "Corroborating fraud indicators.",
                        "source_file": "invoice_1003.txt",
                        "processed_at": "2026-09-21T02:40:37+00:00",
                        "revision": None,
                        "line_hash": "abc123",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    meta = restore(archive, db_path=env)

    assert meta["restored"] == 1
    assert ledger_provenance(env) == {"grok-4.6": 1}
    with connect(env) as conn:
        row = find_prior_invoices(conn, "INV-1003")[0]
    assert row.decision == "rejected"
    assert row.llm_mode == "grok-4.6"
    assert row.processed_at == "2026-09-21T02:40:37+00:00"


def test_a_missing_archive_is_reported_clearly(tmp_path):
    from invoice_agents.archive import ArchiveError, load_archive

    with pytest.raises(ArchiveError, match="No archived run"):
        load_archive(tmp_path / "absent.json")


def test_provenance_of_a_pre_tracking_ledger_is_unknown_not_absent(tmp_path, monkeypatch):
    """An older database has no llm_mode column. Reporting its rows as 'unknown' rather
    than 'none' is what stops the demo from silently overwriting a live run."""
    import sqlite3

    from invoice_agents.archive import ledger_provenance

    db = tmp_path / "old.db"
    initialize(db, reset=True)
    raw = sqlite3.connect(db)
    raw.execute("ALTER TABLE invoice_ledger DROP COLUMN llm_mode")
    raw.execute(
        "INSERT INTO invoice_ledger (invoice_number, decision, processed_at) "
        "VALUES ('INV-OLD', 'approved', '2026-01-01')"
    )
    raw.commit()
    raw.close()

    assert ledger_provenance(db) == {"unknown": 1}
