"""Tests for the local web console."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from invoice_agents.config import get_settings
from invoice_agents.inventory import initialize
from invoice_agents.web import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A console backed by a throwaway database, offline."""
    path = tmp_path / "web.db"
    initialize(path, reset=True)
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


def test_overview_reports_the_backend_and_the_database(client):
    body = client.get("/api/overview").json()

    assert body["llm_mode"] == "mock"
    assert body["db_ready"] is True
    assert body["items"] == 7
    assert body["vendors"] == 12
    assert body["documents"] >= 25


def test_inbox_lists_every_supported_format(client):
    rows = client.get("/api/invoices").json()
    formats = {row["format"] for row in rows}

    assert {"txt", "json", "csv", "xml", "pdf"} <= formats
    assert all(row["filename"] for row in rows)


def test_inventory_and_vendors_are_served(client):
    inventory = client.get("/api/inventory").json()
    vendors = client.get("/api/vendors").json()

    assert {row["item"] for row in inventory} >= {"WidgetA", "FakeItem"}
    assert "Fraudster LLC" not in {row["name"] for row in vendors}


def test_ledger_starts_empty(client):
    assert client.get("/api/ledger").json() == []


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def test_raw_document_is_served_as_text(client):
    body = client.get("/api/invoices/invoice_1001.txt/raw").json()

    assert body["format"] == "txt"
    assert "Widgets Inc." in body["raw_text"]


def test_pdf_is_extracted_with_its_damage_intact(client):
    """The console shows what the pipeline sees, not a cleaned-up version of it."""
    body = client.get("/api/invoices/invoice_1012.pdf/raw").json()

    assert body["format"] == "pdf"
    assert "2O26" in body["raw_text"]


def test_a_missing_document_is_a_404(client):
    assert client.get("/api/invoices/nope.txt/raw").status_code == 404


@pytest.mark.parametrize(
    "target",
    ["..%2F..%2Fpyproject.toml", "C:%5CWindows%5Cwin.ini", "%2Fetc%2Fpasswd"],
)
def test_the_api_will_not_read_outside_the_inbox(client, target):
    """Filenames come from the URL, so they are treated as hostile."""
    assert client.get(f"/api/invoices/{target}/raw").status_code == 404


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------


def test_processing_returns_the_whole_run(client):
    body = client.post("/api/invoices/invoice_1003.txt/process").json()

    assert body["decision"]["decision"] == "rejected"
    assert body["decision"]["revised"] is True
    assert body["counts"]["critical"] == 3
    assert body["payment"]["status"] == "withheld"
    assert len(body["audit_log"]) > 5
    assert body["reconciliation"]["is_consistent"] is True


def test_a_clean_invoice_reports_a_payment_reference(client):
    body = client.post("/api/invoices/invoice_1001.txt/process").json()

    assert body["decision"]["decision"] == "approved"
    assert body["payment"]["reference"].startswith("PAY-")


def test_processing_writes_to_the_ledger_and_the_inbox_reflects_it(client):
    client.post("/api/invoices/invoice_1003.txt/process")

    ledger = client.get("/api/ledger").json()
    assert [row["invoice_number"] for row in ledger] == ["INV-1003"]
    assert ledger[0]["decision"] == "rejected"

    inbox = {row["filename"]: row for row in client.get("/api/invoices").json()}
    assert inbox["invoice_1003.txt"]["decision"] == "rejected"


def test_processing_a_missing_document_is_a_404(client):
    assert client.post("/api/invoices/nope.txt/process").status_code == 404


def test_the_ledger_can_be_cleared(client):
    client.post("/api/invoices/invoice_1001.txt/process")
    assert client.delete("/api/ledger").json()["cleared"] == 1
    assert client.get("/api/ledger").json() == []


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


def test_the_page_and_its_assets_are_served(client):
    assert "INVOICE" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_a_missing_database_is_reported_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "absent.db"))
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()

    client = TestClient(app)
    assert client.get("/api/overview").json()["db_ready"] is False
    response = client.get("/api/inventory")
    assert response.status_code == 503
    assert "init_db" in response.json()["detail"]

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Recorded reasoning
# ---------------------------------------------------------------------------


def test_the_inbox_carries_the_last_decisions_reasoning(client):
    """Reading a past verdict should cost nothing. Without this the only way to see why
    an invoice was refused is to pay for another full model run."""
    client.post("/api/invoices/invoice_1003.txt/process")

    row = next(
        r for r in client.get("/api/invoices").json() if r["filename"] == "invoice_1003.txt"
    )

    assert row["decision"] == "rejected"
    assert row["reason"]
    assert len(row["reason"]) > 80
    assert row["processed_at"]


def test_an_unprocessed_document_has_no_reasoning(client):
    row = next(
        r for r in client.get("/api/invoices").json() if r["filename"] == "invoice_1001.txt"
    )
    assert row["decision"] is None
    assert row["reason"] is None


def test_the_ledger_serves_full_reasoning(client):
    client.post("/api/invoices/invoice_1013.json/process")
    entry = client.get("/api/ledger").json()[0]

    assert entry["reason"]
    assert not entry["reason"].endswith("...")


def test_long_rationales_are_stored_whole(tmp_path, monkeypatch):
    """Rationales were once cut at 500 characters, which truncated three of 25 mid-word."""
    from invoice_agents.inventory import connect, find_prior_invoices
    from invoice_agents.payment import MAX_REASON_CHARS, finalize_node
    from invoice_agents.schemas import ApprovalDecision, Decision, InvoiceData, LineItem
    from invoice_agents.state import initial_state

    path = tmp_path / "reason.db"
    initialize(path, reset=True)
    monkeypatch.setenv("DB_PATH", str(path))
    get_settings.cache_clear()

    long_rationale = "This invoice needs review because " + ("x" * 900)
    state = initial_state(source_path="x.json", source_format="json", raw_text="{}")
    state["invoice"] = InvoiceData(
        invoice_number="INV-LONG",
        vendor_name="Widgets Inc.",
        line_items=[LineItem(item="WidgetA", quantity=1, unit_price=250.0)],
        total=250.0,
    )
    state["decision"] = ApprovalDecision(
        decision=Decision.NEEDS_REVIEW, rationale=long_rationale
    )
    finalize_node(state)

    with connect(path) as conn:
        stored = find_prior_invoices(conn, "INV-LONG")[0].reason

    assert len(stored) == len(long_rationale)
    assert MAX_REASON_CHARS > 500
    get_settings.cache_clear()
