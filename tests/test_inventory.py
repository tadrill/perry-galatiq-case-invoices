"""Tests for the mock inventory database.

The item and vendor names here are copied verbatim out of data/invoices/ -- these are
the real strings the pipeline has to survive, not invented ones.
"""

from __future__ import annotations

import pytest

from invoice_agents.inventory import (
    check_stock,
    compute_line_hash,
    connect,
    find_prior_invoices,
    initialize,
    lookup_item,
    lookup_vendor,
    normalize_item,
    record_invoice,
)


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "test_inventory.db"
    initialize(db, reset=True)
    with connect(db) as connection:
        yield connection


# --------------------------------------------------------------------------
# Name normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WidgetA", "widgeta"),
        ("Widget A", "widgeta"),  # INV-1012, OCR split the SKU
        ("WidgetA (rush order)", "widgeta"),  # INV-1010, qualifier on the line
        ("  gadget-x ", "gadgetx"),
        ("", ""),
    ],
)
def test_normalize_item_collapses_cosmetic_variation(raw, expected):
    assert normalize_item(raw) == expected


def test_normalize_item_preserves_sku_identity():
    """A different trailing character is a different product, not a typo."""
    assert normalize_item("WidgetC") != normalize_item("WidgetA")


# --------------------------------------------------------------------------
# Item lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["WidgetA", "Widget A", "WidgetA (rush order)"])
def test_cosmetic_variants_resolve_exactly(conn, name):
    """These must never reach the agent -- the answer is unambiguous."""
    result = lookup_item(conn, name)
    assert result.found
    assert result.exact.item == "WidgetA"


def test_unknown_sku_is_not_auto_resolved(conn):
    """INV-1016's WidgetC scores 0.857 against WidgetA. Silently booking it against
    WidgetA would be exactly the class of error this system exists to eliminate, so it
    comes back inexact with its near misses for the agent to adjudicate."""
    result = lookup_item(conn, "WidgetC")

    assert not result.found
    assert result.exact is None
    candidates = {c.item for c in result.candidates}
    assert {"WidgetA", "WidgetB"} <= candidates
    assert all(c.score < 1.0 for c in result.candidates)
    assert result.to_dict()["resolution_required"] is True


@pytest.mark.parametrize("name", ["SuperGizmo", "MegaSprocket"])
def test_genuinely_absent_items_return_no_candidates(conn, name):
    """INV-1008's items resemble nothing in the catalog; no misleading near misses."""
    result = lookup_item(conn, name)
    assert not result.found
    assert result.candidates == ()


# --------------------------------------------------------------------------
# Stock checks
# --------------------------------------------------------------------------


def test_aggregated_quantity_catches_split_lines(conn):
    """INV-1013 bills WidgetA on three separate lines: 15 + 5 + 2 = 22 against 15.

    Every individual line passes. Only the aggregate fails, which is why callers must
    sum per SKU before asking.
    """
    assert check_stock(conn, "WidgetA", 15).status == "ok"
    assert check_stock(conn, "WidgetA", 5).status == "ok"

    aggregate = check_stock(conn, "WidgetA", 22)
    assert aggregate.status == "insufficient"
    assert aggregate.shortfall == 7


def test_quantity_exactly_at_stock_is_allowed(conn):
    """INV-1005 orders 10 WidgetB against 10 in stock -- a boundary, not a breach."""
    assert check_stock(conn, "WidgetB", 10).status == "ok"


def test_stockout_and_fraud_are_distinguishable(conn):
    """A backordered item and a delisted one both have zero stock but mean different
    things; collapsing them into one status would lose the fraud signal."""
    assert check_stock(conn, "CoolantPro", 5).status == "out_of_stock"
    assert check_stock(conn, "FakeItem", 100).status == "discontinued"


def test_unknown_item_reports_full_shortfall(conn):
    result = check_stock(conn, "SuperGizmo", 12)
    assert result.status == "unknown_item"
    assert result.shortfall == 12


# --------------------------------------------------------------------------
# Vendor lookup
# --------------------------------------------------------------------------


def test_approved_vendor_resolves(conn):
    result = lookup_vendor(conn, "Widgets Inc.")
    assert result.found
    assert result.exact.status == "approved"


def test_misspelled_vendor_surfaces_as_near_miss(conn):
    """INV-1012 writes 'Distributers'. High score, but still the agent's call."""
    result = lookup_vendor(conn, "QuickShip Distributers")
    assert not result.found
    assert result.candidates[0].name == "QuickShip Distributors"
    assert result.candidates[0].score > 0.9


def test_former_trading_name_matches_via_alias(conn):
    result = lookup_vendor(conn, "FastShip Ltd.")
    assert result.candidates[0].name == "QuickShip Distributors"


@pytest.mark.parametrize("name", ["Fraudster LLC", "NoProd Industries", ""])
def test_unapproved_vendors_are_unknown(conn, name):
    """INV-1003, INV-1008 and INV-1009's blank vendor must all fail verification."""
    result = lookup_vendor(conn, name)
    assert not result.found


# --------------------------------------------------------------------------
# Ledger / duplicate detection
# --------------------------------------------------------------------------


def test_ledger_starts_empty(conn):
    assert find_prior_invoices(conn, "INV-1004") == []


def test_reused_invoice_number_is_detected(conn):
    """INV-1004_revised reuses INV-1004's number with three line items instead of two."""
    record_invoice(
        conn, invoice_number="INV-1004", vendor_name="Precision Parts Ltd.",
        total=1890.00, decision="approved",
    )

    prior = find_prior_invoices(conn, "INV-1004")
    assert len(prior) == 1
    assert prior[0].total == 1890.00
    assert prior[0].decision == "approved"


def test_resubmission_under_new_number_is_caught_by_content_hash(conn):
    """The other half of the duplicate problem: same goods, fresh invoice number."""
    lines = [("WidgetA", 3, 250.00), ("WidgetB", 2, 500.00)]
    digest = compute_line_hash(lines)

    record_invoice(
        conn, invoice_number="INV-1004", total=1890.00,
        line_hash=digest, decision="approved",
    )

    assert find_prior_invoices(conn, "INV-9999") == []
    assert len(find_prior_invoices(conn, "INV-9999", line_hash=digest)) == 1


def test_line_hash_ignores_order_and_spelling():
    """A resubmission that reorders lines or respells an item still collides."""
    original = compute_line_hash([("WidgetA", 3, 250.00), ("WidgetB", 2, 500.00)])
    shuffled = compute_line_hash([("WidgetB", 2, 500.00), ("Widget A", 3, 250.00)])
    assert original == shuffled


def test_line_hash_separates_genuinely_different_invoices():
    """INV-1004 vs its revision: the added GadgetX line must change the fingerprint."""
    original = compute_line_hash([("WidgetA", 3, 250.00), ("WidgetB", 2, 500.00)])
    revised = compute_line_hash(
        [("WidgetA", 3, 250.00), ("WidgetB", 2, 500.00), ("GadgetX", 5, 750.00)]
    )
    assert original != revised


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_initialize_is_idempotent(tmp_path):
    db = tmp_path / "idempotent.db"
    first = initialize(db)
    second = initialize(db)
    assert first == second


def test_reset_clears_the_ledger(tmp_path):
    db = tmp_path / "reset.db"
    initialize(db, reset=True)
    with connect(db) as conn:
        record_invoice(conn, invoice_number="INV-1001", decision="approved")

    initialize(db, reset=True)
    with connect(db) as conn:
        assert find_prior_invoices(conn, "INV-1001") == []
