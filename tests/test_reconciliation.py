"""Tests for the deterministic arithmetic node.

Most cases build an InvoiceData directly rather than going through extraction, so the
arithmetic is tested against known inputs instead of whatever the extractor happened to
produce. The last section runs the real documents end to end.
"""

from __future__ import annotations

import pytest

from invoice_agents.agents import extract_node
from invoice_agents.config import PROJECT_ROOT
from invoice_agents.ingestion import load_document
from invoice_agents.reconciliation import (
    MAX_EXTRACTION_ATTEMPTS,
    aggregate_quantities,
    integrity_findings,
    reconcile,
    reconcile_node,
    route_after_reconciliation,
)
from invoice_agents.schemas import InvoiceData, LineItem, ReconciliationResult, Severity
from invoice_agents.state import initial_state

INVOICE_DIR = PROJECT_ROOT / "data" / "invoices"


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    from invoice_agents.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def invoice(**overrides) -> InvoiceData:
    defaults = {
        "invoice_number": "INV-TEST",
        "vendor_name": "Test Vendor",
        "line_items": [LineItem(item="WidgetA", quantity=10, unit_price=250.0)],
    }
    return InvoiceData(**{**defaults, **overrides})


def extracted(filename: str) -> InvoiceData:
    document = load_document(INVOICE_DIR / filename)
    state = initial_state(
        source_path=str(document.path),
        source_format=document.source_format,
        raw_text=document.raw_text,
    )
    return extract_node(state)["invoice"]


# ---------------------------------------------------------------------------
# Line recomputation
# ---------------------------------------------------------------------------


def test_line_is_recomputed_from_quantity_and_price():
    result = reconcile(invoice(subtotal=2500.0, total=2500.0))
    line = result.lines[0]
    assert line.computed_amount == 2500.0
    assert line.matches


def test_stated_line_total_that_disagrees_is_flagged():
    result = reconcile(
        invoice(
            line_items=[LineItem(item="WidgetA", quantity=10, unit_price=250.0, amount=2400.0)],
            subtotal=2400.0,
            total=2400.0,
        )
    )
    assert not result.lines[0].matches
    assert result.lines[0].delta == -100.0
    assert any("line 1" in d for d in result.discrepancies)


def test_missing_stated_amount_is_not_a_discrepancy():
    """INV-1001 prints no line totals. Nothing to contradict, so nothing to report."""
    result = reconcile(invoice(subtotal=2500.0, total=2500.0))
    assert result.lines[0].stated_amount is None
    assert result.is_consistent


def test_recomputation_repairs_an_unreadable_line_total():
    """INV-1012 prints '$3,500.O0' with a letter O. The extractor yields null for the
    stated amount; 7 x 500 recovers the real figure and the invoice still balances."""
    result = reconcile(
        invoice(
            line_items=[
                LineItem(item="Widget A", quantity=12, unit_price=250.0, amount=3000.0),
                LineItem(item="WidgetB", quantity=7, unit_price=500.0, amount=None),
                LineItem(item="Gadget X", quantity=4, unit_price=750.0, amount=3000.0),
            ],
            subtotal=9500.0,
            tax_rate=0.05,
            tax_amount=475.0,
            total=9975.0,
        )
    )
    assert result.lines[1].computed_amount == 3500.0
    assert result.computed_subtotal == 9500.0
    assert result.is_consistent


# ---------------------------------------------------------------------------
# Subtotal, tax, total
# ---------------------------------------------------------------------------


def test_subtotal_mismatch_is_reported_with_the_difference():
    """INV-1009: states 1000.00 against lines summing to -250.00."""
    result = reconcile(
        invoice(
            line_items=[
                LineItem(item="WidgetA", quantity=-5, unit_price=250.0),
                LineItem(item="WidgetB", quantity=2, unit_price=500.0),
            ],
            subtotal=1000.0,
            tax_rate=0.0,
            tax_amount=0.0,
            total=-250.0,
        )
    )
    assert not result.is_consistent
    assert result.computed_subtotal == -250.0
    assert any("difference 1,250.00" in d for d in result.discrepancies)


def test_tax_is_computed_from_the_rate():
    result = reconcile(
        invoice(
            line_items=[LineItem(item="WidgetA", quantity=10, unit_price=250.0)],
            subtotal=2500.0,
            tax_rate=0.08,
            tax_amount=200.0,
            total=2700.0,
        )
    )
    assert result.computed_tax == 200.0
    assert result.is_consistent


def test_tax_uses_exact_decimal_arithmetic():
    """21040 * 0.07 is 1472.8000000000002 in binary floating point. Money is Decimal
    here precisely so the recomputation reports 1472.80 and not an artifact."""
    result = reconcile(
        invoice(
            line_items=[LineItem(item="WidgetA", quantity=1, unit_price=21040.0)],
            subtotal=21040.0,
            tax_rate=0.07,
            tax_amount=1472.80,
            total=22512.80,
        )
    )
    assert result.computed_tax == 1472.80
    assert result.is_consistent


def test_shipping_is_included_in_the_total():
    """INV-1010 adds a 150.00 shipping line outside the taxed subtotal."""
    result = reconcile(
        invoice(
            line_items=[
                LineItem(item="WidgetA", quantity=8, unit_price=250.0),
                LineItem(item="WidgetB", quantity=4, unit_price=500.0),
                LineItem(item="GadgetX", quantity=2, unit_price=750.0),
                LineItem(item="WidgetA (rush order)", quantity=4, unit_price=300.0),
            ],
            subtotal=6700.0,
            tax_amount=335.0,
            shipping=150.0,
            total=7185.0,
        )
    )
    assert result.computed_subtotal == 6700.0
    assert result.computed_total == 7185.0
    assert result.is_consistent


def test_a_wrong_tax_figure_is_reported_once_not_twice():
    """The total is built from the STATED tax, so one bad tax figure produces one
    discrepancy rather than cascading into a second, misleading total mismatch."""
    result = reconcile(
        invoice(
            line_items=[LineItem(item="WidgetA", quantity=10, unit_price=250.0)],
            subtotal=2500.0,
            tax_rate=0.08,
            tax_amount=999.0,  # should be 200.00
            total=3499.0,  # internally consistent with the wrong tax
        )
    )
    assert len(result.discrepancies) == 1
    assert "tax" in result.discrepancies[0]


def test_absent_stated_subtotal_is_not_compared():
    """INV-1002 prints only a total. A missing figure cannot contradict anything."""
    result = reconcile(
        invoice(
            line_items=[LineItem(item="GadgetX", quantity=20, unit_price=750.0)],
            subtotal=None,
            total=15000.0,
        )
    )
    assert result.is_consistent
    assert result.computed_subtotal == 15000.0


@pytest.mark.parametrize(
    ("stated_total", "consistent"),
    [(2500.00, True), (2500.01, True), (2499.99, True), (2500.02, False), (2400.00, False)],
)
def test_one_cent_of_rounding_drift_is_tolerated(stated_total, consistent):
    """Vendors round tax by their own conventions; a cent is not worth blocking a
    payment over, but two is a real disagreement."""
    result = reconcile(invoice(subtotal=2500.0, total=stated_total))
    assert result.is_consistent is consistent


# ---------------------------------------------------------------------------
# Per-SKU aggregation
# ---------------------------------------------------------------------------


def test_aggregation_sums_repeated_skus():
    """INV-1013 bills WidgetA on three lines: 15 + 5 + 2."""
    totals = aggregate_quantities(
        [
            LineItem(item="WidgetA", quantity=15, unit_price=250.0),
            LineItem(item="WidgetB", quantity=10, unit_price=500.0),
            LineItem(item="WidgetA", quantity=5, unit_price=240.0),
            LineItem(item="WidgetA", quantity=2, unit_price=250.0),
        ]
    )
    assert totals == {"WidgetA": 22.0, "WidgetB": 10.0}


def test_aggregation_groups_cosmetic_spelling_variants():
    """'WidgetA', 'Widget A' and 'WidgetA (rush order)' are one product on one shelf."""
    totals = aggregate_quantities(
        [
            LineItem(item="WidgetA", quantity=8, unit_price=250.0),
            LineItem(item="WidgetA (rush order)", quantity=4, unit_price=300.0),
            LineItem(item="Widget A", quantity=3, unit_price=250.0),
        ]
    )
    assert totals == {"WidgetA": 15.0}


def test_aggregation_keeps_distinct_skus_apart():
    """WidgetC is not WidgetA, however similar it looks."""
    totals = aggregate_quantities(
        [
            LineItem(item="WidgetA", quantity=4, unit_price=250.0),
            LineItem(item="WidgetC", quantity=3, unit_price=350.0),
        ]
    )
    assert totals == {"WidgetA": 4.0, "WidgetC": 3.0}


def test_aggregation_key_is_the_first_spelling_seen():
    totals = aggregate_quantities(
        [
            LineItem(item="Widget A", quantity=1, unit_price=250.0),
            LineItem(item="WidgetA", quantity=1, unit_price=250.0),
        ]
    )
    assert list(totals) == ["Widget A"]


# ---------------------------------------------------------------------------
# Data integrity (must not drive the retry loop)
# ---------------------------------------------------------------------------


def test_negative_quantity_is_an_integrity_finding_not_a_discrepancy():
    """A re-read of INV-1009 reports -5 every time, so routing it back to the extractor
    would burn an attempt for nothing."""
    result = reconcile(
        invoice(
            line_items=[LineItem(item="WidgetA", quantity=-5, unit_price=250.0)],
            subtotal=-1250.0,
            total=-1250.0,
        )
    )
    assert result.is_consistent
    assert not result.discrepancies
    codes = [f.code for f in result.integrity_findings]
    assert "data.negative_quantity" in codes
    assert result.integrity_findings[0].severity is Severity.CRITICAL


@pytest.mark.parametrize(
    ("line", "code"),
    [
        (LineItem(item="WidgetA", quantity=0, unit_price=250.0), "data.zero_quantity"),
        (LineItem(item="WidgetA", quantity=2.5, unit_price=250.0), "data.fractional_quantity"),
        (LineItem(item="WidgetA", quantity=2, unit_price=None), "data.missing_unit_price"),
        (LineItem(item="WidgetA", quantity=2, unit_price=-5.0), "data.negative_unit_price"),
        (LineItem(item="  ", quantity=2, unit_price=250.0), "data.missing_item_name"),
    ],
)
def test_integrity_problems_are_detected(line, code):
    result = reconcile(invoice(line_items=[line], subtotal=None, total=None))
    assert code in [f.code for f in result.integrity_findings]


def test_clean_invoice_has_no_integrity_findings():
    result = reconcile(invoice(subtotal=2500.0, total=2500.0))
    assert result.integrity_findings == []


# ---------------------------------------------------------------------------
# Node and routing
# ---------------------------------------------------------------------------


def test_node_writes_the_result_and_logs():
    state = initial_state(source_path="x.json", source_format="json", raw_text="{}")
    state["invoice"] = invoice(subtotal=2500.0, total=2500.0)

    update = reconcile_node(state)

    assert update["reconciliation"].is_consistent
    assert any(entry.event == "reconciled" for entry in update["audit_log"])


def test_node_tolerates_a_failed_extraction():
    state = initial_state(source_path="x.json", source_format="json", raw_text="{}")
    update = reconcile_node(state)

    assert update["reconciliation"] is None
    assert any(entry.event == "skipped" for entry in update["audit_log"])


def test_consistent_arithmetic_proceeds():
    state = initial_state(source_path="x", source_format="json", raw_text="{}")
    state["invoice"] = invoice()
    state["reconciliation"] = ReconciliationResult(is_consistent=True)
    state["extraction_attempts"] = 1
    assert route_after_reconciliation(state) == "validate"


def test_first_mismatch_routes_back_to_the_extractor():
    state = initial_state(source_path="x", source_format="json", raw_text="{}")
    state["invoice"] = invoice()
    state["reconciliation"] = ReconciliationResult(is_consistent=False, discrepancies=["x"])
    state["extraction_attempts"] = 1
    assert route_after_reconciliation(state) == "extract"


def test_a_persistent_mismatch_proceeds_rather_than_looping():
    """INV-1013's total is genuinely wrong, so re-reading will never reconcile it. After
    the bounded retry it moves forward to be judged, not retried into the ground."""
    state = initial_state(source_path="x", source_format="json", raw_text="{}")
    state["invoice"] = invoice()
    state["reconciliation"] = ReconciliationResult(is_consistent=False, discrepancies=["x"])
    state["extraction_attempts"] = MAX_EXTRACTION_ATTEMPTS
    assert route_after_reconciliation(state) == "validate"


def test_failed_extraction_retries_then_aborts():
    state = initial_state(source_path="x", source_format="json", raw_text="{}")
    state["extraction_attempts"] = 1
    assert route_after_reconciliation(state) == "extract"

    state["extraction_attempts"] = MAX_EXTRACTION_ATTEMPTS
    assert route_after_reconciliation(state) == "abort"


def test_integrity_findings_helper_is_safe_without_a_result():
    state = initial_state(source_path="x", source_format="json", raw_text="{}")
    assert integrity_findings(state) == []


# ---------------------------------------------------------------------------
# Against the real documents
# ---------------------------------------------------------------------------


def test_clean_invoices_reconcile():
    for filename in ("invoice_1001.txt", "invoice_1004.json", "invoice_1005.json"):
        result = reconcile(extracted(filename))
        assert result.is_consistent, f"{filename}: {result.discrepancies}"


def test_inv_1013_grand_total_is_fifty_dollars_wrong():
    """Its lines sum correctly and its 7% tax is right, but the stated total is not the
    sum of the two. An LLM-only check would have agreed with itself and passed it."""
    result = reconcile(extracted("invoice_1013.json"))

    assert result.computed_subtotal == 21040.00
    assert result.computed_tax == 1472.80
    assert result.computed_total == 22512.80
    assert result.stated_total == 22562.80
    assert not result.is_consistent
    assert any("total:" in d for d in result.discrepancies)


def test_inv_1013_aggregates_past_every_stock_level():
    result = reconcile(extracted("invoice_1013.json"))
    assert result.aggregated_quantities == {"WidgetA": 22.0, "WidgetB": 18.0, "GadgetX": 9.0}


def test_inv_1009_reports_both_a_bad_subtotal_and_a_bad_quantity():
    result = reconcile(extracted("invoice_1009.json"))

    assert not result.is_consistent
    assert any("subtotal" in d for d in result.discrepancies)
    assert "data.negative_quantity" in [f.code for f in result.integrity_findings]


def test_inv_1003_arithmetic_is_impeccable():
    """100 x 1000 really is 100000. The fraud in INV-1003 is not arithmetic, which is
    why catching it is the validator's job, not this node's."""
    result = reconcile(extracted("invoice_1003.txt"))
    assert result.is_consistent


# ---------------------------------------------------------------------------
# Plausibility: arithmetically perfect invoices that are still wrong
# ---------------------------------------------------------------------------


def integrity_codes(result) -> list[str]:
    return [f.code for f in result.integrity_findings]


def test_negative_tax_rate_is_critical():
    """INV-1017 dresses a rebate up as negative tax. The invoice adds up perfectly."""
    result = reconcile(
        invoice(subtotal=2500.0, tax_rate=-0.05, tax_amount=-125.0, total=2375.0)
    )
    assert result.is_consistent
    assert "tax.negative_rate" in integrity_codes(result)
    assert result.integrity_findings[0].severity is Severity.CRITICAL


def test_absurd_stated_tax_rate_is_flagged():
    """INV-1018 charges 85%. Nothing about the arithmetic objects."""
    result = reconcile(
        invoice(subtotal=2500.0, tax_rate=0.85, tax_amount=2125.0, total=4625.0)
    )
    assert result.is_consistent
    assert "tax.implausible_rate" in integrity_codes(result)


def test_absurd_implied_tax_rate_is_flagged():
    """INV-1021 states no rate at all -- only an amount. Deriving it is the only way to
    notice that the amount is 40% of the goods."""
    result = reconcile(invoice(subtotal=2500.0, tax_amount=1000.0, total=3500.0))

    found = next(f for f in result.integrity_findings if f.code == "tax.implausible_rate")
    assert found.evidence["derived"] is True
    assert "implied by" in found.message


@pytest.mark.parametrize("rate", [0.0, 0.05, 0.06, 0.07, 0.08, 0.10])
def test_ordinary_tax_rates_are_silent(rate):
    """Every rate appearing in the supplied invoices must pass without comment."""
    result = reconcile(
        invoice(subtotal=2500.0, tax_rate=rate, tax_amount=2500.0 * rate,
                total=2500.0 * (1 + rate))
    )
    assert not [c for c in integrity_codes(result) if c.startswith("tax.")]


def test_tax_check_survives_a_zero_subtotal():
    """Dividing by the subtotal to derive a rate must not blow up on an empty invoice."""
    result = reconcile(invoice(line_items=[], subtotal=0.0, tax_amount=0.0, total=0.0))
    assert not [c for c in integrity_codes(result) if c.startswith("tax.")]


def test_the_same_goods_billed_twice_is_flagged():
    """INV-1019 bills WidgetB x4 at 500.00 on two separate lines. The total is correct,
    stock is not breached, and nothing else in the pipeline notices."""
    result = reconcile(
        invoice(
            line_items=[
                LineItem(item="WidgetA", quantity=3, unit_price=250.0),
                LineItem(item="WidgetB", quantity=4, unit_price=500.0),
                LineItem(item="WidgetB", quantity=4, unit_price=500.0),
            ],
            subtotal=4750.0,
            total=4750.0,
        )
    )
    found = next(f for f in result.integrity_findings if f.code == "data.duplicate_line")
    assert found.evidence["first_line"] == 2
    assert found.evidence["duplicate_line"] == 3


def test_a_repeated_sku_at_a_different_price_is_not_a_duplicate():
    """INV-1013's volume discounts repeat WidgetA three times legitimately."""
    result = reconcile(
        invoice(
            line_items=[
                LineItem(item="WidgetA", quantity=15, unit_price=250.0),
                LineItem(item="WidgetA", quantity=5, unit_price=240.0, note="Volume"),
                LineItem(item="WidgetA", quantity=2, unit_price=250.0, note="Replacement"),
            ],
            subtotal=5450.0,
            total=5450.0,
        )
    )
    assert "data.duplicate_line" not in integrity_codes(result)


def test_duplicate_detection_sees_through_spelling():
    """'Widget A' and 'WidgetA' at the same quantity and price are the same line twice."""
    result = reconcile(
        invoice(
            line_items=[
                LineItem(item="WidgetA", quantity=2, unit_price=250.0),
                LineItem(item="Widget A", quantity=2, unit_price=250.0),
            ],
            subtotal=1000.0,
            total=1000.0,
        )
    )
    assert "data.duplicate_line" in integrity_codes(result)


def test_outsized_freight_is_flagged():
    """INV-1020 bills 4,800 of freight on 1,000 of goods. Shipping is checked against
    nothing -- no catalog price, no stock level -- which is what makes it attractive."""
    result = reconcile(invoice(subtotal=2500.0, shipping=25000.0, total=27500.0))

    found = next(
        f for f in result.integrity_findings if f.code == "charges.disproportionate"
    )
    assert "1000%" in found.message or "%" in found.message


def test_ordinary_freight_is_silent():
    """INV-1010 bills 150.00 of shipping on 6,700.00 of goods."""
    result = reconcile(invoice(subtotal=6700.0, shipping=150.0, total=6850.0))
    assert "charges.disproportionate" not in integrity_codes(result)


def test_negative_grand_total_is_critical():
    result = reconcile(
        invoice(
            line_items=[LineItem(item="WidgetA", quantity=10, unit_price=250.0)],
            subtotal=2500.0,
            total=-2500.0,
        )
    )
    assert "data.negative_total" in integrity_codes(result)


def test_a_clean_invoice_trips_none_of_the_plausibility_checks():
    result = reconcile(
        invoice(subtotal=2500.0, tax_rate=0.08, tax_amount=200.0, shipping=50.0, total=2750.0)
    )
    assert result.integrity_findings == []


# ---------------------------------------------------------------------------
# The new sample invoices, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "code"),
    [
        ("invoice_1017.json", "tax.negative_rate"),
        ("invoice_1018.csv", "tax.implausible_rate"),
        ("invoice_1019.json", "data.duplicate_line"),
        ("invoice_1020.txt", "charges.disproportionate"),
        ("invoice_1021.json", "tax.implausible_rate"),
    ],
)
def test_each_new_sample_invoice_trips_its_check(filename, code):
    """Every one of these reconciles arithmetically, so only the plausibility checks
    stand between them and payment."""
    result = reconcile(extracted(filename))

    assert result.is_consistent, f"{filename} should add up: {result.discrepancies}"
    assert code in integrity_codes(result)
