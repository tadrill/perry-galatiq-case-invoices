"""Tests that InvoiceState's reducers behave as the design assumes.

The whole communication model rests on this: fields several nodes contribute to must
append, and single-author fields must overwrite. Getting an `Annotated[..., operator.add]`
wrong is silent -- the graph runs, and the validator's findings simply vanish when the
approver writes its own update. So it is asserted against a real compiled graph rather
than taken on trust.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from invoice_agents.schemas import Finding, InvoiceData, Severity
from invoice_agents.state import InvoiceState, initial_state, log


def finding(code: str) -> Finding:
    return Finding(code=code, severity=Severity.WARNING, message=code)


def build(*nodes) -> object:
    """Compile a linear graph over InvoiceState from the given node functions."""
    graph = StateGraph(InvoiceState)
    previous = START
    for index, node in enumerate(nodes):
        name = f"n{index}"
        graph.add_node(name, node)
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()


def run(*nodes) -> InvoiceState:
    app = build(*nodes)
    return app.invoke(
        initial_state(source_path="x.json", source_format="json", raw_text="{}")
    )


def test_audit_log_accumulates_across_nodes():
    """Without the reducer, the second node's update would discard the first's entry."""
    result = run(
        lambda state: {"audit_log": [log("first", "ran")]},
        lambda state: {"audit_log": [log("second", "ran")]},
    )

    assert [entry.node for entry in result["audit_log"]] == ["first", "second"]


def test_findings_accumulate_across_nodes():
    result = run(
        lambda state: {"findings": [finding("a"), finding("b")]},
        lambda state: {"findings": [finding("c")]},
    )

    assert [f.code for f in result["findings"]] == ["a", "b", "c"]


def test_errors_accumulate_across_nodes():
    result = run(
        lambda state: {"errors": ["first failure"]},
        lambda state: {"errors": ["second failure"]},
    )

    assert result["errors"] == ["first failure", "second failure"]


def test_invoice_is_overwritten_not_appended():
    """Single-author fields must replace. A re-extraction supersedes its predecessor
    rather than leaving two invoices in state."""
    result = run(
        lambda state: {"invoice": InvoiceData(invoice_number="FIRST")},
        lambda state: {"invoice": InvoiceData(invoice_number="SECOND")},
    )

    assert result["invoice"].invoice_number == "SECOND"


def test_counters_overwrite_so_increments_are_not_double_counted():
    """extraction_attempts is read-modify-write. An additive reducer here would make
    the retry bound fire at the wrong time."""
    result = run(
        lambda state: {"extraction_attempts": state.get("extraction_attempts", 0) + 1},
        lambda state: {"extraction_attempts": state.get("extraction_attempts", 0) + 1},
    )

    assert result["extraction_attempts"] == 2


def test_a_node_may_return_only_the_keys_it_owns():
    """total=False: nodes return partial updates and everything else is left alone."""
    result = run(
        lambda state: {"invoice": InvoiceData(invoice_number="INV-1")},
        lambda state: {"audit_log": [log("second", "ran")]},
    )

    assert result["invoice"].invoice_number == "INV-1"
    assert result["source_path"] == "x.json"
    assert len(result["audit_log"]) == 1


def test_a_node_returning_nothing_leaves_state_intact():
    result = run(
        lambda state: {"findings": [finding("a")]},
        lambda state: {},
    )

    assert [f.code for f in result["findings"]] == ["a"]
