"""The LangGraph pipeline.

                     ┌──────────────────────────────┐
                     │      (arithmetic mismatch)   │
                     ▼                              │
    START ──▶ extract ──▶ reconcile ──┬─────────────┘
                                      ├──▶ validate ──▶ approve ◀──┐
                                      │                    │       │
                                      │                    ▼       │
                                      │                 critique ──┤ (revision
                                      │                    │       │  recommended)
                                      └──▶ finalize ◀──────┘       │
                                              │                    │
                                              ▼                    │
                                             END                   │

Two cycles, and both exist for a reason rather than for decoration.

The first sends a transcription whose arithmetic does not add up back to the extractor,
once. A mismatch is ambiguous between a misread and a genuinely inconsistent document,
and re-reading is what tells them apart. INV-1013 does not reconcile on either pass --
its stated total really is $50 wrong -- so it moves on and is judged on that basis.

The second lets a critic overturn a draft approval. It is a graph edge rather than a
loop inside one node so that a changed decision appears in the audit trail as two
separate events with the argument between them.

Both are bounded by counters in state. Without them, an invoice that can never reconcile
and a critic that can always find one more objection would each run forever.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents.approver import approve_node, critique_node, route_after_critique
from .agents.extractor import extract_node
from .agents.validator import validate_node
from .payment import finalize_node
from .reconciliation import reconcile_node, route_after_reconciliation
from .state import InvoiceState, initial_state

#: Hard ceiling on node executions, independent of the per-loop counters. A backstop
#: against a routing bug turning into a hang, not part of the intended control flow.
RECURSION_LIMIT = 40


def build_graph() -> Any:
    """Compile the invoice processing graph."""
    builder = StateGraph(InvoiceState)

    builder.add_node("extract", extract_node)
    builder.add_node("reconcile", reconcile_node)
    builder.add_node("validate", validate_node)
    builder.add_node("approve", approve_node)
    builder.add_node("critique", critique_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "reconcile")

    # Re-read on an arithmetic mismatch; give up if extraction never produced anything.
    builder.add_conditional_edges(
        "reconcile",
        route_after_reconciliation,
        {"extract": "extract", "validate": "validate", "abort": "finalize"},
    )

    builder.add_edge("validate", "approve")
    builder.add_edge("approve", "critique")

    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"approve": "approve", "finalize": "finalize"},
    )

    builder.add_edge("finalize", END)

    return builder.compile()


def process_invoice(
    *, source_path: str, source_format: str, raw_text: str, graph: Any = None
) -> InvoiceState:
    """Run one invoice end to end and return its final state.

    The returned state is the complete record of the run: what was extracted, what the
    arithmetic said, every finding, the decision and its reasoning, and the audit log.
    """
    compiled = graph or build_graph()
    return compiled.invoke(
        initial_state(
            source_path=source_path, source_format=source_format, raw_text=raw_text
        ),
        config={"recursion_limit": RECURSION_LIMIT},
    )
