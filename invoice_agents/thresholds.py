"""Every tunable number in the pipeline, grouped by who would want to move it.

Nothing here is read from the environment, deliberately. These are policy, not deployment
configuration: changing one changes what the system decides about somebody's money, so it
belongs in review like any other code change. Values that genuinely are
environment-dependent -- API keys, model ids, file paths -- live in `config.py`.
"""

from __future__ import annotations

from decimal import Decimal

# ---------------------------------------------------------------------------
# Approval policy — what a finance team decides
# ---------------------------------------------------------------------------

#: At or above this, an invoice gets extra scrutiny: size alone will not refuse it, but it
#: loses the benefit of the doubt, and it cannot be auto-approved while any warning is
#: still open. The assignment sets this at 10,000.
SCRUTINY_THRESHOLD = 10_000.0


# ---------------------------------------------------------------------------
# Validation tolerances — what an AP team decides
# ---------------------------------------------------------------------------

#: How far a line's unit price may drift from the catalog before it is worth a human's
#: attention. INV-1013's volume discounts sit at -4% and pass; INV-1010's rush line is
#: +20% and does not. Tightening this floods the review queue with ordinary commercial
#: variation; loosening it lets quiet price creep through.
PRICE_TOLERANCE = 0.10

#: "Net 30" rarely means exactly 30 days to a vendor's billing system, so a due date this
#: many days either side of what the terms imply is accepted without comment. At 5 days,
#: every supplied invoice passes except INV-1002, which is dated and due the same day
#: while claiming Net 30.
DUE_DATE_TOLERANCE_DAYS = 5

#: Sales tax above this is not a rate anyone charges on industrial parts. Set clear of the
#: highest legitimate rate in the sample set (10%) so ordinary invoices never trip it, and
#: low enough to catch a padded tax line dressed up as a statutory charge.
MAX_PLAUSIBLE_TAX_RATE = Decimal("0.25")

#: Freight above this share of the goods is worth confirming. Shipping is the easiest place
#: to inflate a bill because, unlike a line item, nothing validates it against a catalog
#: price or a stock level.
MAX_SHIPPING_RATIO = Decimal("0.25")

#: Vendors round tax by their own conventions, so figures within a cent of each other are
#: treated as agreeing. Two cents is a real disagreement.
MONEY_TOLERANCE = Decimal("0.01")


# ---------------------------------------------------------------------------
# Catalog matching — what a data owner decides
# ---------------------------------------------------------------------------

#: Similarity below this is noise and is never offered to the agent as a candidate.
#: "WidgetC" scores 0.857 against WidgetA, well above; "SuperGizmo" scores against nothing.
CANDIDATE_FLOOR = 0.55

#: How many near misses accompany a failed exact match. More than a handful is not evidence,
#: it is a list.
CANDIDATE_LIMIT = 3


# ---------------------------------------------------------------------------
# Loop bounds — engineering. Each of these stops something running forever.
# ---------------------------------------------------------------------------

#: Extractor passes allowed: one read plus one re-read after an arithmetic mismatch. Enough
#: to tell a misread from an inconsistent document. INV-1013 can never reconcile, so an
#: unbounded loop would spin on it.
MAX_EXTRACTION_ATTEMPTS = 2

#: Critique passes allowed. Two means the draft is reviewed, may be revised once, and the
#: revision is reviewed. A critic can always find one more thing to say.
MAX_CRITIQUE_ROUNDS = 2

#: Repairs of malformed model output inside a single extraction, with the validation error
#: fed back each time.
MAX_REPAIRS = 2

#: Tool-calling rounds before the validator is cut off. Four tools and a handful of
#: unresolved names; beyond this it is looping, not investigating.
MAX_TOOL_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Safety limits — engineering backstops, not part of the intended control flow
# ---------------------------------------------------------------------------

#: Hard ceiling on node executions per invoice, independent of the loop counters above.
#: Guards against a routing bug becoming a hang.
RECURSION_LIMIT = 40

#: A single document is truncated past this, so a pathological file cannot exhaust the
#: context window.
MAX_DOCUMENT_CHARS = 100_000

#: Rationales are stored whole because they are the record a human acts on. This cap only
#: stops a runaway model response from bloating the ledger.
MAX_REASON_CHARS = 4_000


# ---------------------------------------------------------------------------
# Offline stand-ins — affect only the no-API-key path, never a live run
# ---------------------------------------------------------------------------

#: Above this similarity the offline validator calls a near miss a misspelling. "QuickShip
#: Distributers" scores 0.952 and resolves; "WidgetC" scores 0.857 and does not. A real
#: model reasons about *why* two strings differ rather than comparing a float.
MOCK_RESOLUTION_THRESHOLD = 0.90

#: What the offline approver adds up to decide fraud is corroborated rather than merely
#: suspected, at which point it rejects instead of referring to a human.
MOCK_FRAUD_WEIGHTS: dict[str, int] = {
    "vendor.unapproved": 2,
    "vendor.blocked": 2,
    "stock.discontinued": 2,
    "duplicate.resubmission": 2,
    "duplicate.same_content": 2,
    "catalog.unknown_item": 1,
    "vendor.missing": 1,
}

#: The score at which those weights amount to a refusal.
MOCK_CONCLUSIVE_SCORE = 3
