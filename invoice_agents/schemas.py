"""Typed payloads carried through the agent graph.

`InvoiceData` does double duty: it is the extractor's structured-output contract AND the
type of the `invoice` field in graph state. One schema, so there is no translation layer
between what the model promises and what the graph stores, and a malformed extraction
fails as a Pydantic ValidationError the extractor can retry against.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Decision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


# ---------------------------------------------------------------------------
# Extraction output
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    """One billed line, transcribed exactly as printed.

    `amount` is what the document CLAIMS this line costs. It is never computed here --
    Python recomputes quantity x unit_price downstream and compares. If the extractor
    were allowed to do that multiplication, the comparison would be checking the model
    against itself and would catch nothing.
    """

    model_config = ConfigDict(extra="forbid")

    item: str = Field(
        description=(
            "Item name copied VERBATIM from the document, including any spacing or "
            "qualifier as printed, e.g. 'Widget A' or 'WidgetA (rush order)'. Do not "
            "correct, expand or normalize it."
        )
    )
    quantity: float = Field(
        description="Quantity as printed. Negative if the document shows a negative."
    )
    unit_price: float | None = Field(
        default=None, description="Price per unit as printed, null if absent."
    )
    amount: float | None = Field(
        default=None,
        description=(
            "Line total AS STATED on the document, or null if not shown or unreadable. "
            "Never calculate this yourself."
        ),
    )
    note: str | None = Field(
        default=None,
        description="Any qualifier printed against the line, e.g. 'Volume discount'.",
    )


class InvoiceData(BaseModel):
    """Everything transcribed from one invoice document."""

    model_config = ConfigDict(extra="forbid")

    invoice_number: str | None = Field(
        default=None, description="Invoice identifier as printed, e.g. 'INV-1001' or '1002'."
    )
    revision: str | None = Field(
        default=None, description="Revision marker if the document declares one, e.g. 'R1'."
    )

    vendor_name: str | None = Field(
        default=None, description="Vendor/supplier name as printed. Null if blank or absent."
    )
    vendor_address: str | None = Field(default=None)

    invoice_date_raw: str | None = Field(
        default=None, description="Invoice date exactly as printed, e.g. '26-Jan-2O26'."
    )
    invoice_date: date | None = Field(
        default=None,
        description=(
            "invoice_date_raw normalized to YYYY-MM-DD. Null if it is not a resolvable "
            "calendar date."
        ),
    )
    due_date_raw: str | None = Field(
        default=None, description="Due date exactly as printed, e.g. 'yesterday'."
    )
    due_date: date | None = Field(
        default=None,
        description=(
            "due_date_raw normalized to YYYY-MM-DD. Null if it is not a resolvable "
            "calendar date -- a relative word like 'yesterday' is NOT resolvable."
        ),
    )

    line_items: list[LineItem] = Field(
        default_factory=list,
        description=(
            "Every billed line, in document order. Keep repeated items as separate "
            "entries; do not merge lines that name the same product."
        ),
    )

    subtotal: float | None = Field(default=None, description="Subtotal AS STATED.")
    tax_rate: float | None = Field(
        default=None, description="Tax rate as a decimal fraction, e.g. 0.08 for 8%."
    )
    tax_amount: float | None = Field(default=None, description="Tax AS STATED.")
    shipping: float | None = Field(
        default=None, description="Shipping/freight charge AS STATED, if itemized separately."
    )
    total: float | None = Field(default=None, description="Grand total AS STATED.")

    currency: str = Field(
        default="USD", description="ISO currency code. Infer from symbols or an explicit code."
    )
    payment_terms: str | None = Field(default=None, description="e.g. 'Net 30'.")
    notes: str | None = Field(
        default=None, description="Free-text notes printed on the document, verbatim."
    )

    unreadable_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Names of fields that were OCR-damaged, ambiguous, or could not be "
            "transcribed confidently -- including ones where an unambiguous correction "
            "was applied, so that the correction stays auditable."
        ),
    )

    @field_validator("currency", mode="before")
    @classmethod
    def _default_currency(cls, value: Any) -> Any:
        """An absent currency means USD rather than a validation failure."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return "USD"
        return str(value).strip().upper()

    @field_validator("vendor_name", "invoice_number", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """INV-1009 ships an empty vendor string; treat it as missing, not as a name."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


# ---------------------------------------------------------------------------
# Downstream stage payloads
# ---------------------------------------------------------------------------


class LineRecomputation(BaseModel):
    """Python's arithmetic verdict on a single line."""

    item: str
    quantity: float
    unit_price: float | None
    stated_amount: float | None
    computed_amount: float | None
    delta: float | None = None
    matches: bool = True


class Finding(BaseModel):
    """One thing wrong with an invoice."""

    code: str = Field(description="Stable machine code, e.g. 'stock.insufficient'.")
    severity: Severity
    message: str
    field: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReconciliationResult(BaseModel):
    """Deterministic arithmetic over the extracted invoice.

    Produced in pure Python. An arithmetic mismatch is ambiguous between "the extractor
    misread the document" and "the vendor's arithmetic is wrong", which is why it routes
    back to the extractor once before being treated as a real discrepancy.

    `discrepancies` therefore holds *only* arithmetic failures -- they are the ones a
    re-read could plausibly fix. Data-integrity problems like a negative quantity go in
    `integrity_findings` instead: re-reading INV-1009 will faithfully report -5 every
    time, so letting it drive the retry edge would just burn attempts.

    Those findings live here rather than being appended straight to state because this
    node runs again after each retry; accumulating them would duplicate every issue once
    per attempt. Held on the result, they are overwritten atomically and folded into
    state exactly once downstream.
    """

    lines: list[LineRecomputation] = Field(default_factory=list)

    computed_subtotal: float | None = None
    stated_subtotal: float | None = None
    computed_tax: float | None = None
    stated_tax: float | None = None
    shipping: float | None = None
    computed_total: float | None = None
    stated_total: float | None = None

    #: Per-SKU totals, summed across every line naming the same product. INV-1013 bills
    #: WidgetA on three lines; only the aggregate breaches stock.
    aggregated_quantities: dict[str, float] = Field(default_factory=dict)

    is_consistent: bool = True
    discrepancies: list[str] = Field(default_factory=list)
    integrity_findings: list[Finding] = Field(default_factory=list)


class ItemAdjudication(BaseModel):
    """The validator's ruling on an item name the catalog could not resolve outright.

    This is the judgment call the whole validator exists for. "Widget A" never reaches
    here -- it normalizes to an exact hit. "WidgetC" does, scoring 0.857 against both
    WidgetA and WidgetB, and the right answer is that a different trailing character in
    a SKU means a different product, not a typo.
    """

    model_config = ConfigDict(extra="forbid")

    invoice_name: str = Field(description="The item name as printed on the invoice.")
    verdict: Literal["same_product", "different_product", "unknown"] = Field(
        description=(
            "same_product: a misspelling of a catalog SKU. different_product: resembles "
            "a catalog SKU but is a distinct item. unknown: not in the catalog at all."
        )
    )
    resolved_sku: str | None = Field(
        default=None, description="The catalog SKU, only when verdict is same_product."
    )
    reasoning: str = Field(description="One or two sentences justifying the verdict.")


class VendorAdjudication(BaseModel):
    """The validator's ruling on a vendor name that did not match the approved list."""

    model_config = ConfigDict(extra="forbid")

    invoice_name: str
    verdict: Literal["same_vendor", "different_vendor", "unknown"]
    resolved_vendor: str | None = Field(default=None)
    reasoning: str


class RiskSignal(BaseModel):
    """A soft signal the validator judged worth surfacing.

    Deliberately open-ended: pressure language, a total sitting just under an approval
    threshold, a vendor whose banking details recently changed. These are the things a
    rule cannot enumerate ahead of time, which is why a model is asked about them.
    """

    model_config = ConfigDict(extra="forbid")

    signal: str = Field(description="Short label, e.g. 'urgency pressure in notes'.")
    severity: Severity
    evidence: str = Field(description="What in the document supports this.")


class ValidationVerdict(BaseModel):
    """Structured output from the validator agent.

    The agent returns judgments, not findings. Python turns these into `Finding` objects
    with stable codes, so the finding vocabulary stays fixed rather than drifting with
    whatever the model felt like calling something today.
    """

    model_config = ConfigDict(extra="forbid")

    item_adjudications: list[ItemAdjudication] = Field(default_factory=list)
    vendor_adjudication: VendorAdjudication | None = Field(default=None)
    risk_signals: list[RiskSignal] = Field(default_factory=list)
    summary: str = Field(description="Two or three sentences for a human reviewer.")


class ApprovalDecision(BaseModel):
    """The approver's verdict."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    rationale: str = Field(
        description="Why, in a few sentences, addressed to whoever releases the money."
    )
    driving_findings: list[str] = Field(
        default_factory=list,
        description=(
            "The exact `code` of every finding that drove this decision, copied from the "
            "findings you were given. Cite only codes that appear in that list. If nothing "
            "drove the decision because nothing was flagged, leave this empty."
        ),
    )
    policy_flags: list[str] = Field(
        default_factory=list, description="Policy conditions that applied to this invoice."
    )
    revised: bool = Field(
        default=False, description="True when a critique pass changed the first draft."
    )


class Critique(BaseModel):
    """A reviewer's second look at a draft approval decision.

    Separating the critique from the decision is what makes the reflection loop worth
    running. A model asked to "reconsider" its own answer in one breath tends to restate
    it; asked to argue against it as a distinct task, it finds the things it glossed
    over the first time.
    """

    model_config = ConfigDict(extra="forbid")

    concerns: list[str] = Field(
        default_factory=list,
        description="Specific problems with the draft: findings it under-weighted, "
        "reasoning that does not follow, policy it overlooked.",
    )
    recommend_revision: bool = Field(
        description="True only if a concern is serious enough to change the decision."
    )
    reasoning: str = Field(description="One or two sentences supporting the assessment.")


class PaymentResult(BaseModel):
    status: str
    vendor: str | None = None
    amount: float | None = None
    currency: str = "USD"
    reference: str | None = None
    detail: str | None = None


class LogEntry(BaseModel):
    """One step of the run, accumulated across the graph into an audit trail."""

    node: str
    event: str
    detail: str | None = None
    severity: Severity = Severity.INFO
    elapsed_ms: float | None = None
    at: datetime = Field(default_factory=datetime.now)
