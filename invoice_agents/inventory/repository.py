"""Read/write access to the mock inventory database.

These functions are what the validator agent's tools wrap. The contract that matters:
a lookup NEVER silently resolves an inexact name. It reports an exact hit, or it reports
scored candidates and lets the agent decide.

That distinction is the whole design. "Widget A" normalizes to an exact hit on WidgetA
and never troubles the model. "WidgetC" scores 0.86 against WidgetA -- high enough that
auto-resolution would quietly book a phantom product against the wrong SKU, which is
precisely the error rate this system exists to remove. So it comes back inexact, with
its near misses attached, for the agent to reason about.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .database import utcnow
from .naming import normalize_item, normalize_vendor

#: Candidates scoring below this are noise and are not shown to the agent.
CANDIDATE_FLOOR = 0.55

#: Default number of near misses returned alongside a failed exact match.
CANDIDATE_LIMIT = 3


@dataclass(frozen=True)
class ItemRecord:
    item: str
    display_name: str
    stock: int
    unit_price: float | None
    currency: str
    category: str | None
    status: str
    notes: str | None
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VendorRecord:
    name: str
    status: str
    payment_terms: str | None
    currency: str
    also_known_as: str | None
    notes: str | None
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Lookup:
    """Result of a name lookup. `exact` is None when nothing matched outright."""

    query: str
    normalized: str
    exact: ItemRecord | VendorRecord | None
    candidates: tuple[ItemRecord | VendorRecord, ...] = field(default=())

    @property
    def found(self) -> bool:
        return self.exact is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized": self.normalized,
            "exact_match": self.exact.to_dict() if self.exact else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "resolution_required": self.exact is None and bool(self.candidates),
        }


@dataclass(frozen=True)
class StockCheck:
    item: str
    requested: int
    available: int
    status: str  # ok | insufficient | out_of_stock | discontinued | unknown_item
    shortfall: int = 0
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LedgerEntry:
    id: int
    invoice_number: str | None
    revision: str | None
    vendor_name: str | None
    total: float | None
    currency: str
    line_hash: str | None
    source_path: str | None
    decision: str | None
    reason: str | None
    llm_mode: str | None
    processed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _item_from_row(row: sqlite3.Row, score: float = 1.0) -> ItemRecord:
    return ItemRecord(
        item=row["item"],
        display_name=row["display_name"],
        stock=row["stock"],
        unit_price=row["unit_price"],
        currency=row["currency"],
        category=row["category"],
        status=row["status"],
        notes=row["notes"],
        score=round(score, 3),
    )


def _vendor_from_row(row: sqlite3.Row, score: float = 1.0) -> VendorRecord:
    return VendorRecord(
        name=row["name"],
        status=row["status"],
        payment_terms=row["payment_terms"],
        currency=row["currency"],
        also_known_as=row["also_known_as"],
        notes=row["notes"],
        score=round(score, 3),
    )


def lookup_item(conn: sqlite3.Connection, name: str, *, limit: int = CANDIDATE_LIMIT) -> Lookup:
    """Resolve an invoice line's item name against the catalog."""
    key = normalize_item(name)
    if not key:
        return Lookup(query=name, normalized=key, exact=None)

    row = conn.execute("SELECT * FROM inventory WHERE normalized = ?", (key,)).fetchone()
    if row is not None:
        return Lookup(query=name, normalized=key, exact=_item_from_row(row))

    scored = (
        (_score(key, r["normalized"]), r)
        for r in conn.execute("SELECT * FROM inventory").fetchall()
    )
    candidates = sorted(
        (_item_from_row(r, s) for s, r in scored if s >= CANDIDATE_FLOOR),
        key=lambda c: c.score,
        reverse=True,
    )[:limit]
    return Lookup(query=name, normalized=key, exact=None, candidates=tuple(candidates))


def lookup_vendor(conn: sqlite3.Connection, name: str, *, limit: int = CANDIDATE_LIMIT) -> Lookup:
    """Resolve an invoice's vendor against the approved-vendor list."""
    key = normalize_vendor(name)
    if not key:
        return Lookup(query=name, normalized=key, exact=None)

    row = conn.execute("SELECT * FROM vendors WHERE normalized = ?", (key,)).fetchone()
    if row is not None:
        return Lookup(query=name, normalized=key, exact=_vendor_from_row(row))

    rows = conn.execute("SELECT * FROM vendors").fetchall()
    scored = (
        (
            max(
                _score(key, row["normalized"]),
                _score(key, normalize_vendor(row["also_known_as"] or "")),
            ),
            row,
        )
        for row in rows
    )
    candidates = sorted(
        (_vendor_from_row(r, s) for s, r in scored if s >= CANDIDATE_FLOOR),
        key=lambda c: c.score,
        reverse=True,
    )[:limit]
    return Lookup(query=name, normalized=key, exact=None, candidates=tuple(candidates))


def check_stock(conn: sqlite3.Connection, item: str, quantity: int) -> StockCheck:
    """Compare a requested quantity against the shelf.

    `quantity` is expected to be the TOTAL across every line naming this SKU. An invoice
    that splits one SKU over several lines (INV-1013 bills WidgetA three times) passes a
    per-line check while busting the real stock level, so aggregation happens upstream.
    """
    found = lookup_item(conn, item)
    record = found.exact
    if not isinstance(record, ItemRecord):
        return StockCheck(
            item=item,
            requested=quantity,
            available=0,
            status="unknown_item",
            shortfall=quantity,
            note="Not in catalog.",
        )

    if record.status == "discontinued":
        status, shortfall = "discontinued", quantity
    elif record.stock == 0:
        status, shortfall = "out_of_stock", quantity
    elif quantity > record.stock:
        status, shortfall = "insufficient", quantity - record.stock
    else:
        status, shortfall = "ok", 0

    return StockCheck(
        item=record.item,
        requested=quantity,
        available=record.stock,
        status=status,
        shortfall=shortfall,
        note=record.notes,
    )


def compute_line_hash(lines: Sequence[tuple[str, float, float]]) -> str:
    """Fingerprint an invoice's economic content from its (item, qty, unit_price) triples.

    Order-insensitive and name-normalized, so a resubmission that reorders lines or
    respells an item still collides with the original.
    """
    canonical = sorted(
        f"{normalize_item(item)}|{float(qty):.4f}|{float(price):.4f}"
        for item, qty, price in lines
    )
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()[:32]


def find_prior_invoices(
    conn: sqlite3.Connection, invoice_number: str | None, *, line_hash: str | None = None
) -> list[LedgerEntry]:
    """Find earlier submissions sharing this invoice number, or the same content.

    Catches both halves of the duplicate problem: INV-1004_revised reuses the invoice
    number with a different total, while a plain resubmission under a fresh number keeps
    the same line hash.
    """
    if not invoice_number and not line_hash:
        return []

    clause = "invoice_number = ?"
    params: list[Any] = [invoice_number]
    if line_hash:
        clause += " OR line_hash = ?"
        params.append(line_hash)

    rows = conn.execute(
        f"SELECT * FROM invoice_ledger WHERE {clause} ORDER BY processed_at DESC",
        params,
    ).fetchall()
    return [LedgerEntry(**dict(row)) for row in rows]


def record_invoice(
    conn: sqlite3.Connection,
    *,
    invoice_number: str | None = None,
    vendor_name: str | None = None,
    total: float | None = None,
    currency: str = "USD",
    revision: str | None = None,
    line_hash: str | None = None,
    source_path: str | None = None,
    decision: str | None = None,
    reason: str | None = None,
    llm_mode: str | None = None,
) -> int:
    """Append this run's outcome to the ledger. Returns the new row id."""
    cursor = conn.execute(
        """
        INSERT INTO invoice_ledger
            (invoice_number, revision, vendor_name, total, currency,
             line_hash, source_path, decision, reason, llm_mode, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            invoice_number,
            revision,
            vendor_name,
            total,
            currency,
            line_hash,
            source_path,
            decision,
            reason,
            llm_mode,
            utcnow(),
        ),
    )
    return int(cursor.lastrowid or 0)
