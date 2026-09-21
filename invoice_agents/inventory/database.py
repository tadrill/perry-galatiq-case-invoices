"""Connection handling and schema/seed initialization for the mock inventory DB."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .naming import normalize_item, normalize_vendor
from .seed import INVENTORY_SEED, VENDOR_SEED

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

#: Tables owned by this module, in safe drop order.
TABLES = ("invoice_ledger", "vendors", "inventory")


def utcnow() -> str:
    """Timestamp string used across every table."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection with row access by name, committing on clean exit.

    The legacy database is shared by every agent in the graph, so each unit of work
    gets its own short-lived connection rather than one long-lived global.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize(db_path: str | Path, *, reset: bool = False) -> dict[str, int]:
    """Create the schema and load seed data. Idempotent.

    Args:
        db_path: File to create or open. Persisted to disk so every agent sees it.
        reset: Drop existing tables first. Clears the ledger, which is what you want
            when re-running a demo that depends on duplicate detection.

    Returns:
        Row counts per table after seeding.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with connect(db_path) as conn:
        if reset:
            for table in TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")

        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        stamp = utcnow()

        conn.executemany(
            """
            INSERT INTO inventory
                (item, normalized, display_name, stock, unit_price,
                 currency, category, status, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item) DO UPDATE SET
                stock      = excluded.stock,
                unit_price = excluded.unit_price,
                status     = excluded.status,
                updated_at = excluded.updated_at
            """,
            [
                (item, normalize_item(item), display, stock, price,
                 currency, category, status, notes, stamp)
                for item, display, stock, price, currency, category, status, notes
                in INVENTORY_SEED
            ],
        )

        conn.executemany(
            """
            INSERT INTO vendors
                (name, normalized, status, payment_terms, currency,
                 also_known_as, notes, approved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                status        = excluded.status,
                payment_terms = excluded.payment_terms
            """,
            [
                (name, normalize_vendor(name), status, terms, currency, aka, notes, stamp)
                for name, status, terms, currency, aka, notes in VENDOR_SEED
            ],
        )

        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in TABLES
        }
