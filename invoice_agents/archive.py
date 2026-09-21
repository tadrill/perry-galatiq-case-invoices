"""Load an archived pipeline run back into the ledger.

The offline demo executes the graph for real, but with stand-ins where the model would
be, so the rationales it writes are a scoring table's rather than a model's. They read
the same. Somebody browsing the console has no way to tell which they are looking at --
which is exactly the mistake this module exists to stop.

Replaying an archived live run puts genuine model reasoning in front of a reader without
spending an API call, and every restored row is stamped with the backend that produced it
so the console can say so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import get_settings
from .inventory import connect, initialize, record_invoice

ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
DEFAULT_ARCHIVE = ARCHIVE_DIR / "grok-4.6-full-run.json"


class ArchiveError(RuntimeError):
    """The archive is missing or unusable."""


def load_archive(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_ARCHIVE
    if not path.exists():
        raise ArchiveError(
            f"No archived run at {path}. Produce one with: python scripts/demo.py --live"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveError(f"{path.name} is not valid JSON: {exc}") from exc

    if not payload.get("ledger"):
        raise ArchiveError(f"{path.name} contains no ledger rows.")
    return payload


def restore(path: Path | None = None, *, db_path: Path | None = None) -> dict[str, Any]:
    """Rebuild the database and replay an archived run into its ledger.

    Returns:
        The archive's metadata, plus a `restored` count.
    """
    settings = get_settings()
    db_path = db_path or settings.resolved_db_path
    payload = load_archive(path)

    initialize(db_path, reset=True)
    invoice_dir = settings.resolved_invoice_dir

    with connect(db_path) as conn:
        for entry in payload["ledger"]:
            source = entry.get("source_file")
            record_invoice(
                conn,
                invoice_number=entry.get("invoice_number"),
                revision=entry.get("revision"),
                vendor_name=entry.get("vendor_name"),
                total=entry.get("total"),
                currency=entry.get("currency") or "USD",
                line_hash=entry.get("line_hash"),
                source_path=str(invoice_dir / source) if source else None,
                decision=entry.get("decision"),
                reason=entry.get("reason"),
                llm_mode=payload.get("model", "grok"),
            )
        # The archive carries its own timestamps; keep them rather than stamping now,
        # so the console shows when the run actually happened.
        for entry in payload["ledger"]:
            if entry.get("processed_at") and entry.get("invoice_number"):
                conn.execute(
                    "UPDATE invoice_ledger SET processed_at = ? "
                    "WHERE invoice_number = ? AND reason = ?",
                    (entry["processed_at"], entry["invoice_number"], entry.get("reason")),
                )

    return {**{k: v for k, v in payload.items() if k != "ledger"},
            "restored": len(payload["ledger"])}


def ledger_provenance(db_path: Path | None = None) -> dict[str, int]:
    """How many ledger rows came from each backend. Empty when there is no ledger.

    A database written before provenance tracking existed has no `llm_mode` column, so
    its rows come back as "unknown". Callers guarding against data loss should treat that
    as *possibly* live rather than definitely not: the whole point of the guard is that a
    rationale gives no clue who wrote it, and an older ledger gives even less.
    """
    db_path = db_path or get_settings().resolved_db_path
    if not db_path.exists():
        return {}

    with connect(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(invoice_ledger)")}
        if "llm_mode" not in columns:
            total = conn.execute("SELECT COUNT(*) AS n FROM invoice_ledger").fetchone()["n"]
            return {"unknown": total} if total else {}

        rows = conn.execute(
            "SELECT COALESCE(llm_mode, 'unknown') AS m, COUNT(*) AS n "
            "FROM invoice_ledger GROUP BY m"
        ).fetchall()
    return {row["m"]: row["n"] for row in rows}
