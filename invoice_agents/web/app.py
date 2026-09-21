"""Local web console for the invoice pipeline.

Read-only over the inventory database and the invoice inbox, plus one write path: you can
run an invoice through the pipeline from the browser and watch what comes back.

Runs on localhost only. There is no authentication and none is wanted -- this reads a mock
database and a folder of sample documents on the machine it is started from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..archive import ledger_provenance
from ..config import get_settings
from ..graph import build_graph, process_invoice
from ..ingestion import SUPPORTED_SUFFIXES, DocumentLoadError, load_document
from ..inventory import connect
from ..schemas import Severity

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Invoice Pipeline Console", docs_url="/api/docs")

#: Compiling the graph is cheap but not free, and every run reuses it.
_graph: Any = None


def graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _invoice_dir() -> Path:
    return get_settings().resolved_invoice_dir


def _db_path() -> Path:
    return get_settings().resolved_db_path


def _require_db() -> Path:
    path = _db_path()
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No database at {path}. Run: python scripts/init_db.py",
        )
    return path


def _rows(query: str, params: tuple = ()) -> list[dict[str, Any]]:
    with connect(_require_db()) as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Flatten a finished run into something the browser can render."""
    invoice = state.get("invoice")
    reconciliation = state.get("reconciliation")
    decision = state.get("decision")
    critique = state.get("critique")
    payment = state.get("payment")
    findings = list(state.get("findings") or [])

    return {
        "source_path": state.get("source_path"),
        "invoice": invoice.model_dump(mode="json") if invoice else None,
        "reconciliation": (
            reconciliation.model_dump(mode="json") if reconciliation else None
        ),
        "findings": [f.model_dump(mode="json") for f in findings],
        "counts": {
            "total": len(findings),
            "critical": sum(1 for f in findings if f.severity is Severity.CRITICAL),
            "warning": sum(1 for f in findings if f.severity is Severity.WARNING),
            "info": sum(1 for f in findings if f.severity is Severity.INFO),
        },
        "decision": decision.model_dump(mode="json") if decision else None,
        "critique": critique.model_dump(mode="json") if critique else None,
        "payment": payment.model_dump(mode="json") if payment else None,
        "audit_log": [entry.model_dump(mode="json") for entry in state.get("audit_log") or []],
        "errors": list(state.get("errors") or []),
        "extraction_attempts": state.get("extraction_attempts", 0),
        "critique_rounds": state.get("critique_rounds", 0),
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    """Header stats: how the backend is configured and what is in the database."""
    settings = get_settings()
    inbox = [
        p for p in _invoice_dir().iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    stats: dict[str, Any] = {
        "llm_mode": settings.llm_mode,
        "model": settings.grok_model if settings.llm_mode == "grok" else "offline replay",
        "db": _db_path().name,
        "db_ready": _db_path().exists(),
        "documents": len(inbox),
        "formats": sorted({p.suffix.lstrip(".") for p in inbox}),
    }

    if stats["db_ready"]:
        counts = _rows(
            "SELECT (SELECT COUNT(*) FROM inventory) AS items,"
            " (SELECT COUNT(*) FROM vendors) AS vendors,"
            " (SELECT COUNT(*) FROM invoice_ledger) AS processed"
        )[0]
        stats.update(counts)
        stats["provenance"] = ledger_provenance()
        stats["decisions"] = {
            row["decision"] or "unknown": row["n"]
            for row in _rows(
                "SELECT decision, COUNT(*) AS n FROM invoice_ledger GROUP BY decision"
            )
        }
    return stats


@app.get("/api/invoices")
def invoices() -> list[dict[str, Any]]:
    """The inbox, annotated with each document's most recent processing outcome."""
    ledger: dict[str, dict[str, Any]] = {}
    if _db_path().exists():
        for row in _rows("SELECT * FROM invoice_ledger ORDER BY processed_at ASC"):
            if row.get("source_path"):
                ledger[Path(row["source_path"]).name] = row

    results = []
    for path in sorted(_invoice_dir().iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        last = ledger.get(path.name)
        results.append(
            {
                "filename": path.name,
                "format": path.suffix.lstrip("."),
                "bytes": path.stat().st_size,
                "invoice_number": last["invoice_number"] if last else None,
                "vendor": last["vendor_name"] if last else None,
                "total": last["total"] if last else None,
                "currency": last["currency"] if last else None,
                "decision": last["decision"] if last else None,
                "reason": last["reason"] if last else None,
                "llm_mode": last["llm_mode"] if last else None,
                "revision": last["revision"] if last else None,
                "processed_at": last["processed_at"] if last else None,
            }
        )
    return results


@app.get("/api/invoices/{filename}/raw")
def raw_document(filename: str) -> dict[str, Any]:
    """The source document as text. PDFs come back as extracted text."""
    path = (_invoice_dir() / Path(filename).name).resolve()
    if not path.is_file() or _invoice_dir().resolve() not in path.parents:
        raise HTTPException(status_code=404, detail=f"No such invoice: {filename}")

    try:
        document = load_document(path)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "filename": path.name,
        "format": document.source_format,
        "raw_text": document.raw_text,
        "lines": document.raw_text.count("\n") + 1,
    }


@app.post("/api/invoices/{filename}/process")
def process(filename: str) -> dict[str, Any]:
    """Run one invoice through the full graph.

    Deliberately a plain `def`, so FastAPI runs it in a worker thread and the server stays
    responsive while a live model call takes its time.
    """
    path = (_invoice_dir() / Path(filename).name).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such invoice: {filename}")

    _require_db()

    try:
        document = load_document(path)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    state = process_invoice(
        source_path=str(document.path),
        source_format=document.source_format,
        raw_text=document.raw_text,
        graph=graph(),
    )
    return _serialize_state(state)


@app.get("/api/inventory")
def inventory() -> list[dict[str, Any]]:
    return _rows("SELECT * FROM inventory ORDER BY status, category, item")


@app.get("/api/vendors")
def vendors() -> list[dict[str, Any]]:
    return _rows(
        "SELECT * FROM vendors ORDER BY CASE status"
        " WHEN 'blocked' THEN 0 WHEN 'unverified' THEN 1 ELSE 2 END, name"
    )


@app.get("/api/ledger")
def ledger() -> list[dict[str, Any]]:
    return _rows("SELECT * FROM invoice_ledger ORDER BY processed_at DESC, id DESC")


@app.delete("/api/ledger")
def clear_ledger() -> dict[str, Any]:
    """Forget processing history, so duplicate detection starts clean again."""
    with connect(_require_db()) as conn:
        removed = conn.execute("SELECT COUNT(*) FROM invoice_ledger").fetchone()[0]
        conn.execute("DELETE FROM invoice_ledger")
    return {"cleared": removed}


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api wins)
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
