"""One command that sets everything up and processes the whole invoice inbox.

    python scripts/demo.py             # run all 25 invoices offline, then report
    python scripts/demo.py --serve     # ...and open the web console on the result
    python scripts/demo.py --archived  # replay the recorded live Grok run instead
    python scripts/demo.py --live      # run against the real Grok API

Defaults to offline replay, so it needs no API key and no network. Every extraction is
real `grok-4.6` output recorded earlier and replayed deterministically; the arithmetic,
database checks, policy and payment logic all execute for real either way.

Offline it finishes in a couple of seconds. With --live it is roughly 90s per invoice.
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.box import ROUNDED
from rich.rule import Rule
from rich.table import Table

from invoice_agents.archive import ArchiveError, ledger_provenance, restore
from invoice_agents.config import get_settings
from invoice_agents.console import FAIL, OK, WARN, console
from invoice_agents.graph import build_graph, process_invoice
from invoice_agents.ingestion import SUPPORTED_SUFFIXES, DocumentLoadError, load_document
from invoice_agents.inventory import initialize
from invoice_agents.state import summarize

STYLE = {"approved": "ok", "rejected": "bad", "needs_review": "warn", None: "bad"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="start the web console afterwards")
    parser.add_argument("--live", action="store_true", help="call the real Grok API")
    parser.add_argument(
        "--archived",
        action="store_true",
        help="replay the recorded live run instead of executing the pipeline",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite a ledger holding live-model results"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.archived and args.live:
        parser.error("--archived replays a recorded run; it cannot also be --live")

    if not args.live:
        # An env var beats the .env file, so this holds even if a key is configured.
        os.environ["LLM_MODE"] = "mock"
        get_settings.cache_clear()

    settings = get_settings()
    if args.live and not settings.has_api_key:
        console.print(
            f"[bad]{FAIL}[/] --live needs XAI_API_KEY. Copy .env.example to .env and add "
            f"a key, or drop --live to use offline replay."
        )
        return 1

    console.print(Rule("[bold]Invoice pipeline — full run[/]", style="muted"))
    console.print(
        f"[muted]llm {settings.llm_mode}"
        + (f" ({settings.grok_model})" if settings.llm_mode == "grok" else " (recorded replay)")
        + f"  ·  db {settings.resolved_db_path.name}[/]\n"
    )

    # Rebuilding the database drops the ledger. That is right for a demo and wrong for a
    # database holding a live run someone spent 40 minutes and real quota producing, so
    # say what is about to be lost rather than discovering it afterwards.
    existing = ledger_provenance(settings.resolved_db_path)
    at_risk = {mode: n for mode, n in existing.items() if mode != "mock"}
    if at_risk and not args.force:
        unknown_only = set(at_risk) == {"unknown"}
        origin = (
            "of unknown origin — this ledger predates provenance tracking, so there is no "
            "way to tell whether a model produced them"
            if unknown_only
            else f"from {', '.join(sorted(at_risk))}"
        )
        console.print(
            f"[warn]{WARN}[/] The ledger holds [bold]{sum(at_risk.values())}[/] "
            f"decision(s) {origin}."
        )
        console.print(
            "[muted]    Rebuilding discards them. Replay a recorded run with --archived, "
            "or pass --force to overwrite.[/]"
        )
        return 1

    if args.archived:
        try:
            meta = restore()
        except ArchiveError as exc:
            console.print(f"[bad]{FAIL}[/] {exc}")
            return 1
        console.print(
            f"[ok]{OK}[/] replayed [bold]{meta['restored']}[/] decisions recorded from "
            f"[field]{meta.get('model', 'the live API')}[/] "
            f"[muted]({meta.get('started', '?')})[/]"
        )
        console.print(
            "[muted]  These are the model's own rationales, not the offline stand-in's. "
            "Nothing was executed.[/]\n"
        )
        if not args.serve:
            console.print("[muted]Browse them with:  python scripts/serve.py[/]")
            return 0
        return _serve(args.port)

    counts = initialize(settings.resolved_db_path, reset=True)
    console.print(
        f"[ok]{OK}[/] database built  "
        + "  ".join(f"[muted]{k}={v}[/]" for k, v in counts.items())
    )

    paths = sorted(
        p
        for p in settings.resolved_invoice_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        console.print(f"[bad]{FAIL}[/] No invoices in {settings.resolved_invoice_dir}")
        return 1

    console.print(f"[ok]{OK}[/] processing {len(paths)} documents\n")

    graph = build_graph()
    table = Table(box=ROUNDED, header_style="bold")
    table.add_column("document", no_wrap=True, min_width=26)
    table.add_column("invoice", no_wrap=True, min_width=9)
    table.add_column("total", justify="right", no_wrap=True, min_width=10)
    table.add_column("decision", no_wrap=True, min_width=12)
    table.add_column("why", overflow="ellipsis", no_wrap=True, max_width=52)

    tally: dict[str, int] = {}
    for path in paths:
        try:
            document = load_document(path)
            state = process_invoice(
                source_path=str(document.path),
                source_format=document.source_format,
                raw_text=document.raw_text,
                graph=graph,
            )
        except DocumentLoadError as exc:
            table.add_row(path.name, "—", "—", "[bad]error[/]", str(exc)[:60])
            tally["error"] = tally.get("error", 0) + 1
            continue

        summary = summarize(state)
        decision = summary["decision"]
        tally[decision or "error"] = tally.get(decision or "error", 0) + 1
        decision_obj = state.get("decision")
        why = (decision_obj.rationale.split(". ")[0] if decision_obj else "no decision")[:120]

        table.add_row(
            path.name,
            summary["invoice_number"] or "—",
            f"{summary['total']:,.2f}" if summary["total"] is not None else "—",
            f"[{STYLE.get(decision, 'bad')}]{(decision or 'error').replace('_', ' ')}[/]",
            why,
        )

    console.print(table)
    console.print(
        "  "
        + "   ".join(
            f"[{STYLE.get(k, 'bad')}]{v} {k.replace('_', ' ')}[/]"
            for k, v in sorted(tally.items())
        )
    )
    console.print(
        f"\n[muted]Every decision, with its reasoning, is now in "
        f"{settings.resolved_db_path.name}.[/]"
    )

    if not args.serve:
        console.print(
            "[muted]Browse it with:  python scripts/serve.py    "
            "(or re-run this with --serve)[/]"
        )
        return 0

    return _serve(args.port)


def _serve(port: int) -> int:
    """Start the console. Shared by the normal path and --archived."""
    import uvicorn

    console.print(
        f"\n[ok]{OK}[/] console on [field]http://127.0.0.1:{port}[/]  "
        f"[muted]ctrl-c to stop[/]\n"
    )
    uvicorn.run("invoice_agents.web.app:app", host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
