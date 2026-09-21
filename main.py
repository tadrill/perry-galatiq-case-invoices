"""Invoice processing pipeline — command line entry point.

    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --all                      # the whole inbox, with a summary
    python main.py --invoice_path=... -v      # show the full audit trail
    python main.py --invoice_path=... --json  # machine-readable, for piping

Run scripts/init_db.py first to build the inventory database.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.box import ROUNDED
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from invoice_agents.config import get_settings
from invoice_agents.console import FAIL, OK, WARN, console
from invoice_agents.graph import build_graph, process_invoice
from invoice_agents.ingestion import SUPPORTED_SUFFIXES, DocumentLoadError, load_document
from invoice_agents.inventory import connect
from invoice_agents.schemas import Decision, Severity
from invoice_agents.state import InvoiceState, summarize

DECISION_STYLE = {
    Decision.APPROVED: ("ok", OK),
    Decision.REJECTED: ("bad", FAIL),
    Decision.NEEDS_REVIEW: ("warn", WARN),
}

SEVERITY_STYLE = {
    Severity.CRITICAL: "bad",
    Severity.WARNING: "warn",
    Severity.INFO: "muted",
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_header(state: InvoiceState) -> None:
    invoice = state.get("invoice")
    if invoice is None:
        console.print("  [bad]Extraction produced no invoice.[/]")
        return

    total = (
        f"{invoice.total:,.2f} {invoice.currency}" if invoice.total is not None else "no total"
    )
    title = Text()
    title.append(invoice.invoice_number or "UNNUMBERED", style="bold")
    title.append("  ·  ")
    title.append(invoice.vendor_name or "NO VENDOR", style="field")

    detail = (
        f"issued {invoice.invoice_date or invoice.invoice_date_raw or '?'}"
        f"  ·  due {invoice.due_date or invoice.due_date_raw or '?'}"
        f"  ·  {invoice.payment_terms or 'no terms'}"
        f"  ·  {len(invoice.line_items)} line(s)"
    )

    grid = Table.grid(expand=True)
    grid.add_column(overflow="ellipsis")
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(title, Text(total, style="bold"))
    grid.add_row(Text(detail, style="dim"), "")
    console.print(grid)

    attempts = state.get("extraction_attempts", 0)
    if attempts > 1:
        console.print(
            f"[warn]{WARN}[/] re-read {attempts} times — the arithmetic did not reconcile "
            f"on the first pass"
        )


def render_arithmetic(state: InvoiceState) -> None:
    reconciliation = state.get("reconciliation")
    if reconciliation is None:
        return

    if reconciliation.is_consistent:
        console.print(
            f"\n[ok]{OK}[/] arithmetic  [muted]subtotal "
            f"{reconciliation.computed_subtotal:,.2f}, total "
            f"{reconciliation.computed_total:,.2f}[/]"
        )
        return

    console.print(f"\n[bad]{FAIL}[/] arithmetic does not reconcile")
    for discrepancy in reconciliation.discrepancies:
        console.print(f"    [bad]·[/] {discrepancy}")


def render_findings(state: InvoiceState) -> None:
    findings = list(state.get("findings") or [])
    if not findings:
        console.print(f"\n[ok]{OK}[/] no findings")
        return

    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    ranked = sorted(findings, key=lambda f: order[f.severity])

    table = Table(box=ROUNDED, show_header=True, header_style="bold", expand=True)
    table.add_column("", width=1, no_wrap=True)
    table.add_column("code", style="field", no_wrap=True)
    table.add_column("detail", overflow="fold")

    for finding in ranked:
        style = SEVERITY_STYLE[finding.severity]
        marker = {"bad": FAIL, "warn": WARN, "muted": "·"}[style]
        table.add_row(f"[{style}]{marker}[/]", finding.code, finding.message)

    criticals = sum(1 for f in findings if f.severity is Severity.CRITICAL)
    label = f"Findings ({len(findings)}"
    label += f", {criticals} critical)" if criticals else ")"
    console.print(f"\n[bold]{label}[/]")
    console.print(table)


def render_decision(state: InvoiceState) -> None:
    decision = state.get("decision")
    if decision is None:
        console.print(f"\n[bad]{FAIL} No decision was reached.[/]")
        return

    style, marker = DECISION_STYLE[decision.decision]
    heading = f"{marker} {decision.decision.value.upper().replace('_', ' ')}"
    if decision.revised:
        heading += "   [muted](revised after critique)[/]"

    body = Text(decision.rationale)
    console.print()
    console.print(
        Panel(body, title=f"[{style}]{heading}[/]", title_align="left", box=ROUNDED, padding=(0, 1))
    )

    critique = state.get("critique")
    if critique is not None and critique.concerns:
        console.print("[muted]critique:[/]")
        for concern in critique.concerns:
            console.print(f"  [muted]· {concern}[/]")

    payment = state.get("payment")
    if payment is None:
        return
    if payment.status == "success":
        console.print(
            f"\n[ok]{OK}[/] paid [bold]{payment.amount:,.2f} {payment.currency}[/] to "
            f"{payment.vendor}  [muted]{payment.reference}[/]"
        )
    else:
        console.print(f"\n[warn]{WARN}[/] payment {payment.status}")


def render_trail(state: InvoiceState) -> None:
    console.print("\n[bold]Audit trail[/]")

    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="muted", no_wrap=True, width=8)
    grid.add_column(no_wrap=True)
    grid.add_column(style="muted", overflow="fold", ratio=1)

    for entry in state.get("audit_log") or []:
        style = SEVERITY_STYLE[entry.severity]
        elapsed = f"{entry.elapsed_ms:.0f}ms" if entry.elapsed_ms else ""
        grid.add_row(
            elapsed, f"[{style}]{entry.node}.{entry.event}[/]", entry.detail or ""
        )

    console.print(grid)

    for error in state.get("errors") or []:
        console.print(f"  [bad]{FAIL} {error}[/]")


def render(state: InvoiceState, *, verbose: bool) -> None:
    render_header(state)
    render_arithmetic(state)
    render_findings(state)
    render_decision(state)
    if verbose:
        render_trail(state)


def render_summary(results: list[tuple[str, InvoiceState]]) -> None:
    table = Table(box=ROUNDED, show_header=True, header_style="bold", expand=True)
    table.add_column("invoice", no_wrap=True)
    table.add_column("vendor", overflow="ellipsis", max_width=26, no_wrap=True)
    table.add_column("total", justify="right", no_wrap=True)
    table.add_column("decision", no_wrap=True)
    table.add_column("findings", justify="right", no_wrap=True)

    tally: dict[str, int] = {}
    for name, state in results:
        summary = summarize(state)
        decision = summary["decision"]
        tally[decision or "error"] = tally.get(decision or "error", 0) + 1

        style = (
            DECISION_STYLE[Decision(decision)][0] if decision else "bad"
        )
        total = (
            f"{summary['total']:,.2f}" if summary["total"] is not None else "—"
        )
        counts = str(summary["findings"])
        if summary["critical"]:
            counts += f" [bad]({summary['critical']}!)[/]"

        table.add_row(
            name,
            summary["vendor"] or "—",
            total,
            f"[{style}]{(decision or 'error').replace('_', ' ')}[/]",
            counts,
        )

    console.print()
    console.print(Rule("[bold]Summary[/]"))
    console.print(table)

    console.print(
        "  "
        + "   ".join(
            f"[{DECISION_STYLE[Decision(k)][0] if k in {d.value for d in Decision} else 'bad'}]"
            f"{v} {k.replace('_', ' ')}[/]"
            for k, v in sorted(tally.items())
        )
    )

    render_failures(results)


def render_failures(results: list[tuple[str, InvoiceState]]) -> None:
    """Say what went wrong, and where.

    A batch that quietly drops documents is worse than one that fails loudly: the count
    at the bottom looks fine and nobody goes looking. Anything that errored gets named
    here, whether or not it still reached a decision.
    """
    failures = [
        (name, state)
        for name, state in results
        if state.get("errors") or state.get("decision") is None
    ]
    if not failures:
        return

    console.print(f"\n[bold]Failures ({len(failures)})[/]")
    for name, state in failures:
        errors = state.get("errors") or ["pipeline ended without a decision"]
        # A retried extraction records one error per attempt. Both are true, but the
        # reader only needs to be told once.
        for error in dict.fromkeys(errors):
            console.print(f"  [bad]{FAIL}[/] {name}")
            console.print(f"    [muted]{error[:150]}[/]")


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


def run_one(path: Path, graph: Any) -> InvoiceState | None:
    try:
        document = load_document(path)
    except DocumentLoadError as exc:
        console.print(f"[bad]{FAIL}[/] {exc}")
        return None

    return process_invoice(
        source_path=str(document.path),
        source_format=document.source_format,
        raw_text=document.raw_text,
        graph=graph,
    )


def discover(invoice_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in invoice_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process invoices through the multi-agent pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--invoice_path", help="a single invoice file to process")
    parser.add_argument("--all", action="store_true", help="process every invoice in the inbox")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the audit trail")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument(
        "--reset-ledger",
        action="store_true",
        help="clear processed-invoice history first, so duplicate detection starts fresh",
    )
    args = parser.parse_args()

    if not args.invoice_path and not args.all:
        parser.error("give --invoice_path=FILE or --all")

    settings = get_settings()

    if not settings.resolved_db_path.exists():
        console.print(
            f"[bad]{FAIL}[/] No inventory database at {settings.resolved_db_path}.\n"
            f"    Run: python scripts/init_db.py"
        )
        return 1

    if args.reset_ledger:
        with connect(settings.resolved_db_path) as conn:
            conn.execute("DELETE FROM invoice_ledger")
        if not args.json:
            console.print("[muted]ledger cleared[/]\n")

    paths = (
        discover(settings.resolved_invoice_dir) if args.all else [Path(args.invoice_path)]
    )
    if not paths:
        console.print(f"[bad]{FAIL}[/] No invoices found.")
        return 1

    if not args.json:
        console.print(
            f"[muted]llm: {settings.llm_mode}"
            + (f" ({settings.grok_model})" if settings.llm_mode == "grok" else "")
            + f"  ·  db: {settings.resolved_db_path.name}[/]"
        )

    graph = build_graph()
    results: list[tuple[str, InvoiceState]] = []
    failures = 0

    for path in paths:
        if not args.json:
            console.print()
            console.print(Rule(f"[bold]{path.name}[/]", align="left", style="muted"))

        state = run_one(path, graph)
        if state is None:
            failures += 1
            continue

        results.append((path.name, state))
        if not args.json:
            render(state, verbose=args.verbose)

    if args.json:
        payload = [{**summarize(state), "file": name} for name, state in results]
        # Straight to stdout, unstyled and unwrapped, so it survives a pipe.
        print(json.dumps(payload, indent=2, default=str))
    elif len(results) > 1:
        render_summary(results)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
