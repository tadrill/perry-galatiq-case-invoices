"""Create and seed the mock legacy inventory database.

    python scripts/init_db.py           # create if absent, refresh seed rows
    python scripts/init_db.py --reset   # drop everything first, including the ledger
"""

from __future__ import annotations

import argparse
import sys

from rich.table import Table

from invoice_agents.config import get_settings
from invoice_agents.console import OK, console
from invoice_agents.inventory import connect, initialize


def render(db_path) -> None:
    with connect(db_path) as conn:
        inventory = Table(title="inventory", title_justify="left", header_style="bold")
        for column in ("item", "stock", "unit price", "status", "note"):
            inventory.add_column(column)
        for row in conn.execute("SELECT * FROM inventory ORDER BY category, item"):
            stock_style = "red" if row["stock"] == 0 else "green"
            price = f"{row['unit_price']:,.2f}" if row["unit_price"] is not None else "--"
            inventory.add_row(
                row["item"],
                f"[{stock_style}]{row['stock']}[/]",
                price,
                row["status"],
                (row["notes"] or "").split(".")[0][:52] or "--",
            )

        vendors = Table(title="vendors", title_justify="left", header_style="bold")
        for column in ("name", "status", "terms", "ccy"):
            vendors.add_column(column)
        for row in conn.execute("SELECT * FROM vendors ORDER BY status, name"):
            style = {"approved": "green", "unverified": "yellow", "blocked": "red"}[
                row["status"]
            ]
            vendors.add_row(
                row["name"],
                f"[{style}]{row['status']}[/]",
                row["payment_terms"] or "--",
                row["currency"],
            )

        ledger_count = conn.execute("SELECT COUNT(*) FROM invoice_ledger").fetchone()[0]

    console.print(inventory)
    console.print(vendors)
    console.print(f"\n[bold]invoice_ledger:[/] {ledger_count} row(s) — fills as invoices process.")
    console.print(
        "[dim]Absent by design: Fraudster LLC (INV-1003), NoProd Industries (INV-1008) — "
        "both should fail vendor verification.[/]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop existing tables first")
    parser.add_argument("--db-path", default=None, help="override DB_PATH")
    args = parser.parse_args()

    settings = get_settings()
    db_path = args.db_path or settings.resolved_db_path

    counts = initialize(db_path, reset=args.reset)
    console.print(
        f"[ok]{OK}[/] {'Rebuilt' if args.reset else 'Initialized'} [bold]{db_path}[/]  "
        + "  ".join(f"{table}={n}" for table, n in counts.items())
        + "\n"
    )
    render(db_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
