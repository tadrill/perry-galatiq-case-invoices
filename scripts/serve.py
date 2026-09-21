"""Start the local web console.

    python scripts/serve.py               # http://127.0.0.1:8000
    python scripts/serve.py --port 9000
    python scripts/serve.py --reload      # auto-restart while editing

Binds to localhost only. There is no authentication, because it serves a mock database
and a folder of sample invoices from the machine it is started on.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from invoice_agents.config import get_settings
from invoice_agents.console import OK, WARN, console


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.resolved_db_path.exists():
        console.print(
            f"[warn]{WARN}[/] No database at {settings.resolved_db_path} — "
            f"run [field]python scripts/init_db.py[/] first.\n"
        )

    console.print(f"[ok]{OK}[/] console on [field]http://{args.host}:{args.port}[/]")
    console.print(
        f"[muted]  llm {settings.llm_mode}"
        + (f" ({settings.grok_model})" if settings.llm_mode == "grok" else "")
        + f"  ·  db {settings.resolved_db_path.name}  ·  ctrl-c to stop[/]\n"
    )

    uvicorn.run(
        "invoice_agents.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
