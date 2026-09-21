"""Shared Rich console.

Windows consoles still default to cp1252, which raises UnicodeEncodeError the moment
any non-ASCII glyph is printed -- a check mark is enough. Reconfiguring the streams to
UTF-8 at import time keeps the CLI legible on Windows, macOS and Linux alike, so this
module is the one place output is constructed.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

THEME = Theme(
    {
        "ok": "bold green",
        "warn": "bold yellow",
        "bad": "bold red",
        "muted": "dim",
        "field": "cyan",
    }
)

OK = "✓"
FAIL = "✗"
WARN = "!"


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


_force_utf8()

console = Console(theme=THEME, soft_wrap=False)
