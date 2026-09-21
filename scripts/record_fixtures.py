"""Record extraction fixtures from live Grok runs, so mock mode replays real output.

    python scripts/record_fixtures.py                      # every invoice lacking one
    python scripts/record_fixtures.py --invoice invoice_1012
    python scripts/record_fixtures.py --overwrite           # re-record everything

Requires XAI_API_KEY. Run it once after adding a key; from then on the whole pipeline
runs offline against genuine model output rather than anyone's guess at it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from invoice_agents.agents.extractor import build_extractor, build_messages
from invoice_agents.agents.mock_extraction import FIXTURE_DIR
from invoice_agents.config import Settings, get_settings
from invoice_agents.console import FAIL, OK, console
from invoice_agents.ingestion import DocumentLoadError, load_document
from invoice_agents.llm import LLMConfigurationError
from invoice_agents.state import initial_state

#: When an invoice exists in several formats, record the one the vendor actually sent.
#: PDFs are lossier, so they get their own fixture keyed by format instead of
#: overwriting the canonical one.
FORMAT_PRIORITY = {".json": 0, ".xml": 1, ".csv": 2, ".txt": 3, ".pdf": 4}


def fixture_name(path: Path, seen_stems: set[str]) -> str:
    """Canonical source keeps the bare stem; later formats get a format-suffixed name."""
    if path.stem in seen_stems:
        return f"{path.stem}.{path.suffix.lstrip('.')}"
    return path.stem


def discover(invoice_dir: Path, only: str | None) -> list[Path]:
    paths = [
        p
        for p in sorted(invoice_dir.iterdir())
        if p.is_file() and p.suffix.lower() in FORMAT_PRIORITY
    ]
    if only:
        paths = [p for p in paths if p.stem == only or p.name == only]
    return sorted(paths, key=lambda p: (p.stem, FORMAT_PRIORITY[p.suffix.lower()]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invoice", default=None, help="record just this stem or filename")
    parser.add_argument("--overwrite", action="store_true", help="re-record existing fixtures")
    args = parser.parse_args()

    base = get_settings()
    # Recording is pointless against the mock; force the live backend.
    settings = Settings(
        llm_mode="grok",
        xai_api_key=base.xai_api_key,
        xai_base_url=base.xai_base_url,
        grok_model=base.grok_model,
    )

    try:
        chain = build_extractor(settings)
    except LLMConfigurationError as exc:
        console.print(f"[bad]{FAIL}[/] {exc}")
        return 1

    paths = discover(base.resolved_invoice_dir, args.invoice)
    if not paths:
        console.print(f"[bad]{FAIL}[/] No invoices matched.")
        return 1

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"Recording with [field]{settings.grok_model}[/] → {FIXTURE_DIR}\n")

    seen_stems: set[str] = set()
    recorded = skipped = failed = 0

    for path in paths:
        name = fixture_name(path, seen_stems)
        seen_stems.add(path.stem)
        target = FIXTURE_DIR / f"{name}.json"

        if target.exists() and not args.overwrite:
            console.print(f"  [muted]· {path.name:28} exists, skipping[/]")
            skipped += 1
            continue

        try:
            document = load_document(path)
        except DocumentLoadError as exc:
            console.print(f"  [bad]{FAIL}[/] {path.name:28} {exc}")
            failed += 1
            continue

        state = initial_state(
            source_path=str(document.path),
            source_format=document.source_format,
            raw_text=document.raw_text,
        )

        try:
            invoice = chain.invoke(build_messages(state))
        except Exception as exc:  # noqa: BLE001 - report and continue to the next file
            console.print(f"  [bad]{FAIL}[/] {path.name:28} {type(exc).__name__}: {exc}")
            failed += 1
            continue

        payload = {
            "_provenance": f"recorded from {path.name} via {settings.grok_model}",
            "_recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **invoice.model_dump(mode="json"),
        }
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        console.print(
            f"  [ok]{OK}[/] {path.name:28} → {target.name:28} "
            f"[muted]{len(invoice.line_items)} line(s), total={invoice.total}[/]"
        )
        recorded += 1

    console.print(
        f"\n[bold]{recorded} recorded[/], {skipped} skipped, "
        + (f"[bad]{failed} failed[/]" if failed else "0 failed")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
