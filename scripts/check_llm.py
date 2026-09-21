"""Verify the configured LLM backend before running the pipeline.

    python scripts/check_llm.py

Checks three things in the order they break: the model constructs, it answers a plain
prompt, and it honours a structured-output schema. The extractor depends on all three,
so failing here is much easier to read than failing mid-graph.
"""

from __future__ import annotations

import sys
import time

from pydantic import BaseModel, Field

from invoice_agents.config import get_settings
from invoice_agents.console import FAIL, OK, WARN, console
from invoice_agents.llm import LLMConfigurationError, Purpose, get_llm


class Probe(BaseModel):
    """Tiny schema to confirm the structured-output path works end to end."""

    vendor: str = Field(description="The vendor name exactly as written")
    total: float = Field(description="The invoice total as a number")


PROBE_TEXT = "Vendor: Widgets Inc.\nTotal Amount: $5,000.00"


def main() -> int:
    settings = get_settings()

    console.print("[bold]Configuration[/]")
    console.print(f"  mode      {settings.llm_mode}")
    console.print(f"  model     {settings.grok_model}")
    console.print(f"  base url  {settings.xai_base_url}")
    key_state = "[green]set[/]" if settings.has_api_key else "[yellow]not set[/]"
    console.print(f"  api key   {key_state}\n")

    if settings.llm_mode == "mock":
        console.print(
            f"[warn]{WARN}[/] LLM_MODE=mock — no network call made. "
            "Set LLM_MODE=grok in .env to test the live API.\n"
        )

    try:
        llm = get_llm(Purpose.EXTRACTOR, settings)
    except LLMConfigurationError as exc:
        console.print(f"[bad]{FAIL}[/] {exc}")
        return 1

    console.print("[bold]Probes[/]")

    started = time.perf_counter()
    try:
        reply = llm.invoke("Reply with the single word: ready")
    except Exception as exc:  # noqa: BLE001 - surface whatever the backend raised
        console.print(f"  [bad]{FAIL}[/] plain completion — {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    text = str(reply.content).strip().replace("\n", " ")[:60]
    console.print(f"  [ok]{OK}[/] plain completion   {elapsed:5.2f}s  → {text!r}")

    started = time.perf_counter()
    try:
        parsed = llm.with_structured_output(Probe).invoke(
            f"Extract the vendor and total from this invoice fragment:\n\n{PROBE_TEXT}"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [bad]{FAIL}[/] structured output — {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    console.print(
        f"  [ok]{OK}[/] structured output {elapsed:5.2f}s  → "
        f"vendor={parsed.vendor!r} total={parsed.total}"
    )

    if parsed.total != 5000.0:
        console.print(
            f"\n[warn]{WARN}[/] Expected total 5000.0, got {parsed.total}. "
            "The connection works, but extraction fidelity looks off."
        )

    console.print(f"\n[ok]{OK} {settings.llm_mode} backend ready.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
