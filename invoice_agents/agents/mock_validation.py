"""Offline stand-in for the validator agent.

Two call shapes to satisfy, matching what the node actually does:

* the tool loop -- returns a reply with no tool calls, so the loop exits on its first
  turn and the node proceeds to ask for a verdict
* the structured verdict -- builds a `ValidationVerdict` from the database

The adjudication here is a similarity threshold, and that is a crude imitation of the
thing it stands in for. A model reasons about *why* two strings differ -- a space inside
a SKU is typography, a different trailing letter is a different part number -- whereas
this just compares a float. It happens to reach the right verdict on every invoice in
the sample set, but the threshold is doing none of the thinking that earns that answer.
Record real fixtures once a key is available if you want the genuine behaviour.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from ..config import get_settings
from ..inventory import connect, lookup_item, lookup_vendor
from ..policy import SCRUTINY_THRESHOLD
from ..schemas import (
    ItemAdjudication,
    RiskSignal,
    Severity,
    ValidationVerdict,
    VendorAdjudication,
)
from ..thresholds import MOCK_RESOLUTION_THRESHOLD as RESOLUTION_THRESHOLD

#: Phrases that pressure a clerk into paying without checking.
PRESSURE_PHRASES = (
    "urgent",
    "immediately",
    "wire transfer",
    "penalties",
    "at once",
    "avoid penalty",
)

_ITEM_RE = re.compile(r"UNRESOLVED ITEM:\s*(.*?)\s+\(nearest:")
_VENDOR_RE = re.compile(r"UNRESOLVED VENDOR:\s*(.*?)\s+\(nearest:")
_TOTAL_RE = re.compile(r"total:\s+([\d,]+\.\d{2})\s+([A-Z]{3})")
_NOTES_RE = re.compile(r"^\s*notes:\s*(.+)$", re.MULTILINE)


def _prompt_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


def _adjudicate_items(prompt: str, conn: Any) -> list[ItemAdjudication]:
    rulings: list[ItemAdjudication] = []

    for name in _ITEM_RE.findall(prompt):
        found = lookup_item(conn, name)
        best = found.candidates[0] if found.candidates else None

        if best is not None and best.score >= RESOLUTION_THRESHOLD:
            rulings.append(
                ItemAdjudication(
                    invoice_name=name,
                    verdict="same_product",
                    resolved_sku=best.item,
                    reasoning=(
                        f"Differs from catalog SKU {best.item} only cosmetically "
                        f"(similarity {best.score:.3f})."
                    ),
                )
            )
        elif best is not None:
            rulings.append(
                ItemAdjudication(
                    invoice_name=name,
                    verdict="different_product",
                    resolved_sku=None,
                    reasoning=(
                        f"Resembles {best.item} (similarity {best.score:.3f}) but the "
                        f"difference is in the SKU itself, not its typography, so this "
                        f"is a distinct part number rather than a misspelling."
                    ),
                )
            )
        else:
            rulings.append(
                ItemAdjudication(
                    invoice_name=name,
                    verdict="unknown",
                    resolved_sku=None,
                    reasoning="Nothing in the catalog resembles this item name.",
                )
            )

    return rulings


def _adjudicate_vendor(prompt: str, conn: Any) -> VendorAdjudication | None:
    match = _VENDOR_RE.search(prompt)
    if not match:
        return None

    name = match.group(1)
    found = lookup_vendor(conn, name)
    best = found.candidates[0] if found.candidates else None

    if best is not None and best.score >= RESOLUTION_THRESHOLD:
        return VendorAdjudication(
            invoice_name=name,
            verdict="same_vendor",
            resolved_vendor=best.name,
            reasoning=(
                f"Spelling variant of approved vendor {best.name} "
                f"(similarity {best.score:.3f})."
            ),
        )

    return VendorAdjudication(
        invoice_name=name,
        verdict="unknown",
        resolved_vendor=None,
        reasoning=(
            "No approved vendor resembles this name. An invoice from an unlisted "
            "vendor has not been cleared for payment."
        ),
    )


def _risk_signals(prompt: str) -> list[RiskSignal]:
    signals: list[RiskSignal] = []

    notes_match = _NOTES_RE.search(prompt)
    if notes_match:
        notes = notes_match.group(1)
        hits = [phrase for phrase in PRESSURE_PHRASES if phrase in notes.lower()]
        if hits:
            signals.append(
                RiskSignal(
                    signal="urgency and pressure language in the invoice notes",
                    severity=Severity.CRITICAL,
                    evidence=(
                        f"Notes contain {', '.join(repr(h) for h in hits)}. Pressure to "
                        f"pay without the usual checks is a standard invoice-fraud tactic."
                    ),
                )
            )

    total_match = _TOTAL_RE.search(prompt)
    if total_match:
        total = float(total_match.group(1).replace(",", ""))
        if SCRUTINY_THRESHOLD * 0.9 <= total < SCRUTINY_THRESHOLD:
            shortfall = SCRUTINY_THRESHOLD - total
            signals.append(
                RiskSignal(
                    signal="total sits just below the additional-scrutiny threshold",
                    severity=Severity.WARNING,
                    evidence=(
                        f"Total of {total:,.2f} is {shortfall:,.2f} under the "
                        f"{SCRUTINY_THRESHOLD:,.0f} threshold. Invoices priced to just "
                        f"miss a review gate deserve the review anyway."
                    ),
                )
            )

    return signals


def _summarize(
    items: list[ItemAdjudication],
    vendor: VendorAdjudication | None,
    signals: list[RiskSignal],
) -> str:
    parts: list[str] = []

    unresolved = [r.invoice_name for r in items if r.verdict != "same_product"]
    corrected = [r.invoice_name for r in items if r.verdict == "same_product"]
    if unresolved:
        parts.append(
            f"{len(unresolved)} item name(s) are not catalog products: "
            f"{', '.join(unresolved)}."
        )
    if corrected:
        parts.append(f"Resolved {', '.join(corrected)} to catalog SKUs.")
    if vendor and vendor.verdict == "same_vendor":
        parts.append(f"Vendor resolved to {vendor.resolved_vendor}.")
    elif vendor:
        parts.append(f"Vendor {vendor.invoice_name!r} is not on the approved list.")
    if signals:
        parts.append(f"{len(signals)} risk signal(s) raised.")
    if not parts:
        parts.append("Nothing required adjudication beyond the deterministic checks.")

    return " ".join(parts)


def mock_validate(
    messages: list[BaseMessage], schema: Any = None, tools: Any = None
) -> Any:
    """Offline validator. Signature matches the MockHandler contract."""
    if schema is None:
        # Tool-loop turn: decline to call anything so the loop exits immediately.
        return AIMessage(
            content="Reviewed the deterministic findings; proceeding to a verdict."
        )

    prompt = _prompt_text(messages)

    with connect(get_settings().resolved_db_path) as conn:
        items = _adjudicate_items(prompt, conn)
        vendor = _adjudicate_vendor(prompt, conn)

    signals = _risk_signals(prompt)

    return ValidationVerdict(
        item_adjudications=items,
        vendor_adjudication=vendor,
        risk_signals=signals,
        summary=_summarize(items, vendor, signals),
    )
