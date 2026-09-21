"""Match-key normalization for item and vendor names.

Invoices name the same thing many ways: "WidgetA", "Widget A" (OCR inserts a space),
"WidgetA (rush order)" (a qualifier on the line). Normalizing to a match key resolves
those deterministically, so the LLM never has to adjudicate a case with an obvious answer.

Deliberately conservative: it collapses spacing, punctuation and parentheticals, and
nothing else. "WidgetC" does NOT normalize to "WidgetA" -- a different trailing character
in a SKU is a different product, and that judgment belongs to the validator agent, which
sees the near-miss candidates and their scores.
"""

from __future__ import annotations

import re

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_item(raw: str) -> str:
    """Reduce an item name to its match key.

    >>> normalize_item("Widget A")
    'widgeta'
    >>> normalize_item("WidgetA (rush order)")
    'widgeta'
    >>> normalize_item("WidgetC")
    'widgetc'
    """
    if not raw:
        return ""
    lowered = raw.lower()
    without_qualifier = _PARENTHETICAL.sub(" ", lowered)
    return _NON_ALNUM.sub("", without_qualifier)


def normalize_vendor(raw: str) -> str:
    """Reduce a vendor name to its match key.

    Legal suffixes are kept -- "Widgets Inc." and "Widgets LLC" are different entities,
    and silently merging them would defeat the point of vendor verification.

    >>> normalize_vendor("Widgets Inc.")
    'widgetsinc'
    >>> normalize_vendor("QuickShip Distributers")
    'quickshipdistributers'
    """
    if not raw:
        return ""
    return _NON_ALNUM.sub("", raw.lower())
