"""Seed data for the mock legacy inventory database.

The first four SKUs and their stock levels are fixed by the assignment -- the sample
invoices are built against them. Everything else is chosen to make the validator's job
a real one rather than a lookup:

  CoolantPro   a legitimate stockout (active, stock 0), so "out of stock" and
               "fraudulent SKU" are distinguishable rather than one bucket
  FakeItem     discontinued with no price, the fraud canary INV-1003 reaches for
  vendors      Fraudster LLC and NoProd Industries are deliberately ABSENT, so the
               two suspect invoices also fail vendor verification
  QuickShip    seeded with the correct spelling "Distributors"; INV-1012 spells it
               "Distributers", which the fuzzy matcher surfaces as a near-miss
"""

from __future__ import annotations

# (item, display_name, stock, unit_price, currency, category, status, notes)
INVENTORY_SEED: tuple[tuple, ...] = (
    # --- Required by the assignment; stock levels are load-bearing for the test cases ---
    ("WidgetA", "Widget A", 15, 250.00, "USD", "widgets", "active", None),
    ("WidgetB", "Widget B", 10, 500.00, "USD", "widgets", "active", None),
    ("GadgetX", "Gadget X", 5, 750.00, "USD", "gadgets", "active", None),
    (
        "FakeItem",
        "Fake Item",
        0,
        None,
        "USD",
        None,
        "discontinued",
        (
            "Delisted 2025-11; no active supplier contract. Appearances on inbound "
            "invoices have historically been fraudulent."
        ),
    ),
    # --- Catalog realism: gives the lookup tool meaningful negatives to return ---
    ("PistonKit", "Piston Kit", 40, 120.00, "USD", "components", "active", None),
    ("BearingSet", "Bearing Set", 100, 45.00, "USD", "components", "active", None),
    (
        "CoolantPro",
        "Coolant Pro",
        0,
        80.00,
        "USD",
        "consumables",
        "active",
        "Backordered, restock expected 2026-02-15. Genuine stockout, not a bad SKU.",
    ),
)

# (name, status, payment_terms, currency, also_known_as, notes)
VENDOR_SEED: tuple[tuple, ...] = (
    ("Widgets Inc.", "approved", "Net 15", "USD", None, None),
    ("Gadgets Co.", "approved", "Net 30", "USD", None, None),
    ("Precision Parts Ltd.", "approved", "Net 30", "USD", None, None),
    ("Global Supply Chain Partners", "approved", "Net 60", "USD", None, None),
    ("Acme Industrial Supplies", "approved", "Net 15", "USD", None, None),
    ("MegaWidgets Corp", "approved", "Net 30", "USD", None, None),
    ("Consolidated Materials Group", "approved", "Net 30", "USD", None, None),
    ("Summit Manufacturing Co.", "approved", "Net 30", "USD", None, None),
    ("Atlas Industrial Supply", "approved", "Net 60", "USD", None, None),
    ("TechParts International", "approved", "Net 30", "EUR", None, None),
    ("Reliable Components Inc.", "approved", "Net 30", "USD", None, None),
    (
        "QuickShip Distributors",
        "unverified",
        "Net 30",
        "USD",
        "FastShip Ltd.",
        (
            "Rebranded from FastShip Ltd. in 2025-12. Banking details changed at the "
            "same time; re-verification pending."
        ),
    ),
    # Intentionally absent: "Fraudster LLC" (INV-1003), "NoProd Industries" (INV-1008).
)
