-- Mock legacy inventory database for the invoice processing pipeline.
--
-- The assignment's minimum schema is `inventory(item TEXT PRIMARY KEY, stock INTEGER)`.
-- This is a strict superset: those two columns keep their names and meanings, and the
-- additions exist to support validation checks the base schema cannot express.
--
--   unit_price  -> price-variance checks (INV-1010 bills WidgetA at $300 "rush",
--                  INV-1013 at $240 "volume discount", INV-1014 at EUR 225)
--   status      -> separates a discontinued/bogus SKU from a legitimately empty shelf,
--                  so FakeItem (fraud) and CoolantPro (real stockout) read differently
--   normalized  -> match key for messy OCR names ("Widget A", "WidgetA (rush order)")
--   vendors     -> unknown-vendor detection (Fraudster LLC, NoProd Industries are absent)
--   ledger      -> duplicate/revision detection (INV-1004 vs INV-1004_revised)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS inventory (
    item         TEXT PRIMARY KEY,
    normalized   TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    stock        INTEGER NOT NULL CHECK (stock >= 0),
    unit_price   REAL CHECK (unit_price IS NULL OR unit_price >= 0),
    currency     TEXT NOT NULL DEFAULT 'USD',
    category     TEXT,
    status       TEXT NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active', 'discontinued')),
    notes        TEXT,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inventory_normalized ON inventory (normalized);

CREATE TABLE IF NOT EXISTS vendors (
    name          TEXT PRIMARY KEY,
    normalized    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'approved'
                  CHECK (status IN ('approved', 'unverified', 'blocked')),
    payment_terms TEXT,
    currency      TEXT NOT NULL DEFAULT 'USD',
    also_known_as TEXT,
    notes         TEXT,
    approved_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_vendors_normalized ON vendors (normalized);

-- Append-only record of every run the pipeline has attempted. Written at the end of
-- each one; read at the start of the next to catch resubmissions.
--
-- invoice_number is nullable on purpose. A run that fails before extraction produces one
-- has no number to record, and the alternative -- writing nothing -- makes a failure less
-- visible than a success in a table whose whole job is traceability.
CREATE TABLE IF NOT EXISTS invoice_ledger (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number TEXT,
    revision       TEXT,
    vendor_name    TEXT,
    total          REAL,
    currency       TEXT NOT NULL DEFAULT 'USD',
    line_hash      TEXT,
    source_path    TEXT,
    decision       TEXT CHECK (decision IN ('approved', 'rejected', 'needs_review', 'error')),
    reason         TEXT,
    -- Which backend produced this decision: the model, or the offline stand-in. Without
    -- it the two are indistinguishable when reading a rationale back, and a replayed
    -- scoring table reads exactly like considered judgment.
    llm_mode       TEXT,
    processed_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_invoice_number ON invoice_ledger (invoice_number);
CREATE INDEX IF NOT EXISTS idx_ledger_line_hash     ON invoice_ledger (line_hash);
