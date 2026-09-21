"""Mock legacy inventory database: schema, seed data, and access layer."""

from .database import SCHEMA_PATH, connect, initialize, utcnow
from .naming import normalize_item, normalize_vendor
from .repository import (
    CANDIDATE_FLOOR,
    ItemRecord,
    LedgerEntry,
    Lookup,
    StockCheck,
    VendorRecord,
    check_stock,
    compute_line_hash,
    find_prior_invoices,
    lookup_item,
    lookup_vendor,
    record_invoice,
)

__all__ = [
    "CANDIDATE_FLOOR",
    "SCHEMA_PATH",
    "ItemRecord",
    "LedgerEntry",
    "Lookup",
    "StockCheck",
    "VendorRecord",
    "check_stock",
    "compute_line_hash",
    "connect",
    "find_prior_invoices",
    "initialize",
    "lookup_item",
    "lookup_vendor",
    "normalize_item",
    "normalize_vendor",
    "record_invoice",
    "utcnow",
]
