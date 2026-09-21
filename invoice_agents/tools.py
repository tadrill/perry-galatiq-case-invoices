"""LangChain tools over the legacy inventory database.

These are the validator agent's hands. Each opens its own short-lived SQLite connection,
which is cheap and keeps the tools stateless -- a tool that held a connection would have
to be constructed per run and torn down on failure.

The contract inherited from the repository layer matters more than the plumbing: a lookup
reports an exact hit or it reports scored near misses, and never resolves an inexact name
on the agent's behalf. The tool surfaces the evidence; the agent rules on it.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from .config import get_settings
from .inventory import connect
from .inventory import find_prior_invoices as _find_prior_invoices
from .inventory import lookup_item as _lookup_item
from .inventory import lookup_vendor as _lookup_vendor
from .inventory.repository import check_stock as _check_stock


def _db_path():
    return get_settings().resolved_db_path


@tool
def lookup_item(name: str) -> str:
    """Look up an item name in the inventory catalog.

    Returns an exact match when the name resolves unambiguously, otherwise the nearest
    catalog entries with similarity scores from 0 to 1. An inexact result is NOT a
    match -- it is evidence for you to weigh. A high score can still be a different
    product: SKUs that differ only in a trailing character are usually distinct items,
    not typos.

    Args:
        name: The item name exactly as printed on the invoice.
    """
    with connect(_db_path()) as conn:
        return json.dumps(_lookup_item(conn, name).to_dict(), default=str)


@tool
def lookup_vendor(name: str) -> str:
    """Look up a vendor in the approved-vendor list.

    Returns an exact match, or the nearest approved vendors with similarity scores.
    Also matches against former trading names. A vendor absent from this list has not
    been approved for payment. Check the `status` field: 'approved', 'unverified' (for
    example a recent rebrand with changed banking details) or 'blocked'.

    Args:
        name: The vendor name exactly as printed on the invoice.
    """
    with connect(_db_path()) as conn:
        return json.dumps(_lookup_vendor(conn, name).to_dict(), default=str)


@tool
def check_stock(item: str, quantity: float) -> str:
    """Check whether a quantity of an item can be fulfilled from stock.

    Pass the TOTAL quantity across every line naming this product, not a single line's
    quantity. Invoices routinely split one SKU over several lines, and each line can sit
    within stock while the sum does not.

    Returns a status of: ok, insufficient, out_of_stock (a genuine stockout of a real
    product), discontinued (the SKU is delisted, which on an inbound invoice is a fraud
    signal rather than a supply problem), or unknown_item.

    Args:
        item: The item name or catalog SKU.
        quantity: Total quantity requested across the whole invoice.
    """
    with connect(_db_path()) as conn:
        return json.dumps(_check_stock(conn, item, int(quantity)).to_dict(), default=str)


@tool
def find_prior_invoices(invoice_number: str) -> str:
    """Check the ledger for earlier submissions of this invoice number.

    A hit means this invoice number has been processed before. Compare the totals: an
    identical total is a resubmission, a different total is either a revision or an
    attempt to be paid twice for overlapping goods.

    Args:
        invoice_number: The invoice identifier as printed.
    """
    with connect(_db_path()) as conn:
        prior = _find_prior_invoices(conn, invoice_number)
        return json.dumps([entry.to_dict() for entry in prior], default=str)


#: Every tool the validator agent may call.
VALIDATOR_TOOLS = [lookup_item, lookup_vendor, check_stock, find_prior_invoices]

TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in VALIDATOR_TOOLS}
