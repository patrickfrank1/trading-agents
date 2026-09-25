"""Macro vendor registry.

Provides a unified interface to fetch macro data from multiple sources.
All vendors are optional — the system gracefully degrades when a vendor
is unavailable or its API key is not configured.

Supported vendors:
  fred      — Federal Reserve Economic Data (requires FRED_API_KEY)
  oecd      — OECD Stats (no key required)
  worldbank — World Bank Open Data (no key required)
  ecb       — European Central Bank SDMX (no key required)
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

from .fred_vendor import fetch_fred_data, format_fred_report
from .oecd_vendor import fetch_oecd_data, format_oecd_report
from .worldbank_vendor import fetch_worldbank_data, format_worldbank_report
from .ecb_vendor import fetch_ecb_data, format_ecb_report

VENDORS = {
    "fred": {
        "fetch": fetch_fred_data,
        "format": format_fred_report,
        "env_key": "FRED_API_KEY",
        "requires_key": True,
    },
    "oecd": {
        "fetch": fetch_oecd_data,
        "format": format_oecd_report,
        "env_key": None,
        "requires_key": False,
    },
    "worldbank": {
        "fetch": fetch_worldbank_data,
        "format": format_worldbank_report,
        "env_key": None,
        "requires_key": False,
    },
    "ecb": {
        "fetch": fetch_ecb_data,
        "format": format_ecb_report,
        "env_key": None,
        "requires_key": False,
    },
}


def is_vendor_available(vendor_name: str) -> bool:
    cfg = VENDORS.get(vendor_name)
    if cfg is None:
        return False
    if cfg["requires_key"]:
        key = os.environ.get(cfg["env_key"], "")
        return bool(key)
    return True


def get_available_vendors() -> list:
    return [name for name, cfg in VENDORS.items() if is_vendor_available(name)]


def fetch_vendor_data(vendor_name: str, **kwargs) -> dict:
    cfg = VENDORS.get(vendor_name)
    if cfg is None:
        return {"error": f"Unknown vendor: {vendor_name}"}
    if not is_vendor_available(vendor_name):
        key_hint = f" (set {cfg['env_key']})" if cfg["env_key"] else ""
        return {"error": f"{vendor_name} not available{key_hint}"}
    try:
        return cfg["fetch"](**kwargs)
    except ImportError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.warning("Failed to fetch from %s: %s", vendor_name, e)
        return {"error": str(e)}


def format_vendor_report(vendor_name: str, data: dict) -> str:
    cfg = VENDORS.get(vendor_name)
    if cfg is None:
        return f"Unknown vendor: {vendor_name}"
    if "error" in data:
        return f"# {vendor_name.upper()} Data\n\nError: {data['error']}\n"
    return cfg["format"](data)
