"""OECD (Organisation for Economic Co-operation and Development) vendor.

Fetches key macro indicators via the OECD REST API (SDMX-JSON).
No API key required.  Results are cached on disk with a 7-day TTL.
"""

import logging
from datetime import datetime

import requests

from .cache import cached_fetch, rate_limited_iter

logger = logging.getLogger(__name__)

OECD_API = "https://stats.oecd.org/SDMX-JSON/data"

OECD_SERIES = [
    {
        "key": "usa_gdp_quarterly",
        "subject": "QNA",
        "filter": "USA.GDPV.Q",
        "label": "US Real GDP (Quarterly, millions USD, SA)",
    },
    {
        "key": "usa_cpi_monthly",
        "subject": "PRICES_CPI",
        "filter": "USA.CPALTT01.M",
        "label": "US CPI All Items (Index, 2015=100)",
    },
    {
        "key": "usa_unemployment",
        "subject": "STLABOUR",
        "filter": "USA.LRUN64T.ST",
        "label": "US Unemployment Rate (%)",
    },
    {
        "key": "usa_industrial_production",
        "subject": "MEI",
        "filter": "USA.PRODME.M",
        "label": "US Industrial Production Index",
    },
    {
        "key": "usa_trade_balance",
        "subject": "MEI",
        "filter": "USA.BLSHDT.G",
        "label": "US Trade Balance (millions USD)",
    },
]

_RATE_LIMIT_DELAY = 3.0


def _fetch_series(subject: str, filter_str: str, timeout: int = 15) -> dict:
    url = f"{OECD_API}/{subject}/{filter_str}/all"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _parse_observations(data: dict) -> list:
    observations = []
    try:
        obs_map = (
            data.get("dataSets", [{}])[0]
            .get("observations", {})
        )
        if not obs_map:
            return []

        for obs_key in sorted(obs_map.keys(), reverse=True):
            values = obs_map[obs_key]
            val = values[0] if values else None
            if val is None or val == "":
                continue
            parts = obs_key.split(":")
            period = parts[-1] if parts else obs_key
            try:
                observations.append({"period": period, "value": float(val)})
            except (ValueError, TypeError):
                continue
    except (IndexError, KeyError, TypeError):
        pass
    return observations


def _do_fetch() -> dict:
    result = {"vendor": "oecd", "fetched_at": datetime.utcnow().isoformat() + "Z"}

    for series_cfg in rate_limited_iter(OECD_SERIES, delay=_RATE_LIMIT_DELAY):
        key = series_cfg["key"]
        try:
            logger.info("OECD: fetching %s", key)
            data = _fetch_series(series_cfg["subject"], series_cfg["filter"])
            obs = _parse_observations(data)
            if obs:
                result[key] = {
                    "label": series_cfg["label"],
                    "latest": obs[0],
                    "history": obs[:6],
                }
            else:
                result[key] = {
                    "label": series_cfg["label"],
                    "error": "No observations returned",
                }
        except requests.exceptions.Timeout:
            logger.warning("OECD: timeout fetching %s", key)
            result[key] = {"label": series_cfg["label"], "error": "Timeout (15s)"}
        except Exception as e:
            logger.warning("OECD: failed to fetch %s: %s", key, e)
            result[key] = {"label": series_cfg["label"], "error": str(e)}

    return result


def fetch_oecd_data(force_refresh: bool = False) -> dict:
    return cached_fetch("oecd_data.json", _do_fetch, force_refresh=force_refresh)


def format_oecd_report(data: dict) -> str:
    lines = []
    lines.append("# OECD Macroeconomic Indicators")
    lines.append(f"*Source: OECD Stats | Fetched: {data.get('fetched_at', 'unknown')}*")
    lines.append("")

    any_data = False
    for series_cfg in OECD_SERIES:
        key = series_cfg["key"]
        d = data.get(key, {})
        if not d or "error" in d:
            continue
        any_data = True
        label = d.get("label", key)
        latest = d.get("latest", {})
        history = d.get("history", [])
        lines.append(f"## {label}")
        if latest:
            val = f"{latest.get('value', 'N/A'):,.2f}" if latest["value"] > 100 else f"{latest.get('value', 'N/A'):.4f}"
            lines.append(f"- **Latest**: {val} ({latest.get('period', 'N/A')})")
        if history and len(history) > 1:
            lines.append("  Recent values:")
            for obs in history[:5]:
                val = f"{obs['value']:,.2f}" if obs["value"] > 100 else f"{obs['value']:.4f}"
                lines.append(f"  - {obs['period']}: {val}")
        lines.append("")

    if not any_data:
        lines.append("*No data returned from OECD. The API may be temporarily unavailable.*")
        lines.append("")

    return "\n".join(lines)
