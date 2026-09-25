"""World Bank macro data vendor.

Uses the World Bank REST API v2 (no API key required).
Results are cached on disk with a 7-day TTL.

Documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

import logging
from datetime import datetime

import requests

from .cache import cached_fetch, rate_limited_iter

logger = logging.getLogger(__name__)

WB_API = "https://api.worldbank.org/v2"

WB_INDICATORS = {
    "gdp_growth": {"code": "NY.GDP.MKTP.KD.ZG", "label": "GDP Growth (annual %)"},
    "inflation": {"code": "FP.CPI.TOTL.ZG", "label": "Inflation, Consumer Prices (annual %)"},
    "unemployment": {"code": "SL.UEM.TOTL.ZS", "label": "Unemployment (% of Labor Force)"},
    "real_interest_rate": {"code": "FR.INR.RINR", "label": "Real Interest Rate (%)"},
    "trade_pct_gdp": {"code": "NE.TRD.GNFS.ZS", "label": "Trade (% of GDP)"},
    "fdi_net_inflows": {"code": "BX.KLT.DINV.WD.GD.ZS", "label": "FDI, Net Inflows (% of GDP)"},
    "govt_debt_pct_gdp": {"code": "GC.DOD.TOTL.GD.ZS", "label": "Govt Debt (% of GDP)"},
    "exchange_rate": {"code": "PA.NUS.FCRF", "label": "Exchange Rate (LCU per USD)"},
    "gdp_current_usd": {"code": "NY.GDP.MKTP.CD", "label": "GDP, Current US$"},
}

DEFAULT_COUNTRY = "USA"

_RATE_LIMIT_DELAY = 2.0


def _wb_request(country: str, indicator: str, per_page: int = 20) -> list:
    url = f"{WB_API}/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": per_page, "date": "2000:2025", "MRV": 5}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, list) and len(body) >= 2:
        return body[1]
    return []


def _do_fetch(country: str) -> dict:
    result = {"vendor": "worldbank", "country": country, "fetched_at": datetime.utcnow().isoformat() + "Z"}

    for key, cfg in rate_limited_iter(WB_INDICATORS.items(), delay=_RATE_LIMIT_DELAY):
        try:
            logger.info("World Bank: fetching %s for %s", key, country)
            records = _wb_request(country, cfg["code"])
            if records:
                latest = None
                for r in records:
                    val = r.get("value")
                    if val is not None:
                        latest = {
                            "date": r.get("date", ""),
                            "value": float(val),
                        }
                        break
                history = [
                    {"date": r["date"], "value": float(r["value"])}
                    for r in records
                    if r.get("value") is not None
                ][:5]
                result[key] = {
                    "label": cfg["label"],
                    "latest": latest,
                    "history": history,
                }
            else:
                result[key] = {"label": cfg["label"], "error": "No data"}
        except Exception as e:
            logger.warning("World Bank: failed to fetch %s: %s", key, e)
            result[key] = {"label": cfg["label"], "error": str(e)}

    return result


def fetch_worldbank_data(country: str = None, force_refresh: bool = False) -> dict:
    if country is None:
        country = DEFAULT_COUNTRY

    def _fetch():
        return _do_fetch(country)

    return cached_fetch("worldbank_data.json", _fetch, force_refresh=force_refresh)


def format_worldbank_report(data: dict) -> str:
    lines = []
    country = data.get("country", "USA")
    lines.append(f"# World Bank Macro Data — {country}")
    lines.append(f"*Source: World Bank Open Data | Fetched: {data.get('fetched_at', 'unknown')}*")
    lines.append("")

    for key in WB_INDICATORS:
        d = data.get(key, {})
        if not d or "error" in d:
            continue
        label = d.get("label", key)
        latest = d.get("latest")
        history = d.get("history", [])
        lines.append(f"## {label}")
        if latest:
            fmt = f"{latest['value']:,.2f}" if abs(latest["value"]) < 1000 else f"{latest['value']:,.0f}"
            lines.append(f"- **Latest**: {fmt} ({latest.get('date', 'N/A')})")
        if history and len(history) > 1:
            lines.append("  Recent values:")
            for h in history:
                fmt = f"{h['value']:,.2f}" if abs(h["value"]) < 1000 else f"{h['value']:,.0f}"
                lines.append(f"  - {h['date']}: {fmt}")
        lines.append("")

    return "\n".join(lines)
