"""FRED (Federal Reserve Economic Data) vendor for macro indicators.

Optional — only queried when ``FRED_API_KEY`` is set.  Results are cached
on disk with a 7-day TTL so subsequent runs within a week reuse the same
data without hitting the network.

All series are free; request an API key at
https://fred.stlouisfed.org/docs/api/api_key.html
"""

import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from .cache import cached_fetch, rate_limited_iter

SERIES_IDS = {
    "cpi": "CPIAUCSL",
    "pce": "PCEPI",
    "real_gdp": "GDP",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "fed_funds_rate": "FEDFUNDS",
    "fed_target_upper": "DFEDTARU",
    "fed_target_lower": "DFEDTARL",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "treasury_3mo": "DGS3MO",
    "yield_curve_10y_2y": "T10Y2Y",
    "vix": "VIXCLS",
    "housing_starts": "HOUST",
    "median_home_price": "MSPUS",
    "manufacturing_employment": "MANEMP",
    "consumer_sentiment": "UMCSENT",
    "industrial_production": "INDPRO",
}

_RATE_LIMIT_DELAY = 2.0


def _get_fred_client(api_key: str):
    try:
        from fredapi import Fred
        return Fred(api_key=api_key)
    except ImportError:
        raise ImportError(
            "fredapi is required for FRED data. "
            "Install it with: pip install fredapi"
        )


def _do_fetch(api_key: str, look_back_months: int) -> dict:
    fred = _get_fred_client(api_key)
    end = datetime.today()
    start = end - timedelta(days=look_back_months * 31)
    result = {"vendor": "fred", "fetched_at": datetime.utcnow().isoformat() + "Z"}

    for key, series_id in rate_limited_iter(SERIES_IDS.items(), delay=_RATE_LIMIT_DELAY):
        try:
            logger.info("FRED: fetching %s (%s)", key, series_id)
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            if s is not None and not s.empty:
                last = s.dropna().iloc[-1]
                prev = s.dropna().iloc[-2] if len(s.dropna()) >= 2 else None
                entry = {
                    "series_id": series_id,
                    "latest_date": str(s.dropna().index[-1].date()),
                    "latest_value": float(last),
                }
                if prev is not None:
                    try:
                        entry["change"] = round(float(last) - float(prev), 4)
                    except (TypeError, ValueError):
                        pass
                result[key] = entry
            else:
                result[key] = {"series_id": series_id, "error": "No data returned"}
        except Exception as e:
            logger.warning("FRED: failed to fetch %s: %s", series_id, e)
            result[key] = {"series_id": series_id, "error": str(e)}

    _add_derived(result)
    return result


def fetch_fred_data(api_key: str = None, look_back_months: int = 12, force_refresh: bool = False) -> dict:
    if not api_key:
        api_key = os.environ.get("FRED_API_KEY", "")

    def _fetch():
        return _do_fetch(api_key, look_back_months)

    return cached_fetch("fred_data.json", _fetch, force_refresh=force_refresh)


def _add_derived(data: dict):
    derived = {}
    t10 = data.get("treasury_10y", {})
    t2 = data.get("treasury_2y", {})
    t3m = data.get("treasury_3mo", {})
    if "latest_value" in t10 and "latest_value" in t2:
        derived["10y_2y_spread"] = round(t10["latest_value"] - t2["latest_value"], 4)
        derived["curve_shape"] = "normal" if derived["10y_2y_spread"] > 0 else "inverted"
    if "latest_value" in t10 and "latest_value" in t3m:
        derived["10y_3mo_spread"] = round(t10["latest_value"] - t3m["latest_value"], 4)
    ff = data.get("fed_funds_rate", {})
    tu = data.get("fed_target_upper", {})
    tl = data.get("fed_target_lower", {})
    if "latest_value" in ff:
        derived["fed_funds"] = ff["latest_value"]
    if "latest_value" in tu and "latest_value" in tl:
        derived["fed_target_range"] = f"{tl['latest_value']:.2f}% - {tu['latest_value']:.2f}%"
        derived["fed_target_midpoint"] = round((tu["latest_value"] + tl["latest_value"]) / 2, 4)
        derived["fed_target_width"] = round(tu["latest_value"] - tl["latest_value"], 4)
    if "latest_value" in ff and "latest_value" in tu and "latest_value" in tl:
        within = tl["latest_value"] <= ff["latest_value"] <= tu["latest_value"]
        derived["fed_funds_within_target"] = within
        derived["fed_funds_vs_midpoint_bps"] = round((ff["latest_value"] - derived["fed_target_midpoint"]) * 100, 1)
    cpi = data.get("cpi", {})
    if "change" in cpi:
        derived["cpi_monthly_change"] = cpi["change"]
    ur = data.get("unemployment", {})
    if "latest_value" in ur:
        derived["unemployment_rate"] = ur["latest_value"]
    gdp = data.get("real_gdp", {})
    if "latest_value" in gdp and "change" in gdp:
        derived["real_gdp_qoq_change"] = gdp["change"]
    data["derived"] = derived


def format_fred_report(data: dict) -> str:
    lines = []
    lines.append("# FRED Macroeconomic Indicators")
    lines.append(f"*Source: Federal Reserve Economic Data | Fetched: {data.get('fetched_at', 'unknown')}*")
    lines.append("")

    derived = data.get("derived", {})

    lines.append("## Inflation & Prices")
    for key, label in [("cpi", "CPI (All Urban)"), ("pce", "PCE Price Index")]:
        d = data.get(key, {})
        if "error" in d:
            lines.append(f"- **{label}**: Error ({d['error']})")
        elif "latest_value" in d:
            lines.append(f"- **{label}**: {d['latest_value']:.2f} (as of {d['latest_date']})")
    if "cpi_monthly_change" in derived:
        lines.append(f"- **CPI Monthly Change**: {derived['cpi_monthly_change']}")
    lines.append("")

    lines.append("## Monetary Policy")
    ff = data.get("fed_funds_rate", {})
    if "latest_value" in ff and "error" not in ff:
        lines.append(f"- **Fed Funds Rate (Effective)**: {ff['latest_value']:.2f}% (as of {ff['latest_date']})")
    if "fed_target_range" in derived:
        lines.append(f"- **FOMC Target Range**: {derived['fed_target_range']}")
    if "fed_target_width" in derived:
        lines.append(f"- **Target Range Width**: {derived['fed_target_width']:.2f}%")
    if "fed_funds_within_target" in derived:
        status = "within" if derived["fed_funds_within_target"] else "outside"
        lines.append(f"- **Effective Rate vs Target**: {status} range")
    if "fed_funds_vs_midpoint_bps" in derived:
        lines.append(f"- **Effective Rate vs Midpoint**: {derived['fed_funds_vs_midpoint_bps']:+.1f} bps")
    t10 = data.get("treasury_10y", {})
    if "latest_value" in t10 and "error" not in t10:
        lines.append(f"- **10Y Treasury**: {t10['latest_value']:.2f}%")
    t2 = data.get("treasury_2y", {})
    if "latest_value" in t2 and "error" not in t2:
        lines.append(f"- **2Y Treasury**: {t2['latest_value']:.2f}%")
    if "10y_2y_spread" in derived:
        lines.append(f"- **10Y-2Y Spread**: {derived['10y_2y_spread']:.2f}% ({derived.get('curve_shape', '')})")
    lines.append("")

    lines.append("## Labor Market")
    for key, label in [("unemployment", "Unemployment Rate"), ("nonfarm_payrolls", "Nonfarm Payrolls"), ("manufacturing_employment", "Manufacturing Employment")]:
        d = data.get(key, {})
        if "error" in d:
            lines.append(f"- **{label}**: Error")
        elif "latest_value" in d:
            unit = "%" if key == "unemployment" else "K"
            lines.append(f"- **{label}**: {d['latest_value']:,.0f}{unit} (as of {d['latest_date']})")
    lines.append("")

    lines.append("## Growth & Production")
    gdp = data.get("real_gdp", {})
    if "latest_value" in gdp and "error" not in gdp:
        lines.append(f"- **Real GDP**: {gdp['latest_value']:,.1f} B (as of {gdp['latest_date']})")
    ip = data.get("industrial_production", {})
    if "latest_value" in ip and "error" not in ip:
        lines.append(f"- **Industrial Production Index**: {ip['latest_value']:.2f} (as of {ip['latest_date']})")
    lines.append("")

    lines.append("## Housing")
    hs = data.get("housing_starts", {})
    if "latest_value" in hs and "error" not in hs:
        lines.append(f"- **Housing Starts**: {hs['latest_value']:,.0f}K (SAAR, {hs['latest_date']})")
    mp = data.get("median_home_price", {})
    if "latest_value" in mp and "error" not in mp:
        lines.append(f"- **Median Home Price**: ${mp['latest_value']:,.0f} ({mp['latest_date']})")
    lines.append("")

    lines.append("## Sentiment")
    cs = data.get("consumer_sentiment", {})
    if "latest_value" in cs and "error" not in cs:
        lines.append(f"- **Consumer Sentiment (U. Michigan)**: {cs['latest_value']:.1f} ({cs['latest_date']})")
    lines.append("")

    return "\n".join(lines)
