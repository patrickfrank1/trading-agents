"""ECB (European Central Bank) macro data vendor.

Fetches key Eurozone indicators via the ECB SDMX REST API (no API key
required).  Results are cached on disk with a 7-day TTL.

API reference: https://data.ecb.europa.eu/help/api/data
"""

import logging
from datetime import datetime

import requests

from .cache import cached_fetch, rate_limited_iter

logger = logging.getLogger(__name__)

ECB_SDMX_BASE = "https://data-api.ecb.europa.eu/service/data"

ECB_DATASETS = {
    "deposit_facility": {
        "key": "FM/D.U2.EUR.4F.KR.DFR.LEV",
        "label": "ECB Deposit Facility Rate (%)",
    },
    "eonia": {
        "key": "FM/M.U2.EUR.4F.MM.UONSTR.HSTA",
        "label": "Euro Short-Term Rate (€STR) (%)",
    },
    "euribor_3m": {
        "key": "FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
        "label": "3-Month EURIBOR (%)",
    },
    "lending_facility": {
        "key": "FM/D.U2.EUR.4F.KR.MLFR.LEV",
        "label": "ECB Marginal Lending Facility Rate (%)",
    },
    "hicp_inflation": {
        "key": "ICP/M.U2.N.000000.4.ANR",
        "label": "Euro Area HICP Inflation (Y/Y % change)",
    },
    "industrial_production": {
        "key": "STS/M.I9.N.PROD.NS0010.4.000",
        "label": "Euro Area Industrial Production Index",
    },
    "retail_trade": {
        "key": "STS/M.I9.W.TOVV.2G4700.4.000",
        "label": "Euro Area Retail Trade Index",
    },
    "unemployment": {
        "key": "AME/A.EA20.1.0.0.0.ZUTN",
        "label": "Euro Area Unemployment Rate (%)",
    },
}

_RATE_LIMIT_DELAY = 3.0


def _ecb_request(dataflow_key: str) -> dict:
    url = f"{ECB_SDMX_BASE}/{dataflow_key}"
    params = {"format": "jsondata"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_ecb_observations(data: dict) -> list:
    observations = []
    try:
        series_map = data.get("dataSets", [{}])[0].get("series", {})
        time_dims = (
            data.get("structure", {})
            .get("dimensions", {})
            .get("observation", [])
        )
        if not series_map or not time_dims:
            return []

        time_dim = time_dims[0]
        time_values = time_dim.get("values", [])

        for _skey, sdata in series_map.items():
            obs_list = sdata.get("observations", {})
            for obs_idx_str, values in sorted(
                obs_list.items(), key=lambda x: int(x[0]), reverse=True
            ):
                obs_idx = int(obs_idx_str)
                if obs_idx < len(time_values):
                    time_id = time_values[obs_idx].get("id", str(obs_idx))
                else:
                    time_id = str(obs_idx)
                val = values[0] if values else None
                if val is None or val == "":
                    continue
                try:
                    observations.append({"period": time_id, "value": float(val)})
                except (ValueError, TypeError):
                    continue
    except (IndexError, KeyError, TypeError):
        pass
    return observations


def _do_fetch() -> dict:
    result = {"vendor": "ecb", "fetched_at": datetime.utcnow().isoformat() + "Z"}

    for key, cfg in rate_limited_iter(ECB_DATASETS.items(), delay=_RATE_LIMIT_DELAY):
        try:
            logger.info("ECB: fetching %s", key)
            data = _ecb_request(cfg["key"])
            observations = _parse_ecb_observations(data)
            if observations:
                result[key] = {
                    "label": cfg["label"],
                    "latest": observations[0],
                    "history": observations[:5],
                }
            else:
                result[key] = {"label": cfg["label"], "error": "No data"}
        except Exception as e:
            logger.warning("ECB: failed to fetch %s: %s", key, e)
            result[key] = {"label": cfg["label"], "error": str(e)}

    return result


def fetch_ecb_data(force_refresh: bool = False) -> dict:
    return cached_fetch("ecb_data.json", _do_fetch, force_refresh=force_refresh)


def format_ecb_report(data: dict) -> str:
    lines = []
    lines.append("# ECB / Eurozone Macroeconomic Indicators")
    lines.append(f"*Source: European Central Bank SDMX | Fetched: {data.get('fetched_at', 'unknown')}*")
    lines.append("")

    lines.append("## ECB Policy Rates")
    for key in ["deposit_facility", "eonia", "euribor_3m", "lending_facility"]:
        d = data.get(key, {})
        if not d or "error" in d:
            continue
        latest = d.get("latest", {})
        if latest:
            lines.append(f"- **{d['label']}**: {latest.get('value', 'N/A')}% ({latest.get('period', 'N/A')})")
    lines.append("")

    lines.append("## Euro Area Economy")
    for key in ["hicp_inflation", "unemployment", "industrial_production", "retail_trade"]:
        d = data.get(key, {})
        if not d or "error" in d:
            continue
        latest = d.get("latest", {})
        history = d.get("history", [])
        label = d.get("label", key)
        if latest:
            lines.append(f"- **{label}**: {latest.get('value', 'N/A')} ({latest.get('period', 'N/A')})")
        if history and len(history) > 1:
            for obs in history[:4]:
                lines.append(f"  - {obs['period']}: {obs['value']}")
    lines.append("")

    return "\n".join(lines)
