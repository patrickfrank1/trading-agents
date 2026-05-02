"""ECB (European Central Bank) macro data vendor.

Fetches key Eurozone indicators via the ECB SDMX REST API (no API key
required).  Results are cached on disk with a 7-day TTL.
"""

import logging
from datetime import datetime

import requests

from .cache import cached_fetch, rate_limited_iter

logger = logging.getLogger(__name__)

ECB_SDMX_BASE = "https://sdw-wsrest.ecb.europa.eu/service/data"

ECB_DATASETS = {
    "euribor_3m": {
        "key": "FM/D.U2.EUR.4F.KR.MRR_FR.LEV",
        "label": "3-Month EURIBOR (%)",
    },
    "eonia": {
        "key": "FM/D.U2.EUR.4F.KR.DF.LEV",
        "label": "EONIA / Euro Short-Term Rate (%)",
    },
    "deposit_facility": {
        "key": "FM/D.U2.EUR.4F.KR.DFT.LEV",
        "label": "ECB Deposit Facility Rate (%)",
    },
    "lending_facility": {
        "key": "FM/D.U2.EUR.4F.KR.LFT.LEV",
        "label": "ECB Marginal Lending Facility Rate (%)",
    },
    "industrial_production": {
        "key": "STS/M.I8.Y.U2.15.0000.4.ANR",
        "label": "Euro Area Industrial Production (Y/Y % change)",
    },
    "retail_trade": {
        "key": "STS/M.RT.Y.U2.15.0000.4.ANR",
        "label": "Euro Area Retail Trade (Y/Y % change)",
    },
    "hicp_inflation": {
        "key": "ICP/M.U2.N.000000.4.ANR",
        "label": "Euro Area HICP Inflation (Y/Y % change)",
    },
    "unemployment": {
        "key": "LFS/Q.U.N.S14.S.C.LT.GD.A._Z.0000.V._T._Z.E._Z._Z.SV_START._T._Z.SV_END",
        "label": "Euro Area Unemployment Rate (%)",
    },
}

_RATE_LIMIT_DELAY = 3.0


def _ecb_request(dataflow_key: str) -> dict:
    url = f"{ECB_SDMX_BASE}/{dataflow_key}"
    params = {"detail": "dataonly", "format": "sdmx-json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_ecb_observations(data: dict) -> list:
    observations = []
    try:
        obs_map = data.get("data", {}).get("dataSets", [{}])[0].get("observations", {})
        if not obs_map:
            return []
        dims = data.get("data", {}).get("structure", {}).get("dimensions", {}).get("observation", [])
        time_idx = len(dims) - 1

        for obs_key in sorted(obs_map.keys(), reverse=True):
            parts = obs_key.split(":")
            time_period = parts[time_idx] if time_idx < len(parts) else parts[-1]
            values = obs_map[obs_key]
            val = values[0] if values else None
            if val is None or val == "":
                continue
            try:
                observations.append({"period": time_period, "value": float(val)})
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
