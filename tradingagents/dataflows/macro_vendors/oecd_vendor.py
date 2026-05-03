"""OECD (Organisation for Economic Co-operation and Development) vendor.

Fetches key macro indicators via the OECD SDMX-REST API.
No API key required.  Results are cached on disk with a 7-day TTL.

Uses the new sdmx.oecd.org endpoint which respects startPeriod/endPeriod
filters, dramatically reducing response sizes compared to the legacy endpoint.
"""

import logging
import random
import time
from datetime import datetime, timedelta

import requests

from .cache import cached_fetch, rate_limited_iter

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/19.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 OPR/126.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 OPR/125.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/19.0 Safari/605.1.15",
]

_ACCEPT_LANGS = [
    "en-US,en;q=0.9",
    "en-US,en;q=0.8",
    "en,en-US;q=0.9",
    "en-GB,en;q=0.9,en-US;q=0.8",
    "en-US,en;q=0.9,fr;q=0.7",
    "en-US,en;q=0.9,de;q=0.7",
]


def _make_headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": random.choice(_ACCEPT_LANGS),
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "Referer": "https://data-explorer.oecd.org/",
    }


OECD_SERIES = [
    {
        "key": "usa_gdp_quarterly",
        "subject": "QNA",
        "filter": "USA.GDPV.Q",
        "dataflow": "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD",
        "dataflow_base": "https://sdmx.oecd.org/public/rest/data/",
        "country": "USA",
        "label": "US Real GDP Growth (Quarterly %, SA)",
    },
    {
        "key": "usa_cpi_monthly",
        "subject": "PRICES_CPI",
        "filter": "USA.CPALTT01.M",
        "freq_filter": "M",
        "dataflow": "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL",
        "dataflow_base": "https://sdmx.oecd.org/public/rest/data/",
        "country": "USA",
        "label": "US CPI All Items (Index, 2015=100)",
    },
    {
        "key": "usa_unemployment",
        "subject": "MEI",
        "filter": "USA.LRUN64TT.M",
        "dataflow": "OECD,DF_MEI",
        "dataflow_base": "https://sdmx.oecd.org/archive/rest/data/",
        "country": "USA",
        "label": "US Unemployment Rate (%)",
    },
]

_RATE_LIMIT_DELAY = 12.0


def _time_period_map(structure_list: list) -> dict[str, str]:
    mapping = {}
    for struct in structure_list:
        for dim in struct.get("dimensions", {}).get("observation", []):
            if dim.get("id") == "TIME_PERIOD":
                for i, v in enumerate(dim.get("values", [])):
                    mapping[str(i)] = v.get("id", str(i))
    return mapping


def _build_dim_maps(structure_list: list) -> dict[int, dict[int, str]]:
    result = {}
    for struct in structure_list:
        for dim in struct.get("dimensions", {}).get("series", []):
            kp = dim.get("keyPosition")
            if kp is not None:
                result[kp] = {i: v.get("id") for i, v in enumerate(dim.get("values", []))}
    return result


def _find_country_index(structure_list: list, country: str = "USA") -> tuple[int, int] | None:
    for struct in structure_list:
        for dim in struct.get("dimensions", {}).get("series", []):
            did = dim.get("id")
            if did in ("REF_AREA", "LOCATION"):
                key_pos = dim.get("keyPosition", 0)
                for i, v in enumerate(dim.get("values", [])):
                    if v.get("id") == country:
                        return (key_pos, i)
    return None


def _fetch_series(series_cfg: dict, timeout: int = 90) -> dict:
    start_period = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-01")
    end_period = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-12")
    subject = series_cfg["subject"]
    dataflow = series_cfg["dataflow"]
    base = series_cfg["dataflow_base"]

    if "MEI" in dataflow and "archive" in base:
        url = f"{base}{dataflow}?startPeriod={start_period}&endPeriod={end_period}"
    else:
        url = f"{base}{dataflow}/?startPeriod={start_period}&endPeriod={end_period}"

    for attempt in range(4):
        headers = _make_headers()
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.exceptions.ConnectionError:
            wait = 5 * (attempt + 1) + random.uniform(0, 3)
            logger.warning("OECD connection error on %s, retrying in %.0fs", subject, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            wait = 20 * (attempt + 1) + random.uniform(5, 15)
            logger.warning("OECD 429 on %s, retrying in %.0fs (attempt %d)", subject, wait, attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        logger.info("OECD %s: %s bytes", subject, len(resp.content))
        data = resp.json()
        return data
    return {"error": "Max retries exceeded"}


def _sort_key(period: str) -> str:
    p = period.strip()
    parts = p.split("-")
    if len(parts) == 2 and len(parts[0]) == 4:
        base = parts[0]
        q_map = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}
        q = q_map.get(parts[1], parts[1])
        return f"{base}-{q}"
    return p


def _normalise(data: dict) -> tuple[dict, list]:
    if "dataSets" in data and "structure" in data:
        structures = [data["structure"]] if isinstance(data.get("structure"), dict) else data.get("structure", [])
        return data, structures
    if "data" in data and "dataSets" in data["data"]:
        inner = data["data"]
        structures = inner.get("structures", [])
        return inner, structures
    return data, []


def _parse_observations(data: dict, series_cfg: dict) -> list:
    observations = []
    try:
        ds_dict, structures = _normalise(data)
        data_sets = ds_dict.get("dataSets", [])
        if not data_sets:
            return []

        ds = data_sets[0]
        tp_map = _time_period_map(structures)
        dim_maps = _build_dim_maps(structures)
        country_ref = _find_country_index(structures, series_cfg.get("country", "USA"))

        obs_map = ds.get("observations", {})

        if obs_map:
            for obs_key in sorted(obs_map.keys(), reverse=True):
                values = obs_map[obs_key]
                val = values[0] if values else None
                if val is None or val == "":
                    continue
                period = tp_map.get(obs_key, obs_key)
                try:
                    observations.append({"period": period, "value": float(val)})
                except (ValueError, TypeError):
                    continue
            return observations

        series_map = ds.get("series", {})
        if not series_map:
            return []

        subject = series_cfg.get("subject", "")
        filter_str = series_cfg.get("filter", "")
        filter_parts = [p for p in filter_str.split(".") if p != "USA" and p != "all"]
        subject_filter_code = filter_parts[0] if filter_parts else None
        freq_filter = series_cfg.get("freq_filter", None)

        freq_kp = None
        if freq_filter:
            for struct in structures:
                for dim in struct.get("dimensions", {}).get("series", []):
                    if dim.get("id") == "FREQ":
                        freq_kp = dim.get("keyPosition")
                        break

        seen_periods: dict[str, float] = {}
        for series_key, series_data in series_map.items():
            if not isinstance(series_data, dict):
                continue

            parts = series_key.split(":")

            if country_ref is not None:
                key_pos, country_idx = country_ref
                if key_pos >= len(parts) or parts[key_pos] != str(country_idx):
                    continue

            if freq_filter and freq_kp is not None and freq_kp < len(parts):
                freq_id = dim_maps.get(freq_kp, {}).get(int(parts[freq_kp]), "")
                if freq_id != freq_filter:
                    continue

            if subject == "MEI" and subject_filter_code:
                subj_pos = 1
                if subj_pos < len(parts):
                    subj_map = dim_maps.get(subj_pos, {})
                    subj_id = subj_map.get(int(parts[subj_pos]), "")
                    if subj_id != subject_filter_code:
                        continue
                meas_pos = 2
                if meas_pos < len(parts):
                    meas_map = dim_maps.get(meas_pos, {})
                    meas_id = meas_map.get(int(parts[meas_pos]), "")
                    if meas_id not in ("STSA", "ST"):
                        continue

            series_obs = series_data.get("observations", {})
            for obs_key, values in series_obs.items():
                if obs_key in seen_periods:
                    continue
                val = values[0] if values else None
                if val is None or val == "":
                    continue
                try:
                    seen_periods[obs_key] = float(val)
                except (ValueError, TypeError):
                    continue

        observations = [
            {"period": tp_map.get(k, k), "value": v}
            for k, v in sorted(
                seen_periods.items(),
                key=lambda x: _sort_key(tp_map.get(x[0], x[0])),
                reverse=True,
            )
        ]
    except (IndexError, KeyError, TypeError):
        pass
    return observations


def _do_fetch() -> dict:
    result = {"vendor": "oecd", "fetched_at": datetime.utcnow().isoformat() + "Z"}

    for series_cfg in rate_limited_iter(OECD_SERIES, delay=_RATE_LIMIT_DELAY):
        key = series_cfg["key"]
        try:
            logger.info("OECD: fetching %s", key)
            data = _fetch_series(series_cfg)
            if "error" in data:
                result[key] = {"label": series_cfg["label"], "error": data["error"]}
                continue
            obs = _parse_observations(data, series_cfg)
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
            result[key] = {"label": series_cfg["label"], "error": "Timeout (90s)"}
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
