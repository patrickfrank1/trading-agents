"""Macro market data fetching with JSON-file caching (7-day TTL).

Fetches prices, yields, and derived metrics for broad macro markets
(Treasuries, gold, oil, commodities, housing, equity breadth) using
yfinance.  Results are cached on disk so that multiple runs within a
week reuse the same data without hitting the network.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from .config import get_config
from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)

MACRO_TICKERS = {
    "treasury_13w": "^IRX",
    "treasury_5y": "^FVX",
    "treasury_10y": "^TNX",
    "treasury_30y": "^TYX",
    "gold": "GC=F",
    "wti_oil": "CL=F",
    "brent_oil": "BZ=F",
    "commodities_etf": "DBC",
    "homebuilders": "XHB",
    "home_construction": "ITB",
    "reits": "VNQ",
    "sp500_equal_weight": "RSP",
    "sp500_cap_weight": "SPY",
    "vix": "^VIX",
    "russell_2000": "^RUT",
}

CACHE_TTL_DAYS = 7
CACHE_FILENAME = "macro_market_data.json"


def _cache_path() -> str:
    config = get_config()
    cache_dir = config["data_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, CACHE_FILENAME)


def _is_cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        mtime = os.path.getmtime(path)
        age = time.time() - mtime
        return age < CACHE_TTL_DAYS * 86400
    except OSError:
        return False


def _load_cache(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _pct_change(current, past):
    if past is None or past == 0 or pd.isna(past):
        return None
    try:
        return round(((current - past) / past) * 100, 2)
    except (TypeError, ZeroDivisionError):
        return None


def _sma(series: pd.Series, window: int):
    if len(series) < window:
        return None
    return round(series.rolling(window=window).mean().iloc[-1], 4)


def _rsi(series: pd.Series, window: int = 14):
    if len(series) < window + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)


def _fetch_single_ticker(ticker: str, period: str = "6mo") -> pd.DataFrame:
    data = yf_retry(lambda: yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True,
    ))
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def _build_ticker_section(name: str, ticker: str, hist: pd.DataFrame) -> dict:
    section = {"ticker": ticker}

    if hist.empty:
        section["error"] = f"No data returned for {ticker}"
        return section

    last = hist.iloc[-1]
    current_close = float(last["Close"])
    section["current_price"] = round(current_close, 4)
    section["latest_date"] = str(hist.index[-1].date())

    hist_len = len(hist)

    if hist_len >= 22:
        price_30d_ago = float(hist.iloc[-22]["Close"])
        section["price_30d_ago"] = round(price_30d_ago, 4)
        section["change_30d_pct"] = _pct_change(current_close, price_30d_ago)

    if hist_len >= 66:
        price_90d_ago = float(hist.iloc[-66]["Close"])
        section["price_90d_ago"] = round(price_90d_ago, 4)
        section["change_90d_pct"] = _pct_change(current_close, price_90d_ago)

    closes = hist["Close"]
    sma_50 = _sma(closes, 50)
    sma_200 = _sma(closes, 200)
    if sma_50 is not None:
        section["sma_50"] = sma_50
        section["price_vs_sma_50_pct"] = _pct_change(current_close, sma_50)
    if sma_200 is not None:
        section["sma_200"] = sma_200
        section["price_vs_sma_200_pct"] = _pct_change(current_close, sma_200)

    rsi_val = _rsi(closes)
    if rsi_val is not None:
        section["rsi_14"] = rsi_val

    if sma_50 is not None and sma_200 is not None:
        section["sma_cross"] = "golden" if sma_50 > sma_200 else "death"

    return section


def _fetch_all_data() -> dict:
    result = {}
    result["fetched_at"] = datetime.utcnow().isoformat() + "Z"

    history_cache = {}

    for key, ticker in MACRO_TICKERS.items():
        try:
            logger.info("Fetching macro data for %s (%s)", key, ticker)
            hist = _fetch_single_ticker(ticker, period="6mo")
            history_cache[key] = hist
            result[key] = _build_ticker_section(key, ticker, hist)
        except Exception as e:
            logger.warning("Failed to fetch %s (%s): %s", key, ticker, e)
            result[key] = {"ticker": ticker, "error": str(e)}

    result["derived"] = _compute_derived_metrics(result, history_cache)

    return result


def _compute_derived_metrics(data: dict, history_cache: dict) -> dict:
    derived = {}

    tn = data.get("treasury_10y", {})
    fv = data.get("treasury_5y", {})
    ty = data.get("treasury_30y", {})
    ir = data.get("treasury_13w", {})

    if "error" not in tn and "current_price" in tn:
        derived["10y_yield"] = tn["current_price"]
        if "error" not in fv and "current_price" in fv:
            derived["5y_10y_spread"] = round(fv["current_price"] - tn["current_price"], 4)
            derived["yield_curve_5y_10y"] = "normal" if derived["5y_10y_spread"] > 0 else "inverted"
        if "error" not in ty and "current_price" in ty:
            derived["10y_30y_spread"] = round(tn["current_price"] - ty["current_price"], 4)
        if "error" not in ir and "current_price" in ir:
            derived["2y_proxy_spread"] = round(ir["current_price"] - tn["current_price"], 4)
        if "error" not in tn and "change_30d_pct" in tn:
            derived["10y_yield_30d_change_pct"] = tn["change_30d_pct"]
            direction = "rising" if tn["change_30d_pct"] > 0 else "falling" if tn["change_30d_pct"] < 0 else "stable"
            derived["10y_yield_trend"] = direction

    cl = data.get("wti_oil", {})
    bz = data.get("brent_oil", {})
    if "error" not in cl and "error" not in bz:
        derived["wti_brent_spread"] = round(cl["current_price"] - bz["current_price"], 4)

    rsp = data.get("sp500_equal_weight", {})
    spy = data.get("sp500_cap_weight", {})
    if "error" not in rsp and "error" not in spy and spy["current_price"] > 0:
        ratio = rsp["current_price"] / spy["current_price"]
        derived["rsp_spy_ratio"] = round(ratio, 6)
        derived["rsp_spy_interpretation"] = (
            "Equal-weight outperforming (broad participation)" if ratio > 0.92
            else "Cap-weight dominating (concentration in mega-caps)"
        )

        rsp_hist = history_cache.get("sp500_equal_weight")
        spy_hist = history_cache.get("sp500_cap_weight")
        if rsp_hist is not None and spy_hist is not None and len(rsp_hist) >= 22 and len(spy_hist) >= 22:
            try:
                ratio_30d_ago = float(rsp_hist.iloc[-22]["Close"]) / float(spy_hist.iloc[-22]["Close"])
                derived["rsp_spy_ratio_30d_ago"] = round(ratio_30d_ago, 6)
                derived["rsp_spy_ratio_change_30d_pct"] = _pct_change(ratio, ratio_30d_ago)
            except (IndexError, TypeError, ZeroDivisionError):
                pass

    vix = data.get("vix", {})
    if "error" not in vix:
        derived["vix_level"] = vix["current_price"]
        if vix["current_price"] < 15:
            derived["vix_regime"] = "low (complacent)"
        elif vix["current_price"] < 25:
            derived["vix_regime"] = "moderate (normal)"
        elif vix["current_price"] < 35:
            derived["vix_regime"] = "elevated (anxious)"
        else:
            derived["vix_regime"] = "high (fearful)"

    rut = data.get("russell_2000", {})
    if "error" not in rut and "change_90d_pct" in rut:
        derived["russell_2000_90d_change_pct"] = rut["change_90d_pct"]
        direction = "outperforming" if rut["change_90d_pct"] > 0 else "underperforming"
        derived["small_cap_momentum"] = direction

    xhb = data.get("homebuilders", {})
    vnq = data.get("reits", {})
    if "error" not in xhb and "change_30d_pct" in xhb:
        derived["homebuilders_30d_change_pct"] = xhb["change_30d_pct"]
    if "error" not in vnq and "change_30d_pct" in vnq:
        derived["reits_30d_change_pct"] = vnq["change_30d_pct"]

    gold = data.get("gold", {})
    if "error" not in gold:
        if "sma_cross" in gold:
            derived["gold_trend"] = f"{gold['sma_cross']} cross (50/200 SMA)"
        if "rsi_14" in gold:
            derived["gold_rsi"] = gold["rsi_14"]

    return derived


def fetch_macro_market_data(force_refresh: bool = False) -> dict:
    path = _cache_path()

    if not force_refresh and _is_cache_valid(path):
        logger.info("Using cached macro market data from %s", path)
        try:
            return _load_cache(path)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Cache corrupted, re-fetching macro market data")

    logger.info("Fetching fresh macro market data")
    data = _fetch_all_data()
    _save_cache(path, data)
    return data


def format_macro_market_report(data: dict) -> str:
    lines = []
    lines.append("# Macro Market Overview")
    fetched_at = data.get("fetched_at", "unknown")
    lines.append(f"*Data fetched: {fetched_at}*")
    lines.append("")

    derived = data.get("derived", {})

    lines.append("## US Treasury Market")
    if "10y_yield" in derived:
        lines.append(f"- **10-Year Yield**: {derived['10y_yield']}%")
    if "5y_10y_spread" in derived:
        curve = derived.get("yield_curve_5y_10y", "unknown")
        lines.append(f"- **5Y-10Y Spread**: {derived['5y_10y_spread']}% ({curve})")
    if "10y_30y_spread" in derived:
        lines.append(f"- **10Y-30Y Spread**: {derived['10y_30y_spread']}%")
    if "10y_yield_trend" in derived:
        lines.append(f"- **10Y Yield Trend (30d)**: {derived['10y_yield_trend']} ({derived.get('10y_yield_30d_change_pct', 'N/A')}%)")
    tn = data.get("treasury_30y", {})
    if "current_price" in tn and "error" not in tn:
        lines.append(f"- **30-Year Yield**: {tn['current_price']}%")
    fv = data.get("treasury_5y", {})
    if "current_price" in fv and "error" not in fv:
        lines.append(f"- **5-Year Yield**: {fv['current_price']}%")
    lines.append("")

    lines.append("## Gold (GC=F)")
    gold = data.get("gold", {})
    if "error" not in gold and "current_price" in gold:
        lines.append(f"- **Current Price**: ${gold['current_price']:,.2f}")
        if "change_30d_pct" in gold:
            lines.append(f"- **30-Day Change**: {gold['change_30d_pct']}%")
        if "change_90d_pct" in gold:
            lines.append(f"- **90-Day Change**: {gold['change_90d_pct']}%")
        if "price_vs_sma_50_pct" in gold:
            lines.append(f"- **vs 50 SMA**: {gold['price_vs_sma_50_pct']}%")
        if "price_vs_sma_200_pct" in gold:
            lines.append(f"- **vs 200 SMA**: {gold['price_vs_sma_200_pct']}%")
        if "gold_trend" in derived:
            lines.append(f"- **Trend**: {derived['gold_trend']}")
        if "gold_rsi" in derived:
            lines.append(f"- **RSI(14)**: {derived['gold_rsi']}")
        if "sma_cross" in gold:
            lines.append(f"- **50/200 SMA Cross**: {gold['sma_cross']}")
    else:
        lines.append(f"- Error: {gold.get('error', 'No data')}")
    lines.append("")

    lines.append("## Oil Market")
    wti = data.get("wti_oil", {})
    brent = data.get("brent_oil", {})
    if "error" not in wti and "current_price" in wti:
        lines.append(f"- **WTI Crude**: ${wti['current_price']:,.2f}")
        if "change_30d_pct" in wti:
            lines.append(f"- **WTI 30-Day Change**: {wti['change_30d_pct']}%")
        if "rsi_14" in wti:
            lines.append(f"- **WTI RSI(14)**: {wti['rsi_14']}")
    if "error" not in brent and "current_price" in brent:
        lines.append(f"- **Brent Crude**: ${brent['current_price']:,.2f}")
        if "change_30d_pct" in brent:
            lines.append(f"- **Brent 30-Day Change**: {brent['change_30d_pct']}%")
    if "wti_brent_spread" in derived:
        lines.append(f"- **WTI-Brent Spread**: ${derived['wti_brent_spread']:,.2f}")
    lines.append("")

    lines.append("## Broad Commodities (DBC)")
    dbc = data.get("commodities_etf", {})
    if "error" not in dbc and "current_price" in dbc:
        lines.append(f"- **DBC ETF Price**: ${dbc['current_price']:,.2f}")
        if "change_30d_pct" in dbc:
            lines.append(f"- **30-Day Change**: {dbc['change_30d_pct']}%")
        if "change_90d_pct" in dbc:
            lines.append(f"- **90-Day Change**: {dbc['change_90d_pct']}%")
        if "sma_cross" in dbc:
            lines.append(f"- **50/200 SMA Cross**: {dbc['sma_cross']}")
        if "rsi_14" in dbc:
            lines.append(f"- **RSI(14)**: {dbc['rsi_14']}")
    else:
        lines.append(f"- Error: {dbc.get('error', 'No data')}")
    lines.append("")

    lines.append("## Housing & Real Estate")
    xhb = data.get("homebuilders", {})
    itb = data.get("home_construction", {})
    vnq = data.get("reits", {})
    if "error" not in xhb and "current_price" in xhb:
        lines.append(f"- **Homebuilders (XHB)**: ${xhb['current_price']:,.2f} ({xhb.get('change_30d_pct', 'N/A')}% 30d)")
    if "error" not in itb and "current_price" in itb:
        lines.append(f"- **Home Construction (ITB)**: ${itb['current_price']:,.2f} ({itb.get('change_30d_pct', 'N/A')}% 30d)")
    if "error" not in vnq and "current_price" in vnq:
        lines.append(f"- **REITs (VNQ)**: ${vnq['current_price']:,.2f} ({vnq.get('change_30d_pct', 'N/A')}% 30d)")
    if "homebuilders_30d_change_pct" in derived:
        lines.append(f"- **Homebuilder Trend**: {'positive' if derived['homebuilders_30d_change_pct'] > 0 else 'negative'} momentum")
    if "reits_30d_change_pct" in derived:
        lines.append(f"- **REIT Trend**: {'positive' if derived['reits_30d_change_pct'] > 0 else 'negative'} momentum")
    lines.append("")

    lines.append("## Equity Market Breadth")
    if "rsp_spy_ratio" in derived:
        lines.append(f"- **RSP/SPY Ratio**: {derived['rsp_spy_ratio']:.4f}")
        lines.append(f"- **Interpretation**: {derived.get('rsp_spy_interpretation', 'N/A')}")
        if "rsp_spy_ratio_30d_ago" in derived:
            lines.append(f"- **Ratio 30d Ago**: {derived['rsp_spy_ratio_30d_ago']:.4f}")
        if "rsp_spy_ratio_change_30d_pct" in derived:
            lines.append(f"- **Ratio 30d Change**: {derived['rsp_spy_ratio_change_30d_pct']}%")
    rsp = data.get("sp500_equal_weight", {})
    spy = data.get("sp500_cap_weight", {})
    if "error" not in rsp and "change_30d_pct" in rsp:
        lines.append(f"- **RSP (Equal-Weight) 30d**: {rsp['change_30d_pct']}%")
    if "error" not in spy and "change_30d_pct" in spy:
        lines.append(f"- **SPY (Cap-Weight) 30d**: {spy['change_30d_pct']}%")
    if "vix_level" in derived:
        lines.append(f"- **VIX**: {derived['vix_level']} ({derived.get('vix_regime', 'N/A')})")
    if "small_cap_momentum" in derived:
        lines.append(f"- **Russell 2000 (90d)**: {derived.get('russell_2000_90d_change_pct', 'N/A')}% ({derived['small_cap_momentum']})")
    lines.append("")

    return "\n".join(lines)
