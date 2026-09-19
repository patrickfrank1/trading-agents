"""Quarterly macro panel builder for the causal Bayesian macro model.

Unlike the snapshot-style macro tools (``macro_market_data``,
``macro_vendors``), this module fetches **full history** and builds a
versioned, quarterly-frequency panel cached as Parquet under
``<data_cache_dir>/macro_panel``.

Design (see ``assets/bayesian_causal_model_implementation_plan.md``):

- FRED series (API key required) are aggregated to quarterly frequency with
  per-series rules (mean / end-of-period).
- Asset prices come from yfinance full history and are converted to
  quarterly total returns.
- Every column carries a **publication lag** (in quarters). The panel
  returned by :func:`build_panel` is real-time aligned: row ``t`` of a
  column contains only information that was publicly available by the end
  of quarter ``t``. This is the look-ahead guard for out-of-sample
  evaluation.
- ``panel.full`` keeps the contemporaneous alignment (observation-date
  based) for diagnostics; ``panel.data`` is the real-time aligned panel
  used by the models.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------


class FREDSeriesSpec:
    __slots__ = ("series_id", "agg", "lag_q")

    def __init__(self, series_id: str, agg: str, lag_months: int):
        self.series_id = series_id
        self.agg = agg
        self.lag_q = int(np.ceil(lag_months / 3.0))


#: name -> (FRED series id, quarterly aggregation rule, publication lag in months)
FRED_PANEL_SERIES = {
    # Rates & policy (daily/weekly market data; published same day)
    "fed_funds": ("FEDFUNDS", "mean", 1),
    "treasury_3m": ("DGS3MO", "mean", 0),
    "treasury_2y": ("DGS2", "mean", 0),
    "treasury_10y": ("DGS10", "mean", 0),
    "treasury_30y": ("DGS30", "mean", 0),
    "real_yield_10y": ("DFII10", "mean", 0),
    "breakeven_10y": ("T10YIE", "mean", 0),
    "breakeven_5y": ("T5YIE", "mean", 0),
    "mortgage_rate_30y": ("MORTGAGE30US", "mean", 1),
    "fed_balance_sheet": ("WALCL", "eop", 0),
    "vix": ("VIXCLS", "mean", 0),
    "usd_broad": ("DTWEXBGS", "eop", 0),
    # Inflation & prices
    "cpi": ("CPIAUCSL", "eop", 1),
    "cpi_core": ("CPILFESL", "eop", 1),
    "pce": ("PCEPI", "eop", 1),
    "pce_core": ("PCEPILFE", "eop", 1),
    "infl_exp_mich": ("MICH", "mean", 0),
    "consumer_sentiment": ("UMCSENT", "mean", 0),
    "avg_hourly_earnings": ("CES0500000003", "mean", 1),
    "unit_labour_costs": ("ULCNFB", "mean", 2),
    "ppi": ("PPIACO", "eop", 1),
    # Growth & labour
    "real_gdp": ("GDPC1", "eop", 1),
    "nominal_gdp": ("GDP", "eop", 1),
    "real_potential_gdp": ("GDPPOT", "eop", 1),
    "productivity": ("OPHNFB", "mean", 2),
    "participation_rate": ("CIVPART", "mean", 1),
    "unemployment": ("UNRATE", "mean", 1),
    "payrolls": ("PAYEMS", "eop", 1),
    "industrial_production": ("INDPRO", "eop", 1),
    "fixed_investment": ("GPDI", "mean", 1),
    "corporate_profits": ("CP", "mean", 2),
    # Fiscal (quarterly NIPA aggregates; released with a sizeable lag)
    "federal_receipts": ("FGRECPT", "mean", 3),
    "federal_outlays": ("FGEXPND", "mean", 3),
    "federal_interest": ("A091RC1Q027SBEA", "mean", 3),
    "debt_to_gdp": ("GFDEGDQ188S", "eop", 3),
    "gross_federal_debt": ("GFDEBTN", "eop", 3),
    # Housing
    "housing_starts": ("HOUST", "mean", 1),
    "housing_starts_1f": ("HOUST1F", "mean", 1),
    "permits": ("PERMIT", "mean", 1),
    "case_shiller": ("CSUSHPINSA", "eop", 2),
    "median_home_price": ("MSPUS", "eop", 3),
    "household_mortgage_debt": ("HHMSDODNS", "eop", 3),
    "household_net_worth": ("TNWBSHNO", "eop", 3),
    # Credit & financial conditions
    "nfci": ("NFCI", "mean", 0),
    "ig_oas": ("BAMLC0A0CM", "mean", 0),
    "hy_oas": ("BAMLH0A0HYM2", "mean", 0),
    "sloos_ci_tightening": ("DRTSCILM", "mean", 1),
    "bank_loans": ("TOTLL", "eop", 0),
    "delinq_consumer": ("DRALACBS", "mean", 3),
    "delinq_business": ("DRBLACBS", "mean", 3),
    "delinq_cre": ("DRCRELEXLACBS", "mean", 3),
    "consumer_credit": ("TOTALSL", "eop", 1),
}

#: asset name -> yfinance ticker (quarterly end-of-quarter prices -> returns)
YF_PANEL_TICKERS = {
    "gold": "GC=F",
    "sp500": "SPY",
    "treasury_bond": "TLT",
    "reits": "VNQ",
    "usd": "DX-Y.NYB",
    "wti_oil": "CL=F",
    "vix": "^VIX",
    "hy_credit": "HYG",
    "ig_credit": "LQD",
    "bank_equity": "KRE",
}

GPR_CSV_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.csv"

FRED_RATE_LIMIT_DELAY = 0.75

_DERIVED_CACHE: list = []


def derived_series(parents: list, extra_lag: int = 0):
    """Register the decorated derived-series builder.

    ``parents`` are panel column names the series is computed from; the
    publication lag of the derived column is the maximum parent lag plus
    ``extra_lag`` (for functions that shift internally).
    """

    def decorator(fn):
        fn.derived_meta = {"parents": parents, "extra_lag": extra_lag}
        _DERIVED_CACHE.append(fn)
        return fn

    return decorator


def _qoq_annualized(s: pd.Series) -> pd.Series:
    return (s / s.shift(1)) ** 4 - 1.0


# ---------------------------------------------------------------------------
# Derived series (computed on the contemporaneous panel)
# ---------------------------------------------------------------------------


@derived_series(["real_gdp", "real_potential_gdp"])
def _d_output_gap(df):
    pot = df["real_potential_gdp"]
    return (df["real_gdp"] - pot) / pot * 100.0


@derived_series(["real_yield_10y", "treasury_10y", "breakeven_10y"])
def _d_real_yield(df):
    ry = df["real_yield_10y"].copy()
    fallback = df["treasury_10y"] - df["breakeven_10y"]
    return ry.fillna(fallback)


@derived_series(["real_yield_10y"], extra_lag=1)
def _d_real_yield_chg(df):
    return df["real_yield_10y"].diff()


@derived_series(["breakeven_10y"], extra_lag=1)
def _d_breakeven_chg(df):
    return df["breakeven_10y"].diff()


@derived_series(["cpi"], extra_lag=1)
def _d_cpi_qoq_ann(df):
    return _qoq_annualized(df["cpi"]) * 100.0


@derived_series(["cpi_core"], extra_lag=1)
def _d_cpi_core_qoq_ann(df):
    return _qoq_annualized(df["cpi_core"]) * 100.0


@derived_series(["cpi_qoq_ann"], extra_lag=1)
def _d_inflation_surprise(df):
    realized = df["cpi_qoq_ann"]
    expectation = realized.rolling(8, min_periods=4).mean().shift(1)
    return realized - expectation


@derived_series(["corporate_profits"], extra_lag=1)
def _d_profits_growth(df):
    return _qoq_annualized(df["corporate_profits"]) * 100.0


@derived_series(["federal_receipts", "federal_outlays", "nominal_gdp"])
def _d_primary_balance_pct_gdp(df):
    return (df["federal_receipts"] - df["federal_outlays"]) / df["nominal_gdp"] * 100.0


@derived_series(["primary_balance_pct_gdp"], extra_lag=1)
def _d_fiscal_impulse(df):
    return -df["primary_balance_pct_gdp"].diff()


@derived_series(["federal_interest", "gross_federal_debt"])
def _d_r_eff(df):
    return 4.0 * df["federal_interest"] / df["gross_federal_debt"].shift(1) * 100.0


@derived_series(["mortgage_rate_30y"], extra_lag=1)
def _d_mortgage_rate_chg(df):
    return df["mortgage_rate_30y"].diff()


@derived_series(["nfci"], extra_lag=1)
def _d_nfci_chg(df):
    return df["nfci"].diff()


@derived_series(["hy_oas"], extra_lag=1)
def _d_hy_oas_chg(df):
    return df["hy_oas"].diff()


@derived_series(["fed_balance_sheet"], extra_lag=1)
def _d_qe_flow(df):
    return df["fed_balance_sheet"].pct_change()


def _derived_registry():
    return {fn.__name__[3:]: fn for fn in _DERIVED_CACHE}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _resample_quarterly(s: pd.Series, agg: str) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    q = s.to_period("Q")
    if agg == "mean":
        out = q.groupby(level=0).mean()
    elif agg == "eop":
        out = q.groupby(level=0).last()
    elif agg == "sum":
        out = q.groupby(level=0).sum()
    else:
        raise ValueError(f"Unknown aggregation rule: {agg}")
    out.index = out.index.to_timestamp(how="end").normalize()
    return out


def _fetch_fred_histories(api_key: str, start: str) -> dict:
    import time

    from fredapi import Fred

    fred = Fred(api_key=api_key)
    out = {}
    for i, (name, (series_id, _agg, _lag)) in enumerate(FRED_PANEL_SERIES.items()):
        if i:
            time.sleep(FRED_RATE_LIMIT_DELAY)
        try:
            s = fred.get_series(series_id, observation_start=start)
            if s is None or s.empty:
                logger.warning("FRED panel: no data for %s (%s)", name, series_id)
                continue
            out[name] = s.astype(float)
        except Exception as e:  # per-series failure is non-fatal
            logger.warning("FRED panel: failed to fetch %s: %s", series_id, e)
    return out


def _fetch_yf_quarterly(start: str) -> pd.DataFrame:
    from .stockstats_utils import yf_retry

    frames = {}
    for name, ticker in YF_PANEL_TICKERS.items():
        try:
            hist = yf_retry(lambda t=ticker: yf.Ticker(t).history(start=start, auto_adjust=True))
            if hist is None or hist.empty:
                logger.warning("YF panel: no data for %s (%s)", name, ticker)
                continue
            close = hist["Close"].dropna()
            q = close.to_period("Q").groupby(level=0).last()
            q.index = q.index.to_timestamp(how="end").normalize()
            frames[name] = q.astype(float)
        except Exception as e:
            logger.warning("YF panel: failed to fetch %s: %s", ticker, e)
    return pd.DataFrame(frames)


def _fetch_gpr(start: str) -> pd.Series:
    try:
        df = pd.read_csv(GPR_CSV_URL, parse_dates=["DATE"], index_col="DATE")
        s = df["GPR"].astype(float).loc[start:]
        return _resample_quarterly(s, "mean")
    except Exception as e:
        logger.warning("GPR index unavailable (%s); column omitted", e)
        return None


def build_full_panel(start: str | None = None, fetch_gpr: bool = True) -> pd.DataFrame:
    """Build the contemporaneous (observation-date aligned) quarterly panel."""
    start = start or get_config().get("macro_panel_start", "1990-01-01")

    fred = {}
    api_key = os.environ.get("FRED_API_KEY", "")
    if api_key:
        fred = _fetch_fred_histories(api_key, start)
    else:
        logger.warning("FRED_API_KEY not set; panel will contain only market data")

    df = pd.DataFrame(fred)
    if not df.empty:
        df.index = pd.to_datetime(df.index)

    yf_q = _fetch_yf_quarterly(start)
    if not yf_q.empty:
        df = df.join(yf_q, how="outer") if not df.empty else yf_q.copy()

    if fetch_gpr:
        gpr = _fetch_gpr(start)
        if gpr is not None:
            df["gpr"] = gpr

    if df.empty:
        raise RuntimeError("Panel is empty: no data sources succeeded")

    idx = pd.period_range(pd.Period(start, freq="Q"), df.index.max().to_period("Q"), freq="Q")
    idx_ts = idx.to_timestamp(how="end").normalize()
    df = df.reindex(idx_ts)
    df.index.name = "quarter"

    df = _apply_fred_aggregation(df, start)
    df = _add_asset_returns(df)
    df = _add_derived(df)
    return df


def _apply_fred_aggregation(df: pd.DataFrame, start: str) -> pd.DataFrame:
    for name, spec in FRED_PANEL_SERIES.items():
        if name not in df.columns:
            continue
        col = df[name].dropna()
        if col.empty:
            continue
        s = col.loc[(col.index >= pd.Timestamp(start))]
        df[name] = _resample_quarterly(s, spec.agg).reindex(df.index)
    return df


def _add_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    for name in YF_PANEL_TICKERS:
        if name in df.columns:
            df[f"{name}_ret"] = df[name].pct_change() * 100.0
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    registry = _derived_registry()
    for name, fn in registry.items():
        try:
            missing = [p for p in fn.derived_meta["parents"] if p not in df.columns]
            if missing:
                logger.warning("Derived %s skipped; missing parents %s", name, missing)
                continue
            df[name] = fn(df)
        except Exception as e:
            logger.warning("Derived series %s failed: %s", name, e)
    return df


def publication_lag_quarters(columns: list) -> dict:
    """Real-time publication lag (in quarters) for each panel column."""
    lags = {
        name: int(np.ceil(lag_months / 3.0))
        for name, (_sid, _agg, lag_months) in FRED_PANEL_SERIES.items()
    }
    for name in YF_PANEL_TICKERS:
        lags[name] = 0
        lags[f"{name}_ret"] = 0
    lags["gpr"] = 0
    registry = _derived_registry()
    for name, fn in registry.items():
        parent_lags = [lags.get(p, 0) for p in fn.derived_meta["parents"]]
        lags[name] = int(max(parent_lags)) + fn.derived_meta["extra_lag"]
    return {c: lags.get(c, 0) for c in columns}


def real_time_align(df: pd.DataFrame) -> pd.DataFrame:
    """Shift each column back by its publication lag (look-ahead guard)."""
    lags = publication_lag_quarters(list(df.columns))
    rt = df.copy()
    for col, lag in lags.items():
        if lag:
            rt[col] = df[col].shift(lag)
    return rt


# ---------------------------------------------------------------------------
# Versioned cache
# ---------------------------------------------------------------------------


@dataclass
class PanelBundle:
    data: pd.DataFrame          # real-time aligned (look-ahead guarded)
    full: pd.DataFrame          # contemporaneous alignment (diagnostics)
    manifest: dict = field(default_factory=dict)
    path: str | None = None


def panel_dir() -> str:
    cfg = get_config()
    d = cfg.get("macro_panel_dir") or os.path.join(cfg["data_cache_dir"], "macro_panel")
    os.makedirs(d, exist_ok=True)
    return d


def model_dir() -> str:
    cfg = get_config()
    d = cfg.get("macro_model_dir") or os.path.join(cfg["data_cache_dir"], "macro_models")
    os.makedirs(d, exist_ok=True)
    return d


MANIFEST_FILENAME = "manifest.json"


def _latest_manifest_path() -> str:
    return os.path.join(panel_dir(), MANIFEST_FILENAME)


def _save_parquet_or_pickle(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path)
    except (ImportError, OSError):
        df.to_pickle(path + ".pkl")


def _load_parquet_or_pickle(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.read_pickle(path + ".pkl")


def save_panel(bundle: PanelBundle) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = os.path.join(panel_dir(), f"panel_{stamp}.parquet")
    _save_parquet_or_pickle(bundle.data, path)
    full_path = os.path.join(panel_dir(), f"panel_full_{stamp}.parquet")
    _save_parquet_or_pickle(bundle.full, full_path)
    manifest = dict(bundle.manifest or {})
    manifest.update({"saved_at": stamp, "data_file": os.path.basename(path), "full_file": os.path.basename(full_path)})
    with open(_latest_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    bundle.path = path
    return path


def load_latest_panel(max_age_days: float | None = None) -> PanelBundle | None:
    max_age = max_age_days if max_age_days is not None else get_config().get("macro_panel_max_age_days", 7)
    mpath = _latest_manifest_path()
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    age_days = time.time() - os.path.getmtime(mpath)
    if max_age is not None and age_days > max_age * 86400:
        return None
    data_path = os.path.join(panel_dir(), manifest["data_file"])
    full_path = os.path.join(panel_dir(), manifest.get("full_file", manifest["data_file"]))
    if not (os.path.exists(data_path) or os.path.exists(data_path + ".pkl")):
        return None
    data = _load_parquet_or_pickle(data_path)
    full = _load_parquet_or_pickle(full_path) if os.path.exists(full_path) or os.path.exists(full_path + ".pkl") else data
    return PanelBundle(data=data, full=full, manifest=manifest, path=data_path)


def build_panel(start: str | None = None, force_refresh: bool = False) -> PanelBundle:
    """Build (or load cached) quarterly macro panel.

    Returns a :class:`PanelBundle` whose ``data`` attribute is the
    real-time aligned panel safe for out-of-sample modelling.
    """
    if not force_refresh:
        cached = load_latest_panel()
        if cached is not None:
            logger.info("Using cached macro panel %s", cached.path)
            return cached
    full = build_full_panel(start=start)
    data = real_time_align(full)
    manifest = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "start": start or get_config().get("macro_panel_start", "1990-01-01"),
        "n_quarters": int(len(full)),
        "last_quarter": str(full.index[-1].date()),
        "n_columns": int(full.shape[1]),
        "fred_series_fetched": sorted(set(full.columns) & set(FRED_PANEL_SERIES)),
        "warnings": sorted(
            [n for n in FRED_PANEL_SERIES if n not in full.columns]
            + [n for n in YF_PANEL_TICKERS if n not in full.columns]
        ),
    }
    bundle = PanelBundle(data=data, full=full, manifest=manifest)
    save_panel(bundle)
    return bundle
