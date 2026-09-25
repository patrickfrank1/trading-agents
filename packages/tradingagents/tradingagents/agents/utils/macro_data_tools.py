from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta

import os

import numpy as np
import yfinance as yf

from tradingagents.dataflows.yfinance_news import _extract_article_data
from tradingagents.dataflows.stockstats_utils import yf_retry
from tradingagents.dataflows.macro_market_data import (
    fetch_macro_market_data,
    format_macro_market_report,
)
from tradingagents.dataflows.macro_timeseries import load_latest_panel, model_dir
from tradingagents.models.macro_bayes.common import load_model
from tradingagents.models.macro_bayes.simulate import build_last_state_row
from tradingagents.dataflows.macro_vendors import (
    fetch_vendor_data,
    format_vendor_report,
    get_available_vendors,
)
from tradingagents.agents.utils.tool_errors import safe_tool


def _load_cbn_artifacts():
    """Load (panel bundle, asset-regression fits or None, gold fit or None)."""
    from tradingagents.models.macro_bayes.asset_regressions import load_asset_regressions
    from tradingagents.models.macro_bayes.v1_gold import GoldModelFit

    bundle = load_latest_panel()
    fits = None
    gold_fit = None
    assets_file = os.path.join(model_dir(), "asset_regressions.pkl")
    if os.path.exists(assets_file):
        fits = load_asset_regressions(assets_file)
    gold_file = os.path.join(model_dir(), "v1_gold.pkl")
    if os.path.exists(gold_file):
        gold_fit = GoldModelFit.from_payload(load_model(gold_file))
    return bundle, fits, gold_fit


def _run_counterfactual(bundle, fits, gold_fit, shocks: str, stock_beta: float, horizon: int) -> str:
    from tradingagents.models.macro_bayes.asset_regressions import (
        counterfactual_samples,
        parse_shocks,
        render_counterfactual_report,
    )

    if not fits:
        return (
            "Counterfactual mode unavailable: no fitted asset regressions found.\n"
            "Fit them offline with: `uv run python packages/tradingagents/scripts/fit_macro_model.py --model assets`."
        )

    valid_names = {f for fit in fits.values() for f in fit.features}
    sds = {}
    for fit in fits.values():
        for f in valid_names & set(fit.features):
            sds[f] = fit.scaler.sds[f]
    try:
        parsed = parse_shocks(shocks, valid_names, sds)
    except ValueError as e:
        return (
            f"Invalid counterfactual specification: {e}\n"
            "Format: 'var:value' (native units: percentage points for rate/changes, "
            "percent for returns) or 'var:Nsd'. Example: 'real_yield_chg:0.27'."
        )

    prices = {}
    for col in ("gold", "sp500", "reits", "treasury_bond"):
        if col in bundle.full.columns:
            prices[col] = float(bundle.full[col].iloc[-1])
    row = bundle.full.iloc[-1].copy()
    row.name = bundle.full.index[-1].date()
    results = counterfactual_samples(fits, bundle.full, parsed, stock_beta=stock_beta)

    joint_block = None
    joint_file = os.path.join(model_dir(), "joint_v2v3.pkl")
    if horizon > 1 and os.path.exists(joint_file):
        from tradingagents.models.macro_bayes.joint import JointModelFit
        from tradingagents.models.macro_bayes.simulate import (
            cumulative_returns,
            simulate_joint,
        )

        jfit = JointModelFit.from_payload(load_model(joint_file))
        shock_sd, exog_sd = {}, {}
        for name, spec in parsed.items():
            if name in jfit.endog_vars:
                shock_sd[name] = spec if spec["unit"] == "sd" else {
                    "value": spec["value"] / jfit.scaler_endog.sds[name], "unit": "sd"}
            elif name in jfit.exog_vars:
                exog_sd[name] = spec if spec["unit"] == "sd" else {
                    "value": spec["value"] / jfit.scaler_exog.sds[name], "unit": "sd"}
        spec = {}
        if shock_sd:
            spec["shock"] = {k: v["value"] for k, v in shock_sd.items()}
        if exog_sd:
            spec["exog"] = {k: v["value"] for k, v in exog_sd.items()}
        base_paths = simulate_joint(jfit, build_last_state_row(bundle.data), horizon=horizon, scenario="baseline", seed=7)
        shock_paths = simulate_joint(jfit, build_last_state_row(bundle.data), horizon=horizon, scenario=spec, seed=7)
        jlines = [
            f"Multi-quarter path (joint VARX, {horizon} quarters, shock applied at the",
            "first forecast quarter; reduced-form — rate-shock responses pool",
            "growth-driven and policy-driven yield moves):",
            "",
            "| Asset | Baseline | With shock | Effect |",
            "|---|---|---|---|",
        ]
        for var in ("gold_ret", "sp500_ret", "treasury_bond_ret", "reits_ret"):
            if var not in jfit.endog_vars:
                continue
            cb = cumulative_returns(base_paths, jfit, var)
            cs = cumulative_returns(shock_paths, jfit, var)
            jlines.append(
                f"| {var.removesuffix('_ret')} | {np.median(cb):+.1f}% | "
                f"{np.median(cs):+.1f}% | {np.median(cs) - np.median(cb):+.2f}pp |"
            )
        joint_block = "\n".join(jlines)

    return render_counterfactual_report(results, parsed, prices, horizon=horizon, joint_block=joint_block)


def get_macro_causal_forecast_impl(
    horizon_quarters: int = 8,
    scenario: str = "baseline",
    model: str = "joint",
    shocks: str = "",
    stock_beta: float = 0.0,
) -> str:
    """Render the causal Bayesian macro model's forward scenario report.

    Named scenarios and counterfactuals (``shocks``) are supported; models
    are fitted offline (``packages/tradingagents/scripts/fit_macro_model.py``) and this runs a
    posterior-predictive simulation (NumPy only — no PyMC, no network).
    """
    from tradingagents.models.macro_bayes.simulate import (
        render_joint_forecast,
        render_scenario_list,
        render_v1_forecast,
    )
    from tradingagents.models.macro_bayes.joint import JointModelFit
    from tradingagents.models.macro_bayes.v1_gold import GoldModelFit

    horizon = int(min(max(horizon_quarters, 1), 12))

    bundle, fits, gold_fit = _load_cbn_artifacts()
    if bundle is None:
        return (
            "Causal Bayesian macro model not available: no quarterly panel found.\n"
            "Build it offline with: `uv run python packages/tradingagents/scripts/fit_macro_model.py` "
            "(requires FRED_API_KEY and the optional `model` extra: pymc, arviz, pyarrow)."
        )

    if shocks:
        return _run_counterfactual(bundle, fits, gold_fit, shocks, stock_beta, horizon)

    model_file = os.path.join(model_dir(), "joint_v2v3.pkl" if model == "joint" else "v1_gold.pkl")
    if not os.path.exists(model_file):
        return (
            f"Causal Bayesian macro model not available: {os.path.basename(model_file)} "
            "not found.\nFit the model offline with: `uv run python packages/tradingagents/scripts/fit_macro_model.py` "
            "(models persist under ~/.tradingagents/cache/macro_models)."
        )

    if model == "joint" and scenario not in SCENARIO_NAMES:
        return f"Unknown scenario '{scenario}'.\n\n{render_scenario_list()}"

    try:
        payload = load_model(model_file)
        row = build_last_state_row(bundle.data)
        if model == "joint":
            fit = JointModelFit.from_payload(payload)
            return render_joint_forecast(fit, row, horizon=horizon, scenario=scenario,
                                         stock_beta=stock_beta)
        fit = GoldModelFit.from_payload(payload)
        return render_v1_forecast(fit, row)
    except Exception as e:
        return f"Causal Bayesian macro model forecast failed: {e}"


SCENARIO_NAMES = ("baseline", "hawkish", "inflation_shock", "risk_off", "productivity_boom")


@tool
@safe_tool
def get_macro_causal_forecast(
    horizon_quarters: Annotated[int, "Forecast horizon in quarters (1-12)"] = 8,
    scenario: Annotated[str, "baseline | hawkish | inflation_shock | risk_off | productivity_boom"] = "baseline",
    model: Annotated[str, "joint (multi-asset VARX) or v1 (gold only)"] = "joint",
    shocks: Annotated[str, "Counterfactual overrides, e.g. 'real_yield_chg:0.27' (native units, pp for rate changes / % for returns) or 'real_yield_chg:1sd'. Comma-separated. Overrides the scenario parameter."] = "",
    stock_beta: Annotated[float, "If > 0, add a beta-adjusted single-stock equity row (SPX surrogate)"] = 0.0,
) -> str:
    """
    Run the causal Bayesian macro model to generate probabilistic forward
    scenarios — and custom counterfactuals — for gold, equities (S&P 500),
    Treasury bonds, and REITs over a quarterly horizon.

    The model is a stability-constrained quarterly VARX(1) plus per-asset
    contemporaneous regressions, estimated with PyMC on free public data
    (FRED + yfinance) with sign-informed priors and Student-t errors.
    Fitted offline; this tool loads the cached posterior and simulates.

    SCOPE
    - Quarterly granularity: results are cumulative returns over the full
      horizon, not intra-quarter paths (an FOMC-day question cannot be
      answered with this tool).
    - Asset universe is fixed: gold, S&P 500 (SPY), Treasury bonds (TLT),
      REITs (VNQ). Use stock_beta to add a beta-adjusted equity surrogate
      row for an individual stock (linear single-factor approximation;
      idiosyncratic risk and beta instability are ignored).
    - Inputs are free public data; consensus surprises are proxied by
      realized-minus-trend inflation, credit spreads by the HYG/LQD ratio.

    QUERIES
    - Named scenarios: scenario = baseline | hawkish | inflation_shock |
      risk_off | productivity_boom, horizon 1-12 quarters.
    - Custom counterfactuals: pass shocks as comma-separated
      'variable:value' entries, e.g. 'real_yield_chg:0.27' (+0.27pp on the
      10y real yield, the historical response to a 50bp hike) or
      'real_yield_chg:1sd'. Variables: real_yield_chg, usd_ret,
      inflation_surprise, nfci_chg, profits_growth, mortgage_rate_chg.
      For a Fed rate hike, shock real_yield_chg and STATE the pass-through
      assumption — it is an input to the query, not modeled.
      Counterfactuals use a same-window conditional model: the shock and
      the asset return occur over the same quarter.
    - model: "joint" for the multi-asset VARX model (default), or "v1"
      for the gold-only one-quarter-ahead regression.
    - One-quarter counterfactuals use the contemporaneous regression layer
      (recommended for 'what if' questions); horizons > 1 quarter
      additionally show the joint VARX path.

    INTERPRETATION
    - The signal is the DIFFERENCE between the scenario/counterfactual and
      baseline, not the absolute level: level forecasts embed the drift of
      a mostly bull sample and are not price targets.
    - Quote probabilities and intervals, not point predictions.

    Args:
        horizon_quarters: Forecast horizon in quarters (1-12, default 8)
        scenario: baseline, hawkish, inflation_shock, risk_off,
            productivity_boom (default baseline; ignored when shocks given)
        model: "joint" (default) or "v1" (gold-only, one quarter)
        shocks: Counterfactual overrides in native units or sd
            (e.g. "real_yield_chg:0.27,usd_ret:1sd"); empty = none
        stock_beta: Equity surrogate beta (> 0 adds the row, default off)

    Returns:
        str: A formatted markdown report with the scenario table
    """
    return get_macro_causal_forecast_impl(
        horizon_quarters=horizon_quarters,
        scenario=scenario,
        model=model,
        shocks=shocks,
        stock_beta=stock_beta,
    )


@tool
@safe_tool
def get_macro_sensitivity() -> str:
    """
    Show the causal Bayesian macro model's estimated driver sensitivities:
    how each asset's NEXT-QUARTER return responds to a +1 standard
    deviation move in each macro driver (change in the 10y real yield,
    USD, oil, inflation surprise, mortgage rate, profit growth, NFCI
    tightening), with 90% posterior intervals.

    Use this to understand which channels the model thinks matter per
    asset (e.g. REITs are the most rate-sensitive; equities react more to
    financial-conditions tightening than to the yield move itself) and to
    translate a concrete move (e.g. +50bp hike = +0.27pp real yield =
    ~0.9sd) into an approximate return effect.

    No arguments. Uses the fitted per-asset regressions and the V1 gold
    model; if they are not fitted yet it returns setup instructions.

    Returns:
        str: A formatted markdown sensitivity table
    """
    from tradingagents.models.macro_bayes.asset_regressions import render_sensitivity_table

    bundle, fits, gold_fit = _load_cbn_artifacts()
    if bundle is None:
        return (
            "Sensitivity table unavailable: no quarterly panel found.\n"
            "Build it offline with: `uv run python packages/tradingagents/scripts/fit_macro_model.py`."
        )
    if not fits and gold_fit is None:
        return (
            "Sensitivity table unavailable: no fitted models found.\n"
            "Fit them offline with: `uv run python packages/tradingagents/scripts/fit_macro_model.py` "
            "(fits the V1 gold model, the per-asset regressions, and the joint model)."
        )
    return render_sensitivity_table(fits or {}, gold_fit=gold_fit)

def _search_macro_news(queries, curr_date, look_back_days, limit):
    all_news = []
    seen_titles = set()

    for query in queries:
        search = yf_retry(lambda q=query: yf.Search(
            query=q,
            news_count=limit,
            enable_fuzzy_query=True,
        ))

        if search.news:
            for article in search.news:
                if "content" in article:
                    data = _extract_article_data(article)
                    title = data["title"]
                else:
                    title = article.get("title", "")

                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_news.append(article)

        if len(all_news) >= limit:
            break

    if not all_news:
        return f"No macro economic news found for {curr_date}"

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    news_str = ""
    for article in all_news[:limit]:
        if "content" in article:
            data = _extract_article_data(article)
            if data.get("pub_date"):
                pub_naive = data["pub_date"].replace(tzinfo=None) if hasattr(data["pub_date"], "replace") else data["pub_date"]
                if pub_naive > curr_dt + relativedelta(days=1):
                    continue
            title = data["title"]
            publisher = data["publisher"]
            link = data["link"]
            summary = data["summary"]
        else:
            title = article.get("title", "No title")
            publisher = article.get("publisher", "Unknown")
            link = article.get("link", "")
            summary = ""

        news_str += f"### {title} (source: {publisher})\n"
        if summary:
            news_str += f"{summary}\n"
        if link:
            news_str += f"Link: {link}\n"
        news_str += "\n"

    return news_str



@tool
@safe_tool
def get_cpi_data(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back for CPI data"] = 30,
    limit: Annotated[int, "Maximum number of articles to return"] = 10,
) -> str:
    """
    Retrieve Consumer Price Index (CPI) related economic news and data.
    CPI measures the average change over time in the prices paid by urban consumers
    for a market basket of consumer goods and services. It is a key indicator of
    inflation and is closely watched by the Federal Reserve for monetary policy decisions.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 30)
        limit (int): Maximum number of articles to return (default 10)
    Returns:
        str: A formatted string containing CPI-related news and data
    """
    queries = [
        "CPI consumer price index inflation",
        "US inflation rate report",
        "consumer prices Bureau of Labor Statistics",
    ]

    news = _search_macro_news(queries, curr_date, look_back_days, limit)

    header = (
        f"## CPI (Consumer Price Index) Data\n"
        f"Period: {curr_date} (last {look_back_days} days)\n\n"
    )
    return header + news


@tool
@safe_tool
def get_fomc_data(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back for FOMC data"] = 30,
    limit: Annotated[int, "Maximum number of articles to return"] = 10,
) -> str:
    """
    Retrieve Federal Open Market Committee (FOMC) related economic news and data.
    The FOMC is the monetary policy-making body of the Federal Reserve System.
    It sets the target for the federal funds rate and conducts open market operations.
    FOMC meetings and decisions are critical for understanding the direction of
    US monetary policy, interest rates, and their impact on financial markets.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 30)
        limit (int): Maximum number of articles to return (default 10)
    Returns:
        str: A formatted string containing FOMC-related news and data
    """
    queries = [
        "FOMC Federal Reserve interest rate decision",
        "Federal Reserve monetary policy meeting",
        "Fed funds rate decision",
    ]

    news = _search_macro_news(queries, curr_date, look_back_days, limit)

    header = (
        f"## FOMC (Federal Open Market Committee) Data\n"
        f"Period: {curr_date} (last {look_back_days} days)\n\n"
    )
    return header + news


@tool
@safe_tool
def get_nonfarm_payrolls_data(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back for NFP data"] = 30,
    limit: Annotated[int, "Maximum number of articles to return"] = 10,
) -> str:
    """
    Retrieve Non-farm Payrolls (NFP) related economic news and data.
    Non-farm Payrolls measures the change in the number of people employed during
    the previous month, excluding the farming industry. It is one of the most
    closely watched economic indicators and is released monthly by the Bureau
    of Labor Statistics. Strong NFP numbers typically signal a robust economy
    and may influence Federal Reserve policy decisions.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 30)
        limit (int): Maximum number of articles to return (default 10)
    Returns:
        str: A formatted string containing NFP-related news and data
    """
    queries = [
        "nonfarm payrolls jobs report employment",
        "US jobs data labor market unemployment",
        "employment situation Bureau of Labor Statistics",
    ]

    news = _search_macro_news(queries, curr_date, look_back_days, limit)

    header = (
        f"## Non-farm Payrolls (NFP) Data\n"
        f"Period: {curr_date} (last {look_back_days} days)\n\n"
    )
    return header + news


@tool
@safe_tool
def get_macro_market_data() -> str:
    """
    Retrieve a comprehensive snapshot of broad macro market conditions
    including US Treasury yields and yield curve shape, gold, oil (WTI and
    Brent), broad commodities, housing/real estate ETFs, and equity market
    breadth (RSP/SPY ratio, VIX, Russell 2000).  Data is cached for up to
    7 days since it is independent of any individual ticker.

    Use this tool to understand the macro environment beyond CPI, FOMC,
    and employment data.  The report covers:
    - Treasury yields (13W, 5Y, 10Y, 30Y) and yield curve spreads
    - Gold price, trend, and RSI
    - WTI and Brent crude oil prices and spread
    - Broad commodities ETF (DBC)
    - Housing market proxies (XHB homebuilders, ITB, VNQ REITs)
    - Equity breadth (RSP/SPY ratio, VIX, Russell 2000 momentum)

    Returns:
        str: A formatted markdown report with current macro market conditions
    """
    data = fetch_macro_market_data()
    return format_macro_market_report(data)


@tool
@safe_tool
def get_fred_economic_data(
    look_back_months: Annotated[int, "Number of months of history to fetch"] = 12,
) -> str:
    """
    Retrieve official US economic indicators from the Federal Reserve Economic
    Data (FRED) database. Requires FRED_API_KEY environment variable.

    Covers: CPI, PCE, Real GDP, unemployment rate, nonfarm payrolls, Fed funds
    rate, Treasury yields (2Y/10Y/3MO), yield curve spread, VIX, housing
    starts, median home prices, manufacturing employment, consumer sentiment,
    and industrial production.

    Returns:
        str: A formatted markdown report with the latest values and trends
    """
    import os
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        available = get_available_vendors()
        return (
            "FRED API key not configured. Set FRED_API_KEY environment variable.\n"
            "Request a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            f"\nCurrently available macro vendors: {available}"
        )
    data = fetch_vendor_data("fred", api_key=api_key, look_back_months=look_back_months)
    return format_vendor_report("fred", data)


@tool
@safe_tool
def get_oecd_data() -> str:
    """
    Retrieve key macro indicators from the OECD (Organisation for Economic
    Co-operation and Development) for the US, Eurozone, Japan, UK, China, and
    Germany. No API key required.

    Covers: GDP growth, unemployment rate, CPI inflation, long-term interest
    rates, industrial production, and retail trade.

    Returns:
        str: A formatted markdown report with latest OECD indicators
    """
    data = fetch_vendor_data("oecd")
    return format_vendor_report("oecd", data)


@tool
@safe_tool
def get_world_bank_data(
    country: Annotated[str, "ISO country code (e.g. USA, CHN, GBR, DEU, JPN)"] = "USA",
) -> str:
    """
    Retrieve macro indicators from the World Bank Open Data API for a given
    country. No API key required.

    Covers: GDP growth, inflation, unemployment, real interest rate, trade as
    % of GDP, FDI net inflows, government debt as % of GDP, exchange rate,
    and GDP in current US dollars.

    Args:
        country: ISO 3166 country code (default: USA)

    Returns:
        str: A formatted markdown report with World Bank indicators
    """
    data = fetch_vendor_data("worldbank", country=country)
    return format_vendor_report("worldbank", data)


@tool
@safe_tool
def get_ecb_data() -> str:
    """
    Retrieve Eurozone macro indicators from the European Central Bank via
    SDMX. No API key required.

    Covers: ECB policy rates (deposit facility, EURIBOR, EONIA, lending
    facility), HICP inflation, unemployment, industrial production, and
    retail trade for the euro area.

    Returns:
        str: A formatted markdown report with ECB / Eurozone indicators
    """
    data = fetch_vendor_data("ecb")
    return format_vendor_report("ecb", data)


# ---------------------------------------------------------------------------
# FX rates (no-login, via yfinance FX pairs)
# ---------------------------------------------------------------------------

#: Standard USD pairs covered when no explicit list is given.
DEFAULT_FX_PAIRS = "EURUSD,USDJPY,GBPUSD,USDCNY,USDCAD,USDMXN,USDCHF,USDKRW"

#: Trading-day lookbacks approximating 1 / 3 / 12 months.
_FX_WINDOWS = {"1M": 21, "3M": 63, "12M": 252}


def _fx_change(closes, lookback: int):
    if closes is None or len(closes) < lookback + 1:
        return None
    base = float(closes.iloc[-(lookback + 1)])
    if base == 0:
        return None
    return float(closes.iloc[-1]) / base - 1.0


@tool
@safe_tool
def get_fx_rates(
    pairs: Annotated[
        str,
        "comma-separated FX pairs as <BASE><QUOTE> (e.g. 'EURUSD,USDJPY'); "
        "defaults to the major USD pairs",
    ] = DEFAULT_FX_PAIRS,
) -> str:
    """
    Retrieve current FX rates and 1/3/12-month percentage changes for
    major currency pairs. No API key required.

    Use this to ground currency analysis: dollar strength/weakness,
    EUR/JPY/CNY moves relevant to exporters, importers, and companies
    with foreign revenue. Pair this with the geographic segment data
    (Fundamentals analyst) to assess translation and input-cost
    sensitivity for a specific company.

    Args:
        pairs: Comma-separated currency pairs (e.g. "EURUSD,USDJPY").
            Each pair must be a 6-letter code like EURUSD; XAUUSD and
            other Yahoo "=-X" symbols also work.

    Returns:
        str: A formatted markdown table of rates and changes
    """
    raw = [p.strip().upper() for p in (pairs or "").split(",") if p.strip()]
    pairs_list = raw or [p.strip() for p in DEFAULT_FX_PAIRS.split(",")]
    pairs_list = pairs_list[:10]

    lines = [
        "# FX Rates & Trends",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| Pair | Spot | 1M | 3M | 12M |",
        "|---|---|---|---|---|",
    ]
    any_data = False
    for pair in pairs_list:
        symbol = pair if pair.endswith("=X") else f"{pair}=X"
        try:
            closes = yf_retry(
                lambda s=symbol: yf.Ticker(s).history(period="13mo")
            )["Close"]
        except Exception:
            closes = None
        if closes is None or len(closes) < 30:
            lines.append(f"| {pair} | N/A | N/A | N/A | N/A |")
            continue
        any_data = True
        spot = float(closes.iloc[-1])
        cells = [f"{spot:.4f}"]
        for lookback in _FX_WINDOWS.values():
            chg = _fx_change(closes, lookback)
            cells.append("N/A" if chg is None else f"{chg * 100:+.1f}%")
        lines.append(f"| {pair} | " + " | ".join(cells) + " |")

    if not any_data:
        lines.append(
            "\nNo FX data could be retrieved for the requested pairs "
            f"({', '.join(pairs_list)})."
        )
    else:
        lines.append(
            "\nInterpretation: for USD-quoted pairs like EURUSD, a positive "
            "change means the foreign currency strengthened against the "
            "dollar (a dollar tailwind for US exporters, headwind for US-"
            "revenue/foreign-cost importers). For USDCNY-style pairs a rise "
            "means the dollar strengthened."
        )
    return "\n".join(lines)
