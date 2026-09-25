"""Equity-intelligence tools added to close the analytical gaps surfaced in
post-run report audits (e.g. the ORCL analysis):

- get_analyst_estimates        — sell-side consensus / revisions / price targets
- get_credit_and_debt_detail   — debt structure, interest coverage, maturity hints
- get_customer_concentration   — RPO / backlog / customer-concentration disclosures
- get_short_interest           — short % of float, days to cover
- get_institutional_holders    — 13F ownership and flow
- get_option_positioning       — open interest, put/call, IV around key levels
- get_earnings_calendar        — next earnings + guidance/beat-miss history
- get_capital_allocation       — multi-year buybacks, dividends, share count
- get_governance               — ownership concentration, insider/founder stakes

These call yfinance / SEC EDGAR directly (like the option-Greeks and macro
tools), so they need no API key beyond what yfinance already requires. Every
call is defensive: a missing attribute degrades to "N/A" rather than crashing
the pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

import yfinance as yf
from langchain_core.tools import tool

from tradingagents.dataflows.stockstats_utils import yf_retry
from tradingagents.agents.utils.tool_errors import safe_tool

logger = logging.getLogger("tradingagents.tools.equity_intel")


def _fmt_num(v):
    if v is None:
        return "N/A"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.2f}"


def _safe_info(ticker_obj: yf.Ticker) -> dict:
    try:
        return yf_retry(lambda: ticker_obj.info) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 1. Sell-side analyst consensus estimates & revisions
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_analyst_estimates(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve sell-side analyst consensus: price targets, EPS/revenue estimates,
    recommendation distribution, and the earnings surprise history.

    Use this to independently test whether a valuation multiple (e.g. a forward
    P/E implying EPS will double) is actually the consensus trajectory or a
    misprint, and to sanity-check DCF / EPV outputs against the Street.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted report with consensus estimates, targets, and revisions
    """
    t = yf.Ticker(ticker.upper())
    info = _safe_info(t)
    lines = [
        f"# Analyst Consensus for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Price Targets & Recommendations",
    ]
    target_fields = [
        ("Target Mean Price", "targetMeanPrice"),
        ("Target Median Price", "targetMedianPrice"),
        ("Target High Price", "targetHighPrice"),
        ("Target Low Price", "targetLowPrice"),
        ("Current Price", "currentPrice"),
        ("Recommendation", "recommendationKey"),
        ("Number of Analysts", "numberOfAnalystOpinions"),
        ("Recommendation Mean", "recommendationMean"),
    ]
    for label, key in target_fields:
        val = info.get(key)
        if val is not None:
            lines.append(f"  {label}: {val}")
    implied = None
    mean_target = info.get("targetMeanPrice")
    price = info.get("currentPrice")
    if mean_target and price:
        try:
            implied = (float(mean_target) - float(price)) / float(price) * 100
            lines.append(f"  Implied Upside to Mean Target: {implied:+.1f}%")
        except (TypeError, ValueError):
            pass

    # EPS / revenue estimates (yfinance earnings_estimate / revenue_estimate tables)
    for label, attr in (
        ("EPS Trend", "earnings_estimate"),
        ("Revenue Estimates", "revenue_estimate"),
    ):
        try:
            df = yf_retry(lambda a=attr: getattr(t, a))
        except Exception:
            df = None
        if df is None:
            continue
        try:
            if hasattr(df, "empty") and not df.empty:
                lines.append(f"\n## {label} (most recent rows)")
                lines.append(df.tail(4).to_csv())
        except Exception:
            continue

    # Earnings surprise history
    try:
        eh = yf_retry(lambda: t.earnings_history)
        if eh is not None and hasattr(eh, "empty") and not eh.empty:
            lines.append("\n## Earnings Surprise History")
            lines.append(eh.tail(8).to_csv())
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Credit & debt detail
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_credit_and_debt_detail(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Retrieve debt structure and credit-relevant metrics: total debt, long/
    short-term split, debt-to-equity, net debt, interest coverage, and the
    annual debt trajectory.

    Use this to resolve load-bearing refinancing-risk claims (e.g. "debt
    maturities out to 2066", "interest coverage 4.5x") against actual filings
    rather than letting debaters assert them unchecked.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date in YYYY-MM-DD format (optional)
    Returns:
        str: Formatted report with debt structure and coverage metrics
    """
    t = yf.Ticker(ticker.upper())
    info = _safe_info(t)
    lines = [
        f"# Credit & Debt Detail for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Headline Leverage (from info)",
    ]
    for label, key in (
        ("Total Debt", "totalDebt"),
        ("Total Debt to Equity", "debtToEquity"),
        ("Total Debt to Capital", "totalDebtToCapital"),
        ("Total Cash", "totalCash"),
        ("Net Debt", "netDebt"),
        ("Quick Ratio", "quickRatio"),
        ("Current Ratio", "currentRatio"),
    ):
        val = info.get(key)
        if val is not None:
            lines.append(f"  {label}: {val}")

    # Balance sheet: long-term vs current debt trajectory
    try:
        bs = yf_retry(lambda: t.balance_sheet)
        if bs is not None and hasattr(bs, "empty") and not bs.empty:
            debt_rows = [r for r in bs.index if "debt" in str(r).lower() or "Long Term" in str(r)]
            if debt_rows:
                sub = bs.loc[debt_rows]
                lines.append("\n## Debt Line Items (annual, most recent periods)")
                lines.append(sub.to_csv())
    except Exception:
        pass

    # Interest coverage from income statement
    try:
        inc = yf_retry(lambda: t.income_stmt)
        if inc is not None and hasattr(inc, "empty") and not inc.empty:
            def _pick(candidates):
                for c in candidates:
                    for idx in inc.index:
                        if c.lower() in str(idx).lower():
                            return inc.loc[idx]
                return None
            ebit = _pick(["Operating Income", "EBIT"])
            interest = _pick(["Interest Expense"])
            if ebit is not None and interest is not None:
                latest_ebit = ebit.iloc[0] if hasattr(ebit, "iloc") else ebit
                latest_int = interest.iloc[0] if hasattr(interest, "iloc") else interest
                try:
                    cov = float(latest_ebit) / abs(float(latest_int)) if float(latest_int) else None
                    lines.append("\n## Interest Coverage")
                    lines.append(f"  Operating Income (latest): {_fmt_num(latest_ebit)}")
                    lines.append(f"  Interest Expense (latest): {_fmt_num(latest_int)}")
                    if cov is not None:
                        lines.append(f"  Interest Coverage: {cov:.2f}x")
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
    except Exception:
        pass

    lines.append(
        "\nNote: Credit ratings and CDS spreads are not available via free "
        "data. See the macro market report for the Treasury / credit-spread "
        "environment, and the 10-K filing for the full debt-maturity schedule."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Short interest
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_short_interest(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve short-interest metrics: short % of float, short ratio (days to
    cover), shares short, and month-over-month change.

    Use this to assess positioning / squeeze risk around sharp drawdowns.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted short-interest report
    """
    t = yf.Ticker(ticker.upper())
    info = _safe_info(t)
    lines = [
        f"# Short Interest for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    any_data = False
    for label, key in (
        ("Short % of Float", "shortPercentOfFloat"),
        ("Short Ratio (Days to Cover)", "shortRatio"),
        ("Shares Short", "sharesShort"),
        ("Shares Short Prior Month", "sharesShortPriorMonth"),
        ("Short Prior Month Date", "sharesShortPreviousMonthDate"),
    ):
        val = info.get(key)
        if val is not None:
            any_data = True
            if "Percent" in label and val:
                try:
                    lines.append(f"  {label}: {float(val) * 100:.2f}%")
                    continue
                except (TypeError, ValueError):
                    pass
            lines.append(f"  {label}: {val}")
    if not any_data:
        lines.append("  No short-interest data available via yfinance for this ticker.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Institutional holders / 13F flows
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_institutional_holders(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve institutional ownership: major-holder percentages and the top
    institutional holders with quarter-over-quarter share changes (13F flows).

    Use this to see who is buying/selling during a drawdown and how
    concentrated ownership is.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted institutional-ownership report
    """
    t = yf.Ticker(ticker.upper())
    lines = [
        f"# Institutional Holders for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    try:
        mh = yf_retry(lambda: t.major_holders)
        if mh is not None and hasattr(mh, "empty") and not mh.empty:
            lines.append("## Major Holders")
            lines.append(mh.to_csv(index=False, header=False))
    except Exception:
        pass
    try:
        ih = yf_retry(lambda: t.institutional_holders)
        if ih is not None and hasattr(ih, "empty") and not ih.empty:
            lines.append("\n## Top Institutional Holders (with qtr-over-qtr change)")
            lines.append(ih.head(15).to_csv(index=False))
        else:
            lines.append("\nNo institutional holder detail available.")
    except Exception:
        lines.append("\nNo institutional holder detail available.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Options positioning (open interest, put/call, IV)
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_option_positioning(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """Retrieve options positioning: total open interest, put/call OI ratio,
    implied volatility, and the strikes with the largest open interest for the
    nearest expirations.

    Use this to read positioning around key price levels (supports/resistance,
    max-pain) — complementary to the option-Greeks tool which only gives delta/
    gamma.

    Args:
        symbol (str): Ticker symbol of the company
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
    Returns:
        str: Formatted options-positioning report
    """
    t = yf.Ticker(symbol.upper())
    expiries = yf_retry(lambda: t.options)
    if not expiries:
        return f"No options chain available for {symbol}."
    lines = [
        f"# Options Positioning for {symbol.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Available expirations: {', '.join(expiries[:4])}{' ...' if len(expiries) > 4 else ''}\n",
    ]
    for expiry in expiries[:2]:
        try:
            chain = yf_retry(lambda e=expiry: t.option_chain(e))
        except Exception:
            continue
        calls = chain.calls if chain.calls is not None and not chain.calls.empty else None
        puts = chain.puts if chain.puts is not None and not chain.puts.empty else None
        if calls is None and puts is None:
            continue
        call_oi = float(calls["openInterest"].sum()) if calls is not None else 0.0
        put_oi = float(puts["openInterest"].sum()) if puts is not None else 0.0
        total_oi = call_oi + put_oi
        pc_ratio = (put_oi / call_oi) if call_oi else None
        call_iv = float(calls["impliedVolatility"].mean()) if calls is not None and not calls["impliedVolatility"].empty else None
        put_iv = float(puts["impliedVolatility"].mean()) if puts is not None and not puts["impliedVolatility"].empty else None

        lines.append(f"## Expiry {expiry}")
        lines.append(f"  Total Open Interest: {total_oi:,.0f}")
        lines.append(f"  Call OI: {call_oi:,.0f} | Put OI: {put_oi:,.0f}")
        if pc_ratio is not None:
            lines.append(f"  Put/Call OI Ratio: {pc_ratio:.3f}")
        if call_iv is not None:
            lines.append(f"  Avg Call IV: {call_iv:.2%}")
        if put_iv is not None:
            lines.append(f"  Avg Put IV: {put_iv:.2%}")

        # Top OI strikes (potential support/resistance / max-pain proxies)
        for side, df in (("Call", calls), ("Put", puts)):
            if df is None or df.empty:
                continue
            top = df.nlargest(5, "openInterest")[["strike", "openInterest", "impliedVolatility"]]
            lines.append(f"  Top {side} OI strikes:")
            for _, row in top.iterrows():
                lines.append(
                    f"    strike {row['strike']:.2f}  OI {int(row['openInterest']):,}  IV {row['impliedVolatility']:.2%}"
                )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Earnings calendar + guidance/beat-miss history
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_earnings_calendar(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve the earnings calendar (next reported earnings date) and the
    recent earnings beat/miss history with EPS actual vs estimate.

    Use this for near-term catalyst timing and to gauge management's guidance
    track record.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted earnings-calendar and history report
    """
    t = yf.Ticker(ticker.upper())
    lines = [
        f"# Earnings Calendar for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    try:
        cal = yf_retry(lambda: t.calendar)
        if cal:
            lines.append("## Next Earnings")
            if isinstance(cal, dict):
                for k, v in cal.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(str(cal))
    except Exception:
        pass
    try:
        eh = yf_retry(lambda: t.earnings_history)
        if eh is not None and hasattr(eh, "empty") and not eh.empty:
            lines.append("\n## Earnings History (actual vs estimate)")
            cols = [c for c in ("epsActual", "epsEstimate", "epsDifference", "quarter") if c in eh.columns]
            lines.append(eh.tail(8)[cols].to_csv(index=False) if cols else eh.tail(8).to_csv())
    except Exception:
        pass
    try:
        ed = yf_retry(lambda: t.earnings_dates)
        if ed is not None and hasattr(ed, "empty") and not ed.empty:
            lines.append("\n## Upcoming Earnings Dates")
            lines.append(ed.head(4).to_csv())
    except Exception:
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Capital allocation history
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_capital_allocation_history(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve the multi-year capital-allocation record: share buybacks
    (repurchase of capital), dividends paid, share-count drift, and stock
    splits.

    Use this to test whether "the equity is a call option on growth" framing
    is being engineered by buybacks, and to judge capital-allocation
    discipline over time.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted capital-allocation history report
    """
    t = yf.Ticker(ticker.upper())
    lines = [
        f"# Capital Allocation History for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    try:
        cf = yf_retry(lambda: t.cashflow)
        if cf is not None and hasattr(cf, "empty") and not cf.empty:
            wanted = [r for r in cf.index if any(
                k in str(r).lower() for k in
                ("repurchase", "dividend paid", "capital expenditure", "stock based comp")
            )]
            if wanted:
                lines.append("## Annual Cash-Flow Allocation (most recent periods)")
                lines.append(cf.loc[wanted].to_csv())
    except Exception:
        pass
    try:
        splits = yf_retry(lambda: t.splits)
        if splits is not None and hasattr(splits, "empty") and not splits.empty:
            lines.append("\n## Stock Splits")
            lines.append(splits.tail(5).to_csv())
    except Exception:
        pass
    try:
        divs = yf_retry(lambda: t.dividends)
        if divs is not None and hasattr(divs, "empty") and not divs.empty:
            lines.append("\n## Dividends (recent)")
            lines.append(divs.tail(8).to_csv())
    except Exception:
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Governance & ownership concentration
# ---------------------------------------------------------------------------


@tool
@safe_tool
def get_governance(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve governance / ownership-concentration data: major-holder
    percentages, top institutional holders, and insider ownership signals.

    Use this to flag concentrated founder/insider control (e.g. a founder
    owning ~40% of a mega-cap) and to assess governance risk.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted governance / ownership report
    """
    t = yf.Ticker(ticker.upper())
    info = _safe_info(t)
    lines = [
        f"# Governance & Ownership for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    held_pct = info.get("heldPercentInsiders")
    inst_pct = info.get("heldPercentInstitutions")
    if held_pct is not None:
        lines.append(f"  % Held by Insiders: {float(held_pct) * 100:.2f}%")
    if inst_pct is not None:
        lines.append(f"  % Held by Institutions: {float(inst_pct) * 100:.2f}%")
    try:
        mh = yf_retry(lambda: t.major_holders)
        if mh is not None and hasattr(mh, "empty") and not mh.empty:
            lines.append("\n## Major Holders Breakdown")
            lines.append(mh.to_csv(index=False, header=False))
    except Exception:
        pass
    try:
        ih = yf_retry(lambda: t.institutional_holders)
        if ih is not None and hasattr(ih, "empty") and not ih.empty:
            lines.append("\n## Top Institutional Holders")
            lines.append(ih.head(10).to_csv(index=False))
    except Exception:
        pass
    lines.append(
        "\nNote: For founder/dual-class-share and board-independence detail, "
        "consult the 10-K Item 1 (Business) and the DEF 14A proxy statement."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Sector-relative momentum & factor positioning
# ---------------------------------------------------------------------------

#: GICS sector (as reported by yfinance ``info['sector']``) -> liquid
#: sector ETF used as the relative-strength benchmark.
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

#: Trading-day lookbacks approximating 1 / 3 / 6 / 12 months.
_MOMENTUM_WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}


def _total_return(closes, lookback: int):
    """Total return over the last *lookback* closes, or None if too short."""
    if closes is None or len(closes) < lookback + 1:
        return None
    base = float(closes.iloc[-(lookback + 1)])
    if base == 0:
        return None
    return float(closes.iloc[-1]) / base - 1.0


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:+.1f}%"


def _sector_etf_for(ticker_obj: yf.Ticker) -> tuple:
    """Return ``(sector, etf)`` for the ticker; ETF falls back to SPY."""
    sector = _safe_info(ticker_obj).get("sector") or "Unknown"
    return sector, SECTOR_ETF_MAP.get(sector, "SPY")


def _max_drawdown(closes) -> float | None:
    if closes is None or len(closes) < 2:
        return None
    running_max = closes.cummax()
    dd = (closes / running_max) - 1.0
    return float(dd.min())


@tool
@safe_tool
def get_relative_momentum_vs_sector(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Compare the stock's momentum against its sector ETF over 1/3/6/12
    month windows, plus the stock's position in its 52-week range and its
    current valuation multiples (context for factor/style positioning).

    Use this to see whether the stock is leading or lagging its sector,
    whether momentum is broad or idiosyncratic, and where price sits in
    its annual range. This is a technical, timing-oriented signal — it
    does not speak to business quality.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted relative-momentum report
    """
    t = yf.Ticker(ticker.upper())
    sector, etf = _sector_etf_for(t)

    hist = None
    bench = None
    try:
        hist = yf_retry(lambda: t.history(period="13mo"))["Close"]
        bench = yf_retry(lambda: yf.Ticker(etf).history(period="13mo"))["Close"]
    except Exception:
        pass

    lines = [
        f"# Relative Momentum vs Sector for {ticker.upper()}",
        f"Sector: {sector} | Benchmark ETF: {etf}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    if hist is None or len(hist) < 30:
        lines.append("Price history unavailable for this ticker.")
        return "\n".join(lines)

    lines.append("| Window | Stock | Sector ETF | Relative |")
    lines.append("|---|---|---|---|")
    for label, lookback in _MOMENTUM_WINDOWS.items():
        r = _total_return(hist, lookback)
        b = _total_return(bench, lookback)
        rel = None if r is None or b is None else r - b
        lines.append(f"| {label} | {_pct(r)} | {_pct(b)} | {_pct(rel)} |")

    hi = float(hist.iloc[-252:].max()) if len(hist) >= 30 else float(hist.max())
    lo = float(hist.iloc[-252:].min()) if len(hist) >= 30 else float(hist.min())
    last = float(hist.iloc[-1])
    if hi > lo:
        pos = (last - lo) / (hi - lo)
        lines.append(
            f"\n52-week range position: {pos * 100:.0f}% "
            f"(last {last:.2f}, 52w low {lo:.2f}, high {hi:.2f})"
        )

    sma50 = hist.tail(50).mean()
    sma200 = hist.tail(200).mean() if len(hist) >= 200 else None
    trend = "above" if last > float(sma50) else "below"
    lines.append(f"Price is {trend} its 50-day average ({float(sma50):.2f})")
    if sma200 is not None:
        trend200 = "above" if last > float(sma200) else "below"
        lines.append(f"Price is {trend200} its 200-day average ({float(sma200):.2f})")

    info = _safe_info(t)
    pe = info.get("trailingPE")
    ps = info.get("priceToSalesTrailing12Months")
    roe = info.get("returnOnEquity")
    margin = info.get("profitMargins")
    if any(v is not None for v in (pe, ps, roe, margin)):
        lines.append("\nCurrent multiples (context; compare via peer tools):")
        if pe is not None:
            lines.append(f"  Trailing P/E: {pe:.1f}")
        if ps is not None:
            lines.append(f"  P/S: {ps:.2f}")
        if roe is not None:
            lines.append(f"  ROE: {roe * 100:.1f}%")
        if margin is not None:
            lines.append(f"  Profit margin: {margin * 100:.1f}%")

    lines.append(
        "\nNote: relative strength is a timing/positioning input. Use it to "
        "inform entry/exit context — do not treat it as the directional thesis."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. Dilution profile (share count trajectory, SBC vs buybacks)
# ---------------------------------------------------------------------------


def _annualized_share_growth(shares) -> float | None:
    """Annualized % change in share count across a (possibly sparse) series."""
    if shares is None or len(shares) < 2:
        return None
    first, last = float(shares.iloc[0]), float(shares.iloc[-1])
    if first <= 0:
        return None
    years = (shares.index[-1] - shares.index[0]).days / 365.25
    if years <= 0:
        return None
    return ((last / first) ** (1.0 / years) - 1.0) * 100.0


@tool
@safe_tool
def get_dilution_profile(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve the share-dilution profile: reported share-count trajectory
    (5 years), annualized dilution rate, and the stock-based-compensation
    vs. buyback offset from the cash-flow statement.

    Use this to quantify whether per-share value is being created or
    eroded: aggressive SBC without offsetting buybacks silently dilutes
    shareholders even when headline EPS grows.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted dilution profile report
    """
    t = yf.Ticker(ticker.upper())
    lines = [
        f"# Dilution Profile for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    shares = None
    try:
        shares = yf_retry(lambda: t.get_shares_full(start="2019-01-01"))
    except Exception:
        pass
    if shares is not None and hasattr(shares, "empty") and not shares.empty:
        growth = _annualized_share_growth(shares)
        lines.append("## Share count trajectory (year-end samples)")
        sampled = shares.resample("YE").last().dropna()
        for idx, val in sampled.tail(6).items():
            lines.append(f"  {idx.year}: {val:,.0f}")
        if growth is not None:
            verdict = (
                "dilutive" if growth > 2.0 else
                "anti-dilutive (net buybacks exceed issuance)" if growth < -1.0
                else "roughly flat"
            )
            lines.append(
                f"\nAnnualized share-count change: {growth:+.2f}%/yr ({verdict})"
            )
    else:
        lines.append("Share-count history unavailable for this ticker.")

    try:
        cf = yf_retry(lambda: t.cashflow)
        if cf is not None and hasattr(cf, "empty") and not cf.empty:
            sbc_rows = [r for r in cf.index if "stock based comp" in str(r).lower()]
            bb_rows = [r for r in cf.index if "repurchase" in str(r).lower()]
            if sbc_rows or bb_rows:
                lines.append("\n## SBC vs buybacks (annual, cash-flow statement)")
                for r in (sbc_rows + bb_rows):
                    lines.append(cf.loc[[r]].to_csv(header=False).strip())
                if sbc_rows and bb_rows:
                    latest = cf.columns[0]
                    sbc = cf.loc[sbc_rows[0], latest]
                    bb = cf.loc[bb_rows[0], latest]
                    try:
                        net = float(bb) + float(sbc)  # repurchase is negative
                        label = "offsets SBC" if net <= 0 else "does NOT offset SBC"
                        lines.append(
                            f"\nMost recent year: buybacks {_fmt_num(abs(float(bb)))} vs "
                            f"SBC {_fmt_num(abs(float(sbc)))} → buybacks {label}."
                        )
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass

    lines.append(
        "\nNote: share-count figures reflect reported snapshots (buyback "
        "settlement lags announcement). Cross-check with "
        "get_capital_allocation_history for the multi-year allocation view."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. Macro-regime analogs (Fed cycle performance)
# ---------------------------------------------------------------------------


def _fetch_fedfunds_monthly() -> "object | None":
    """Monthly Fed funds rate series from FRED's keyless CSV endpoint."""
    import io

    import requests

    try:
        resp = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("FEDFUNDS fetch failed: %s", exc)
        return None
    import pandas as pd

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().set_index("date").sort_index()
    return df["rate"]


def _classify_fed_regimes(fedfunds, min_regimes: int = 4):
    """Group the Fed funds series into HIKING / CUTTING / PAUSE regimes.

    A regime changes when the 6-month change in the fed funds rate exceeds
    +/-0.25pp. Returns a list of dicts with direction, start, end, and
    total rate move, most recent last.
    """
    mom = fedfunds.diff(6)
    regimes = []
    current_dir = None
    start = None
    for date, val in mom.dropna().items():
        if val > 0.25:
            new_dir = "HIKING"
        elif val < -0.25:
            new_dir = "CUTTING"
        else:
            new_dir = current_dir if current_dir in ("HIKING", "CUTTING") else "PAUSE"
        if new_dir != current_dir:
            if current_dir is not None and start is not None:
                regimes.append(
                    {
                        "direction": current_dir,
                        "start": start,
                        "end": date,
                    }
                )
            current_dir = new_dir
            start = date
    if current_dir is not None and start is not None:
        regimes.append(
            {"direction": current_dir, "start": start, "end": fedfunds.index[-1]}
        )
    # Drop leading PAUSE stubs / merge tiny fragments: keep regimes with data
    return [r for r in regimes if r["end"] > r["start"]][-min_regimes:]


def _regime_performance(closes, start, end):
    """Total return and max drawdown of *closes* within [start, end]."""
    # yfinance returns tz-aware indexes; FRED regime bounds are naive.
    if getattr(closes.index, "tz", None) is not None:
        closes = closes.tz_localize(None)
    window = closes[(closes.index >= start) & (closes.index <= end)]
    if len(window) < 2:
        return None
    ret = float(window.iloc[-1]) / float(window.iloc[0]) - 1.0
    return ret, _max_drawdown(window)


@tool
@safe_tool
def get_regime_analog(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Show how the stock and its sector ETF performed during recent Fed
    hiking / cutting cycles: total return, max drawdown, and recovery
    context for each of the last ~4 rate regimes.

    Use this to ground historical analogies in data (e.g. "how did this
    name behave during the 2022 hiking cycle?") instead of asserting
    unvalidated comparisons like 'this is just Meta 2022'.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: Formatted regime-analog report
    """
    t = yf.Ticker(ticker.upper())
    _, etf = _sector_etf_for(t)
    fedfunds = _fetch_fedfunds_monthly()
    if fedfunds is None or len(fedfunds) < 24:
        return (
            "[get_regime_analog] Fed funds data unavailable; cannot classify "
            "regimes right now."
        )

    regimes = _classify_fed_regimes(fedfunds)
    if not regimes:
        return "[get_regime_analog] Could not classify any regimes."

    hist = None
    bench = None
    try:
        hist = yf_retry(lambda: t.history(period="max"))["Close"]
        bench = yf_retry(lambda: yf.Ticker(etf).history(period="max"))["Close"]
    except Exception:
        pass

    lines = [
        f"# Fed-Regime Analogs for {ticker.upper()} (benchmark: {etf})",
        "",
        "| Regime | Start | End | Rate move | Stock | Sector ETF |",
        "|---|---|---|---|---|---|",
    ]
    any_row = False
    for r in regimes:
        start, end = r["start"], r["end"]
        rate_move = float(fedfunds.loc[end]) - float(fedfunds.loc[start])
        sr = _regime_performance(hist, start, end) if hist is not None else None
        br = _regime_performance(bench, start, end) if bench is not None else None
        if sr is None and br is None:
            continue  # regime predates listing data
        stock_txt = (
            f"{sr[0] * 100:+.0f}% (maxDD {sr[1] * 100:.0f}%)" if sr else "N/A"
        )
        bench_txt = (
            f"{br[0] * 100:+.0f}% (maxDD {br[1] * 100:.0f}%)" if br else "N/A"
        )
        lines.append(
            f"| {r['direction']} | {start.date()} | {end.date()} | "
            f"{rate_move:+.2f}pp | {stock_txt} | {bench_txt} |"
        )
        any_row = True

    if not any_row:
        lines.append(
            f"\nNo regime data overlaps {ticker.upper()}'s trading history — "
            "the listing is too recent for analogs."
        )
    else:
        lines.append(
            "\nHow to use: when arguing by analogy (e.g. 'like 2022'), check "
            "whether the stock's actual behavior in the matching regime row "
            "supports the analogy. Rate direction is the regime driver; pair "
            "with the macro report for inflation/growth context of each cycle."
        )
    return "\n".join(lines)
