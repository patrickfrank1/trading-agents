"""Multi-method valuation engines for the fundamentals analyst.

Each ``run_*`` function computes a valuation using a different methodology
and returns a Markdown report.  Tool wrappers live in
``fundamental_data_tools.py``.
"""

from __future__ import annotations

import math

import yfinance as yf

from tradingagents.agents.utils.dcf import (
    _compute_cagr,
    _estimate_wacc,
    _fetch_risk_free_rate,
    _find_row,
    _fmt,
    _get_historical_annual,
    _latest,
    _pct,
    _pct_signed,
    _safe_get,
    _to_clean_list,
)


def _current_price(info):
    return (
        _safe_get(info, "currentPrice")
        or _safe_get(info, "regularMarketPrice")
        or _safe_get(info, "previousClose")
        or 0.0
    )


def _shares(info):
    return _safe_get(info, "sharesOutstanding", 0) or 0


def _heading(name, ticker):
    return f"## {name}: {_safe_get(yf.Ticker(ticker).info, 'longName', ticker)} ({ticker.upper()})"


def _kv(label, value):
    return f"| {label} | {value} |"


def _section(title):
    return f"\n### {title}\n"


def _table_row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _upside(fv, price):
    return ((fv / price) - 1) * 100 if price > 0 else 0


def _fetch_historical(ticker_obj):
    return _get_historical_annual(ticker_obj)


def _net_debt(historical):
    return _latest(historical.get("total_debt", [])) - _latest(historical.get("cash", []))


# ---------------------------------------------------------------------------
# 1. Comparable Company Analysis
# ---------------------------------------------------------------------------

def run_comps_analysis(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    historical = _fetch_historical(ticker_obj)

    multiples = {}
    for label, key in [
        ("Trailing P/E", "trailingPE"),
        ("Forward P/E", "forwardPE"),
        ("P/B", "priceToBook"),
        ("P/S", "priceToSalesTrailing12Months"),
        ("EV/EBITDA", "enterpriseToEbitda"),
        ("EV/Revenue", "enterpriseToRevenue"),
    ]:
        val = _safe_get(info, key)
        if val is not None and not math.isnan(val):
            multiples[label] = val

    net_debt = _net_debt(historical)
    total_revenue = _safe_get(info, "totalRevenue", 0) or 0
    ebitda = _safe_get(info, "ebitda", 0) or 0
    net_income = _safe_get(info, "netIncomeToCommon", 0) or 0
    book_value = _safe_get(info, "bookValue", 0) or 0
    eps = _safe_get(info, "trailingEps", 0) or 0
    fwd_eps = _safe_get(info, "forwardEps", 0) or 0

    implied = {}
    if eps > 0 and "Trailing P/E" in multiples:
        implied["P/E"] = multiples["Trailing P/E"] * eps
    if fwd_eps > 0 and "Forward P/E" in multiples:
        implied["P/E (Fwd)"] = multiples["Forward P/E"] * fwd_eps
    if book_value > 0 and "P/B" in multiples:
        implied["P/B"] = multiples["P/B"] * book_value
    if total_revenue > 0 and "P/S" in multiples:
        implied["P/S"] = multiples["P/S"] * total_revenue / shares
    if ebitda > 0 and "EV/EBITDA" in multiples:
        implied["EV/EBITDA"] = (multiples["EV/EBITDA"] * ebitda - net_debt) / shares
    if total_revenue > 0 and "EV/Revenue" in multiples:
        implied["EV/Revenue"] = (multiples["EV/Revenue"] * total_revenue - net_debt) / shares

    lines = [
        _heading("Comparable Company Analysis", ticker),
        "",
        f"**Sector:** {_safe_get(info, 'sector', 'N/A')}  |  **Industry:** {_safe_get(info, 'industry', 'N/A')}",
        "",
        _section("Current Trading Multiples"),
        "",
        "| Multiple | Value |",
        "|---|---|",
    ]

    for label, val in multiples.items():
        lines.append(_kv(label, f"{val:.2f}x"))

    div_yield = _safe_get(info, "dividendYield")
    if div_yield and div_yield > 0:
        lines.append(_kv("Dividend Yield", f"{div_yield:.2%}"))

    roe = _safe_get(info, "returnOnEquity")
    if roe is not None:
        lines.append(_kv("ROE", f"{roe:.2%}"))
    margin = _safe_get(info, "profitMargins")
    if margin is not None:
        lines.append(_kv("Net Margin", f"{margin:.2%}"))

    lines += [
        "",
        _section("Key Financial Metrics"),
        "",
        f"- **Market Cap:** {_fmt(_safe_get(info, 'marketCap', 0) or 0)}",
        f"- **Enterprise Value:** {_fmt(_safe_get(info, 'enterpriseValue', 0) or 0)}",
        f"- **Revenue (TTM):** {_fmt(total_revenue)}",
        f"- **EBITDA (TTM):** {_fmt(ebitda)}",
        f"- **Net Income (TTM):** {_fmt(net_income)}",
        f"- **Book Value:** {_fmt(book_value)}",
        f"- **Net Debt:** {_fmt(net_debt)}",
        "",
        _section("Implied Valuation from Multiples"),
        "",
        "| Method | Fair Value/Share | Upside / Downside |",
        "|---|---|---|",
    ]

    vals = []
    for method, fv in implied.items():
        if fv and fv > 0:
            vals.append(fv)
            lines.append(_table_row(method, f"${fv:.2f}", _pct_signed(_upside(fv, price))))

    if vals:
        avg = sum(vals) / len(vals)
        med = sorted(vals)[len(vals) // 2]
        lines += [
            "",
            f"**Mean Implied FV/Share:** ${avg:.2f} ({_pct_signed(_upside(avg, price))})",
            f"**Median Implied FV/Share:** ${med:.2f} ({_pct_signed(_upside(med, price))})",
        ]

    lines += [
        "",
        "> **Note:** This analysis uses the company's own current market multiples. "
        "A full comps analysis would benchmark against a curated peer group. "
        "The values above reflect the market's current pricing of fundamentals.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Precedent Transactions
# ---------------------------------------------------------------------------

def run_precedent_transactions_analysis(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    historical = _fetch_historical(ticker_obj)
    net_debt = _net_debt(historical)

    ebitda = _safe_get(info, "ebitda", 0) or 0
    total_revenue = _safe_get(info, "totalRevenue", 0) or 0
    net_income = _safe_get(info, "netIncomeToCommon", 0) or 0
    book_value = _safe_get(info, "bookValue", 0) or 0
    ev = _safe_get(info, "enterpriseValue", 0) or 0

    current_ev_ebitda = _safe_get(info, "enterpriseToEbitda")
    current_ev_rev = _safe_get(info, "enterpriseToRevenue")
    current_pe = _safe_get(info, "trailingPE")
    current_pb = _safe_get(info, "priceToBook")

    premiums = [0.20, 0.30, 0.40]

    lines = [
        _heading("Precedent Transaction Analysis", ticker),
        "",
        "> **Note:** Actual M&A transaction data is not available from yfinance. "
        "This analysis estimates acquisition value by applying typical control premiums "
        "(20%, 30%, 40%) to the company's current trading multiples.",
        "",
        _section("Current Enterprise & Equity Metrics"),
        "",
        _kv("Current Price", f"${price:.2f}"),
        _kv("Enterprise Value", _fmt(ev)),
        _kv("Net Debt", _fmt(net_debt)),
        _kv("Revenue (TTM)", _fmt(total_revenue)),
        _kv("EBITDA (TTM)", _fmt(ebitda)),
        _kv("Net Income (TTM)", _fmt(net_income)),
        "",
        _section("Implied Acquisition Value"),
        "",
        "| Premium | EV/EBITDA Value | Implied Equity | FV/Share | Upside |",
        "|---|---|---|---|---|",
    ]

    rows_data = []
    if current_ev_ebitda and current_ev_ebitda > 0 and ebitda > 0:
        for prem in premiums:
            implied_ev = ebitda * current_ev_ebitda * (1 + prem)
            implied_eq = implied_ev - net_debt
            fv = implied_eq / shares if shares > 0 else 0
            up = _upside(fv, price)
            rows_data.append((f"{prem:.0%} Premium", implied_ev, implied_eq, fv, up))

    if not rows_data and current_pe and current_pe > 0:
        eps = _safe_get(info, "trailingEps", 0) or 0
        if eps > 0:
            for prem in premiums:
                fv = eps * current_pe * (1 + prem)
                implied_eq = fv * shares
                up = _upside(fv, price)
                rows_data.append((f"{prem:.0%} Premium", None, implied_eq, fv, up))

    for label, imp_ev, imp_eq, fv, up in rows_data:
        ev_str = _fmt(imp_ev) if imp_ev else "N/A"
        lines.append(_table_row(label, ev_str, _fmt(imp_eq), f"${fv:.2f}", _pct_signed(up)))

    lines += [
        "",
        _section("Key Considerations"),
        "",
        "- Control premiums in M&A transactions typically range from 20-40% over market price",
        "- Strategic acquirers may pay above multiples due to synergies",
        "- The actual premium depends on competitive dynamics, regulatory risk, and strategic fit",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Asset-Based Valuation
# ---------------------------------------------------------------------------

def run_asset_based_valuation(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    historical = _fetch_historical(ticker_obj)
    net_debt = _net_debt(historical)

    total_assets = _safe_get(info, "totalAssets", 0) or 0
    total_liab = _safe_get(info, "totalLiab", 0) or 0
    book_value = _safe_get(info, "bookValue", 0) or 0
    intangibles = _safe_get(info, "goodwill", 0) or 0
    total_revenue = _safe_get(info, "totalRevenue", 0) or 0

    net_asset_value = total_assets - total_liab
    tangible_book = net_asset_value - intangibles
    nav_ps = net_asset_value / shares if shares > 0 else 0
    tangible_nav_ps = tangible_book / shares if shares > 0 else 0
    bv_ps = book_value / shares if shares > 0 else 0

    liquidation_discounts = [0.6, 0.7, 0.8]
    going_premiums = [0.0, 0.1, 0.2]

    lines = [
        _heading("Asset-Based Valuation", ticker),
        "",
        _section("Balance Sheet Summary"),
        "",
        _kv("Total Assets", _fmt(total_assets)),
        _kv("Total Liabilities", _fmt(total_liab)),
        _kv("Net Asset Value (NAV)", _fmt(net_asset_value)),
        _kv("Goodwill & Intangibles", _fmt(intangibles)),
        _kv("Tangible Book Value", _fmt(tangible_book)),
        _kv("Net Debt", _fmt(net_debt)),
        "",
        _section("Per-Share Values"),
        "",
        "| Metric | Value | Upside / Downside |",
        "|---|---|---|",
        _table_row("Book Value / Share", f"${bv_ps:.2f}", _pct_signed(_upside(bv_ps, price))),
        _table_row("NAV / Share", f"${nav_ps:.2f}", _pct_signed(_upside(nav_ps, price))),
        _table_row("Tangible NAV / Share", f"${tangible_nav_ps:.2f}", _pct_signed(_upside(tangible_nav_ps, price))),
        "",
        _section("Liquidation Scenario (Discounted Assets)"),
        "",
        "| Discount | Liquidation NAV | FV/Share | Upside / Downside |",
        "|---|---|---|---|",
    ]

    for disc in liquidation_discounts:
        liq_nav = tangible_book * disc
        liq_ps = liq_nav / shares if shares > 0 else 0
        lines.append(_table_row(f"{disc:.0%}", _fmt(liq_nav), f"${liq_ps:.2f}", _pct_signed(_upside(liq_ps, price))))

    lines += [
        "",
        _section("Going Concern Premium Scenarios"),
        "",
        "| Premium | Adjusted NAV | FV/Share | Upside / Downside |",
        "|---|---|---|---|",
    ]

    for prem in going_premiums:
        adj_nav = net_asset_value * (1 + prem)
        adj_ps = adj_nav / shares if shares > 0 else 0
        lines.append(_table_row(f"{prem:.0%}", _fmt(adj_nav), f"${adj_ps:.2f}", _pct_signed(_upside(adj_ps, price))))

    lines += [
        "",
        _section("Key Considerations"),
        "",
        "- Asset-based valuation is most relevant for asset-heavy firms (banks, REITs, holding companies)",
        "- Book value may understate market value of appreciated assets or overstate depreciated assets",
        "- Goodwill and intangibles should be excluded for liquidation analysis",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Dividend Discount Model (DDM)
# ---------------------------------------------------------------------------

def run_ddm_analysis(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    beta = _safe_get(info, "beta", 1.0) or 1.0
    risk_free = _fetch_risk_free_rate()
    erp = 0.055
    cost_of_equity = risk_free + beta * erp

    div_rate = _safe_get(info, "trailingAnnualDividendRate", 0) or 0
    div_yield = _safe_get(info, "trailingAnnualDividendYield", 0) or 0
    payout = _safe_get(info, "payoutRatio", 0) or 0
    five_yr_avg_yield = _safe_get(info, "fiveYearAvgDividendYield", 0) or 0
    earnings_growth = _safe_get(info, "earningsGrowth", 0) or 0

    historical = _fetch_historical(ticker_obj)
    rev_cagr = _compute_cagr(historical.get("revenue", []))
    if rev_cagr is None:
        rev_cagr = earnings_growth if earnings_growth else 0.03

    sustainable_g = (1 - payout) * (cost_of_equity if cost_of_equity > 0 else 0.10) if payout and payout < 1 else rev_cagr
    sustainable_g = max(-0.02, min(0.08, sustainable_g))

    lines = [
        _heading("Dividend Discount Model (DDM)", ticker),
        "",
        _section("Dividend Metrics"),
        "",
        _kv("Annual Dividend/Share", f"${div_rate:.4f}"),
        _kv("Dividend Yield", f"{div_yield:.2%}"),
        _kv("5-Year Avg Yield", f"{five_yr_avg_yield:.2%}"),
        _kv("Payout Ratio", f"{payout:.1%}" if payout else "N/A"),
        _kv("Earnings Growth", f"{earnings_growth:.1%}" if earnings_growth else "N/A"),
        _kv("Revenue CAGR", f"{rev_cagr:.1%}"),
        "",
        _section("Gordon Growth Model Assumptions"),
        "",
        _kv("Cost of Equity (r)", f"{cost_of_equity:.2%}"),
        _kv("Sustainable Growth (g)", f"{sustainable_g:.2%}"),
        _kv("Risk-Free Rate", f"{risk_free:.2%}"),
        _kv("Beta", f"{beta:.2f}"),
        _kv("Equity Risk Premium", f"{erp:.1%}"),
    ]

    if div_rate <= 0:
        lines += [
            "",
            "> **Warning:** This company does not pay a dividend. "
            "DDM is not applicable. Consider using DCF or other valuation methods instead.",
            "",
        ]
        return "\n".join(lines)

    if cost_of_equity <= sustainable_g:
        lines += [
            "",
            f"> **Warning:** Cost of equity ({cost_of_equity:.2%}) is not greater than "
            f"growth rate ({sustainable_g:.2%}). The Gordon Growth Model is undefined. "
            "Using a capped growth rate.",
            "",
        ]
        sustainable_g = cost_of_equity - 0.01

    gordon_value = div_rate * (1 + sustainable_g) / (cost_of_equity - sustainable_g)

    lines += [
        "",
        _section("Gordon Growth Model Valuation"),
        "",
        _kv("D₁ (Next Year Dividend)", f"${div_rate * (1 + sustainable_g):.4f}"),
        _kv("Intrinsic Value (Gordon)", f"${gordon_value:.2f}"),
        _kv("Current Price", f"${price:.2f}"),
        _kv("Upside / Downside", _pct_signed(_upside(gordon_value, price))),
        "",
        _section("Multi-Stage DDM Scenarios"),
        "",
        "| Stage 1 Growth | Stage 2 Growth | Years | Intrinsic Value | Upside |",
        "|---|---|---|---|---|",
    ]

    stage_configs = [
        (rev_cagr * 1.2, sustainable_g * 0.5, 5),
        (rev_cagr, sustainable_g * 0.5, 5),
        (rev_cagr * 0.8, sustainable_g * 0.5, 5),
        (sustainable_g * 1.5, sustainable_g * 0.3, 5),
    ]

    for g1, g2, years in stage_configs:
        g1 = max(-0.02, min(0.15, g1))
        g2 = max(0.0, min(0.04, g2))
        if cost_of_equity <= g2:
            g2 = cost_of_equity - 0.005
        if cost_of_equity <= g1:
            g1 = cost_of_equity - 0.01

        pv_divs = 0
        d = div_rate
        for y in range(1, years + 1):
            d *= (1 + g1)
            pv_divs += d / (1 + cost_of_equity) ** y

        terminal = d * (1 + g2) / (cost_of_equity - g2)
        pv_terminal = terminal / (1 + cost_of_equity) ** years
        intrinsic = pv_divs + pv_terminal
        lines.append(_table_row(
            f"{g1:.1%}", f"{g2:.1%}", years,
            f"${intrinsic:.2f}", _pct_signed(_upside(intrinsic, price)),
        ))

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Residual Income
# ---------------------------------------------------------------------------

def run_residual_income_analysis(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    beta = _safe_get(info, "beta", 1.0) or 1.0
    risk_free = _fetch_risk_free_rate()
    erp = 0.055
    cost_of_equity = risk_free + beta * erp

    book_value = _safe_get(info, "bookValue", 0) or 0
    roe = _safe_get(info, "returnOnEquity", 0) or 0
    total_equity = _safe_get(info, "totalStockholderEquity", 0) or 0

    historical = _fetch_historical(ticker_obj)
    net_income_list = historical.get("net_income", [])

    if total_equity <= 0:
        bv_ps = book_value
        total_equity = book_value * shares
    else:
        bv_ps = total_equity / shares if shares > 0 else 0

    earnings_ps = net_income_list[0] / shares if net_income_list and net_income_list[0] and shares > 0 else 0
    ri_ps = earnings_ps - cost_of_equity * bv_ps

    rev_cagr = _compute_cagr(historical.get("revenue", []))
    growth = rev_cagr if rev_cagr else 0.03

    persistence_factors = [1.0, 0.8, 0.5, 0.0]
    projection_years = 5

    lines = [
        _heading("Residual Income Valuation", ticker),
        "",
        _section("Inputs"),
        "",
        _kv("Book Value / Share", f"${bv_ps:.2f}"),
        _kv("ROE", f"{roe:.2%}" if roe else "N/A"),
        _kv("Cost of Equity", f"{cost_of_equity:.2%}"),
        _kv("Earnings / Share", f"${earnings_ps:.2f}" if earnings_ps else "N/A"),
        _kv("Residual Income / Share", f"${ri_ps:.2f}" if ri_ps else "N/A"),
        _kv("Revenue CAGR", f"{growth:.1%}"),
        "",
        _section("Forecast"),
        "",
        "| Year | BVPS | Earnings/Share | RI/Share | PV(RI) |",
        "|---|---|---|---|---|",
    ]

    for persist in persistence_factors:
        cum_pv_ri = 0
        bv = bv_ps
        for y in range(1, projection_years + 1):
            eps = roe * bv if roe else earnings_ps * (1 + growth) ** y
            ri = eps - cost_of_equity * bv
            pv_ri = ri / (1 + cost_of_equity) ** y
            cum_pv_ri += pv_ri
            if persist == persistence_factors[0]:
                lines.append(_table_row(
                    y, f"${bv:.2f}", f"${eps:.2f}",
                    f"${ri:.2f}", f"${pv_ri:.2f}",
                ))
            bv += ri * persist

        terminal_ri = ri * persist if persist > 0 else 0
        if persist > 0 and cost_of_equity > 0:
            pv_terminal = terminal_ri / cost_of_equity / (1 + cost_of_equity) ** projection_years
        else:
            pv_terminal = 0
        total_value = bv_ps + cum_pv_ri + pv_terminal
        upside = _upside(total_value, price)
        if persist == persistence_factors[0]:
            lines += [
                "",
                f"| Terminal RI | | | {terminal_ri:.2f} | {pv_terminal:.2f} |",
                "",
            ]

        lines.append("")
        if persist == persistence_factors[0]:
            lines.append(_section("Valuation by Persistence Factor"))
            lines.append("")

    lines += [
        "| Persistence | Total RI Value | PV Terminal | Total Value/Share | Upside |",
        "|---|---|---|---|---|",
    ]

    for persist in persistence_factors:
        cum_pv_ri = 0
        bv = bv_ps
        for y in range(1, projection_years + 1):
            eps = roe * bv if roe else earnings_ps * (1 + growth) ** y
            ri = eps - cost_of_equity * bv
            cum_pv_ri += ri / (1 + cost_of_equity) ** y
            bv += ri * persist

        terminal_ri = ri * persist if persist > 0 else 0
        pv_terminal = (terminal_ri / cost_of_equity / (1 + cost_of_equity) ** projection_years) if persist > 0 and cost_of_equity > 0 else 0
        total_value = bv_ps + cum_pv_ri + pv_terminal
        lines.append(_table_row(
            f"{persist:.0%}", f"${cum_pv_ri:.2f}", f"${pv_terminal:.2f}",
            f"${total_value:.2f}", _pct_signed(_upside(total_value, price)),
        ))

    lines += [
        "",
        "> **Persistence factor** controls how quickly residual income decays toward zero. "
        "100% = economic moat (competitive advantages persist); 0% = no moat.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. LBO Analysis
# ---------------------------------------------------------------------------

def run_lbo_analysis(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    historical = _fetch_historical(ticker_obj)
    net_debt = _net_debt(historical)

    ev = _safe_get(info, "enterpriseValue", 0) or 0
    ebitda = _safe_get(info, "ebitda", 0) or 0
    total_revenue = _safe_get(info, "totalRevenue", 0) or 0
    ebitda_margin = _safe_get(info, "ebitdaMargins", 0) or (ebitda / total_revenue if total_revenue > 0 else 0.2)
    operating_cf = historical.get("operating_cf", [])
    capex_list = historical.get("capex", [])

    rev_cagr = _compute_cagr(historical.get("revenue", []))
    growth = rev_cagr if rev_cagr else 0.05

    entry_multiples = [6.0, 7.0, 8.0]
    exit_multiples = [6.0, 7.0, 8.0]
    hold_years = 5
    interest_rate = 0.065

    lines = [
        _heading("LBO Analysis", ticker),
        "",
        _section("Current Metrics"),
        "",
        _kv("Current Price", f"${price:.2f}"),
        _kv("Enterprise Value", _fmt(ev)),
        _kv("EBITDA (TTM)", _fmt(ebitda)),
        _kv("EV/EBITDA", f"{ev / ebitda:.1f}x" if ebitda > 0 else "N/A"),
        _kv("EBITDA Margin", f"{ebitda_margin:.1%}"),
        _kv("Net Debt", _fmt(net_debt)),
        _kv("Revenue CAGR", f"{growth:.1%}"),
        _kv("Assumed Interest Rate", f"{interest_rate:.1%}"),
        "",
        _section("LBO Scenarios"),
        "",
        "| Entry EV/EBITDA | Equity Check ($B) | Exit Multiple | Exit EV ($B) | IRR | MOIC |",
        "|---|---|---|---|---|---|",
    ]

    equity_checks = [0.40, 0.50, 0.60]
    for entry_mult in entry_multiples:
        for eq_pct in equity_checks:
            entry_ev = ebitda * entry_mult
            entry_eq = entry_ev * eq_pct
            entry_debt = entry_ev * (1 - eq_pct)

            proj_ebitda = ebitda
            debt = entry_debt
            for _ in range(hold_years):
                proj_ebitda *= (1 + growth)
                fcft = proj_ebitda * 0.6 - interest_rate * debt * 0.5
                debt = max(0, debt - max(0, fcft * 0.6))

            exit_evs = []
            for exit_mult in [entry_mult]:
                exit_ev = proj_ebitda * exit_mult
                exit_eq = exit_ev - debt
                if entry_eq > 0:
                    irr = (exit_eq / entry_eq) ** (1.0 / hold_years) - 1
                    moic = exit_eq / entry_eq
                    exit_evs.append((exit_ev, exit_eq, irr, moic))

            if exit_evs:
                ex_ev, ex_eq, irr, moic = exit_evs[0]
                lines.append(_table_row(
                    f"{entry_mult:.1f}x", f"${entry_eq / 1e9:.2f}",
                    f"{entry_mult:.1f}x", f"${ex_ev / 1e9:.2f}",
                    f"{irr:.1%}", f"{moic:.1f}x",
                ))

    lines += [
        "",
        _section("Assumptions"),
        "",
        f"- Hold period: {hold_years} years",
        f"- EBITDA growth: {growth:.1%} per year",
        f"- FCF conversion: ~60% of EBITDA",
        f"- Debt paydown: ~60% of excess FCF",
        f"- Interest rate: {interest_rate:.1%}",
        "- Exit at same multiple as entry (conservative)",
        "",
        "> **Note:** This is a simplified LBO model. A full LBO would include detailed debt scheduling, "
        "management fees, transaction costs, and more granular operational projections.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. VC Method
# ---------------------------------------------------------------------------

def run_vc_valuation(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    historical = _fetch_historical(ticker_obj)

    total_revenue = _safe_get(info, "totalRevenue", 0) or 0
    ebitda = _safe_get(info, "ebitda", 0) or 0
    net_income = _safe_get(info, "netIncomeToCommon", 0) or 0
    ev = _safe_get(info, "enterpriseValue", 0) or 0
    rev_growth = _safe_get(info, "revenueGrowth", 0) or 0

    rev_cagr = _compute_cagr(historical.get("revenue", []))
    if rev_cagr is None:
        rev_cagr = rev_growth if rev_growth else 0.10

    target_returns = [0.25, 0.30, 0.40]
    exit_years = [3, 5, 7]
    exit_multiples = [8.0, 10.0, 12.0, 15.0]

    lines = [
        _heading("VC Method Valuation", ticker),
        "",
        "> The VC method projects a future exit value and discounts it back at a "
        "target rate of return. Useful for high-growth companies where traditional "
        "valuation metrics may not apply.",
        "",
        _section("Current Metrics"),
        "",
        _kv("Revenue (TTM)", _fmt(total_revenue)),
        _kv("EBITDA (TTM)", _fmt(ebitda)),
        _kv("Net Income (TTM)", _fmt(net_income)),
        _kv("Revenue Growth", f"{rev_growth:.1%}" if rev_growth else "N/A"),
        _kv("Revenue CAGR (Historical)", f"{rev_cagr:.1%}"),
        _kv("Current EV/Revenue", f"{ev / total_revenue:.1f}x" if total_revenue > 0 else "N/A"),
        "",
        _section("Exit Valuation Scenarios (Revenue-Based)"),
        "",
        "| Target Return | Exit Year | Exit Multiple | Proj. Revenue | Exit Value | FV/Share | Upside |",
        "|---|---|---|---|---|---|---|",
    ]

    for target_ret in target_returns:
        for yr in exit_years:
            for exit_mult in exit_multiples:
                proj_rev = total_revenue * (1 + rev_cagr) ** yr
                proj_ev = proj_rev * exit_mult
                fv = proj_ev / (1 + target_ret) ** yr
                fv_ps = fv / shares if shares > 0 else 0
                lines.append(_table_row(
                    f"{target_ret:.0%}", yr, f"{exit_mult:.0f}x",
                    _fmt(proj_rev), _fmt(proj_ev),
                    f"${fv_ps:.2f}", _pct_signed(_upside(fv_ps, price)),
                ))

    lines += [
        "",
        _section("Exit Valuation Scenarios (EBITDA-Based)"),
        "",
        "| Target Return | Exit Year | Exit Multiple | Proj. EBITDA | Exit Value | FV/Share | Upside |",
        "|---|---|---|---|---|---|---|",
    ]

    if ebitda > 0:
        ebitda_growth = rev_cagr * 1.2
        ebitda_exit_mults = [12.0, 15.0, 18.0]
        for target_ret in target_returns[:2]:
            for yr in exit_years[:2]:
                for exit_mult in ebitda_exit_mults:
                    proj_ebitda = ebitda * (1 + ebitda_growth) ** yr
                    proj_ev = proj_ebitda * exit_mult
                    fv = proj_ev / (1 + target_ret) ** yr
                    fv_ps = fv / shares if shares > 0 else 0
                    lines.append(_table_row(
                        f"{target_ret:.0%}", yr, f"{exit_mult:.0f}x",
                        _fmt(proj_ebitda), _fmt(proj_ev),
                        f"${fv_ps:.2f}", _pct_signed(_upside(fv_ps, price)),
                    ))
    else:
        lines.append("| — | — | — | No EBITDA data available | — | — | — |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Earnings Power Value (EPV)
# ---------------------------------------------------------------------------

def run_epv_analysis(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    historical = _fetch_historical(ticker_obj)
    net_debt = _net_debt(historical)

    total_revenue = _safe_get(info, "totalRevenue", 0) or 0
    ebitda = _safe_get(info, "ebitda", 0) or 0
    operating_margin = _safe_get(info, "operatingMargins", 0) or 0
    total_assets = _safe_get(info, "totalAssets", 0) or 0

    risk_free = _fetch_risk_free_rate()
    erp = 0.055
    beta = _safe_get(info, "beta", 1.0) or 1.0
    wacc, coe, cod, tax_rate, we, wd = _estimate_wacc(info, historical, risk_free, erp, 0.0)

    op_inc_list = historical.get("operating_cf", [])
    revenue_list = historical.get("revenue", [])

    normalized_ebitda = ebitda
    if op_inc_list and len(op_inc_list) >= 2:
        avg_op_cf = sum(v for v in op_inc_list if v) / len([v for v in op_inc_list if v])
        if avg_op_cf > 0:
            normalized_ebitda = avg_op_cf * 1.1

    normalized_ebit = normalized_ebitda * (1 - tax_rate)
    excess_cash = max(0, -net_debt)

    epv = normalized_ebit / wacc if wacc > 0 else 0
    epv_adjusted = epv - net_debt + excess_cash
    epv_ps = epv / shares if shares > 0 else 0
    epv_adj_ps = epv_adjusted / shares if shares > 0 else 0

    lines = [
        _heading("Earnings Power Value (EPV)", ticker),
        "",
        _section("Inputs"),
        "",
        _kv("WACC", f"{wacc:.2%}"),
        _kv("Cost of Equity", f"{coe:.2%}"),
        _kv("Cost of Debt", f"{cod:.2%}"),
        _kv("Tax Rate", f"{tax_rate:.2%}"),
        _kv("EBITDA (TTM)", _fmt(ebitda)),
        _kv("Normalized EBITDA", _fmt(normalized_ebitda)),
        _kv("Normalized EBIT (After Tax)", _fmt(normalized_ebit)),
        _kv("Net Debt", _fmt(net_debt)),
        _kv("Excess Cash", _fmt(excess_cash)),
        "",
        _section("EPV Calculation"),
        "",
        _kv("EPV = Normalized EBIT / WACC", _fmt(epv)),
        _kv("EPV - Net Debt + Excess Cash", _fmt(epv_adjusted)),
        "",
        _section("Per-Share Values"),
        "",
        "| Metric | Value/Share | Upside / Downside |",
        "|---|---|---|",
        _table_row("EPV / Share", f"${epv_ps:.2f}", _pct_signed(_upside(epv_ps, price))),
        _table_row("Adjusted EPV / Share", f"${epv_adj_ps:.2f}", _pct_signed(_upside(epv_adj_ps, price))),
        _table_row("Current Price", f"${price:.2f}", "—"),
        "",
        _section("Sensitivity to WACC"),
        "",
        "| WACC | EPV/Share | Upside |",
        "|---|---|---|",
    ]

    for wacc_delta in [-0.02, -0.01, 0, 0.01, 0.02]:
        adj_wacc = wacc + wacc_delta
        if adj_wacc <= 0:
            continue
        sens_epv = normalized_ebit / adj_wacc
        sens_adj = sens_epv - net_debt + excess_cash
        sens_ps = sens_adj / shares if shares > 0 else 0
        lines.append(_table_row(
            f"{adj_wacc:.2%}", f"${sens_ps:.2f}", _pct_signed(_upside(sens_ps, price)),
        ))

    lines += [
        "",
        "> **EPV** assumes the company's current level of profitability is sustainable indefinitely "
        "(no growth). It is most useful for value investing when a company trades below its "
        "earnings power. Compare to the current price to assess margin of safety.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Sum-of-the-Parts (SOTP)
# ---------------------------------------------------------------------------

def run_sotp_valuation(ticker: str, curr_date: str | None = None) -> str:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    price = _current_price(info)
    shares = _shares(info)
    historical = _fetch_historical(ticker_obj)
    net_debt = _net_debt(historical)

    total_revenue = _safe_get(info, "totalRevenue", 0) or 0
    ebitda = _safe_get(info, "ebitda", 0) or 0
    net_income = _safe_get(info, "netIncomeToCommon", 0) or 0
    book_value = _safe_get(info, "bookValue", 0) or 0
    ev = _safe_get(info, "enterpriseValue", 0) or 0
    total_assets = _safe_get(info, "totalAssets", 0) or 0

    operating_margin = _safe_get(info, "operatingMargins", 0) or 0
    ebitda_margin = ebitda / total_revenue if total_revenue > 0 else 0.2

    segments = []

    segments.append({
        "name": "Core Operations",
        "description": "Primary revenue-generating business",
        "revenue": total_revenue * 0.85,
        "ebitda": ebitda * 0.85,
        "method": "EV/EBITDA",
        "multiple": _safe_get(info, "enterpriseToEbitda") or 8.0,
    })

    segments.append({
        "name": "Other / Ancillary",
        "description": "Non-core revenue streams, licensing, fees",
        "revenue": total_revenue * 0.15,
        "ebitda": ebitda * 0.15,
        "method": "EV/Revenue",
        "multiple": _safe_get(info, "enterpriseToRevenue") or 2.0,
    })

    holdings = _safe_get(info, "heldToMaturitySecurities", 0) or 0
    investments = _safe_get(info, "totalInvestments", 0) or 0
    other_assets = _safe_get(info, "otherAssets", 0) or 0
    excess_assets = max(0, holdings + investments + other_assets - total_revenue * 0.1)

    if excess_assets > total_revenue * 0.05:
        segments.append({
            "name": "Financial Assets / Investments",
            "description": "Non-operating financial assets",
            "revenue": 0,
            "ebitda": 0,
            "method": "Book Value",
            "multiple": 1.0,
            "book_value": excess_assets,
        })

    lines = [
        _heading("Sum-of-the-Parts (SOTP) Valuation", ticker),
        "",
        "> **Note:** yfinance does not provide detailed segment revenue breakdowns. "
        "This analysis decomposes the company into estimated business components based on "
        "available financial data. A full SOTP would use management-reported segment data.",
        "",
        _section("Company Overview"),
        "",
        _kv("Total Revenue", _fmt(total_revenue)),
        _kv("Total EBITDA", _fmt(ebitda)),
        _kv("Enterprise Value", _fmt(ev)),
        _kv("Net Debt", _fmt(net_debt)),
        _kv("Total Assets", _fmt(total_assets)),
        _kv("Book Value", _fmt(book_value)),
        "",
        _section("Segment Valuations"),
        "",
        "| Segment | Revenue | EBITDA | Method | Multiple | Implied EV |",
        "|---|---|---|---|---|---|",
    ]

    total_implied_ev = 0
    for seg in segments:
        if seg.get("book_value"):
            seg_ev = seg["book_value"]
        else:
            if seg["method"] == "EV/EBITDA" and seg["ebitda"] > 0:
                seg_ev = seg["ebitda"] * seg["multiple"]
            elif seg["method"] == "EV/Revenue" and seg["revenue"] > 0:
                seg_ev = seg["revenue"] * seg["multiple"]
            else:
                seg_ev = 0

        total_implied_ev += seg_ev
        rev_str = _fmt(seg["revenue"]) if seg["revenue"] else "—"
        ebitda_str = _fmt(seg["ebitda"]) if seg["ebitda"] else "—"
        lines.append(_table_row(
            seg["name"], rev_str, ebitda_str,
            seg["method"], f"{seg['multiple']:.1f}x", _fmt(seg_ev),
        ))

    total_equity = total_implied_ev - net_debt
    fv_ps = total_equity / shares if shares > 0 else 0

    lines += [
        "",
        _section("Consolidated SOTP"),
        "",
        _kv("Total Implied Enterprise Value", _fmt(total_implied_ev)),
        _kv("Less: Net Debt", _fmt(net_debt)),
        _kv("Implied Equity Value", _fmt(total_equity)),
        _kv("Implied FV / Share", f"${fv_ps:.2f}"),
        _kv("Current Price", f"${price:.2f}"),
        _kv("Upside / Downside", _pct_signed(_upside(fv_ps, price))),
        "",
    ]

    multiple_scenarios = [
        ("Conservative", 0.8),
        ("Base", 1.0),
        ("Optimistic", 1.2),
    ]

    lines += [
        _section("Sensitivity to Multiples"),
        "",
        "| Scenario | Multiple Adj. | Implied FV/Share | Upside |",
        "|---|---|---|---|",
    ]

    for label, adj in multiple_scenarios:
        total_adj_ev = 0
        for seg in segments:
            adj_mult = seg["multiple"] * adj
            if seg.get("book_value"):
                total_adj_ev += seg["book_value"] * adj
            elif seg["method"] == "EV/EBITDA" and seg["ebitda"] > 0:
                total_adj_ev += seg["ebitda"] * adj_mult
            elif seg["method"] == "EV/Revenue" and seg["revenue"] > 0:
                total_adj_ev += seg["revenue"] * adj_mult

        adj_equity = total_adj_ev - net_debt
        adj_fvps = adj_equity / shares if shares > 0 else 0
        lines.append(_table_row(
            label, f"{adj:.0%}", f"${adj_fvps:.2f}",
            _pct_signed(_upside(adj_fvps, price)),
        ))

    lines.append("")
    return "\n".join(lines)
