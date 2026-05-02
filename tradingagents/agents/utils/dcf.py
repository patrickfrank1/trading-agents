"""DCF (Discounted Cash Flow) analysis engine for company valuation.

Computes three-scenario (pessimistic / base / optimistic) DCF valuations
using FCF-to-firm with WACC discounting and Gordon Growth terminal value.
"""

from dataclasses import dataclass
from datetime import datetime
import math

import yfinance as yf


@dataclass
class DCFScenarioResult:
    scenario_name: str
    revenue_growth_rate: float
    terminal_growth_rate: float
    beta: float
    equity_risk_premium: float
    wacc: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    weight_equity: float
    weight_debt: float
    base_fcf: float
    projected_fcfs: list
    terminal_value: float
    pv_fcfs: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    fair_value_per_share: float
    current_price: float
    upside_pct: float
    risk_free_rate: float
    projection_years: int


def _safe_get(info, key, default=None):
    val = info.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def _fetch_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        data = tnx.history(period="5d")
        if data is not None and not data.empty:
            return float(data["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.045


def _find_row(df, candidates):
    for candidate in candidates:
        matching = [idx for idx in df.index if candidate.lower() in str(idx).lower()]
        if matching:
            return df.loc[matching[0]]
    return None


def _get_historical_annual(ticker_obj):
    result = {
        "operating_cf": [],
        "capex": [],
        "revenue": [],
        "net_income": [],
        "interest_expense": [],
        "tax_provision": [],
        "pretax_income": [],
        "total_debt": [],
        "cash": [],
    }

    try:
        cf = ticker_obj.cashflow
        if cf is not None and not cf.empty:
            row = _find_row(cf, ["Operating Cash Flow", "Cash From Operations", "Operating Activities"])
            if row is not None:
                result["operating_cf"] = _to_clean_list(row.values.tolist())

            row = _find_row(cf, ["Capital Expenditure", "Purchase Of Property Plant Equipment"])
            if row is not None:
                result["capex"] = [abs(v) for v in _to_clean_list(row.values.tolist())]
    except Exception:
        pass

    try:
        inc = ticker_obj.income_stmt
        if inc is not None and not inc.empty:
            for key, candidates in [
                ("revenue", ["Total Revenue", "Revenue"]),
                ("net_income", ["Net Income", "Net Income Common Stockholders"]),
                ("interest_expense", ["Interest Expense"]),
                ("tax_provision", ["Tax Provision", "Income Tax Expense"]),
                ("pretax_income", ["Pretax Income", "Income Before Tax"]),
            ]:
                row = _find_row(inc, candidates)
                if row is not None:
                    result[key] = _to_clean_list(row.values.tolist())
    except Exception:
        pass

    try:
        bs = ticker_obj.balance_sheet
        if bs is not None and not bs.empty:
            ltd = _find_row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligations"])
            std = _find_row(bs, ["Short Term Debt", "Short Term Debt And Capital Lease Obligations", "Current Debt"])
            cash_row = _find_row(bs, ["Cash", "Cash And Cash Equivalents", "Cash And Short Term Investments"])

            if ltd is not None and std is not None:
                result["total_debt"] = [a + b for a, b in zip(
                    _to_clean_list(ltd.values.tolist()),
                    _to_clean_list(std.values.tolist()),
                )]
            elif ltd is not None:
                result["total_debt"] = _to_clean_list(ltd.values.tolist())

            if cash_row is not None:
                result["cash"] = _to_clean_list(cash_row.values.tolist())
    except Exception:
        pass

    return result


def _to_clean_list(values):
    cleaned = []
    for v in values:
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            cleaned.append(float(v))
    return cleaned


def _compute_fcf_series(historical):
    ocf = historical["operating_cf"]
    capex = historical["capex"]
    fcfs = []
    for o, c in zip(ocf, capex):
        fcfs.append(o - c)
    return fcfs


def _compute_cagr(values):
    if len(values) < 2:
        return None
    values = [v for v in values if v > 0]
    if len(values) < 2:
        return None
    n_years = len(values) - 1
    return (values[0] / values[-1]) ** (1.0 / n_years) - 1.0


def _estimate_tax_rate(historical):
    taxes = historical.get("tax_provision", [])
    pretax = historical.get("pretax_income", [])
    rates = []
    for t, p in zip(taxes, pretax):
        if t is not None and p is not None and p > 0:
            rates.append(t / p)
    if rates:
        return sum(rates) / len(rates)
    return 0.21


def _estimate_cost_of_debt(historical):
    interest = historical.get("interest_expense", [])
    debt = historical.get("total_debt", [])
    costs = []
    for exp, d in zip(interest, debt):
        if exp is not None and d is not None and d > 0:
            costs.append(exp / d)
    if costs:
        return sum(costs) / len(costs)
    return 0.06


def _latest(values):
    for v in values:
        if v is not None:
            return v
    return 0.0


def _estimate_wacc(info, historical, risk_free_rate, equity_risk_premium, beta_adjustment):
    beta = _safe_get(info, "beta", 1.0) or 1.0
    beta = max(0.3, beta + beta_adjustment)

    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    market_cap = _safe_get(info, "marketCap", 0) or 0
    total_debt = _latest(historical.get("total_debt", []))
    total_capital = market_cap + total_debt

    if total_capital <= 0:
        return 0.10, cost_of_equity, 0.06, 0.21, 1.0, 0.0

    weight_equity = market_cap / total_capital
    weight_debt = total_debt / total_capital
    cost_of_debt = _estimate_cost_of_debt(historical)
    tax_rate = _estimate_tax_rate(historical)

    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)
    return wacc, cost_of_equity, cost_of_debt, tax_rate, weight_equity, weight_debt


def _fmt(value):
    if value is None or math.isnan(value):
        return "N/A"
    av = abs(value)
    if av >= 1e12:
        return f"${value / 1e12:.2f}T"
    if av >= 1e9:
        return f"${value / 1e9:.2f}B"
    if av >= 1e6:
        return f"${value / 1e6:.2f}M"
    if av >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:.2f}"


def _pct(value):
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def _pct_signed(value):
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def compute_single_dcf(
    ticker,
    scenario_name,
    revenue_growth_rate,
    terminal_growth_rate,
    beta_adjustment=0.0,
    equity_risk_premium=0.055,
    projection_years=5,
):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    historical = _get_historical_annual(ticker_obj)
    risk_free_rate = _fetch_risk_free_rate()

    hist_fcfs = _compute_fcf_series(historical)
    if hist_fcfs and hist_fcfs[0] > 0:
        base_fcf = hist_fcfs[0]
    else:
        base_fcf = _safe_get(info, "freeCashflow", 0) or 0

    if base_fcf <= 0:
        revenue = _safe_get(info, "totalRevenue", 0) or 0
        margin = _safe_get(info, "operatingMargins", 0.15) or 0.15
        base_fcf = revenue * margin * 0.6

    wacc, coe, cod, tax_rate, w_e, w_d = _estimate_wacc(
        info, historical, risk_free_rate, equity_risk_premium, beta_adjustment,
    )

    projected = [base_fcf * (1 + revenue_growth_rate) ** y for y in range(1, projection_years + 1)]

    if wacc <= terminal_growth_rate:
        tv = 0.0
    else:
        tv = projected[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

    pv_fcfs = sum(f / (1 + wacc) ** t for t, f in enumerate(projected, 1))
    pv_tv = tv / (1 + wacc) ** projection_years

    ev = pv_fcfs + pv_tv
    net_debt = _latest(historical.get("total_debt", [])) - _latest(historical.get("cash", []))
    equity = ev - net_debt

    shares = _safe_get(info, "sharesOutstanding", 0) or 0
    fvps = equity / shares if shares > 0 else 0

    current_price = (
        _safe_get(info, "currentPrice")
        or _safe_get(info, "regularMarketPrice")
        or _safe_get(info, "previousClose")
        or 0
    )
    upside = ((fvps / current_price) - 1) * 100 if current_price > 0 else 0

    beta = _safe_get(info, "beta", 1.0) or 1.0

    return DCFScenarioResult(
        scenario_name=scenario_name,
        revenue_growth_rate=revenue_growth_rate,
        terminal_growth_rate=terminal_growth_rate,
        beta=round(beta + beta_adjustment, 2),
        equity_risk_premium=equity_risk_premium,
        wacc=wacc,
        cost_of_equity=coe,
        cost_of_debt=cod,
        tax_rate=tax_rate,
        weight_equity=w_e,
        weight_debt=w_d,
        base_fcf=base_fcf,
        projected_fcfs=projected,
        terminal_value=tv,
        pv_fcfs=pv_fcfs,
        pv_terminal=pv_tv,
        enterprise_value=ev,
        net_debt=net_debt,
        equity_value=equity,
        shares_outstanding=shares,
        fair_value_per_share=fvps,
        current_price=current_price,
        upside_pct=upside,
        risk_free_rate=risk_free_rate,
        projection_years=projection_years,
    )


def run_three_scenario_dcf(ticker, curr_date=None):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    historical = _get_historical_annual(ticker_obj)

    rev_growth = _compute_cagr(historical.get("revenue", []))
    if rev_growth is None:
        rev_growth = 0.05
    rev_growth = max(-0.10, min(0.30, rev_growth))

    base = compute_single_dcf(
        ticker, "Base Case",
        revenue_growth_rate=round(rev_growth, 4),
        terminal_growth_rate=0.025,
        beta_adjustment=0.0,
        equity_risk_premium=0.055,
    )

    pessimistic = compute_single_dcf(
        ticker, "Pessimistic Case",
        revenue_growth_rate=round(max(-0.10, rev_growth - 0.05), 4),
        terminal_growth_rate=0.015,
        beta_adjustment=0.2,
        equity_risk_premium=0.07,
    )

    optimistic = compute_single_dcf(
        ticker, "Optimistic Case",
        revenue_growth_rate=round(min(0.30, rev_growth + 0.05), 4),
        terminal_growth_rate=0.035,
        beta_adjustment=-0.2,
        equity_risk_premium=0.04,
    )

    hist_fcfs = _compute_fcf_series(historical)

    return _format_report(ticker, info, base, pessimistic, optimistic, hist_fcfs)


def _format_report(ticker, info, base, pessimistic, optimistic, hist_fcfs):
    name = _safe_get(info, "longName", ticker)
    lines = [
        f"## Discounted Cash Flow (DCF) Analysis: {name} ({ticker.upper()})",
        "",
        "### Key Assumptions",
        "",
        "| Parameter | Pessimistic | Base | Optimistic |",
        "|---|---|---|---|",
        f"| Revenue Growth (5y) | {pessimistic.revenue_growth_rate:.1%} | {base.revenue_growth_rate:.1%} | {optimistic.revenue_growth_rate:.1%} |",
        f"| Terminal Growth (g) | {pessimistic.terminal_growth_rate:.1%} | {base.terminal_growth_rate:.1%} | {optimistic.terminal_growth_rate:.1%} |",
        f"| Beta | {pessimistic.beta:.2f} | {base.beta:.2f} | {optimistic.beta:.2f} |",
        f"| Equity Risk Premium | {pessimistic.equity_risk_premium:.1%} | {base.equity_risk_premium:.1%} | {optimistic.equity_risk_premium:.1%} |",
        f"| WACC | {pessimistic.wacc:.1%} | {base.wacc:.1%} | {optimistic.wacc:.1%} |",
        f"| Cost of Equity | {pessimistic.cost_of_equity:.1%} | {base.cost_of_equity:.1%} | {optimistic.cost_of_equity:.1%} |",
        f"| Cost of Debt | {pessimistic.cost_of_debt:.1%} | {base.cost_of_debt:.1%} | {optimistic.cost_of_debt:.1%} |",
        f"| Tax Rate | {pessimistic.tax_rate:.1%} | {base.tax_rate:.1%} | {optimistic.tax_rate:.1%} |",
        f"| Risk-Free Rate | {base.risk_free_rate:.2%} | | |",
        f"| Base FCF | {_fmt(pessimistic.base_fcf)} | {_fmt(base.base_fcf)} | {_fmt(optimistic.base_fcf)} |",
        "",
        "### Projected Free Cash Flows (5-Year)",
        "",
        "| Year | Pessimistic | Base | Optimistic |",
        "|---|---|---|---|",
    ]
    for i, (p, b, o) in enumerate(zip(pessimistic.projected_fcfs, base.projected_fcfs, optimistic.projected_fcfs), 1):
        lines.append(f"| {i} | {_fmt(p)} | {_fmt(b)} | {_fmt(o)} |")

    lines += [
        "",
        "### Valuation Bridge",
        "",
        "| Metric | Pessimistic | Base | Optimistic |",
        "|---|---|---|---|",
        f"| Terminal Value | {_fmt(pessimistic.terminal_value)} | {_fmt(base.terminal_value)} | {_fmt(optimistic.terminal_value)} |",
        f"| PV of Projected FCFs | {_fmt(pessimistic.pv_fcfs)} | {_fmt(base.pv_fcfs)} | {_fmt(optimistic.pv_fcfs)} |",
        f"| PV of Terminal Value | {_fmt(pessimistic.pv_terminal)} | {_fmt(base.pv_terminal)} | {_fmt(optimistic.pv_terminal)} |",
        f"| Enterprise Value | {_fmt(pessimistic.enterprise_value)} | {_fmt(base.enterprise_value)} | {_fmt(optimistic.enterprise_value)} |",
        f"| Net Debt | {_fmt(pessimistic.net_debt)} | {_fmt(base.net_debt)} | {_fmt(optimistic.net_debt)} |",
        f"| Equity Value | {_fmt(pessimistic.equity_value)} | {_fmt(base.equity_value)} | {_fmt(optimistic.equity_value)} |",
        "",
        "### Fair Value vs. Current Price",
        "",
        "| Scenario | Fair Value/Share | Current Price | Upside / Downside |",
        "|---|---|---|---|",
        f"| {pessimistic.scenario_name} | ${pessimistic.fair_value_per_share:.2f} | ${base.current_price:.2f} | {_pct_signed(pessimistic.upside_pct)} |",
        f"| **{base.scenario_name}** | **${base.fair_value_per_share:.2f}** | **${base.current_price:.2f}** | **{_pct_signed(base.upside_pct)}** |",
        f"| {optimistic.scenario_name} | ${optimistic.fair_value_per_share:.2f} | ${base.current_price:.2f} | {_pct_signed(optimistic.upside_pct)} |",
        "",
    ]

    if hist_fcfs:
        lines += ["### Historical Free Cash Flow (Annual)", ""]
        for i, fcf in enumerate(hist_fcfs):
            year_label = len(hist_fcfs) - i
            lines.append(f"- Year {year_label}: {_fmt(fcf)}")
        lines.append("")

    return "\n".join(lines)
