from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.agents.utils.dcf import run_three_scenario_dcf
from tradingagents.agents.utils.valuation import (
    run_comps_analysis,
    run_precedent_transactions_analysis,
    run_asset_based_valuation,
    run_ddm_analysis,
    run_residual_income_analysis,
    run_lbo_analysis,
    run_vc_valuation,
    run_epv_analysis,
    run_sotp_valuation,
)


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)


@tool
def compute_dcf_analysis(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Compute a 3-scenario Discounted Cash Flow (DCF) valuation analysis.
    Returns pessimistic, base, and optimistic fair-value estimates with a
    full valuation bridge (WACC, projected FCFs, terminal value, equity value per share).
    This is a deterministic calculation — the LLM should interpret the results
    in its narrative report rather than re-computing them.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted DCF analysis with three scenarios
    """
    return run_three_scenario_dcf(ticker, curr_date)


@tool
def compute_comps_analysis(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Comparable Company Analysis using current market multiples.
    Shows P/E, P/B, P/S, EV/EBITDA, EV/Revenue multiples and derives
    implied fair value per share from each.
    Best for: Public companies with good multiples data.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted comps analysis
    """
    return run_comps_analysis(ticker, curr_date)


@tool
def compute_precedent_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Precedent Transaction valuation applying typical acquisition control premiums
    (20%, 30%, 40%) to current trading multiples to estimate what a strategic
    acquirer might pay.
    Best for: Companies in active M&A sectors or potential acquisition targets.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted precedent transaction analysis
    """
    return run_precedent_transactions_analysis(ticker, curr_date)


@tool
def compute_asset_based_valuation(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Asset-based valuation using Net Asset Value (NAV), Tangible Book Value,
    and liquidation scenarios with various discount factors.
    Best for: Asset-heavy firms (banks, REITs, holding companies).

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted asset-based valuation
    """
    return run_asset_based_valuation(ticker, curr_date)


@tool
def compute_ddm_valuation(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Dividend Discount Model (DDM) valuation using the Gordon Growth Model
    and multi-stage DDM scenarios. Requires the company to pay dividends.
    Best for: Mature, stable dividend-paying stocks.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted DDM analysis
    """
    return run_ddm_analysis(ticker, curr_date)


@tool
def compute_residual_income_valuation(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Residual Income valuation: Book Value + Present Value of future residual income.
    Includes sensitivity to persistence factor (economic moat strength).
    Best for: Banks, financials, and companies where book value is meaningful.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted residual income valuation
    """
    return run_residual_income_analysis(ticker, curr_date)


@tool
def compute_lbo_analysis(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Leveraged Buyout (LBO) analysis modeling entry at various EV/EBITDA multiples,
    debt paydown over a 5-year hold period, and exit IRR/MOIC calculations.
    Best for: Companies with stable, predictable cash flows suitable for buyouts.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted LBO analysis
    """
    return run_lbo_analysis(ticker, curr_date)


@tool
def compute_vc_valuation(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    VC Method valuation: projects future exit value (revenue-based or EBITDA-based)
    and discounts back at target return rates (25%, 30%, 40%).
    Best for: High-growth companies, startups, or pre-profit firms.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted VC method valuation
    """
    return run_vc_valuation(ticker, curr_date)


@tool
def compute_epv_valuation(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Earnings Power Value (EPV): Normalized EBIT / WACC, adjusted for net debt
    and excess cash. Assumes current profitability is sustainable with no growth.
    Best for: Value investing — identifies margin of safety when price < EPV.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted EPV analysis
    """
    return run_epv_analysis(ticker, curr_date)


@tool
def compute_sotp_valuation(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Sum-of-the-Parts (SOTP) valuation: decomposes the company into business
    segments and values each independently, then sums to derive total equity value.
    Best for: Conglomerates and multi-segment companies.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Markdown-formatted SOTP analysis
    """
    return run_sotp_valuation(ticker, curr_date)
