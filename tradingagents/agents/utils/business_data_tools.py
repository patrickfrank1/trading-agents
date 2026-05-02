from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.sec_edgar import get_10k_filing_data


@tool
def get_company_profile(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve a detailed company profile including business description,
    sector, industry, number of employees, country, website, and key business
    characteristics. Use this to understand the company's business model.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A formatted report containing detailed company profile data
    """
    return route_to_vendor("get_company_profile", ticker)


@tool
def get_sector_performance(
    sector: Annotated[str, "sector name to compare against (e.g. Technology, Healthcare)"],
    period: Annotated[str, "time period: '1mo', '3mo', '6mo', '1y', 'ytd'"] = "1y",
) -> str:
    """Retrieve sector performance data including an ETF benchmark, individual
    stock performance within the sector, and relative strength metrics. Use this
    to compare company performance against its sector peers.

    Args:
        sector (str): Sector name (e.g. Technology, Healthcare)
        period (str): Time period for performance comparison (default '1y')
    Returns:
        str: A formatted report containing sector performance data
    """
    return route_to_vendor("get_sector_performance", sector, period)


@tool
def get_peer_comparison(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve key financial metrics for the company's peer group for comparison.
    Includes market cap, P/E, PEG, profit margins, ROE, revenue growth, and other
    metrics. Use this to evaluate competitive positioning within the industry.

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A formatted report with peer comparison data
    """
    return route_to_vendor("get_peer_comparison", ticker)


@tool
def get_10k_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve the most recent 10-K (or 20-F for foreign companies) annual report
    filing from SEC EDGAR. Returns key sections: Business (Item 1), Risk Factors
    (Item 1A), Selected Financial Data (Item 6), Management's Discussion & Analysis
    (Item 7), and Financial Statements (Item 8). Use this to understand the
    company's detailed business model, competitive positioning, risks, and
    management's view of the business directly from regulatory filings.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Key sections from the latest 10-K/20-F annual filing
    """
    return get_10k_filing_data(ticker, curr_date)
