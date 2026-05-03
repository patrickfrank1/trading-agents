from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.sec_edgar import get_10k_filing_data, get_10q_filing_data, get_8k_filing_data, get_20f_filing_data, get_6k_filing_data


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
    Retrieve the last 2 available 10-K (or 20-F for foreign companies) annual report
    filings from SEC EDGAR. Returns key sections: Business (Item 1), Risk Factors
    (Item 1A), Selected Financial Data (Item 6), Management's Discussion & Analysis
    (Item 7), and Financial Statements (Item 8). For 20-F filings (foreign private
    issuers), returns 20-F-specific sections (Key Information, Information on the
    Company, Operating and Financial Review, Financial Information).
    IMPORTANT: If no filings are returned, the company may be a foreign private
    issuer. Try get_20f_filing as a fallback.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Key sections from the last 2 10-K/20-F annual filings
    """
    return get_10k_filing_data(ticker, curr_date)


@tool
def get_10q_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve the last 2 available 10-Q quarterly report filings from SEC EDGAR.
    Returns Financial Statements, Management's Discussion & Analysis (MD&A),
    and Risk Factors. Use this to understand quarterly financial performance,
    segment results, management commentary on recent trends, and seasonal patterns.
    IMPORTANT: If no filings are returned, the company may be a foreign private
    issuer that files 6-K instead. Try get_6k_filing as a fallback.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Key sections from the last 2 10-Q quarterly filings
    """
    return get_10q_filing_data(ticker, curr_date)


@tool
def get_8k_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve the last 2 available 8-K current report filings from SEC EDGAR.
    8-K filings disclose major corporate events such as earnings results,
    acquisitions, divestitures, leadership changes, debt defaults, and other
    material events. Use this to understand recent significant developments
    that may affect the investment thesis.
    IMPORTANT: If no filings are returned, the company may be a foreign private
    issuer that files 6-K instead. Try get_6k_filing as a fallback.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: All identified sections from the last 2 8-K current report filings
    """
    return get_8k_filing_data(ticker, curr_date)


@tool
def get_20f_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve the last 2 available 20-F annual report filings from SEC EDGAR.
    20-F is the annual report form used by foreign private issuers (non-US
    companies) instead of 10-K. Returns key sections: Key Information (Item 3),
    Information on the Company (Item 4), Operating and Financial Review and
    Prospects (Item 5), Directors, Senior Management and Employees (Item 6),
    Major Shareholders and Related Party Transactions (Item 7), and Financial
    Information (Item 8). Use this for non-US companies listed on US exchanges.
    IMPORTANT: If no filings are returned, the company may be a US domestic
    issuer. Try get_10k_filing as a fallback.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Key sections from the last 2 20-F annual filings
    """
    return get_20f_filing_data(ticker, curr_date)


@tool
def get_6k_filing(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve the last 3 available 6-K reports from SEC EDGAR. 6-K is filed by
    foreign private issuers (non-US companies) to disclose material information
    that is the equivalent of both 8-K (current events) and 10-Q (quarterly
    financials) for US companies. Common contents include earnings results,
    interim financial statements, material contracts, and other periodic
    disclosures. Use this for non-US companies listed on US exchanges.
    IMPORTANT: If no filings are returned, the company may be a US domestic
    issuer. Try get_8k_filing for current events or get_10q_filing for
    quarterly reports as a fallback.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (optional)
    Returns:
        str: Content from the last 3 6-K reports
    """
    return get_6k_filing_data(ticker, curr_date)
