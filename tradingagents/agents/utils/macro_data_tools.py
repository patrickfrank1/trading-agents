from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta

import yfinance as yf

from tradingagents.dataflows.yfinance_news import _extract_article_data
from tradingagents.dataflows.stockstats_utils import yf_retry
from tradingagents.dataflows.macro_market_data import (
    fetch_macro_market_data,
    format_macro_market_report,
)
from tradingagents.dataflows.macro_vendors import (
    fetch_vendor_data,
    format_vendor_report,
    get_available_vendors,
)


def _search_macro_news(queries, curr_date, look_back_days, limit):
    all_news = []
    seen_titles = set()

    try:
        for query in queries:
            search = yf_retry(lambda q=query: yf.Search(
                q=q,
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

    except Exception as e:
        return f"Error fetching macro economic news: {str(e)}"


@tool
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
    try:
        data = fetch_macro_market_data()
        return format_macro_market_report(data)
    except Exception as e:
        return f"Error fetching macro market data: {str(e)}"


@tool
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
    try:
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
    except Exception as e:
        return f"Error fetching FRED data: {str(e)}"


@tool
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
    try:
        data = fetch_vendor_data("oecd")
        return format_vendor_report("oecd", data)
    except Exception as e:
        return f"Error fetching OECD data: {str(e)}"


@tool
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
    try:
        data = fetch_vendor_data("worldbank", country=country)
        return format_vendor_report("worldbank", data)
    except Exception as e:
        return f"Error fetching World Bank data: {str(e)}"


@tool
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
    try:
        data = fetch_vendor_data("ecb")
        return format_vendor_report("ecb", data)
    except Exception as e:
        return f"Error fetching ECB data: {str(e)}"
