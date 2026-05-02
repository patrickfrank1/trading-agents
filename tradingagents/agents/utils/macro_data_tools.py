from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta

import yfinance as yf

from tradingagents.dataflows.yfinance_news import _extract_article_data
from tradingagents.dataflows.stockstats_utils import yf_retry


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
