"""Catch-all web search tool for ad-hoc analyst queries.

Unlike every other tool in this package, ``web_search`` is not bound to a
data vendor: it queries DuckDuckGo (no API key, no login) and is meant as
a *fallback* for questions the specialized tools cannot answer — recent
events beyond the vendor news window, competitor/industry context,
regulatory announcements, executive changes, etc.

Import ``web_search`` from :mod:`tradingagents.agents.utils.agent_utils`
and add it to an analyst's ``tools`` list (with ``WEB_SEARCH_INSTRUCTION``
appended to the system prompt) so the analyst can fall back to it when
nothing else fits. The ToolNode for that analyst must also contain the
tool (see ``TradingAgentsGraph._create_tool_nodes``).
"""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.agents.utils.tool_errors import safe_tool
from tradingagents.dataflows.web_search import search_web

WEB_SEARCH_INSTRUCTION = (
    " You also have access to web_search(query, max_results): a general DuckDuckGo"
    " web search that requires no API key. Use it ONLY for ad-hoc questions the"
    " specialized tools above cannot answer (e.g. events outside their date"
    " coverage, company/industry context they do not provide, or cross-checking"
    " a surprising claim). Prefer the specialized tools whenever they cover the"
    " request; do not use web_search as a substitute for them. Never trust a"
    " single snippet: cite the URL of anything load-bearing."
)


@tool
@safe_tool
def web_search(
    query: Annotated[str, "Free-form web search query"],
    max_results: Annotated[int, "Maximum number of results to return (max 20)"] = 8,
) -> str:
    """
    Perform a general DuckDuckGo web search (no API key required).

    Use this ONLY when the specialized tools cannot answer the question.
    Args:
        query (str): Free-form search query
        max_results (int): Maximum number of results to return (default 8, max 20)
    Returns:
        str: Numbered list of results with title, URL, and snippet
    """
    return search_web(query, max_results)
