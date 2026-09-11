"""No-login web search for ad-hoc queries.

Uses DuckDuckGo via the ``ddgs`` package: no API key, no account, no
vendor configuration. This is intentionally separate from the
``data_vendors`` routing in :mod:`tradingagents.dataflows.interface`
because it is a catch-all fallback tool, not a per-category data vendor.

Used by the ``web_search`` LangChain tool in
:mod:`tradingagents.agents.utils.web_search_tools`, which analysts may
call whenever a question is not covered by a specialized tool.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("tradingagents.tools.web_search")

#: Hard ceiling so the LLM cannot request an unreasonable result count.
MAX_RESULTS_LIMIT = 20


def search_web(query: str, max_results: int = 8) -> str:
    """Run a DuckDuckGo web search and return a formatted digest.

    Args:
        query: Free-form search query.
        max_results: Maximum number of results to return (capped at
            ``MAX_RESULTS_LIMIT``).

    Returns:
        A numbered list of ``title`` / ``url`` / ``snippet`` entries.
        On total failure, returns an error string (callers also wrap
        this with ``safe_tool``, so this is belt-and-braces).
    """
    if not query or not query.strip():
        return "[web_search] Empty query."

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 8
    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))

    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "[web_search] The 'ddgs' package is not installed. "
            "Install it with: uv pip install ddgs"
        )

    try:
        raw = DDGS().text(query.strip(), max_results=max_results)
    except Exception as exc:  # network hiccups, rate limits, parser changes
        logger.warning("web_search failed for query %r: %s", query, exc)
        return f"[web_search] Search failed: {exc}"

    if not raw:
        return f"[web_search] No results found for: {query}"

    lines = []
    for i, item in enumerate(raw, start=1):
        title = (item.get("title") or "").strip()
        url = (item.get("href") or item.get("url") or "").strip()
        snippet = (item.get("body") or item.get("snippet") or "").strip()
        lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")

    return "\n\n".join(lines)
