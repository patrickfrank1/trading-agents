"""Tests for the no-login web search fallback tool.

Covers:
- ``search_web`` dataflows function: formatting, empty query, cap on
  max_results, missing-package and search-failure error strings.
- ``web_search`` LangChain tool: schema preserved, safe_tool applied,
  delegates to ``search_web``.
- ``WEB_SEARCH_INSTRUCTION`` prompt hint.
- Wiring: analysts bind the tool when enabled and not when disabled;
  ToolNodes contain the tool when enabled.
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.utils.web_search_tools import (
    WEB_SEARCH_INSTRUCTION,
    web_search,
)
from tradingagents.dataflows.web_search import MAX_RESULTS_LIMIT, search_web
from tradingagents.default_config import DEFAULT_CONFIG


def _fake_results():
    return [
        {"title": "Apple raises guidance", "href": "https://example.com/a", "body": "Apple beat expectations."},
        {"title": "Analysts weigh in", "href": "https://example.com/b", "body": "Street reaction."},
    ]


@pytest.mark.unit
class TestSearchWeb:
    def test_formats_results(self):
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = _fake_results()
            out = search_web("apple news", max_results=2)
        assert "1. Apple raises guidance" in out
        assert "https://example.com/a" in out
        assert "Apple beat expectations." in out
        assert "2. Analysts weigh in" in out
        mock_ddgs.return_value.text.assert_called_once_with("apple news", max_results=2)

    def test_empty_query_returns_error_string(self):
        assert "[web_search]" in search_web("   ")

    def test_max_results_capped(self):
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = _fake_results()
            search_web("query", max_results=999)
        _, kwargs = mock_ddgs.return_value.text.call_args
        assert kwargs["max_results"] == MAX_RESULTS_LIMIT

    def test_bad_max_results_falls_back_to_default(self):
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = _fake_results()
            search_web("query", max_results="nonsense")
        _, kwargs = mock_ddgs.return_value.text.call_args
        assert kwargs["max_results"] == 8

    def test_no_results(self):
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = []
            out = search_web("obscure query")
        assert "No results" in out

    def test_search_failure_returns_error_string(self):
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.side_effect = RuntimeError("rate limited")
            out = search_web("query")
        assert "[web_search] Search failed" in out
        assert "rate limited" in out


@pytest.mark.unit
class TestWebSearchTool:
    def test_tool_schema(self):
        assert web_search.name == "web_search"
        assert "query" in web_search.args
        assert "max_results" in web_search.args

    def test_tool_delegates_to_search_web(self):
        with patch(
            "tradingagents.agents.utils.web_search_tools.search_web"
        ) as mock_fn:
            mock_fn.return_value = "1. result"
            out = web_search.invoke({"query": "test", "max_results": 3})
        assert out == "1. result"
        mock_fn.assert_called_once_with("test", 3)

    def test_tool_survives_exceptions(self):
        with patch(
            "tradingagents.agents.utils.web_search_tools.search_web"
        ) as mock_fn:
            mock_fn.side_effect = RuntimeError("boom")
            out = web_search.invoke({"query": "test"})
        assert "Tool Error" in out

    def test_instruction_mention_fallback_semantics(self):
        assert "web_search" in WEB_SEARCH_INSTRUCTION
        assert "ONLY" in WEB_SEARCH_INSTRUCTION


@pytest.mark.unit
class TestWiring:
    def test_config_defaults_to_enabled(self):
        assert DEFAULT_CONFIG.get("enable_web_search") is True

    def test_tool_nodes_contain_web_search_when_enabled(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph(config=dict(DEFAULT_CONFIG))
        for name, node in graph.tool_nodes.items():
            assert "web_search" in node.tools_by_name, f"missing in {name}"

    def test_tool_nodes_omit_web_search_when_disabled(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = dict(DEFAULT_CONFIG)
        config["enable_web_search"] = False
        graph = TradingAgentsGraph(config=config)
        for name, node in graph.tool_nodes.items():
            assert "web_search" not in node.tools_by_name, f"present in {name}"

    def _run_analyst(self, factory, **kwargs):
        llm = MagicMock()
        llm.bind_tools.return_value.invoke.return_value = MagicMock(
            tool_calls=[], content="report"
        )
        analyst = factory(llm, **kwargs)
        analyst(
            {"messages": [], "trade_date": "2026-01-01", "company_of_interest": "AAPL"}
        )
        return llm.bind_tools.call_args[0][0]

    def test_analysts_bind_web_search_when_enabled(self):
        from tradingagents.agents.analysts.market_analyst import create_market_analyst
        from tradingagents.agents.analysts.news_analyst import create_news_analyst

        for factory in (create_market_analyst, create_news_analyst):
            bound = self._run_analyst(factory, enable_web_search=True)
            assert any(t.name == "web_search" for t in bound)

    def test_analysts_omit_web_search_when_disabled(self):
        from tradingagents.agents.analysts.market_analyst import create_market_analyst
        from tradingagents.agents.analysts.news_analyst import create_news_analyst

        for factory in (create_market_analyst, create_news_analyst):
            bound = self._run_analyst(factory, enable_web_search=False)
            assert all(t.name != "web_search" for t in bound)
