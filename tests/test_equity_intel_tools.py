"""Tests for the equity-intelligence tools integration.

Covers:
- The new tools are registered in the correct analyst ToolNode categories.
- Each analyst binds the expected new tools.
- The customer-concentration regex patterns match real disclosure language.
"""

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
class TestToolRegistration:
    def test_new_tools_in_correct_categories(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # Bypass __init__ (which would create real LLM clients); build nodes only.
        tag = TradingAgentsGraph.__new__(TradingAgentsGraph)
        tag.config = {
            "data_cache_dir": "/tmp/opencode/_ta_cache",
            "results_dir": "/tmp/opencode/_ta_results",
        }
        import os
        os.makedirs(tag.config["data_cache_dir"], exist_ok=True)
        os.makedirs(tag.config["results_dir"], exist_ok=True)
        from tradingagents.dataflows.config import set_config
        set_config(tag.config)

        nodes = tag._create_tool_nodes()

        def names(key):
            return {t.name for t in nodes[key].tools_by_name.values()} | set(
                nodes[key].tools_by_name.keys()
            )

        market = set(nodes["market"].tools_by_name.keys())
        news = set(nodes["news"].tools_by_name.keys())
        fundamentals = set(nodes["fundamentals"].tools_by_name.keys())
        business = set(nodes["business"].tools_by_name.keys())

        assert "get_option_positioning" in market
        assert "get_short_interest" in market
        assert "get_institutional_holders" in news
        for name in (
            "get_analyst_estimates",
            "get_credit_and_debt_detail",
            "get_earnings_calendar",
            "get_capital_allocation_history",
        ):
            assert name in fundamentals, f"{name} missing from fundamentals"
        assert "get_customer_concentration" in business
        assert "get_governance" in business


@pytest.mark.unit
class TestAnalystsBindNewTools:
    def _tool_names(self, create_fn):
        llm = MagicMock()
        llm.bind_tools.return_value = MagicMock()
        node = create_fn(llm)
        # Inspect the closure to find the bound `tools` list is fragile across
        # implementations, so instead we exercise the node and capture the
        # tools passed to bind_tools via a fake chain.
        return llm.bind_tools.call_args

    def test_fundamentals_binds_new_tools(self):
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

        llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = MagicMock(tool_calls=[])
        llm.bind_tools.return_value = fake_chain
        create_fundamentals_analyst(llm)({"trade_date": "2025-01-01", "company_of_interest": "AAPL", "messages": []})
        bound = llm.bind_tools.call_args[0][0]
        names = {t.name for t in bound}
        assert {"get_analyst_estimates", "get_credit_and_debt_detail", "get_earnings_calendar", "get_capital_allocation_history"} <= names

    def test_business_binds_new_tools(self):
        from tradingagents.agents.analysts.business_analyst import create_business_analyst

        llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = MagicMock(tool_calls=[])
        llm.bind_tools.return_value = fake_chain
        create_business_analyst(llm)({"trade_date": "2025-01-01", "company_of_interest": "AAPL", "messages": []})
        names = {t.name for t in llm.bind_tools.call_args[0][0]}
        assert {"get_customer_concentration", "get_governance"} <= names

    def test_market_binds_new_tools(self):
        from tradingagents.agents.analysts.market_analyst import create_market_analyst

        llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = MagicMock(tool_calls=[])
        llm.bind_tools.return_value = fake_chain
        create_market_analyst(llm)({"trade_date": "2025-01-01", "company_of_interest": "AAPL", "messages": []})
        names = {t.name for t in llm.bind_tools.call_args[0][0]}
        assert {"get_option_positioning", "get_short_interest"} <= names

    def test_news_binds_new_tools(self):
        from tradingagents.agents.analysts.news_analyst import create_news_analyst

        llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = MagicMock(tool_calls=[])
        llm.bind_tools.return_value = fake_chain
        create_news_analyst(llm)({"trade_date": "2025-01-01", "company_of_interest": "AAPL", "messages": []})
        names = {t.name for t in llm.bind_tools.call_args[0][0]}
        assert "get_institutional_holders" in names


@pytest.mark.unit
class TestConcentrationPatterns:
    def test_patterns_match_real_disclosure_language(self):
        from tradingagents.dataflows.sec_edgar import _CONCENTRATION_PATTERNS

        samples = [
            "As of May 31, 2025, our remaining performance obligations were $552 billion.",
            "No single customer accounted for more than 10% of total revenue.",
            "Our backlog was $38.9 billion at the end of the quarter.",
            "Customer concentration risk: one customer represented approximately 12% of receivables.",
            "We depend on a small number of customers for a significant portion of revenue.",
        ]
        for s in samples:
            assert any(p.search(s) for p in _CONCENTRATION_PATTERNS), f"no pattern matched: {s}"

    def test_patterns_do_not_match_unrelated_text(self):
        from tradingagents.dataflows.sec_edgar import _CONCENTRATION_PATTERNS

        assert not any(p.search("The company sells software to enterprises globally.") for p in _CONCENTRATION_PATTERNS)
