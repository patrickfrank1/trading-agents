"""Network-free tests for the SEC filing-signal tools.

Exercises the extraction/normalization helpers and the 8-K/Form-4
classification logic on sample text, plus verifies the new tools are
registered in the correct analyst categories and bound by the analysts.
"""

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
class TestExtractionHelpers:
    def test_keyword_passages_finds_and_dedups(self):
        from tradingagents.dataflows.sec_edgar import _extract_keyword_passages
        import re

        text = (
            "The remaining performance obligations were $552 billion. "
            "Of this, 12% is expected to be recognized as revenue within 12 months. "
            "No single customer accounted for more than 10% of revenue."
        )
        patterns = [re.compile(r"remaining\s+performance\s+obligations?", re.IGNORECASE)]
        passages = _extract_keyword_passages(text, patterns, context=120, max_passages=5)
        assert passages and "552 billion" in passages[0]

    def test_format_passage_report_handles_empty(self):
        from tradingagents.dataflows.sec_edgar import _format_passage_report

        report = _format_passage_report(
            "Title", {"form_type": "10-K", "filing_date": "2025-01-01", "accession": "x"}, []
        )
        assert "No relevant disclosures matched" in report
        assert "Title" in report

    def test_sentence_set_normalizes(self):
        from tradingagents.agents.utils.filing_signals_tools import _sentence_set

        s = _sentence_set(
            "Cybersecurity risk continues to grow for the business. "
            "Supply chain concentration is a newly disclosed risk; "
            "Cybersecurity risk continues to grow for the business."
        )
        assert "cybersecurity risk continues to grow for the business." in s
        assert "supply chain concentration is a newly disclosed risk;" in s


@pytest.mark.unit
class TestRiskFactorDiff:
    def test_added_and_removed_detected(self):
        from tradingagents.agents.utils.filing_signals_tools import _sentence_set

        prior = "Cybersecurity risk continues to grow for the business. Legacy on-premise risk remains material. "
        cur = "Cybersecurity risk continues to grow for the business. AI compute demand risk is emerging rapidly. Legacy on-premise risk remains material."
        added = _sentence_set(cur) - _sentence_set(prior)
        removed = _sentence_set(prior) - _sentence_set(cur)
        assert any("ai compute demand risk is emerging rapidly." in a for a in added)
        assert removed == set()


@pytest.mark.unit
class Test8KEventClassification:
    def test_item_codes_extracted(self):
        import re
        from tradingagents.agents.utils.filing_signals_tools import _8K_ITEM_DESCRIPTIONS

        text = "Item 2.02 Results of Operations. Item 5.02 Departure of Officers."
        codes = sorted(set(re.findall(r"item\s+(\d+\.\d+)", text, re.IGNORECASE)))
        assert codes == ["2.02", "5.02"]
        assert _8K_ITEM_DESCRIPTIONS["2.02"] == "Results of Operations (earnings)"
        assert _8K_ITEM_DESCRIPTIONS["5.02"].startswith("Departure")


@pytest.mark.unit
class TestForm4CodeMap:
    def test_codes_meaningful(self):
        from tradingagents.agents.utils.filing_signals_tools import _FORM4_CODES

        assert _FORM4_CODES["P"] == "Open-market purchase"
        assert _FORM4_CODES["S"] == "Open-market sale"
        assert "exercise" in _FORM4_CODES["M"].lower()


@pytest.mark.unit
class TestFilingSignalToolRegistration:
    def _tag(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        import os
        tag = TradingAgentsGraph.__new__(TradingAgentsGraph)
        tag.config = {
            "data_cache_dir": "/tmp/opencode/_ta_cache2",
            "results_dir": "/tmp/opencode/_ta_results2",
        }
        os.makedirs(tag.config["data_cache_dir"], exist_ok=True)
        os.makedirs(tag.config["results_dir"], exist_ok=True)
        from tradingagents.dataflows.config import set_config
        set_config(tag.config)
        return tag

    def test_fundamentals_category_has_filing_tools(self):
        nodes = self._tag()._create_tool_nodes()
        fund = set(nodes["fundamentals"].tools_by_name.keys())
        for name in (
            "get_debt_maturity_schedule",
            "get_rpo_disaggregation",
            "get_segment_geographic_reporting",
            "get_internal_controls",
            "get_stock_based_compensation",
            "get_regulatory_capital",
            "get_institutional_13f_filings",
            "get_insider_form4_activity",
            "get_prospectus_disclosure",
        ):
            assert name in fund, f"{name} missing from fundamentals tool node"

    def test_business_category_has_filing_tools(self):
        nodes = self._tag()._create_tool_nodes()
        biz = set(nodes["business"].tools_by_name.keys())
        for name in (
            "get_risk_factor_changes",
            "get_legal_proceedings",
            "get_cybersecurity_disclosure",
            "get_properties_capacity",
            "get_proxy_governance",
            "get_activist_filings",
            "get_form_8k_events",
        ):
            assert name in biz, f"{name} missing from business tool node"


@pytest.mark.unit
class TestAnalystsBindFilingTools:
    def _run(self, create_fn):
        llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = MagicMock(tool_calls=[])
        llm.bind_tools.return_value = fake_chain
        create_fn(llm)({"trade_date": "2025-01-01", "company_of_interest": "AAPL", "messages": []})
        return {t.name for t in llm.bind_tools.call_args[0][0]}

    def test_fundamentals_binds_filing_tools(self):
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
        names = self._run(create_fundamentals_analyst)
        assert {"get_debt_maturity_schedule", "get_rpo_disaggregation", "get_internal_controls"} <= names

    def test_business_binds_filing_tools(self):
        from tradingagents.agents.analysts.business_analyst import create_business_analyst
        names = self._run(create_business_analyst)
        assert {"get_risk_factor_changes", "get_form_8k_events", "get_activist_filings"} <= names


@pytest.mark.unit
class TestPassageToolFactoryDistinctNames:
    def test_factory_tools_have_distinct_names(self):
        import tradingagents.agents.utils.filing_signals_tools as m
        names = []
        for n in (
            "get_stock_based_compensation", "get_goodwill_intangibles", "get_pension_opeb",
            "get_uncertain_tax_positions", "get_variable_interest_entities",
            "get_regulatory_capital", "get_commitments_contingencies",
        ):
            obj = getattr(m, n)
            assert obj.name == n
            names.append(obj.name)
        assert len(set(names)) == len(names)
