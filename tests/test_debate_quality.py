"""Tests for the adversarial-debate quality controls added to stop the
bull/bear debate from devolving into rhetorical restatement:

- ConditionalLogic convergence gate (early stop) and reconciliation routing
- Facts Snapshot / Fact Check / Fact Reconciliation / Debate Referee node
  no-op behaviour when their feature flags are off
- Graph compiles with the new nodes wired in
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic


def _debate_state(count=0, converged=False):
    return {
        "investment_debate_state": {"count": count},
        "debate_converged": converged,
        "claim_audit": "",
    }


@pytest.mark.unit
class TestDebateConvergenceGate:
    def test_continues_when_below_cap_and_not_converged(self):
        cl = ConditionalLogic(max_debate_rounds=2, enable_fact_check=True)
        assert cl.should_continue_debate(_debate_state(count=2)) == "Bull Researcher"

    def test_stops_at_cap_and_routes_to_fact_check(self):
        cl = ConditionalLogic(max_debate_rounds=2, enable_fact_check=True)
        assert cl.should_continue_debate(_debate_state(count=4)) == "Fact Check"

    def test_stops_early_when_referee_signals_convergence(self):
        cl = ConditionalLogic(max_debate_rounds=5, enable_debate_referee=True)
        assert (
            cl.should_continue_debate(_debate_state(count=2, converged=True))
            == "Fact Check"
        )

    def test_skips_fact_check_when_disabled(self):
        cl = ConditionalLogic(max_debate_rounds=2, enable_fact_check=False)
        assert cl.should_continue_debate(_debate_state(count=4)) == "Research Manager"

    def test_referee_disabled_ignores_convergence_flag(self):
        # Without the referee, convergence can't be trusted, so only the cap binds.
        cl = ConditionalLogic(max_debate_rounds=5, enable_debate_referee=False)
        assert (
            cl.should_continue_debate(_debate_state(count=2, converged=True))
            == "Bull Researcher"
        )


@pytest.mark.unit
class TestFactCheckRouting:
    def test_routes_to_reconciliation_when_needed(self):
        cl = ConditionalLogic(enable_fact_reconciliation=True)
        state = {"claim_audit": "...some audit...\nRECONCILIATION_NEEDED: yes"}
        assert cl.should_continue_after_fact_check(state) == "Fact Reconciliation"

    def test_routes_to_rm_when_not_needed(self):
        cl = ConditionalLogic(enable_fact_reconciliation=True)
        state = {"claim_audit": "RECONCILIATION_NEEDED: no"}
        assert cl.should_continue_after_fact_check(state) == "Research Manager"

    def test_skips_reconciliation_when_disabled(self):
        cl = ConditionalLogic(enable_fact_reconciliation=False)
        state = {"claim_audit": "RECONCILIATION_NEEDED: yes"}
        assert cl.should_continue_after_fact_check(state) == "Research Manager"


@pytest.mark.unit
class TestDisabledNodesAreNoOps:
    def test_facts_snapshot_disabled_returns_empty(self):
        from tradingagents.agents.utils.facts_snapshot import create_facts_snapshot

        node = create_facts_snapshot(MagicMock(), enabled=False)
        out = node({"market_report": "x"})
        assert out == {"facts_snapshot": ""}

    def test_fact_check_disabled_returns_empty(self):
        from tradingagents.agents.utils.fact_check import create_fact_check

        node = create_fact_check(MagicMock(), enabled=False)
        out = node({"investment_debate_state": {"history": "x"}})
        assert out == {"claim_audit": ""}

    def test_debate_referee_disabled_is_noop(self):
        from tradingagents.agents.researchers.debate_referee import create_debate_referee

        node = create_debate_referee(MagicMock(), enabled=False)
        out = node({"investment_debate_state": {"count": 2}})
        assert out == {}


@pytest.mark.unit
class TestDebateRefereeConvergence:
    def test_sets_converged_when_referee_says_yes(self):
        from tradingagents.agents.researchers.debate_referee import create_debate_referee

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content="CONCESSIONS: none\nCONVERGED: yes"
        )
        node = create_debate_referee(llm)
        state = {
            "investment_debate_state": {
                "history": "h",
                "bull_history": "b",
                "bear_history": "r",
                "current_response": "x",
                "judge_decision": "",
                "count": 4,
                "referee_notes": "",
            }
        }
        out = node(state)
        assert out["debate_converged"] is True
        assert "round 2" in out["investment_debate_state"]["referee_notes"]

    def test_does_not_converge_when_referee_says_no(self):
        from tradingagents.agents.researchers.debate_referee import create_debate_referee

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="CONVERGED: no")
        node = create_debate_referee(llm)
        out = node({"investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": "", "judge_decision": "", "count": 2,
            "referee_notes": "",
        }})
        assert out["debate_converged"] is False


@pytest.mark.unit
class TestGraphCompiles:
    def test_graph_compiles_with_new_debate_nodes(self, mock_llm_client):
        from tradingagents.graph.setup import GraphSetup
        from tradingagents.graph.conditional_logic import ConditionalLogic
        from langgraph.prebuilt import ToolNode
        from langchain_core.tools import tool

        @tool
        def _stub_tool(ticker: str) -> str:
            """stub"""
            return "x"

        def _stub():
            return MagicMock()

        cl = ConditionalLogic(max_debate_rounds=2, max_risk_discuss_rounds=2)
        tool_nodes = {k: ToolNode([_stub_tool]) for k in (
            "market", "social", "news", "fundamentals", "macro", "business"
        )}
        gs = GraphSetup(_stub(), _stub(), tool_nodes, cl)
        workflow = gs.setup_graph(["market", "news", "fundamentals", "business"])
        graph = workflow.compile()
        # The new nodes are present in the compiled graph.
        node_names = set(graph.get_graph().nodes.keys())
        for name in (
            "Facts Snapshot", "Bull Researcher", "Bear Researcher",
            "Debate Referee", "Fact Check", "Fact Reconciliation",
            "Research Manager", "Portfolio Manager",
        ):
            assert name in node_names, f"missing node: {name}"
