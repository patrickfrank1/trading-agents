# TradingAgents/graph/setup.py

from typing import Any, Dict, Optional
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.facts_snapshot import create_facts_snapshot
from tradingagents.agents.utils.fact_check import create_fact_check
from tradingagents.agents.utils.fact_reconciliation import create_fact_reconciliation
from tradingagents.agents.researchers.debate_referee import create_debate_referee
from tradingagents.agents.risk_mgmt.risk_debate_referee import create_risk_debate_referee

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        debate_llms: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with required components.

        Args:
            quick_thinking_llm: LLM for analysts, debaters (default), trader.
            deep_thinking_llm: LLM for Research Manager and Portfolio Manager.
            tool_nodes: Per-category ToolNode map.
            conditional_logic: Flow controller (carries the enable_* flags).
            debate_llms: Optional per-debater LLMs keyed by
                bull/bear/aggressive/conservative/neutral. When a key is absent
                the shared quick_thinking_llm is used. Diversifying the debaters
                (different temperatures / personas) stops the debate from being
                one model arguing with itself.
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.debate_llms = debate_llms or {}

    def _debater_llm(self, key: str) -> Any:
        return self.debate_llms.get(key, self.quick_thinking_llm)

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
                - "macro": Macro analyst
                - "business": Business analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        if "macro" in selected_analysts:
            analyst_nodes["macro"] = create_macro_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["macro"] = create_msg_delete()
            tool_nodes["macro"] = self.tool_nodes["macro"]

        if "business" in selected_analysts:
            analyst_nodes["business"] = create_business_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["business"] = create_msg_delete()
            tool_nodes["business"] = self.tool_nodes["business"]

        # Create researcher and manager nodes.
        # Each debater may use a dedicated (e.g. different-temperature) LLM so
        # the two sides are not literally the same model arguing with itself.
        bull_researcher_node = create_bull_researcher(self._debater_llm("bull"))
        bear_researcher_node = create_bear_researcher(self._debater_llm("bear"))
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Debate-quality nodes. Each is a no-op when its feature flag is off,
        # so the graph topology stays fixed while behaviour is config-driven.
        facts_snapshot_node = create_facts_snapshot(
            self.quick_thinking_llm,
            enabled=self.conditional_logic.enable_facts_snapshot,
        )
        debate_referee_node = create_debate_referee(
            self.quick_thinking_llm,
            enabled=self.conditional_logic.enable_debate_referee,
        )
        fact_check_node = create_fact_check(
            self.quick_thinking_llm,
            enabled=self.conditional_logic.enable_fact_check,
        )
        # Reconciliation fetches raw data and reasons over it; use the deep LLM.
        fact_reconciliation_node = create_fact_reconciliation(self.deep_thinking_llm)

        # Risk-debate referee: scores each Aggressive/Conservative/Neutral
        # round for convergence so the risk debate can end early instead of
        # running fixed restating rounds. No-op when its flag is off.
        risk_debate_referee_node = create_risk_debate_referee(
            self.quick_thinking_llm,
            enabled=self.conditional_logic.enable_risk_debate_referee,
        )

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self._debater_llm("aggressive"))
        neutral_analyst = create_neutral_debator(self._debater_llm("neutral"))
        conservative_analyst = create_conservative_debator(self._debater_llm("conservative"))
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Facts Snapshot", facts_snapshot_node)
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Debate Referee", debate_referee_node)
        workflow.add_node("Fact Check", fact_check_node)
        workflow.add_node("Fact Reconciliation", fact_reconciliation_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Risk Debate Referee", risk_debate_referee_node)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst, or to Facts Snapshot if this is the last analyst.
            # Facts Snapshot compiles the canonical numbers block once, before the
            # debate, so every downstream agent argues from the same figures.
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Facts Snapshot")

        # Debate loop: Bull -> Bear -> Debate Referee -> (Bull | Fact Check).
        # The referee scores each completed round for convergence and can end
        # the debate early instead of running fixed, restating rounds.
        workflow.add_edge("Facts Snapshot", "Bull Researcher")
        workflow.add_edge("Bull Researcher", "Bear Researcher")
        workflow.add_edge("Bear Researcher", "Debate Referee")
        workflow.add_conditional_edges(
            "Debate Referee",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Fact Check": "Fact Check",
                "Research Manager": "Research Manager",
            },
        )

        # Post-debate fact-check -> optional reconciliation -> Research Manager.
        # The fact-check audits debate claims against the source reports; when it
        # flags a contradiction resolvable by re-querying raw data, reconciliation
        # resolves it before any decision is made.
        workflow.add_conditional_edges(
            "Fact Check",
            self.conditional_logic.should_continue_after_fact_check,
            {
                "Fact Reconciliation": "Fact Reconciliation",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Fact Reconciliation", "Research Manager")

        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        # After Neutral speaks, route through the Risk Debate Referee (which
        # scores the round for convergence) instead of straight back to
        # Aggressive. The referee then decides: another round or the PM.
        workflow.add_edge("Neutral Analyst", "Risk Debate Referee")
        workflow.add_conditional_edges(
            "Risk Debate Referee",
            self.conditional_logic.should_continue_after_risk_referee,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
