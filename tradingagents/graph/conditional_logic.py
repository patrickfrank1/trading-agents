# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(
        self,
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        enable_facts_snapshot=True,
        enable_debate_referee=True,
        enable_fact_check=True,
        enable_fact_reconciliation=True,
    ):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds
        self.enable_facts_snapshot = enable_facts_snapshot
        self.enable_debate_referee = enable_debate_referee
        self.enable_fact_check = enable_fact_check
        self.enable_fact_reconciliation = enable_fact_reconciliation

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_macro(self, state: AgentState):
        """Determine if macro analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_macro"
        return "Msg Clear Macro"

    def should_continue_business(self, state: AgentState):
        """Determine if business analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_business"
        return "Msg Clear Business"

    def _debate_done(self, state: AgentState) -> bool:
        """True when the bull/bear debate should stop.

        Stops when the round cap is reached OR the referee has signalled
        convergence (both sides only restating themselves). When the referee
        is disabled, falls back to the pure round-cap behaviour.
        """
        cap_reached = (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        )
        if cap_reached:
            return True
        if self.enable_debate_referee and state.get("debate_converged", False):
            return True
        return False

    def should_continue_debate(self, state: AgentState) -> str:
        """Route after the Debate Referee: keep debating or move to fact-check / RM."""
        if self._debate_done(state):
            if self.enable_fact_check:
                return "Fact Check"
            return "Research Manager"
        return "Bull Researcher"

    def should_continue_after_fact_check(self, state: AgentState) -> str:
        """Route after the Fact Check: reconcile a contradiction or go to RM."""
        if not self.enable_fact_reconciliation:
            return "Research Manager"
        audit = state.get("claim_audit", "")
        if "reconciliation_needed: yes" in audit.lower():
            return "Fact Reconciliation"
        return "Research Manager"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
