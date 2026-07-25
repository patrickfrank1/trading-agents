"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_claim_audit_block,
    get_facts_block,
    get_language_instruction,
    get_report_hygiene_instruction,
    get_reports_digest,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        facts_block = get_facts_block(state)
        reports_digest = get_reports_digest(state)
        claim_audit_block = get_claim_audit_block(state)

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Analyst Weighting Priority** — used to assign impact ratings (same priority as the research team):
- Business Analyst — Highest priority: competitive moat, management execution, product strategy, and long-term business value.
- Fundamentals Analyst — Core financial analysis: profitability, valuation, balance sheet strength, and financial health.
- Macro Analyst — Macroeconomic context: Fed policy, inflation, labor markets, and geopolitical factors.
- Market Analyst — Technical indicators and price action for entry/exit timing — NOT for the directional thesis itself.
- News Analyst — Recent news flow, material events, and catalysts, but do not let news override fundamentals.
- Sentiment Analyst — Social media and retail sentiment, supplementary signal only.

**How to use the inputs below:**
- The raw Business and Fundamentals reports are primary evidence alongside the risk debate. The debate is a *filter*, not a substitute.
- The Canonical Facts Snapshot is the single source of truth for numbers; do not re-derive them.
- The Claim Audit lists debate claims flagged as unsupported/contradicted by the source reports — discount them when assigning impact.

**Step 1 — Arguments Table:**
First, compile a markdown table of the key BUY and SELL arguments extracted from the risk debate and supporting context below. Each row must include: the argument, its source analyst type, an impact rating (High / Medium / Low), and whether it supports BUY or SELL. Arguments sourced from Business Analyst or Fundamentals Analyst should generally carry High impact; arguments from Macro, Market, News, or Sentiment Analysts should generally carry Medium or Low impact. This table anchors your final decision in transparent, weighted evidence.

**Step 2 — Weighted Score:**
Compute a single net score from -100 to +100 by weighing the arguments table (High = ±20, Medium = ±10, Low = ±5; positive for BUY, negative for SELL; clamp to [-100, +100]). The score must be auditable: a reader tallying the table should arrive at the same number. Map score to rating band (>=+40 Buy, +15..+39 Overweight, -14..+14 Hold, -39..-15 Underweight, <=-40 Sell). If your final rating falls outside the band implied by the score, you MUST explain the override in the investment thesis — silent overrides are not allowed.

**Step 3 — Probability-Weighted Scenario Table:**
Produce a three-row (Bull / Base / Bear) markdown table with explicit probability weights (summing to 100%), a price target per scenario, and the one-sentence driver. Compute and show the probability-weighted expected price. This forces an explicit view on any binary / "show-me" outcomes instead of leaving them implicit in prose.

**Step 4 — Trade Ticket:**
Resolve the risk team's conflicting sizing and hedge proposals into ONE executable plan. Specify: action (consistent with the rating), position size as % of portfolio, entry/exit levels, hedge structure with concrete strikes and approximate premium (do NOT write "consider a collar" — write "buy 6-mo $110 put / sell $150 call, ~$0 net debit"), and named exit triggers. For a Hold with no new capital, state size = 0% for new capital and give the maintenance plan for existing holders.

**Step 5 — Investment Thesis & Final Decision:**
Provide a concise summary of the key drivers behind your final decision, then deliver the final rating and supporting details. The rating must be consistent with the weighted score band unless an override is explicitly justified.

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}{facts_block}
{reports_digest}
{claim_audit_block}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}{get_report_hygiene_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
            "referee_notes": risk_debate_state.get("referee_notes", ""),
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
