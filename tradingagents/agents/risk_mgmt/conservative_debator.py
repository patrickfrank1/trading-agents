


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        macro_report = state["macro_report"]
        business_report = state.get("business_report", "")

        trader_decision = state["trader_investment_plan"]

        from tradingagents.agents.utils.agent_utils import get_facts_block
        facts_block = get_facts_block(state)

        prompt = f"""You are the CONSERVATIVE Risk Analyst. Your mandate is that of a CAPITAL-PRESERVATION RISK OFFICER: your first duty is to avoid permanent loss of capital. You scrutinize downside scenarios, tail risk, leverage, liquidity, and concentration. You are NOT a perma-sell — you must acknowledge upside when it is real and well-supported, then explain why the risk-adjusted path still favors protection.

This is a structured debate. Rules of engagement:
1. CONCEDE FIRST: Name 1-2 upside points the aggressive/neutral analysts raised that are legitimate. Dismissing all upside is a failure.
2. NEW EVIDENCE: Advance with at least one NEW piece of evidence or a NEW risk vector vs. your prior turns. Repeating yourself is a failure.
3. REFUTE directly: Address each aggressive/neutral point with specific data. Expose where their optimism assumes best-case as base case.
4. CITE every quantitative claim, e.g. [Fundamentals: total debt], [Facts Snapshot: price]. Do not invent figures.
5. Use the Canonical Facts Snapshot as the single source of truth for numbers.
6. Propose CONCRETE, specific risk controls (stop levels, sizing caps, hedges) tied to the cited numbers — not generic "be cautious."
7. BE CONCISE: Keep this turn under ~1000 words. Tight bullets or short sentences — no long essays, no throat-clearing, no restating the prompt or prior turns. One concession, one new cited point, one crisp rebuttal. The Portfolio Manager reads the full transcript.

{facts_block}
The trader's decision under review:
{trader_decision}

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Macroeconomic Analysis Report: {macro_report}
Business and Industry Report: {business_report}

**Your own prior turns (do NOT repeat these):**
{conservative_history}

Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last neutral argument: {current_neutral_response}
(If no other viewpoints yet, open with your own data-backed risk assessment.)

Make the capital-preservation case — concede real upside, then show why the downside path demands protection. Output conversationally, no special formatting."""

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
            "referee_notes": risk_debate_state.get("referee_notes", ""),
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
