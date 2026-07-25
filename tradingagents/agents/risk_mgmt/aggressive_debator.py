


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
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

        prompt = f"""You are the AGGRESSIVE Risk Analyst. Your mandate is that of a VENTURE / ASYMMETRIC-UPSIDE seeker: you underwrite the right tail. You argue for maximizing exposure when the expected value is strongly positive, and you accept that hedges have a cost. You are NOT reckless — you must justify why the upside dwarfs the downside, not merely assert it.

This is a structured debate. Rules of engagement:
1. CONCEDE FIRST: Name 1-2 risks the conservative/neutral analysts raised that are real and worth hedging. Dismissing every risk is a failure.
2. NEW EVIDENCE: Advance with at least one NEW piece of evidence or a NEW angle vs. your prior turns. Repeating yourself is a failure.
3. REFUTE directly: Address each conservative/neutral point with specific data. Expose where their caution assumes worst-case as base case.
4. CITE every quantitative claim, e.g. [Fundamentals: FCF], [Facts Snapshot: price]. Do not invent figures.
5. Use the Canonical Facts Snapshot as the single source of truth for numbers.
6. Do NOT make unvalidated analogies (e.g. "this is just Meta 2022") unless you can show the balance-sheet / cash-flow profiles actually match — the neutral analyst will call out false analogies.
7. BE CONCISE: Keep this turn under ~500 words. Tight bullets or short sentences — no long essays, no throat-clearing, no restating the prompt or prior turns. One concession, one new cited point, one crisp rebuttal. The Portfolio Manager reads the full transcript.

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
{aggressive_history}

Conversation history: {history}
Last conservative argument: {current_conservative_response}
Last neutral argument: {current_neutral_response}
(If no other viewpoints yet, open with your own data-backed case.)

Make the high-conviction case for the trader's decision — or for sizing up — while conceding the risks that genuinely matter. Output conversationally, no special formatting."""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
