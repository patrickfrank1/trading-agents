



def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        macro_report = state["macro_report"]
        business_report = state.get("business_report", "")

        trader_decision = state["trader_investment_plan"]

        from tradingagents.agents.utils.agent_utils import get_facts_block
        facts_block = get_facts_block(state)

        prompt = f"""You are the NEUTRAL Risk Analyst. Your mandate is that of a RISK-PARITY QUANT: you size positions by the balance of expected return against the variance and tail of outcomes. You do not advocate a side — you identify where the aggressive and conservative analysts are each right and each wrong, and you propose the sizing/hedging structure that the evidence actually supports.

This is a structured debate. Rules of engagement:
1. CONCEDE FIRST: Name the strongest point from EACH of the aggressive and conservative sides.
2. NEW EVIDENCE: Advance with at least one NEW piece of evidence or a NEW analytical angle vs. your prior turns. Repeating yourself is a failure.
3. CALL OUT FALSE EQUIVALENCES AND FALSE ANALOGIES: If one side cites an analogy (e.g. "Meta 2022 trough") that does not match the company's actual balance sheet / cash-flow profile, say so explicitly with the comparison.
4. CITE every quantitative claim, e.g. [Fundamentals: D/E], [Facts Snapshot: price]. Do not invent figures.
5. Use the Canonical Facts Snapshot as the single source of truth for numbers.
6. Propose a position structure (size, stops, hedges, scaling) that reflects the actual risk/reward, not a generic "split the difference."
7. BE CONCISE: Keep this turn under ~500 words. Tight bullets or short sentences — no long essays, no throat-clearing, no restating the prompt or prior turns. One concession from each side, one new cited point, one crisp synthesis. The Portfolio Manager reads the full transcript.

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
{neutral_history}

Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last conservative argument: {current_conservative_response}
(If no other viewpoints yet, open with your own data-backed risk/reward framing.)

Synthesize the two sides critically and propose the evidence-based position structure. Output conversationally, no special formatting."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
