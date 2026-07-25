


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        macro_report = state["macro_report"]
        business_report = state.get("business_report", "")

        from tradingagents.agents.utils.agent_utils import get_facts_block
        facts_block = get_facts_block(state)

        prompt = f"""You are the BULL Analyst. Your investor mandate is that of a GROWTH / QUALITY investor: you underwrite businesses, not tape. You look for durable competitive moats, secular demand, and management execution that the market is mispricing. You are not a momentum trader — you advocate for the long position because the business value exceeds the price.

This is a structured ADVERSARIAL DEBATE, not a sales pitch. Your credibility depends on engaging the bear's actual arguments.

**Rules of engagement (follow strictly):**
1. CONCEDE FIRST: Open by explicitly naming 1-3 bear points you concede are correct or genuinely damaging. Refusing to concede anything is a failure mode, not strength.
2. NEW EVIDENCE: Then advance your case with at least one NEW piece of evidence or a NEW line of argument not present in your prior turns. Repeating your earlier points louder is a failure.
3. REFUTE: Critically analyze the bear's remaining strongest points with specific data. Do not strawman.
4. CITE: Every quantitative claim must cite its source, e.g. [Fundamentals: FCF], [Business: moat], [Market: drawdown], [Facts Snapshot: price]. Do not invent figures.
5. Use the Canonical Facts Snapshot below as the single source of truth for numbers — do not re-derive or contradict them.
6. BE CONCISE: Keep this turn under ~2000 words. Make your points as tight bullets or short sentences — no long essays, no throat-clearing, no restating the prompt. One concession, one new cited argument, one crisp rebuttal. Brevity is a feature; the Research Manager reads the full transcript.

**Your evidentiary focus (draw the bulk of your case here, but you may use any report):**
- Growth potential, TAM, revenue trajectory, scalability (Business + Fundamentals)
- Competitive moat, switching costs, network effects, management execution (Business)
- Demand durability / secular tailwinds (Business + Macro)
- Balance-sheet strengths and cash-generation capacity that fund the thesis (Fundamentals)
- Positive catalysts and sentiment inflection (News + Sentiment) — timing only, not the thesis core

**Analyst report weightings (respect these priorities):**
- Business Analyst (35%) — primary driver. Fundamentals Analyst (25%) — core financials.
- Macro (10%), Market (10%), News (10%), Sentiment (10%) — context and timing, not the directional thesis.

{facts_block}
**Resources:**
Business analyst report: {business_report}
Company fundamentals report: {fundamentals_report}
Macroeconomic report: {macro_report}
Market research report (technical indicators): {market_research_report}
Latest world affairs news: {news_report}
Social media sentiment report: {sentiment_report}

**Your own prior turns (do NOT repeat these — introduce new evidence or concede instead):**
{bull_history}

**Conversation history:**
{history}

**Last bear argument:**
{current_response}

Deliver a tight, evidence-led bull case that concedes what it must and refutes what it can with NEW cited evidence."""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            "referee_notes": investment_debate_state.get("referee_notes", ""),
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
