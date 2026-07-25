


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        macro_report = state["macro_report"]
        business_report = state.get("business_report", "")

        from tradingagents.agents.utils.agent_utils import get_facts_block
        facts_block = get_facts_block(state)

        prompt = f"""You are the BEAR Analyst. Your investor mandate is that of a DISTRESSED-CREDIT / VALUE-SAFETY analyst: you scrutinize balance-sheet durability, cash-flow sufficiency, refinancing risk, capital-allocation discipline, and valuation discipline against intrinsic value. You are not a perma-bear — you make the case against the long position because the risks or valuation outweigh the business case.

This is a structured ADVERSARIAL DEBATE, not a doom-scroll. Your credibility depends on engaging the bull's actual arguments.

**Rules of engagement (follow strictly):**
1. CONCEDE FIRST: Open by explicitly naming 1-3 bull points you concede are correct or genuinely strong. Refusing to concede anything is a failure mode.
2. NEW EVIDENCE: Then advance your case with at least one NEW piece of evidence or a NEW line of argument not present in your prior turns. Repeating your earlier points louder is a failure.
3. REFUTE: Critically analyze the bull's remaining strongest points with specific data. Expose over-optimistic assumptions. Do not strawman.
4. CITE: Every quantitative claim must cite its source, e.g. [Fundamentals: total debt], [Business: churn], [Market: drawdown], [Facts Snapshot: price]. Do not invent figures.
5. Use the Canonical Facts Snapshot below as the single source of truth for numbers — do not re-derive or contradict them.
6. BE CONCISE: Keep this turn under ~180 words. Make your points as tight bullets or short sentences — no long essays, no throat-clearing, no restating the prompt. One concession, one new cited argument, one crisp rebuttal. Brevity is a feature; the Research Manager reads the full transcript.

**Your evidentiary focus (draw the bulk of your case here, but you may use any report):**
- Balance-sheet risk: leverage, debt maturity profile, refinancing risk, off-balance-sheet commitments, tangible book value (Fundamentals)
- Cash-flow sufficiency: FCF, capex burden, interest coverage, funding gaps (Fundamentals)
- Valuation discipline: DCF / EPV / comps vs. current price, margin of safety (Fundamentals)
- Competitive threats, execution risk, demand cyclicality (Business)
- Macro / liquidity headwinds that compound the above (Macro) — context, not the core

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
{bear_history}

**Conversation history:**
{history}

**Last bull argument:**
{current_response}

Deliver a tight, evidence-led bear case that concedes what it must and refutes what it can with NEW cited evidence."""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            "referee_notes": investment_debate_state.get("referee_notes", ""),
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
