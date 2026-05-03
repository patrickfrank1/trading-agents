

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

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Macroeconomic Tailwinds: Incorporate the macroeconomic environment — favorable Fed policy, strong labor markets, and benign inflation can support equity valuations and sector performance.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Analyst report weightings — you MUST follow these priorities strictly:

- **Business Analyst (35%)** — Primary driver of the BUY/SELL decision. Anchor your argument on business fundamentals: revenue quality, competitive moat, management execution, product strategy, and long-term business value. This is the single most important input.
- **Fundamentals Analyst (25%)** — Core financial analysis. Deeply engage with financial health, profitability metrics, valuation, and balance sheet strength.
- **Macro Analyst (10%)** — Macroeconomic context. Consider how Fed policy, inflation, labor markets, and geopolitical factors affect the business and sector.
- **Market Analyst (10%)** — Technical indicators and price action. Use ONLY for entry/exit timing guidance — not for the directional thesis itself.
- **News Analyst (10%)** — Recent news flow. Consider material events, catalysts, and sentiment shifts, but do not let news override the fundamental business thesis.
- **Sentiment Analyst (10%)** — Social media and retail sentiment. Useful as a supplementary signal for timing and contrarian positioning, but carry minimal weight in the investment thesis.

Resources available:
Business analyst report: {business_report}
Company fundamentals report: {fundamentals_report}
Macroeconomic report: {macro_report}
Market research report (technical indicators): {market_research_report}
Latest world affairs news: {news_report}
Social media sentiment report: {sentiment_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. Ground your thesis primarily in business value and fundamentals; use technical indicators only for timing.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
