from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_company_profile,
    get_sector_performance,
    get_peer_comparison,
    get_10k_filing,
    get_10q_filing,
    get_8k_filing,
    get_20f_filing,
    get_6k_filing,
)


def create_business_analyst(llm):
    def business_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_company_profile,
            get_sector_performance,
            get_peer_comparison,
            get_10k_filing,
            get_10q_filing,
            get_8k_filing,
            get_20f_filing,
            get_6k_filing,
        ]

        system_message = (
            "You are a senior business analyst and investment researcher. Your job is to "
            "read the last 2 available annual reports, quarterly/interim reports, and "
            "current event reports and build a deep, evidence-based understanding of the "
            "business. Use ALL of the following tools to gather comprehensive data:\n\n"
            "- `get_10k_filing`: Last 2 US annual reports (Item 1 Business, Item 1A Risk Factors, "
            "Item 7 MD&A, Item 8 Financial Statements). Also retrieves 20-F for foreign companies.\n"
            "- `get_20f_filing`: Last 2 foreign annual reports (20-F). Use if the company is a "
            "non-US/foreign private issuer or if get_10k_filing returns incomplete results. "
            "Returns Key Information, Company Description, Operating & Financial Review, "
            "and Financial Information sections.\n"
            "- `get_10q_filing`: Last 2 US quarterly reports for financial data and MD&A.\n"
            "- `get_6k_filing`: Last 3 foreign interim/event reports (6-K). Use if the company "
            "is a foreign private issuer or if get_10q_filing/get_8k_filing return no results. "
            "Contains earnings, interim financials, and material events.\n"
            "- `get_8k_filing`: Last 2 US current reports for recent material events.\n"
            "- `get_company_profile`: Company overview and business description.\n"
            "- `get_peer_comparison`: Financial metrics vs direct competitors.\n"
            "- `get_sector_performance`: Sector/industry benchmark performance.\n\n"
            "After gathering all data, write a comprehensive business analysis report that "
            "answers each of the following questions with specific evidence, numbers, and "
            "quotes from the filings:\n\n"

            "## 1. Business Model\n"
            "What are the main products or services offered by the company? Describe how the "
            "company generates revenue, its cost structure, pricing power, and unit economics.\n\n"

            "## 2. Target Market & Customer Base\n"
            "What is the company's target market and customer base? Describe customer "
            "concentration, geographic mix, and any trends in customer acquisition/retention.\n\n"

            "## 3. Competitive Positioning\n"
            "What are the company's main competitors and how does it differentiate itself "
            "from them? Assess the competitive moat: network effects, switching costs, brand, "
            "patents/IP, economies of scale, regulatory advantages, data advantages, cost "
            "leadership. Rate moat strength (None / Weak / Moderate / Strong / Very Strong).\n\n"

            "## 4. Geographic Performance\n"
            "Which geographic regions does the company operate in and how does it perform "
            "in each region? Discuss revenue mix by geography, growth rates, and regional risks.\n\n"

            "## 5. Segment Growth Trends\n"
            "Which business segments are growing the fastest and which are declining? "
            "Provide specific revenue growth rates by segment from the filings.\n\n"

            "## 6. Segment Profitability\n"
            "Which business segments are profitable and which are not? Provide margins, "
            "operating income, and any segment-level profitability metrics from the filings.\n\n"

            "## 7. Free Cash Flow Analysis\n"
            "What is the company's free cash flow and how has it changed over the last 2 years? "
            "Calculate or extract operating cash flow minus capital expenditures. Discuss FCF "
            "margin trends and conversion from net income.\n\n"

            "## 8. Capital Allocation\n"
            "For what is the company using its free cash flow? Discuss dividends, share "
            "repurchases, M&A, debt repayment, R&D investment, and capital expenditures.\n\n"

            "## 9. Risk Assessment\n"
            "What are the company's main risks and how does it manage them? Cover both "
            "company-specific risks (from Item 1A) and broader industry/macro risks. "
            "Assess how management mitigates each major risk category.\n\n"

            "## 10. Debt & Capital Structure\n"
            "How much debt does the company have and how does it manage its debt? "
            "Discuss total debt, debt maturity profile, interest coverage ratio, credit "
            "ratings, and any off-balance-sheet obligations.\n\n"

            "## 11. Key Assumptions (Bull Case)\n"
            "What assumptions must be true for this investment to work? Identify 3-5 "
            "critical assumptions (e.g., revenue growth rate, margin expansion, market "
            "share gains, macro conditions) and assess the probability of each.\n\n"

            "## 12. Why the Opportunity Exists\n"
            "Why is the opportunity available? Discuss what the market may be "
            "misunderstanding, overreacting to, or ignoring. Consider sentiment, "
            "macro headwinds/tailwinds, and recent 8-K events that may create dislocation.\n\n"

            "## 13. Valuation Assessment\n"
            "What is this business worth, and why is the market pricing it differently? "
            "Compare current valuation multiples (P/E, EV/EBITDA, P/FCF) against historical "
            "averages and peers. Discuss whether the market is assigning a discount or "
            "premium and whether that is justified.\n\n"

            "Conclude with a summary table and actionable investment insights. "
            "Be specific, quantitative, and cite numbers directly from the filings.\n\n"

            "## Investor Readiness Checklist\n\n"
            "You must be able to answer ALL 20 of the following questions after your research. "
            "You are strongly encouraged to explicitly incorporate the answers into your report "
            "wherever relevant. Do not treat these as a separate appendix — weave the evidence "
            "and reasoning directly into the appropriate sections above.\n\n"
            "1. Do I fully understand how this business makes money?\n"
            "2. Is this business within my circle of competence?\n"
            "3. Does the company have a durable competitive advantage (economic moat)?\n"
            "4. Will this business still be relevant and strong in 10–20 years?\n"
            "5. Does the company consistently earn high returns on capital?\n"
            "6. Does the business generate reliable free cash flow?\n"
            "7. Are reported earnings backed by real cash generation?\n"
            "8. Does management act with honesty and integrity?\n"
            "9. Is management skilled at allocating capital?\n"
            "10. Does the company have pricing power?\n"
            "11. Can the business grow without requiring massive ongoing capital expenditures?\n"
            "12. Is the balance sheet strong and debt manageable?\n"
            "13. How resilient are earnings during recessions or downturns?\n"
            "14. Are profit margins durable and defensible?\n"
            "15. Does management think like long-term owners/shareholders?\n"
            "16. Does the company have opportunities to reinvest capital at high returns?\n"
            "17. What are the biggest risks that could permanently impair the business?\n"
            "18. Would I be comfortable owning this company if the stock market closed for 10 years?\n"
            "19. What is the company's intrinsic value?\n"
            "20. Is the stock trading at a meaningful discount to intrinsic value (margin of safety)?"
            + get_language_instruction(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "business_report": report,
        }

    return business_analyst_node
