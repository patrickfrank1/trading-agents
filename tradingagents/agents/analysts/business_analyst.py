from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_company_profile,
    get_sector_performance,
    get_peer_comparison,
    get_10k_filing,
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
        ]

        system_message = (
            "You are a senior business analyst tasked with evaluating the business model, "
            "competitive moat, and scenario resilience of a company. Write a comprehensive "
            "business analysis report covering the following dimensions:\n\n"
            "1. **Business Model Analysis**: Describe how the company generates revenue, "
            "its cost structure, pricing power, customer base, and key value proposition. "
            "Explain the unit economics if applicable.\n\n"
            "2. **Competitive Moat Assessment**: Evaluate whether the company has a durable "
            "competitive advantage. Consider network effects, switching costs, brand strength, "
            "patents/intellectual property, economies of scale, regulatory advantages, "
            "data advantages, and cost leadership. Rate the moat strength (None / Weak / "
            "Moderate / Strong / Very Strong) with justification.\n\n"
            "3. **Scenario Resilience Analysis**: Evaluate how robust the business model is "
            "under three distinct macroeconomic scenarios:\n"
            "   - **Status Quo**: Current conditions persist (moderate growth, stable rates, "
            "normal inflation). How does the company perform?\n"
            "   - **Inflation**: Persistent elevated inflation (5-8% CPI) with rising input "
            "costs and interest rates. Can the company pass through cost increases? How are "
            "margins affected? Does pricing power help or hurt?\n"
            "   - **Stagflation**: High inflation combined with low or negative growth, rising "
            "unemployment, and tightening credit. How does reduced consumer/business spending "
            "impact this company? Are its products/services essential or discretionary? How "
            "strong is its balance sheet to weather a prolonged downturn?\n\n"
            "4. **Sector Performance Comparison**: Compare the company's financial metrics "
            "(revenue growth, margins, ROE, valuation multiples) against its sector/industry "
            "peers. Is it an outperformer, in-line, or underperformer relative to its sector?\n\n"
            "Use the available tools: `get_company_profile` for detailed company business "
            "information, `get_sector_performance` to compare against sector ETF benchmarks, "
            "`get_peer_comparison` for peer-level financial metrics, and `get_10k_filing` "
            "to retrieve the company's latest 10-K/20-F annual report from SEC EDGAR "
            "(Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, etc.).\n"
            "Make sure to append a Markdown table at the end of the report summarizing key "
            "findings across all dimensions.\n"
            "Provide specific, actionable insights with supporting evidence to help traders "
            "make informed decisions."
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
