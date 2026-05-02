from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.macro_data_tools import (
    get_cpi_data,
    get_fomc_data,
    get_nonfarm_payrolls_data,
)


def create_macro_analyst(llm):

    def macro_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_cpi_data,
            get_fomc_data,
            get_nonfarm_payrolls_data,
        ]

        system_message = (
            "You are a Macroeconomic Analyst tasked with analyzing the current macroeconomic environment and its implications for trading and investment decisions. Your role is to gather and synthesize data on key economic indicators and provide a comprehensive macro outlook.\n\n"
            "Use the following tools to gather macroeconomic data:\n"
            "- get_cpi_data(curr_date, look_back_days, limit): Retrieve Consumer Price Index (CPI) data and inflation-related news. CPI measures the average change in prices paid by urban consumers and is a key gauge of inflation. Analyze current inflation trends and their potential impact on monetary policy.\n"
            "- get_fomc_data(curr_date, look_back_days, limit): Retrieve Federal Open Market Committee (FOMC) data and monetary policy news. The FOMC sets the federal funds rate and conducts monetary policy. Analyze recent FOMC decisions, statements, and forward guidance to assess the current and expected monetary policy stance.\n"
            "- get_nonfarm_payrolls_data(curr_date, look_back_days, limit): Retrieve Non-farm Payrolls (NFP) data and labor market news. NFP measures monthly employment changes and is a critical indicator of economic health. Analyze employment trends and their implications for economic growth and policy.\n\n"
            "In your report, please address:\n"
            "1. **Current Inflation Environment**: Analyze CPI trends — is inflation accelerating, decelerating, or stable? What does this mean for purchasing power and consumer spending?\n"
            "2. **Monetary Policy Stance**: Summarize the latest FOMC decisions and forward guidance. Is the Fed hawkish, dovish, or neutral? What are the implications for interest rates and liquidity?\n"
            "3. **Labor Market Health**: Assess NFP trends — is job growth strong, weakening, or contracting? What does this suggest about overall economic momentum?\n"
            "4. **Macroeconomic Outlook**: Synthesize the above indicators into a coherent macro outlook. Consider how these factors interact (e.g., strong jobs + rising inflation may prompt hawkish Fed response).\n"
            "5. **Investment Implications**: Based on your macro analysis, discuss which asset classes, sectors, or investment styles may benefit or suffer in the current environment. Consider interest rate sensitivity, growth vs. value dynamics, and risk appetite.\n"
            "6. **Risks to the Outlook**: Identify key risks that could shift the macro landscape (e.g., unexpected inflation data, geopolitical events, policy surprises).\n\n"
            "Call each tool at least once to gather data before writing your report. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + get_language_instruction()
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
            "macro_report": report,
        }

    return macro_analyst_node
