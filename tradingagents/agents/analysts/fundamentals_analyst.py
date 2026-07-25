from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    compute_dcf_analysis,
    compute_comps_analysis,
    compute_precedent_transactions,
    compute_asset_based_valuation,
    compute_ddm_valuation,
    compute_residual_income_valuation,
    compute_lbo_analysis,
    compute_vc_valuation,
    compute_epv_valuation,
    compute_sotp_valuation,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
    get_report_hygiene_instruction,
    get_analyst_estimates,
    get_credit_and_debt_detail,
    get_earnings_calendar,
    get_capital_allocation_history,
    get_debt_maturity_schedule,
    get_off_balance_sheet_arrangements,
    get_segment_geographic_reporting,
    get_rpo_disaggregation,
    get_critical_accounting_estimates,
    get_internal_controls,
    get_stock_based_compensation,
    get_goodwill_intangibles,
    get_pension_opeb,
    get_uncertain_tax_positions,
    get_variable_interest_entities,
    get_regulatory_capital,
    get_commitments_contingencies,
    get_proved_reserves_mine_safety,
    get_institutional_13f_filings,
    get_insider_form4_activity,
    get_prospectus_disclosure,
)
from tradingagents.dataflows.config import get_config


VALUATION_METHODS_GUIDE = """
You have access to 10 valuation tools. You MUST select between 2 and 4 valuation methods (no more, no fewer).
The `compute_dcf_analysis` tool is MANDATORY — you must always include it.

Choose 1–3 additional methods from the list below based on the company's profile:

| Method | Tool | Best For |
|---|---|---|
| Comps | `compute_comps_analysis` | Public companies with good market multiples |
| Precedent Transactions | `compute_precedent_transactions` | Acquisition targets, active M&A sectors |
| Asset-Based | `compute_asset_based_valuation` | Asset-heavy firms (banks, REITs, holding companies) |
| DDM | `compute_ddm_valuation` | Mature, stable dividend-paying stocks |
| Residual Income | `compute_residual_income_valuation` | Banks, financials, book-value-driven firms |
| LBO | `compute_lbo_analysis` | Stable cash-flow firms suitable for buyouts |
| VC Method | `compute_vc_valuation` | High-growth / pre-profit companies |
| EPV | `compute_epv_valuation` | Value investing — margin of safety analysis |
| SOTP | `compute_sotp_valuation` | Conglomerates, multi-segment companies |

Use the company's sector, dividend policy, growth profile, and financial structure to pick the most appropriate methods.
After running each valuation tool, synthesize the results: compare fair value estimates, identify convergence/divergence, and provide an overall valuation assessment.
"""


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        valuation_tools = [
            compute_dcf_analysis,
            compute_comps_analysis,
            compute_precedent_transactions,
            compute_asset_based_valuation,
            compute_ddm_valuation,
            compute_residual_income_valuation,
            compute_lbo_analysis,
            compute_vc_valuation,
            compute_epv_valuation,
            compute_sotp_valuation,
        ]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_insider_transactions,
            get_analyst_estimates,
            get_credit_and_debt_detail,
            get_earnings_calendar,
            get_capital_allocation_history,
            get_debt_maturity_schedule,
            get_off_balance_sheet_arrangements,
            get_segment_geographic_reporting,
            get_rpo_disaggregation,
            get_critical_accounting_estimates,
            get_internal_controls,
            get_stock_based_compensation,
            get_goodwill_intangibles,
            get_pension_opeb,
            get_uncertain_tax_positions,
            get_variable_interest_entities,
            get_regulatory_capital,
            get_commitments_contingencies,
            get_proved_reserves_mine_safety,
            get_institutional_13f_filings,
            get_insider_form4_activity,
            get_prospectus_disclosure,
        ] + valuation_tools

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + VALUATION_METHODS_GUIDE
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + " Also call `get_analyst_estimates` to capture sell-side consensus EPS/revenue estimates and price targets — use these to test whether valuation multiples (e.g. forward P/E) are consistent with the consensus EPS trajectory."
            + " Call `get_credit_and_debt_detail` to report total debt, the long/short-term split, and interest coverage — quote these figures directly so downstream debt/refinancing claims can be verified."
            + " Call `get_earnings_calendar` for the next earnings date and recent beat/miss history, and `get_capital_allocation_history` for the multi-year buyback/dividend/share-count record."
            + " Pull SEC-filing footnote signals that resolve common debate threads: `get_debt_maturity_schedule` (year-by-year debt maturities — kills refinancing-risk assertions), `get_off_balance_sheet_arrangements` (leases, guarantees, VIE commitments), `get_rpo_disaggregation` (remaining performance obligations split short-term vs long-term — resolves order-book composition debates), `get_segment_geographic_reporting` (segment/geography revenue and income), `get_critical_accounting_estimates` and `get_internal_controls` (management-flagged estimate uncertainty and any material weakness), `get_stock_based_compensation` (SBC cost and dilution overhang), `get_goodwill_intangibles`, `get_pension_opeb`, `get_uncertain_tax_positions`, `get_variable_interest_entities`, `get_regulatory_capital` (banks/insurers), `get_commitments_contingencies`, `get_proved_reserves_mine_safety` (E&P/miners)."
            + " Use `get_institutional_13f_filings` and `get_insider_form4_activity` for raw ownership/insider signals, and `get_prospectus_disclosure` if the company recently filed an S-3/424B (dilution/use of proceeds)."
            + " Not every tool applies to every company — pick the filing signals relevant to this company's sector and capital structure."
            + "\n\n**Valuation hygiene (mandatory):** When a valuation engine prints a ⚠️ QA/sanity warning (e.g. comp-set outlier flag, extreme DCF fair value, >20x spread), surface that warning in your report and treat the flagged output as a bounded estimate, not a precise value. Do NOT silently include a comp-set median you have been told is distorted by hyper-growth peers, and do NOT present a DCF fair value implying >80% downside as if it were settled intrinsic value. Cross-check flagged outputs against EPV, comps, and the business case."
            + get_language_instruction()
            + get_report_hygiene_instruction(),
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
