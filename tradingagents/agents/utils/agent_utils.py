from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.options_greeks_tools import (
    get_option_greeks,
)
from tradingagents.agents.utils.fundamental_data_tools import (
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
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.agents.utils.macro_data_tools import (
    get_cpi_data,
    get_fomc_data,
    get_nonfarm_payrolls_data,
    get_macro_market_data,
    get_fred_economic_data,
    get_oecd_data,
    get_world_bank_data,
    get_ecb_data,
)
from tradingagents.agents.utils.business_data_tools import (
    get_company_profile,
    get_sector_performance,
    get_peer_comparison,
    get_10k_filing,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
