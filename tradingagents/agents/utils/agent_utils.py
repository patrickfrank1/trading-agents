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
    get_10q_filing,
    get_8k_filing,
    get_20f_filing,
    get_6k_filing,
    get_customer_concentration,
)
from tradingagents.agents.utils.equity_intel_tools import (
    get_analyst_estimates,
    get_credit_and_debt_detail,
    get_short_interest,
    get_institutional_holders,
    get_option_positioning,
    get_earnings_calendar,
    get_capital_allocation_history,
    get_governance,
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


def get_facts_block(state: dict) -> str:
    """Return the canonical facts snapshot for prompt injection.

    Downstream agents (debaters, RM, PM, trader) all read this so they argue
    from one reconciled set of numbers instead of each re-fetching slightly
    different ones. Returns an empty string when no snapshot has been computed
    yet (e.g. when the feature is disabled), so prompts stay well-formed.
    """
    snapshot = state.get("facts_snapshot", "") if state else ""
    if not snapshot:
        return ""
    return f"\n**Canonical Facts Snapshot (single source of truth — cite these numbers, do not re-derive them):**\n{snapshot}\n"


def get_claim_audit_block(state: dict) -> str:
    """Return the post-debate claim audit for decision-agent prompts.

    The fact-check node flags debate claims that are unsupported or
    contradicted by the source analyst reports. Injecting it here lets the
    Research Manager and Portfolio Manager discount rhetorical hot air.
    """
    audit = state.get("claim_audit", "") if state else ""
    if not audit:
        return ""
    return (
        "\n**Claim Audit (debate claims flagged as unsupported or contradicted by "
        "the source reports — discount these when deciding):**\n"
        f"{audit}\n"
    )


def get_reports_digest(state: dict) -> str:
    """Return the Business + Fundamentals analyst reports for decision agents.

    The Research Manager and Portfolio Manager used to see only the debate
    history (a lossy rhetorical compression of these same reports). Giving
    them the raw source reports turns the debate into a *filter* over the
    evidence rather than the sole input, which is what stops the final
    decision from being forced to re-derive the analysts' own conclusion.
    """
    if not state:
        return ""
    business = state.get("business_report", "")
    fundamentals = state.get("fundamentals_report", "")
    parts = []
    if business:
        parts.append(f"**Business Analyst report:**\n{business}")
    if fundamentals:
        parts.append(f"**Fundamentals Analyst report:**\n{fundamentals}")
    return ("\n\n".join(parts) + "\n") if parts else ""


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


        
