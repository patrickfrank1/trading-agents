# TradingAgents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config
from tradingagents.agents.utils.tool_errors import tool_error_collector

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_option_greeks,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
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
    get_news,
    get_insider_transactions,
    get_global_news,
    get_cpi_data,
    get_fomc_data,
    get_nonfarm_payrolls_data,
    get_macro_market_data,
    get_fred_economic_data,
    get_oecd_data,
    get_world_bank_data,
    get_ecb_data,
    get_company_profile,
    get_sector_performance,
    get_peer_comparison,
    get_10k_filing,
    get_10q_filing,
    get_8k_filing,
    get_20f_filing,
    get_6k_filing,
    get_customer_concentration,
    get_analyst_estimates,
    get_credit_and_debt_detail,
    get_short_interest,
    get_institutional_holders,
    get_option_positioning,
    get_earnings_calendar,
    get_capital_allocation_history,
    get_governance,
    get_debt_maturity_schedule,
    get_off_balance_sheet_arrangements,
    get_segment_geographic_reporting,
    get_rpo_disaggregation,
    get_risk_factor_changes,
    get_legal_proceedings,
    get_critical_accounting_estimates,
    get_internal_controls,
    get_stock_based_compensation,
    get_goodwill_intangibles,
    get_pension_opeb,
    get_uncertain_tax_positions,
    get_variable_interest_entities,
    get_regulatory_capital,
    get_proved_reserves_mine_safety,
    get_cybersecurity_disclosure,
    get_properties_capacity,
    get_commitments_contingencies,
    get_proxy_governance,
    get_activist_filings,
    get_institutional_13f_filings,
    get_form_8k_events,
    get_insider_form4_activity,
    get_prospectus_disclosure,
)

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
            enable_facts_snapshot=self.config.get("enable_facts_snapshot", True),
            enable_debate_referee=self.config.get("enable_debate_referee", True),
            enable_fact_check=self.config.get("enable_fact_check", True),
            enable_fact_reconciliation=self.config.get("enable_fact_reconciliation", True),
            enable_risk_debate_referee=self.config.get("enable_risk_debate_referee", True),
        )

        # Per-debater LLMs with distinct temperatures so the two sides of each
        # debate are not literally the same model talking to itself. Falls back
        # to the shared quick LLM when the provider rejects temperature or when
        # no per-debater temperatures are configured.
        debate_llms = self._build_debate_llms(llm_kwargs)

        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            debate_llms=debate_llms,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100)
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _build_debate_llms(self, base_llm_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Create per-debater LLM clients with distinct temperatures.

        Diversifying the debaters by temperature (and by the distinct persona
        mandates baked into their prompts) stops each debate from being one
        model arguing with itself in different masks. Returns an empty dict
        when no temperatures are configured, in which case GraphSetup falls
        back to the shared quick-think LLM for every debater.

        Each client is created defensively: if a provider rejects the
        ``temperature`` kwarg, that debater silently falls back to the shared
        quick-think LLM so the pipeline never blocks.
        """
        temperatures = self.config.get("debate_temperatures") or {}
        if not temperatures:
            return {}

        provider = self.config.get("llm_provider", "")
        model = self.config["quick_think_llm"]
        base_url = self.config.get("backend_url")
        debate_llms: Dict[str, Any] = {}
        for key in ("bull", "bear", "aggressive", "conservative", "neutral"):
            temp = temperatures.get(key)
            if temp is None:
                continue
            kwargs = dict(base_llm_kwargs)
            kwargs["temperature"] = temp
            try:
                client = create_llm_client(
                    provider=provider, model=model, base_url=base_url, **kwargs
                )
                debate_llms[key] = client.get_llm()
            except Exception as exc:
                logger.warning(
                    "Could not create %s debater LLM with temperature=%s (%s); "
                    "falling back to shared quick-think LLM",
                    key, temp, exc,
                )
        return debate_llms

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    get_stock_data,
                    get_indicators,
                    get_option_greeks,
                    get_option_positioning,
                    get_short_interest,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    get_institutional_holders,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
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
                    # Equity-intelligence tools that test valuation contradictions
                    get_analyst_estimates,
                    get_credit_and_debt_detail,
                    get_earnings_calendar,
                    get_capital_allocation_history,
                    # SEC-filing footnote signals (financial-statement-level)
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
                ]
            ),
            "macro": ToolNode(
                [
                    # Macroeconomic indicators
                    get_cpi_data,
                    get_fomc_data,
                    get_nonfarm_payrolls_data,
                    # Broad macro market snapshot (Treasury, gold, oil, commodities, housing, breadth)
                    get_macro_market_data,
                    # Institutional macro data vendors
                    get_fred_economic_data,
                    get_oecd_data,
                    get_world_bank_data,
                    get_ecb_data,
                ]
            ),
            "business": ToolNode(
                [
                    # Business model and competitive analysis
                    get_company_profile,
                    get_sector_performance,
                    get_peer_comparison,
                    get_10k_filing,
                    get_10q_filing,
                    get_8k_filing,
                    get_20f_filing,
                    get_6k_filing,
                    get_customer_concentration,
                    get_governance,
                    # SEC-filing qualitative / governance / event signals
                    get_risk_factor_changes,
                    get_legal_proceedings,
                    get_cybersecurity_disclosure,
                    get_properties_capacity,
                    get_proxy_governance,
                    get_activist_filings,
                    get_form_8k_events,
                ]
            ),
        }

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        Returns (raw_return, alpha_return, actual_holding_days) or
        (None, None, None) if price data is unavailable (too recent, delisted,
        or network error).
        """
        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            spy = yf.Ticker("SPY").history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(spy) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(spy) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            spy_ret = float(
                (spy["Close"].iloc[actual_days] - spy["Close"].iloc[0])
                / spy["Close"].iloc[0]
            )
            alpha = raw - spy_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s (will retry next run): %s",
                ticker, trade_date, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(ticker, entry["date"])
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date.

        When ``checkpoint_enabled`` is set in config, the graph is recompiled
        with a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _run_graph(self, company_name, trade_date):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Clear the tool-error collector so each run starts fresh.
        tool_error_collector.clear()

        # Initialize state — inject memory log context for PM.
        past_context = self.memory_log.get_past_context(company_name)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)
            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Drain accumulated tool errors and log a summary.
        tool_errors = tool_error_collector.drain()
        if tool_errors:
            logger.warning(
                "Tool errors during run (%d total): %s",
                len(tool_errors),
                ", ".join(e["tool"] for e in tool_errors),
            )
        final_state.setdefault("tool_errors", tool_errors)
        tool_error_collector.clear()

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "macro_report": final_state["macro_report"],
            "business_report": final_state["business_report"],
            "facts_snapshot": final_state.get("facts_snapshot", ""),
            "claim_audit": final_state.get("claim_audit", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
                "referee_notes": final_state["investment_debate_state"].get(
                    "referee_notes", ""
                ),
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
                "referee_notes": final_state["risk_debate_state"].get(
                    "referee_notes", ""
                ),
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
            "tool_errors": final_state.get("tool_errors", []),
        }

        # Save to file
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
