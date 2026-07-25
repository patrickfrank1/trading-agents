import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 2,
    # The graph now also runs a Facts Snapshot node, a per-round Debate Referee,
    # a post-debate Fact Check, and an optional Fact Reconciliation, so give a
    # little more headroom than the original pipeline needed.
    "max_recur_limit": 200,
    # Adversarial-debate quality controls. These were added to stop the
    # bull/bear and risk debates from devolving into rhetorical restatement:
    #   - enable_facts_snapshot: compute ONE canonical facts block (price,
    #     multiples, debt, FCF, RPO, ...) once after the analysts finish and
    #     inject it into every downstream agent so they argue from the same
    #     numbers instead of each re-fetching slightly different ones.
    #   - enable_debate_referee: a mid-debate referee scores each round for
    #     convergence and can stop the debate early once both sides are only
    #     restating themselves.
    #   - enable_fact_check: after the debate, audit every quantitative /
    #     load-bearing claim against the source analyst reports and flag the
    #     unsupported ones before the Research Manager decides.
    #   - enable_fact_reconciliation: when the fact-check surfaces a material
    #     contradiction that can be resolved by re-querying raw fundamentals
    #     tools, do so and append the reconciled fact before the decision.
    #   - debate_temperatures: per-debater LLM temperatures so the two sides
    #     are not literally the same model talking to itself. Keys: bull,
    #     bear, aggressive, conservative, neutral. Set to None to reuse the
    #     shared quick-think LLM for all debaters.
    "enable_facts_snapshot": True,
    "enable_debate_referee": True,
    "enable_fact_check": True,
    "enable_fact_reconciliation": True,
    "debate_temperatures": {
        "bull": 0.7,
        "bear": 0.3,
        "aggressive": 0.8,
        "conservative": 0.2,
        "neutral": 0.5,
    },
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
        "business_data": "yfinance",          # Options: yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Macro data vendors (environment variables)
    # FRED_API_KEY    — Federal Reserve Economic Data (free)  https://fred.stlouisfed.org/docs/api/api_key.html
    # OECD, World Bank, ECB — no API key required
}
