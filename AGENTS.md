# TradingAgents

Multi-agent LLM financial trading framework (v0.2.4). Agents analyze stocks through a LangGraph pipeline: Analysts → Researchers → Trader → Risk Management → Portfolio Manager.

## Commands

```bash
uv pip install -e .              # install editable
uv run python -m cli.main        # run CLI (interactive)
uv run tradingagents              # same, after install

# non-interactive (fully automated, no prompts)
uv run tradingagents analyze --non-interactive

# specify options via flags
uv run tradingagents analyze --non-interactive \
  -t AAPL -d 2025-04-01 \
  -a market -a news -a fundamentals \
  -p anthropic --shallow-model claude-haiku-4-5 --deep-model claude-opus-4-6 \
  --anthropic-effort high --display-report

# checkpoint / recovery
uv run tradingagents analyze --checkpoint         # enable crash-resume
uv run tradingagents analyze --clear-checkpoints  # wipe saved checkpoints

# tests
uv run pytest                      # all tests
uv run pytest -m unit              # fast, no API calls needed
uv run pytest -m smoke             # sanity checks
uv run pytest tests/test_model_validation.py  # single file

# smoke test structured-output against a real provider (costs tokens)
OPENAI_API_KEY=... uv run python scripts/smoke_structured_output.py openai

# docker
docker compose run --rm tradingagents
```

## opencode agents / commands

- `/compare-stocks <reports-dir>` — runs the `portfolio-comparison` agent (`.opencode/agent/portfolio-comparison.md`): reads `5_portfolio/decision.md` from every report dir under the given directory (handles `run_*`/`batch_*`/`archive` groupings, latest report per ticker wins), ranks equities by investability, treats ETFs as baselines, sizes positions via a fractional-Kelly criterion from each stock's scenario table (trend/probability-of-gain dominates, expected return scales magnitude), and writes `reports/portfolio_comparison_<scope>_<YYYYMMDD>.md` with buy/sell answers, Kelly-derived NAV weightings, top-2 risks/opportunities, and 1Y/5Y expected returns per stock.

No linter, typechecker, or formatter is configured in the project.

- **LLM-friendly docs**: if the `context7` MCP server is available, additional documentation for this project lives under the `tradingagents` context7 source. Use it when you need deeper reference material on agents, schemas, or the LangGraph pipeline.

## Architecture

```
tradingagents/
  graph/           LangGraph orchestration: trading_graph.py is the main entry
  agents/          Agent implementations
    analysts/      Market (entry-timing gate), News, Fundamentals, Macro, Business, Sector Specialist (auto-selected by sector/asset class) — produce reports
    researchers/   Bull/Bear debate
    managers/      Research Manager, Portfolio Manager — produce structured decisions
    trader/        Trader — produces transaction proposals
    risk_mgmt/     Aggressive, Conservative, Neutral debate
    schemas.py     Pydantic schemas for the 3 decision agents (ResearchPlan, TraderProposal, PortfolioDecision)
    utils/         agent_states, agent_utils, memory, rating, structured output helpers
  dataflows/       Data vendor abstraction (yfinance default, alpha_vantage optional)
  llm_clients/     LLM provider factory: openai_client covers openai/xai/deepseek/qwen/glm/ollama/openrouter
  default_config.py All config keys and defaults
cli/               Typer CLI app (entrypoint: cli.main:app)
tests/             pytest (conftest.py auto-injects placeholder API keys)
```

## Key details

- **Config**: all options live in `tradingagents/default_config.py`. The `data_vendors` dict routes data calls; `tool_vendors` overrides per-tool. Set `ALPHA_VANTAGE_API_KEY` to use that vendor (yfinance works with no keys).
- **Structured output**: Research Manager, Trader, and Portfolio Manager return typed Pydantic instances via `llm.with_structured_output()`. Each provider uses its native mode (json_schema, response_schema, tool-use). `render_*()` helpers convert back to markdown for the rest of the system.
- **Rating scale**: 5-tier (Buy/Overweight/Hold/Underweight/Sell) for Research Manager and PM; 3-tier (Buy/Hold/Sell) for Trader. Parsed deterministically — no extra LLM call.
- **Signal extraction**: `SignalProcessor.process_signal()` is now a pure heuristic in `agents/utils/rating.py`; the LLM arg is accepted but ignored for backward compatibility.
- **Memory log**: auto-enabled. Decisions persist to `~/.tradingagents/memory/trading_memory.md`. Override with `TRADINGAGENTS_MEMORY_LOG_PATH`.
- **Causal Bayesian macro model**: quarterly VARX(1) over gold/SPX/bonds/REITs (stability-constrained, sign-informed priors, Student-t), built on the versioned quarterly panel from `tradingagents/dataflows/macro_timeseries.py` (full-history FRED + yfinance, publication-lag look-ahead guard, Parquet cache). Fitted offline: `uv run python scripts/fit_macro_model.py [--model v1|joint|all] [--validate]` (needs `FRED_API_KEY` + the optional `model` extra: `uv pip install -e ".[model]"`). The macro analyst consumes it via `get_macro_causal_forecast` (NumPy-only at runtime). Design + milestones: `assets/bayesian_causal_model_implementation_plan.md`.
- **Checkpoint resume**: opt-in, per-ticker SQLite at `~/.tradingagents/cache/checkpoints/<TICKER>.db`. Override base with `TRADINGAGENTS_CACHE_DIR`.
- **LLM factory**: lazy imports so test collection doesn't pull in heavy SDKs. Provider string is lowercased and matched; `_OPENAI_COMPATIBLE` tuple in `factory.py` lists which providers share the OpenAI client.
- **Environment**: `.env` and `.env.enterprise` are loaded by `cli/main.py` via dotenv. Enterprise keys (Azure) go in `.env.enterprise`.
- **Tests**: conftest.py auto-sets placeholder API keys so tests don't hang waiting for input. `mock_llm_client` fixture patches the factory. Markers: `unit`, `integration`, `smoke`.
