---
description: Compare TradingAgents stock reports in a directory and rank them by investability with portfolio weights.
agent: portfolio-comparison
---

Compare all TradingAgents stock reports found in: $ARGUMENTS

Discover the report directories (handle run_*/batch_*/archive groupings, latest report per ticker wins), extract each 5_portfolio/decision.md, rank the equities by investability, treat ETFs as baselines, and generate the full portfolio comparison report at reports/portfolio_comparison_<scope>_<YYYYMMDD>.md following the agent's report structure. Finish by summarizing the top picks, the avoid list, and where the report file was written.
