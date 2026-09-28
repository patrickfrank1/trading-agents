#!/usr/bin/env bash
#
# Daily autonomous allocator run: research exactly one new investment idea.
#
# Intended for cron. Never trades: the allocator agent is permission-denied from
# `rebalance --execute` and from `mirror`, and this script strips the live
# trading-account credentials from the environment as defence in depth.
#
# Usage:
#   bin/allocator_daily.sh
#
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Defence in depth: make the live trading account unresolvable. Blanked (not
# unset) because the CLIs load .env with override=False, which would otherwise
# repopulate unset keys.
export ALPACA_TRADING_API_KEY="" ALPACA_TRADING_SECRET_KEY=""  # allow: key

cd "$(dirname "$0")/.." || exit 1
mkdir -p .state/logs

exec opencode run --agent value-allocator --auto \
  "Run the DAILY allocator procedure from your instructions: research exactly one new investment idea, persist it to .state, snapshot accounts 1-3, and stop. Never execute trades."
