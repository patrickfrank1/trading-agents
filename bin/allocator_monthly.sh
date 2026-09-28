#!/usr/bin/env bash
#
# Monthly autonomous allocator run: propose rebalances for accounts 2 and 3.
#
# Intended for cron on the first trading day of the month. This only *proposes*
# and notifies; execution happens later and manually via bin/allocator_approve.sh.
# The live trading-account credentials are stripped from the environment.
#
# Usage:
#   bin/allocator_monthly.sh
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
  "Run the MONTHLY allocator procedure from your instructions. Run the refresh gate first; if any held name needs refreshing, stop and report which. Otherwise compare holdings and ideas, produce the account-2 (moderate) and account-3 (high) proposals under .state/proposals/<YYYY-MM>/, dry-run both rebalances, notify, and stop. Never execute trades."
