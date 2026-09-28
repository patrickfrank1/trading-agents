#!/usr/bin/env bash
#
# Monthly autonomous allocator run: propose rebalances for accounts 2 and 3.
#
# Intended for cron over the first few days of the month; it self-limits to the
# first weekday and is idempotent (one proposal per month). This only *proposes*
# and notifies; execution happens manually via bin/allocator_approve.sh. The
# live trading-account credentials are blanked as defence in depth.
#
# Usage:
#   bin/allocator_monthly.sh
#
set -uo pipefail

# cron has a minimal PATH: include the uv and snap (opencode) locations.
export PATH="$HOME/.local/bin:/snap/bin:/usr/local/bin:$PATH"

# Defence in depth: make the live trading account unresolvable. Blanked (not
# unset) because the CLIs load .env with override=False, which would otherwise
# repopulate unset keys.
export ALPACA_TRADING_API_KEY="" ALPACA_TRADING_SECRET_KEY=""  # allow: key

cd "$(dirname "$0")/.." || exit 1
mkdir -p .state/logs

month="$(date +%Y-%m)"

if [ -f ".state/proposals/${month}/PROPOSAL.md" ]; then
  echo "allocator: proposal for ${month} already exists; nothing to do"
  exit 0
fi

# Only run on a weekday; cron fires on days 1-7 so the first weekday wins.
dow="$(date +%u)"
if [ "$dow" -gt 5 ]; then
  echo "allocator: ${month} - weekend; waiting for the first weekday"
  exit 0
fi

opencode run --agent value-allocator --auto \
  "Run the MONTHLY allocator procedure from your instructions. Run the refresh gate first; if any held name needs refreshing, stop and report which. Otherwise compare holdings and ideas, produce the account-2 (moderate) and account-3 (high) proposals under .state/proposals/${month}/, dry-run both rebalances, notify, and stop. Never execute trades."
exit $?
