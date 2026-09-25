#!/usr/bin/env bash
#
# Post-open fill check for Alpaca paper accounts.
#
# Reports, per account, whether any submitted orders are still open (unfilled)
# and what positions resulted. Exits non-zero if anything is still open, so it
# can be used as a cron/alert hook after the market opens.
#
# Usage:
#   bin/check_fills.sh              # accounts 1 and 2 (default)
#   bin/check_fills.sh 1 2 3        # explicit accounts
#   bin/check_fills.sh --json 1 2   # machine-readable output
#
set -uo pipefail

# Ensure uv (usually in ~/.local/bin) is on PATH when run without a login shell,
# e.g. from cron.
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.." || exit 1

json=""
accounts=()
for arg in "$@"; do
  case "$arg" in
    --json) json="--json" ;;
    *) accounts+=("$arg") ;;
  esac
done

if [ ${#accounts[@]} -eq 0 ]; then
  accounts=(1 2)
fi

cmd=(uv run tradingpaperaccount fill-check)
if [ -n "$json" ]; then
  cmd+=("$json")
fi
for acct in "${accounts[@]}"; do
  cmd+=(-a "$acct")
done

exec "${cmd[@]}"
