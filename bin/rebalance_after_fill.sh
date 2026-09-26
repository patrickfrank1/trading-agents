#!/usr/bin/env bash
#
# Rebalance an account to target weights, but only once its open orders have
# filled. Polls the fill check first so it never submits a second order for a
# symbol whose original order is still working (which would double the fill).
#
# Usage:
#   bin/rebalance_after_fill.sh <account> <weights.json> [timeout_minutes]
#
# Exits 0 after a successful rebalance, non-zero if it times out waiting for
# fills or the rebalance fails.
#
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.." || exit 1

account="${1:?usage: rebalance_after_fill.sh <account> <weights.json> [timeout_minutes]}"
weights="${2:?usage: rebalance_after_fill.sh <account> <weights.json> [timeout_minutes]}"
timeout_minutes="${3:-120}"

deadline=$(( $(date +%s) + timeout_minutes * 60 ))

while :; do
  if uv run tradingpaperaccount fill-check -a "$account" >/dev/null 2>&1; then
    echo "account ${account}: no open orders; rebalancing to ${weights}"
    exec uv run tradingpaperaccount rebalance -a "$account" -w "$weights" --execute
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "account ${account}: timed out waiting for orders to fill" >&2
    exit 1
  fi
  sleep 60
done
