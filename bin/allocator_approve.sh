#!/usr/bin/env bash
#
# Human approval gate for a monthly allocator proposal.
#
# Shows the proposal and dry-run plans for paper accounts 2 and 3, asks for
# confirmation, and on approval executes the rebalances. This is the ONLY path
# that runs `rebalance --execute`; the autonomous agent never does. Account 1
# and the live trading account are never touched.
#
# Usage:
#   bin/allocator_approve.sh <YYYY-MM>
#
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Defence in depth: make the live trading account unresolvable. Blanked (not
# unset) because the CLIs load .env with override=False, which would otherwise
# repopulate unset keys.
export ALPACA_TRADING_API_KEY="" ALPACA_TRADING_SECRET_KEY=""  # allow: key

cd "$(dirname "$0")/.." || exit 1

month="${1:?usage: allocator_approve.sh <YYYY-MM>}"
dir=".state/proposals/${month}"

if [ ! -f "${dir}/PROPOSAL.md" ]; then
  echo "error: no proposal at ${dir}/PROPOSAL.md" >&2
  exit 1
fi

for account in 2 3; do
  if [ ! -f "${dir}/weights_${account}.json" ]; then
    echo "error: missing ${dir}/weights_${account}.json" >&2
    exit 1
  fi
done

echo "=============================================================="
echo " Allocator proposal ${month}"
echo "=============================================================="
cat "${dir}/PROPOSAL.md"
echo
echo "--------------------------------------------------------------"
echo " Dry-run plans (nothing submitted yet)"
echo "--------------------------------------------------------------"
for account in 2 3; do
  echo
  echo "### Account ${account}"
  uv run tradingpaperaccount rebalance -a "${account}" \
    -w "${dir}/weights_${account}.json" --json || exit 1
done

echo
printf 'Execute these rebalances on accounts 2 and 3? [y/N]: '
read -r reply
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

case "$reply" in
  y | Y | yes | YES)
    echo "approved - executing" ;;
  *)
    printf '{"ts":"%s","month":"%s","event":"aborted"}\n' "$ts" "$month" \
      >> .state/decisions.jsonl
    echo "aborted: no orders submitted"
    exit 1 ;;
esac

rc=0
for account in 2 3; do
  echo
  echo "### Executing account ${account}"
  uv run tradingpaperaccount rebalance -a "${account}" \
    -w "${dir}/weights_${account}.json" --execute --json || rc=1
done

if [ "$rc" -eq 0 ]; then
  printf '{"ts":"%s","month":"%s","event":"executed","accounts":[2,3]}\n' \
    "$ts" "$month" >> .state/decisions.jsonl
  echo
  echo "done: rebalances submitted for accounts 2 and 3"
  echo "check fills with: bin/check_fills.sh 2 3"
else
  printf '{"ts":"%s","month":"%s","event":"execute_partial_failure","accounts":[2,3]}\n' \
    "$ts" "$month" >> .state/decisions.jsonl
  echo
  echo "error: at least one order failed; inspect the output above" >&2
fi
exit "$rc"
