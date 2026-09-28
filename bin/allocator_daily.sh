#!/usr/bin/env bash
#
# Daily autonomous allocator run: research exactly one new investment idea.
#
# Intended for cron. Never trades: the allocator agent is permission-denied from
# `rebalance --execute` and from `mirror`, and this script blanks the live
# trading-account credentials as defence in depth.
#
# Set ALLOCATOR_DRY_RUN=1 to exercise the plumbing without generating a report:
# `uv run tradingagents` is shadowed by a stub that refuses to run.
#
# Usage:
#   bin/allocator_daily.sh
#   ALLOCATOR_DRY_RUN=1 bin/allocator_daily.sh
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

today="$(date +%Y%m%d)"
if [ -f ".state/logs/daily-${today}.done" ]; then
  echo "allocator: daily run already completed for ${today}; nothing to do"
  exit 0
fi

if [ "${ALLOCATOR_DRY_RUN:-0}" = "1" ]; then
  export ALLOCATOR_DRY_RUN=1
  real_uv="$(command -v uv)"
  shim_dir="$(mktemp -d)"
  cat >"${shim_dir}/uv" <<EOF
#!/usr/bin/env bash
if [ "\$1" = "run" ] && [ "\$2" = "tradingagents" ]; then
  echo "DRY RUN: blocked 'uv run tradingagents \$*'" >&2
  exit 0
fi
exec "${real_uv}" "\$@"
EOF
  chmod +x "${shim_dir}/uv"
  export PATH="${shim_dir}:${PATH}"
  echo "allocator: DRY RUN - tradingagents analyze is blocked"
fi

opencode run --agent value-allocator --auto \
  "Run the DAILY allocator procedure from your instructions: research exactly one new investment idea, persist it to .state, snapshot accounts 1-3, and stop. Never execute trades."
rc=$?

if [ "$rc" -eq 0 ] && [ "${ALLOCATOR_DRY_RUN:-0}" != "1" ]; then
  touch ".state/logs/daily-${today}.done"
fi
exit "$rc"
