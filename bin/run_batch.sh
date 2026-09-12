#!/usr/bin/env bash
#
# bin/run_batch.sh — sequential TradingAgents batch runner (screen-based)
#
# Runs every ticker in a YAML file (default: bin/TICKERS.yaml) one at a time,
# strictly sequentially, inside a single detached `screen` session. You can
# close your terminal while the batch runs.
#
# Reports are written to reports/batch_<YYYYMMDD_HHMMSS>/<TICKER>/ per run.
#
# Usage:
#   bin/run_batch.sh start [options]    start a fresh batch run
#   bin/run_batch.sh status             show progress
#   bin/run_batch.sh attach             attach to the screen session (Ctrl-a d to detach)
#   bin/run_batch.sh stop               kill the batch (current analysis is killed too;
#                                       its --checkpoint data survives for a manual re-run)
#
# Options for start:
#   -f, --file PATH          Tickers YAML file (default: bin/TICKERS.yaml)
#       --provider P         LLM provider        (default: deepseek)
#       --shallow-model M    Quick-thinking model (default: deepseek-flash)
#       --deep-model M       Deep-thinking model (default: deepseek-flash)
#       --research-depth D   shallow|medium|deep (default: deep)
#   -d, --date DATE          Analysis date YYYY-MM-DD (default: none)
#   -a, --analyst NAME       Analyst to include; repeatable (default: CLI defaults)
#   -l, --language LANG      Output language (default: CLI default)
#   -h, --help               Show this help
#
# Examples:
#   bin/run_batch.sh start
#   bin/run_batch.sh start --provider anthropic --shallow-model claude-haiku-4-5 \
#       --deep-model claude-opus-4-6 --anthropic-effort high    # (see note below)
#   bin/run_batch.sh status
#
# Note: flags not listed above (e.g. --anthropic-effort) are not supported by
# this wrapper; run the single-ticker CLI directly for those, or edit the
# DEFAULTS section of this script.
#
# Env overrides:
#   TA_STATE_DIR   state/logs dir   (default: ./.runstate/batch)
#   TA_SESSION     screen session   (default: tabatch)
#   TICKERS_FILE   default tickers file (overridden by -f)

set -euo pipefail

# ---------- defaults ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TICKERS_FILE="${TICKERS_FILE:-$REPO_ROOT/bin/TICKERS.yaml}"
STATE_DIR="${TA_STATE_DIR:-$REPO_ROOT/.runstate/batch}"
SESSION="${TA_SESSION:-tabatch}"

PROVIDER="deepseek"
SHALLOW_MODEL="deepseek-flash"
DEEP_MODEL="deepseek-flash"
RESEARCH_DEPTH="deep"
DATE_OPT=""
LANGUAGE_OPT=""
ANALYSTS=()

BATCH_LOG() { echo "$STATE_DIR/batch.log"; }
STATUS_TSV() { echo "$STATE_DIR/status.tsv"; }

# ---------- helpers ----------
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

session_alive() {
  screen -ls 2>/dev/null | grep -qE "\.${SESSION}\b"
}

yaml_tickers() {
  # print `tickers:` list entries; strips CR, comments, inline values
  awk '
    /^tickers:/ { in_list=1; next }
    in_list && /^[[:space:]]*-/ { sub(/^[[:space:]]*-[[:space:]]*/,""); sub(/[[:space:]]+$/,""); print; next }
    in_list && NF && !/^[[:space:]]*#/ { in_list=0 }
  ' "$1" | tr -d '\r'
}

clog() { # controller log: to screen AND batch.log
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$(BATCH_LOG)" >&2
}

set_status() { # ticker status [rc]
  local t="$1" s="$2" rc="${3:-}" tmp
  tmp="$(mktemp)"
  awk -F'\t' -v t="$t" -v s="$s" -v rc="$rc" 'BEGIN{OFS="\t"} $1==t{$2=s;$3=rc} {print}' \
    "$(STATUS_TSV)" > "$tmp" && mv "$tmp" "$(STATUS_TSV)"
}

# ---------- subcommands ----------
cmd_start() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--file)         TICKERS_FILE="$2"; shift 2 ;;
      --provider)        PROVIDER="$2"; shift 2 ;;
      --shallow-model)   SHALLOW_MODEL="$2"; shift 2 ;;
      --deep-model)      DEEP_MODEL="$2"; shift 2 ;;
      --research-depth)  RESEARCH_DEPTH="$2"; shift 2 ;;
      -d|--date)         DATE_OPT="$2"; shift 2 ;;
      -l|--language)     LANGUAGE_OPT="$2"; shift 2 ;;
      -a|--analyst)      ANALYSTS+=("$2"); shift 2 ;;
      -h|--help)         usage; exit 0 ;;
      *) die "start: unknown option: $1 (see bin/run_batch.sh --help)" ;;
    esac
  done

  session_alive && die "screen session '$SESSION' is already running — check 'bin/run_batch.sh status' or 'stop' first"
  [[ -f "$TICKERS_FILE" ]] || die "tickers file not found: $TICKERS_FILE"

  local tickers_list
  tickers_list="$(yaml_tickers "$TICKERS_FILE")"
  [[ -n "$tickers_list" ]] || die "no tickers found under 'tickers:' in $TICKERS_FILE"

  # fresh run: wipe previous state
  rm -rf "$STATE_DIR"
  mkdir -p "$STATE_DIR/logs"

  local run_dir="$REPO_ROOT/reports/batch_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$run_dir"

  # persist config for the controller (screen does not inherit our shell vars)
  {
    printf "REPO_ROOT=%q\n" "$REPO_ROOT"
    printf "TICKERS_FILE=%q\n" "$TICKERS_FILE"
    printf "RUN_DIR=%q\n" "$run_dir"
    printf "PROVIDER=%q\n" "$PROVIDER"
    printf "SHALLOW_MODEL=%q\n" "$SHALLOW_MODEL"
    printf "DEEP_MODEL=%q\n" "$DEEP_MODEL"
    printf "RESEARCH_DEPTH=%q\n" "$RESEARCH_DEPTH"
    printf "DATE_OPT=%q\n" "$DATE_OPT"
    printf "LANGUAGE_OPT=%q\n" "$LANGUAGE_OPT"
    if [[ ${#ANALYSTS[@]} -gt 0 ]]; then
      printf "ANALYSTS=(%s)\n" "$(printf '%q ' "${ANALYSTS[@]}")"
    else
      printf "ANALYSTS=()\n"
    fi
  } > "$STATE_DIR/run.conf"

  printf '%s\n' "$tickers_list" > "$STATE_DIR/tickers.txt"
  awk '{printf "%s\tpending\t\n", $0}' "$STATE_DIR/tickers.txt" > "$(STATUS_TSV)"

  local n; n="$(wc -l < "$STATE_DIR/tickers.txt")"
  screen -dmS "$SESSION" bash "$0" _controller

  sleep 1
  if ! session_alive; then
    die "failed to start screen session '$SESSION' (check $STATE_DIR/batch.log)"
  fi

  printf 'Batch started: %s ticker(s) from %s\n' "$n" "$TICKERS_FILE"
  printf '  screen session : %s (attach: bin/run_batch.sh attach)\n' "$SESSION"
  printf '  reports        : %s\n' "$run_dir"
  printf '  status         : bin/run_batch.sh status\n'
  printf '  live log       : tail -f %s\n' "$(BATCH_LOG)"
}

cmd__controller() {
  [[ -f "$STATE_DIR/run.conf" ]] || die "no run.conf in $STATE_DIR — use 'bin/run_batch.sh start' first"
  # shellcheck disable=SC1090
  source "$STATE_DIR/run.conf"
  cd "$REPO_ROOT"

  : > "$(BATCH_LOG)"
  clog "=== batch run started ==="
  clog "tickers_file=$TICKERS_FILE  reports=$RUN_DIR"
  clog "provider=$PROVIDER shallow=$SHALLOW_MODEL deep=$DEEP_MODEL depth=$RESEARCH_DEPTH"

  local total=0 n=0 done_ct=0 failed_ct=0 rc t tlog
  total="$(grep -c . "$STATE_DIR/tickers.txt" || true)"

  while IFS= read -r t; do
    [[ -z "$t" ]] && continue
    n=$((n + 1))
    set_status "$t" running
    tlog="$STATE_DIR/logs/$t.log"

    local cmd=(
      uv run tradingagents
      --refresh-rate 0.1
      --non-interactive
      --checkpoint
      --display-report
      --save
      --ticker "$t"
      --save-path "$RUN_DIR/$t"
      --research-depth "$RESEARCH_DEPTH"
      --provider "$PROVIDER"
      --shallow-model "$SHALLOW_MODEL"
      --deep-model "$DEEP_MODEL"
    )
    [[ -n "$DATE_OPT" ]] && cmd+=(--date "$DATE_OPT")
    [[ -n "$LANGUAGE_OPT" ]] && cmd+=(--language "$LANGUAGE_OPT")
    local a
    if [[ ${#ANALYSTS[@]} -gt 0 ]]; then
      for a in "${ANALYSTS[@]}"; do cmd+=(--analyst "$a"); done
    fi

    clog "[$n/$total] $t: starting"
    printf '%q ' "${cmd[@]}" > "$STATE_DIR/last_cmd.txt"; echo >> "$STATE_DIR/last_cmd.txt"

    # run synchronously: wait for completion before the next ticker
    # (if-condition suppresses errexit so a failing ticker doesn't kill the batch)
    if "${cmd[@]}" 2>&1 | tee "$tlog"; then
      rc=0
    else
      rc=${PIPESTATUS[0]}
    fi

    if [[ "$rc" -eq 0 ]]; then
      set_status "$t" done 0
      done_ct=$((done_ct + 1))
      clog "[$n/$total] $t: done (rc=0)"
    else
      set_status "$t" failed "$rc"
      failed_ct=$((failed_ct + 1))
      clog "[$n/$total] $t: FAILED (rc=$rc) — continuing with next ticker"
    fi
  done < "$STATE_DIR/tickers.txt"

  clog "=== batch finished: total=$total done=$done_ct failed=$failed_ct ==="
  clog "reports: $RUN_DIR"
}

cmd_status() {
  if session_alive; then
    echo "session : RUNNING (screen '$SESSION')"
  else
    echo "session : STOPPED"
  fi

  if [[ ! -f "$(STATUS_TSV)" ]]; then
    echo "state   : no batch run found in $STATE_DIR (use 'bin/run_batch.sh start')"
    return 0
  fi
  # shellcheck disable=SC1090
  [[ -f "$STATE_DIR/run.conf" ]] && source "$STATE_DIR/run.conf"
  [[ -n "${RUN_DIR:-}" ]] && echo "reports : $RUN_DIR"

  local done_ct failed_ct running pending total
  total=$(grep -c . "$(STATUS_TSV)")
  done_ct=$(awk -F'\t' '$2=="done"{c++} END{print c+0}' "$(STATUS_TSV)")
  failed_ct=$(awk -F'\t' '$2=="failed"{c++} END{print c+0}' "$(STATUS_TSV)")
  running=$(awk -F'\t' '$2=="running"{print $1}' "$(STATUS_TSV)")
  pending=$(awk -F'\t' '$2=="pending"{c++} END{print c+0}' "$(STATUS_TSV)")

  echo
  printf '%-10s %5s\n' STATUS COUNT
  printf '%-10s %5s\n' done "$done_ct"
  printf '%-10s %5s\n' failed "$failed_ct"
  printf '%-10s %5s\n' pending "$pending"
  printf '%-10s %5s\n' total "$total"

  [[ -n "$running" ]] && echo "current  : $running" || true

  if [[ "$failed_ct" -gt 0 ]]; then
    echo "failed   :"
    awk -F'\t' '$2=="failed"{printf "  %s (rc=%s)\n", $1, $3}' "$(STATUS_TSV)"
  fi

  echo
  echo "live log : tail -f $(BATCH_LOG)"
  echo "per-tkr  : tail -f $STATE_DIR/logs/<TICKER>.log"
}

cmd_stop() {
  if ! session_alive; then
    echo "session '$SESSION' is not running."
    return 0
  fi
  screen -S "$SESSION" -X quit 2>/dev/null || true
  echo "batch stopped."
  echo "The analysis that was in flight was killed; its --checkpoint data survives,"
  echo "so you can re-run that single ticker manually to resume it."
}

usage() { sed -n '2,32p' "$0"; }

# ---------- dispatch ----------
case "${1:-}" in
  start)       shift; cmd_start "$@" ;;
  _controller) cmd__controller ;;
  status)      cmd_status ;;
  attach)      exec screen -r "$SESSION" ;;
  stop)        cmd_stop ;;
  ""|-h|--help|help) usage ;;
  *) die "unknown subcommand: $1 (see bin/run_batch.sh --help)" ;;
esac
