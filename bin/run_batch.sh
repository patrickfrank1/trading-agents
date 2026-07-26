#!/usr/bin/env bash
#
# bin/run_batch.sh — parallel TradingAgents batch runner over bin/TICKERS.yaml
#
# Uses `screen` (detached windows) for parallelization. One controller loop
# (also in a detached screen session) launches up to -N concurrent per-ticker
# screen windows. State is persisted to disk so you can pause, resume, stop,
# and resume after the script or the machine has been killed.
#
# Usage:
#   bin/run_batch.sh start  [-j N] [-f TICKERS.yaml] [-- ANALYSIS_ARGS...]
#   bin/run_batch.sh pause
#   bin/run_batch.sh resume
#   bin/run_batch.sh status
#   bin/run_batch.sh stop
#   bin/run_batch.sh reset [--failed|--all]
#   bin/run_batch.sh logs TICKER
#   bin/run_batch.sh attach          # attach to the controller session
#
# Examples:
#   # default command (the canonical analysis invocation) with 4 parallel jobs
#   bin/run_batch.sh start -j 4
#
#   # override provider/models — everything after `--` replaces the default
#   # per-ticker command (the %%TICKER%% placeholder is substituted)
#   bin/run_batch.sh start -j 8 -- \
#       uv run tradingagents --non-interactive --checkpoint --ticker %%TICKER%% \
#           --provider openai --deep-model gpt-4o
#
# Env:
#   TA_STATE_DIR   where state/logs live (default: ./.runstate/batch)
#   TA_SESSION     screen session name   (default: tabatch)
#
# Notes:
#   - The per-ticker command MUST contain the literal %%TICKER%% placeholder;
#     it is replaced with the ticker symbol when each job is launched. The
#     ticker is NOT auto-appended, so a custom command without %%TICKER%%
#     is rejected.
#   - Completion is detected via a sentinel file written by the wrapper, not
#     by polling screen exit codes, so state survives controller restarts and
#     machine crashes (combined with --checkpoint, killed analyses resume).
#   - After `stop` (or a crash), just run `start` again with no args: it
#     reuses the persisted -j/-f/command, re-queues any tickers that were
#     mid-run when killed, and continues. Pass new -j/-f/-- to override.

set -euo pipefail

# ---------- defaults ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TICKERS_FILE="${TICKERS_FILE:-$REPO_ROOT/bin/TICKERS.yaml}"
STATE_DIR="${TA_STATE_DIR:-$REPO_ROOT/.runstate/batch}"
SESSION="${TA_SESSION:-tabatch}"
CONCURRENCY="${TA_CONCURRENCY:-4}"

mkdir -p "$STATE_DIR/logs"

# canonical per-ticker command (matches the user's requested invocation).
# %%TICKER%% is substituted per ticker.
DEFAULT_CMD=(
  uv run tradingagents
  --refresh-rate 0.1
  --non-interactive
  --checkpoint
  --display-report
  --save
  --ticker %%TICKER%%
  --research-depth deep
  --provider deepseek
  --shallow-model deepseek-v4-pro
  --deep-model deepseek-v4-pro
)

# ---------- helpers ----------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { log "ERROR: $*"; exit 1; }

yaml_tickers() {
  # extract the `tickers:` list entries; robust to leading `- ` / ` - `
  awk '
    /^tickers:/ { in_list=1; next }
    in_list && /^[[:space:]]*-/ { sub(/^[[:space:]]*-[[:space:]]*/,""); print; next }
    in_list && NF && !/^[[:space:]]*#/ { in_list=0 }
  ' "$1"
}

state_file() { echo "$STATE_DIR/state.tsv"; }
pause_file() { echo "$STATE_DIR/PAUSE"; }
stop_file()  { echo "$STATE_DIR/STOP"; }

# state.tsv columns: TICKER \t STATUS \t START_TS \t END_TS
# STATUS ∈ pending | running | done | failed
state_init() {
  local f; f="$(state_file)"
  [[ -f "$f" ]] && return 0
  : > "$f"
  local t
  for t in $(yaml_tickers "$TICKERS_FILE"); do
    printf '%s\tpending\t\t\n' "$t" >> "$f"
  done
  log "initialized state with $(wc -l < "$f") tickers from $TICKERS_FILE"
}

state_get() { # ticker -> status
  awk -F'\t' -v t="$1" '$1==t{print $2; exit}' "$(state_file)"
}
state_set() { # ticker status
  local t="$1" s="$2" tmp
  tmp="$(mktemp)"
  awk -F'\t' -v t="$t" -v s="$s" -v now="$(date +%s)" '
    BEGIN{OFS="\t"}
    $1==t {
      $2=s
      if (s=="running") $3=now; else if (s=="done"||s=="failed") $4=now
      print; next
    }
    {print}
  ' "$(state_file)" > "$tmp" && mv "$tmp" "$(state_file)"
}

count_status() { awk -F'\t' -v s="$1" '$2==s{n++} END{print n+0}' "$(state_file)"; }

# list of window titles in the controller session (one per line, filtered)
session_windows() {
  screen -S "$SESSION" -Q windows 2>/dev/null | tr ' ' '\n' | grep -v '^[0-9*]*$' || true
}

# how many per-ticker screen windows are currently alive
running_windows() {
  session_windows | grep -c "^${SESSION}_ta_" || true
}

# is a specific ticker's window still alive? (prefix match — screen may append
# flag chars to the title)
window_alive() {
  local win; win="$(win_name "$1")"
  session_windows | grep -q "^${win}"
}

# is the controller screen session alive?
controller_alive() {
  screen -ls 2>/dev/null | grep -qE "\.${SESSION}\b" || return 1
}

# per-ticker screen window name
win_name() { echo "${SESSION}_ta_$(echo "$1" | tr -c 'A-Za-z0-9._-' '_')"; }

# sentinel files for completion detection
sent_done()   { echo "$STATE_DIR/done/$1"; }
sent_failed() { echo "$STATE_DIR/failed/$1"; }

# template script (with %%TICKER%% placeholder) and per-ticker job script
template_file() { echo "$STATE_DIR/cmd_template.sh"; }
job_script()    { echo "$STATE_DIR/jobs/$1.sh"; }
wrapper_script(){ echo "$STATE_DIR/wrapper.sh"; }

# Write the wrapper script once (takes ticker as $1). Robust vs. screen's
# arg-splitting because it's a file, not a `bash -c` string. STATE_DIR and
# REPO_ROOT are EMBEDDED at generation time, because new screen windows do NOT
# inherit the controller process's exported environment.
write_wrapper() {
  cat > "$(wrapper_script)" <<WRAP
#!/usr/bin/env bash
set -u
ticker="\$1"
state="$STATE_DIR"
repo="$REPO_ROOT"
job="\$state/jobs/\${ticker}.sh"
log="\$state/logs/\${ticker}.log"
mkdir -p "\$state/done" "\$state/failed" "\$state/logs"
rm -f "\$state/done/\$ticker" "\$state/failed/\$ticker"
cd "\$repo" || { echo 127 > "\$state/failed/\$ticker"; exit 0; }
bash -- "\$job" > "\$log" 2>&1
rc=\$?
if [ "\$rc" -eq 0 ]; then
  touch "\$state/done/\$ticker"
else
  echo "\$rc" > "\$state/failed/\$ticker"
fi
WRAP
  chmod +x "$(wrapper_script)"
}

# Render the per-ticker job script from the template by substituting %%TICKER%%.
# Done as a file (not a string) so the user's quoting/`;`/`$$` survive intact.
render_job_script() {
  local ticker="$1" tpl job
  tpl="$(template_file)"; job="$(job_script "$ticker")"
  mkdir -p "$(dirname "$job")"
  awk -v t="$ticker" '{gsub(/%%TICKER%%/, t); print}' "$tpl" > "$job"
}

launch_ticker() {
  local ticker="$1"
  local win; win="$(win_name "$ticker")"
  render_job_script "$ticker"
  screen -S "$SESSION" -X screen -t "$win" bash "$(wrapper_script)" "$ticker" \
    || die "failed to spawn screen window for $ticker (is the controller session alive?)"
  state_set "$ticker" running
  log "launched $ticker (window: $win)"
}

reconcile() {
  # for any ticker marked `running` whose window is gone: decide done/failed
  local t status
  while IFS=$'\t' read -r t status _ _; do
    [[ "$status" == "running" ]] || continue
    if window_alive "$t"; then
      continue   # still running
    fi
    # window gone — check sentinels
    if [[ -f "$(sent_done "$t")" ]]; then
      state_set "$t" done
      log "$t -> done"
    elif [[ -f "$(sent_failed "$t")" ]]; then
      state_set "$t" failed
      log "$t -> failed (rc=$(cat "$(sent_failed "$t")" 2>/dev/null))"
    else
      # crashed without sentinel (e.g. machine power-off) — requeue
      state_set "$t" pending
      log "$t -> re-queued (no sentinel found, presumed crashed)"
    fi
  done < "$(state_file)"
}

controller_loop() {
  trap 'log "controller received signal, exiting"; exit 0' INT TERM
  local dbg="$STATE_DIR/controller.log"
  : > "$dbg"
  dbg() { echo "[$(date +%H:%M:%S)] $*" >> "$dbg"; }
  while true; do
    # honor STOP
    if [[ -f "$(stop_file)" ]]; then
      log "STOP requested — controller exiting (running ticker jobs continue)"
      rm -f "$(stop_file)"
      exit 0
    fi
    reconcile
    dbg "post-reconcile windows='$(session_windows)' running_windows=$(running_windows) pending=$(count_status pending) done=$(count_status done) failed=$(count_status failed)"

    if [[ -f "$(pause_file)" ]]; then
      sleep 3
      continue
    fi

    local running; running="$(running_windows)"
    local pending; pending="$(count_status pending)"

    if [[ "$pending" -eq 0 ]]; then
      if [[ "$running" -eq 0 ]]; then
        log "all tickers processed — controller done"
        break
      fi
      sleep 3
      continue
    fi

    while [[ "$running" -lt "$CONCURRENCY" && "$pending" -gt 0 ]]; do
      # pick the first pending ticker (stable order)
      local t
      t="$(awk -F'\t' '$2=="pending"{print $1; exit}' "$(state_file)")"
      [[ -z "$t" ]] && break
      launch_ticker "$t"
      running=$((running+1))
      pending=$((pending-1))
      sleep 0.5   # stagger launches a touch
    done

    sleep 3
  done
}

# ---------- subcommands ----------
cmd_start() {
  local cmd_override=0 j_arg="" f_arg=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -j|--concurrency) j_arg="$2"; shift 2 ;;
      -f|--file)        f_arg="$2"; shift 2 ;;
      --)               shift; cmd_override=1; break ;;
      *) die "start: unknown arg: $1" ;;
    esac
  done

  if controller_alive; then
    die "controller session '$SESSION' already running. Use '$0 status' or '$0 stop' first."
  fi

  # Reuse persisted config when restarting after a stop/crash, so the user can
  # just run `start` again without re-specifying -j/-f/--. Explicit flags
  # override; a prior config.env supplies defaults for anything omitted.
  if [[ -f "$STATE_DIR/config.env" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_DIR/config.env"
  fi
  [[ -n "$j_arg" ]] && CONCURRENCY="$j_arg"
  [[ -n "$f_arg" ]] && TICKERS_FILE="$f_arg"

  # Build the per-ticker command template. If a template already exists and the
  # user did not pass `--`, reuse it (preserves a custom command across restarts).
  local tpl; tpl="$(template_file)"
  if [[ "$cmd_override" -eq 1 ]]; then
    [[ "$*" == *"%%TICKER%%"* ]] || die "custom command must contain the %%TICKER%% placeholder"
    : > "$tpl"
    local a
    for a in "$@"; do printf '%q ' "$a" >> "$tpl"; done
    echo >> "$tpl"
  elif [[ ! -f "$tpl" ]]; then
    # first run with no custom command → write the default
    { for a in "${DEFAULT_CMD[@]}"; do printf '%q ' "$a"; done; echo; } > "$tpl"
  fi

  # (re)persist config so the controller reads the resolved values
  cat > "$STATE_DIR/config.env" <<EOF
REPO_ROOT='$REPO_ROOT'
TICKERS_FILE='$TICKERS_FILE'
STATE_DIR='$STATE_DIR'
SESSION='$SESSION'
CONCURRENCY=$CONCURRENCY
EOF

  state_init
  write_wrapper

  log "starting controller: concurrency=$CONCURRENCY, tickers=$TICKERS_FILE"
  log "per-ticker template: $tpl"
  screen -dmS "$SESSION" bash -c "
    set -e
    source '$STATE_DIR/config.env'
    cd \"\$REPO_ROOT\"
    exec '$0' _controller
  "
  sleep 1
  if controller_alive; then
    log "controller session '$SESSION' started. Attach with: $0 attach"
  else
    die "failed to start controller session"
  fi
}

# invoked internally by cmd_start via screen
cmd__controller() {
  # shellcheck disable=SC1090
  source "$STATE_DIR/config.env"
  export REPO_ROOT TICKERS_FILE STATE_DIR SESSION CONCURRENCY
  write_wrapper   # ensure it exists / is up to date
  controller_loop
}

cmd_pause()  { touch "$(pause_file)";  log "paused — running jobs finish, no new ones launched"; }
cmd_resume() { rm -f "$(pause_file)";  log "resumed — new jobs will be launched as slots free up"; }

cmd_stop() {
  rm -f "$(pause_file)"
  touch "$(stop_file)"
  log "stop requested — controller will exit after current poll"
  # give the controller a moment to notice, then tear down the whole session.
  # running ticker jobs are killed; --checkpoint lets them resume on next start.
  sleep 2
  if controller_alive; then
    screen -S "$SESSION" -X quit 2>/dev/null || true
  fi
  rm -f "$(stop_file)"
  log "stopped. Re-run with: $0 start"
}

cmd_status() {
  # load persisted concurrency if available so status reflects the running run
  if [[ -f "$STATE_DIR/config.env" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_DIR/config.env"
  fi
  if controller_alive; then echo "controller: RUNNING (session $SESSION)"; else echo "controller: STOPPED"; fi
  [[ -f "$(pause_file)" ]] && echo "state: PAUSED" || echo "state: ACTIVE"
  echo "concurrency: $CONCURRENCY  (live windows: $(running_windows))"
  echo
  printf '%-12s %8s %12s %12s\n' STATUS COUNT PCT SUMMARY
  local total done failed pending running
  total=$(wc -l < "$(state_file)")
  done=$(count_status done); failed=$(count_status failed)
  pending=$(count_status pending); running=$(count_status running)
  local pct=0; [[ "$total" -gt 0 ]] && pct=$(( (done*100) / total ))
  printf '%-12s %8s %12s\n' "done"    "$done"    "$pct%"
  printf '%-12s %8s\n'         "failed"  "$failed"
  printf '%-12s %8s\n'         "running" "$running"
  printf '%-12s %8s\n'         "pending" "$pending"
  echo "------------------------------------"
  printf '%-12s %8s\n'         "total"   "$total"
  echo
  if [[ "$failed" -gt 0 ]]; then
    echo "Failed tickers:"
    awk -F'\t' '$2=="failed"{print "  "$1}' "$(state_file)"
    echo "  (requeue with: $0 reset --failed)"
  fi
}

cmd_reset() {
  local mode="${1:-failed}"
  if controller_alive; then
    die "controller is running — stop it first: $0 stop"
  fi
  case "$mode" in
    --failed)
      awk -F'\t' '$2=="failed"{print $1}' "$(state_file)" | while read -r t; do
        state_set "$t" pending; rm -f "$(sent_failed "$t")"
      done
      log "re-queued all failed tickers"
      ;;
    --all)
      rm -f "$(state_file)"
      rm -rf "$STATE_DIR/done" "$STATE_DIR/failed" "$STATE_DIR/logs"
      mkdir -p "$STATE_DIR/logs"
      state_init
      log "reset entire state to pending"
      ;;
    *) die "reset: unknown mode '$mode' (use --failed or --all)" ;;
  esac
}

cmd_logs() {
  local t="$1"
  local f="$STATE_DIR/logs/$t.log"
  [[ -f "$f" ]] || die "no log for $t at $f"
  tail -n 200 -f "$f"
}

cmd_attach() {
  screen -r "$SESSION"
}

# ---------- dispatch ----------
case "${1:-}" in
  start)       shift; cmd_start "$@" ;;
  _controller) cmd__controller ;;
  pause)       cmd_pause ;;
  resume)      cmd_resume ;;
  status)      cmd_status ;;
  stop)        cmd_stop ;;
  reset)       shift; cmd_reset "${1:-failed}" ;;
  logs)        shift; cmd_logs "${1:-}" ;;
  attach)      cmd_attach ;;
  ""|-h|--help|help)
    sed -n '2,40p' "$0"
    ;;
  *) die "unknown subcommand: $1" ;;
esac
