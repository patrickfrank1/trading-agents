#!/usr/bin/env bash
# Run tradingagents for every ticker in bin/TICKERS.yaml in parallel,
# inside a detached screen session so it survives terminal close.
#
# Usage:
#   ./bin/run_parallel.sh                  # defaults below
#   MAX_JOBS=8 ./bin/run_parallel.sh
#   PROVIDER=anthropic SHALLOW_MODEL=claude-haiku-4-5 DEEP_MODEL=claude-opus-4-6 ./bin/run_parallel.sh
#   TICKERS_OVERRIDE="ISRG AAPL" ./bin/run_parallel.sh   # run a subset
#
# Managing the session:
#   screen -ls                          # list sessions (name: ta_batch_<ts>)
#   screen -r ta_batch_<ts>             # attach to watch live
#       (Ctrl+A then D to detach again)
#   tail -f logs/parallel_<ts>/_summary.log
#   watch -n2 'cat logs/parallel_*/*.status 2>/dev/null'
#
# Environment:
#   IN_SCREEN=1 is set automatically when relaunching inside screen. If you
#   already want to run inline (no screen), invoke with IN_SCREEN=1 directly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TICKERS_FILE="${SCRIPT_DIR}/TICKERS.yaml"

# --- Config (override via env) ------------------------------------------------
PROVIDER="${PROVIDER:-deepseek}"
SHALLOW_MODEL="${SHALLOW_MODEL:-deepseek-v4-pro}"
DEEP_MODEL="${DEEP_MODEL:-deepseek-v4-pro}"
RESEARCH_DEPTH="${RESEARCH_DEPTH:-deep}"
REFRESH_RATE="${REFRESH_RATE:-0.1}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}"
if [ "${MAX_JOBS}" -lt 4 ] 2>/dev/null; then MAX_JOBS=4; fi

# --- If not yet inside screen, relaunch detached and exit ----------------------
if [ "${IN_SCREEN:-0}" != "1" ]; then
    if ! command -v screen &>/dev/null; then
        echo "ERROR: 'screen' not found. Install it or run with IN_SCREEN=1." >&2
        exit 1
    fi

    TS="$(date +%Y-%m-%d_%H%M%S)"
    SESSION="ta_batch_${TS}"
    RESULTS_DIR="logs/parallel_${TS}"
    mkdir -p "${RESULTS_DIR}"
    LAUNCH_LOG="${RESULTS_DIR}/_launch.log"

    # Re-export every config var so the inner invocation sees them.
    exec env \
        IN_SCREEN=1 \
        RESULTS_DIR="${RESULTS_DIR}" \
        PROVIDER="${PROVIDER}" \
        SHALLOW_MODEL="${SHALLOW_MODEL}" \
        DEEP_MODEL="${DEEP_MODEL}" \
        RESEARCH_DEPTH="${RESEARCH_DEPTH}" \
        REFRESH_RATE="${REFRESH_RATE}" \
        MAX_JOBS="${MAX_JOBS}" \
        TICKERS_OVERRIDE="${TICKERS_OVERRIDE:-}" \
        screen -L -Logfile "${LAUNCH_LOG}" -dmS "${SESSION}" \
            bash "${BASH_SOURCE[0]}"

    # (exec does not return)
fi

# --- Below runs inside the detached screen session ----------------------------
RESULTS_DIR="${RESULTS_DIR:-logs/parallel_$(date +%Y-%m-%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

SUMMARY_FILE="${RESULTS_DIR}/_summary.log"
: > "${SUMMARY_FILE}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${SUMMARY_FILE}"; }

# --- Parse tickers ------------------------------------------------------------
if [ -n "${TICKERS_OVERRIDE:-}" ]; then
    read -r -a TICKERS <<< "${TICKERS_OVERRIDE}"
else
    if command -v yq &>/dev/null; then
        mapfile -t TICKERS < <(yq '.tickers[]' "${TICKERS_FILE}")
    else
        mapfile -t TICKERS < <(grep -E '^\s*-\s+' "${TICKERS_FILE}" | awk '{print $2}')
    fi
fi

TOTAL=${#TICKERS[@]}
if [ "${TOTAL}" -eq 0 ]; then
    log "ERROR: No tickers found in ${TICKERS_FILE}"
    exit 1
fi

log "=== Parallel batch (screen session: ${STY:-<detached>}) ==="
log "  Tickers:    ${TOTAL}"
log "  Max jobs:   ${MAX_JOBS}"
log "  Provider:   ${PROVIDER}"
log "  Models:     ${SHALLOW_MODEL} / ${DEEP_MODEL}"
log "  Depth:      ${RESEARCH_DEPTH}"
log "  Results:    ${RESULTS_DIR}/"
log ""

# --- Per-ticker worker --------------------------------------------------------
run_one() {
    local ticker="$1" results_dir="$2" provider="$3"
    local shallow="$4" deep="$5" depth="$6" refresh="$7"
    local log="${results_dir}/${ticker}.log"
    local status_file="${results_dir}/${ticker}.status"
    local start end elapsed

    start=$(date +%s)
    echo "[$(date +%H:%M:%S)] START ${ticker}" > "${log}"

    if uv run tradingagents \
        --refresh-rate "${refresh}" \
        --non-interactive \
        --checkpoint \
        --display-report \
        --save \
        --save-path "${results_dir}/${ticker}" \
        --ticker "${ticker}" \
        --research-depth "${depth}" \
        --provider "${provider}" \
        --shallow-model "${shallow}" \
        --deep-model "${deep}" \
        >> "${log}" 2>&1; then
        end=$(date +%s); elapsed=$((end - start))
        echo "OK ${ticker} ${elapsed}s" > "${status_file}"
        echo "[$(date +%H:%M:%S)] OK     ${ticker} (${elapsed}s)" >> "${log}"
    else
        end=$(date +%s); elapsed=$((end - start))
        echo "FAILED ${ticker} ${elapsed}s" > "${status_file}"
        echo "[$(date +%H:%M:%S)] FAILED ${ticker} (${elapsed}s)" >> "${log}"
    fi
}
export -f run_one

# --- Launch with concurrency limit -------------------------------------------
OVERALL_START=$(date +%s)
running=0

for ticker in "${TICKERS[@]}"; do
    while [ "${running}" -ge "${MAX_JOBS}" ]; do
        wait -n 2>/dev/null || sleep 1
        running=$(jobs -r | wc -l)
    done

    log "launch ${ticker}"
    run_one "${ticker}" "${RESULTS_DIR}" "${PROVIDER}" \
            "${SHALLOW_MODEL}" "${DEEP_MODEL}" "${RESEARCH_DEPTH}" \
            "${REFRESH_RATE}" &
    running=$((running + 1))
done

wait

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))

# --- Summary ------------------------------------------------------------------
OK=0; FAILED=0
while read -r st _t _s; do
    case "${st}" in
        OK)     OK=$((OK + 1)) ;;
        FAILED) FAILED=$((FAILED + 1)) ;;
    esac
done < <(cat "${RESULTS_DIR}"/*.status 2>/dev/null)

log ""
log "=== Complete ==="
log "  Succeeded:  ${OK}"
log "  Failed:     ${FAILED}"
log "  Total time: ${OVERALL_ELAPSED}s"
log "  Results:    ${RESULTS_DIR}/"

if [ "${FAILED}" -gt 0 ]; then
    log ""
    log "Failed tickers:"
    grep -h "^FAILED" "${RESULTS_DIR}"/*.status | tee -a "${SUMMARY_FILE}"
fi

# Keep the screen window open briefly so an attached user sees the summary.
echo ""
echo "Batch finished. This screen session will close in 10s."
echo "Re-run ./bin/run_parallel.sh for a new batch."
sleep 10
exit $(( FAILED > 0 ? 1 : 0 ))
