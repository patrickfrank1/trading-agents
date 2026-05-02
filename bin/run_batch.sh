#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TICKERS_FILE="${SCRIPT_DIR}/TICKERS.yaml"
RESULTS_DIR="results/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

if ! command -v yq &>/dev/null; then
    TICKERS=($(grep -E '^\s+-\s+' "$TICKERS_FILE" | awk '{print $2}'))
else
    TICKERS=($(yq '.tickers[]' "$TICKERS_FILE"))
fi

TOTAL=${#TICKERS[@]}
echo "=== Batch Research: ${TOTAL} tickers ==="
echo "Results will be saved to: ${RESULTS_DIR}/"
echo ""

OVERALL_START=$(date +%s)
OVERALL_FAILED=0

for i in "${!TICKERS[@]}"; do
    TICKER="${TICKERS[$i]}"
    NUM=$((i + 1))
    echo "-------------------------------------------"
    echo "[$NUM/${TOTAL}] ${TICKER} — $(date +%H:%M:%S)"
    echo "-------------------------------------------"

    TICKER_START=$(date +%s)

    if uv run tradingagents \
        --refresh-rate 0.1 \
        --non-interactive \
        --checkpoint \
        --display-report \
        --save \
        --save-path "$RESULTS_DIR/$TICKER" \
        --ticker "$TICKER" \
        --research-depth deep \
        --provider glm \
        --shallow-model GLM-4.7 \
        --deep-model GLM-5.1 \
        2>&1 | tee "${RESULTS_DIR}/${TICKER}.log"; then
        STATUS="OK"
    else
        STATUS="FAILED"
        OVERALL_FAILED=$((OVERALL_FAILED + 1))
    fi

    TICKER_END=$(date +%s)
    TICKER_ELAPSED=$((TICKER_END - TICKER_START))
    echo ""
    echo "[${TICKER}] ${STATUS} (${TICKER_ELAPSED}s)"
    echo ""
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))
OVERALL_OK=$((TOTAL - OVERALL_FAILED))

echo "==========================================="
echo "=== Batch Complete ==="
echo "  Tickers:      ${TOTAL}"
echo "  Succeeded:    ${OVERALL_OK}"
echo "  Failed:       ${OVERALL_FAILED}"
echo "  Total time:   ${OVERALL_ELAPSED}s"
echo "  Results dir:  ${RESULTS_DIR}/"
echo "==========================================="
