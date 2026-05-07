#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/reports"
KINDLE_EMAIL="${KINDLE_EMAIL:-}"
SMTP_EMAIL="${SMTP_EMAIL:-}"
SEND_EMAIL=false

for arg in "$@"; do
    case "$arg" in
        --send|--email) SEND_EMAIL=true ;;
    esac
done

if ! command -v pandoc &>/dev/null; then
    echo "ERROR: pandoc is required but not installed." >&2
    exit 1
fi

if $SEND_EMAIL; then
    if ! command -v msmtp &>/dev/null; then
        echo "ERROR: msmtp is required with --send but not installed." >&2
        exit 1
    fi
    if [[ -z "$SMTP_EMAIL" ]]; then
        read -r -p "SMTP email: " SMTP_EMAIL
    fi
    if [[ -z "$KINDLE_EMAIL" ]]; then
        read -r -p "Kindle email: " KINDLE_EMAIL
    fi
    read -r -s -p "SMTP password: " SMTP_PASS
    echo ""
    export SMTP_PASS
fi

echo "Looking for complete_report.md files under $REPORTS_DIR ..."

shopt -s nullglob
md_files=("$REPORTS_DIR"/*/complete_report.md)
shopt -u nullglob

if [[ ${#md_files[@]} -eq 0 ]]; then
    echo "No complete_report.md files found."
    exit 0
fi

echo "Found ${#md_files[@]} report(s)"
echo ""

send_epub() {
    local ticker="$1"
    local file="$2"
    local boundary="=====$(uuidgen)====="

    msmtp -t <<EOF
From: ${SMTP_EMAIL}
To: ${KINDLE_EMAIL}
Subject: TradingAgents Report: ${ticker}
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="${boundary}"

--${boundary}
Content-Type: text/plain; charset=utf-8

TradingAgents report for ${ticker}.

--${boundary}
Content-Type: application/epub+zip; name="complete_report_${ticker}.epub"
Content-Disposition: attachment; filename="complete_report_${ticker}.epub"
Content-Transfer-Encoding: base64

$(base64 "$file")
--${boundary}--
EOF
}

COUNT=0
for md_file in "${md_files[@]}"; do
    dir="$(dirname "$md_file")"
    ticker_dir="$(basename "$dir")"
    ticker="${ticker_dir%%_*}"
    epub_file="$dir/complete_report.epub"

    echo "  [$ticker] converting $md_file -> $epub_file"

    pandoc "$md_file" \
        -f markdown \
        -t epub \
        --metadata title="TradingAgents Report: ${ticker}" \
        -o "$epub_file"

    if $SEND_EMAIL; then
        echo "  [$ticker] sending to $KINDLE_EMAIL ..."
        send_epub "$ticker" "$epub_file"
        mkdir -p "$REPORTS_DIR/sent"
        mv "$dir" "$REPORTS_DIR/sent/"
    fi

    COUNT=$((COUNT + 1))
    echo ""
done

echo "Done. Converted $COUNT report(s) to epub."
$SEND_EMAIL && echo "Epubs sent to $KINDLE_EMAIL."
