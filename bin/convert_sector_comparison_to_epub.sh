#!/usr/bin/env bash
set -euo pipefail

: "${KINDLE_EMAIL:?set KINDLE_EMAIL env var}"
: "${SMTP_PASS:?set SMTP_PASS env var}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/reports"
SMTP_EMAIL="patr.frank@gmx.de"

md_files=("$REPORTS_DIR"/sector_comparison_*.md)

if [[ ${#md_files[@]} -eq 0 ]]; then
    echo "No sector_comparison_*.md files found in $REPORTS_DIR"
    exit 0
fi

echo "Found ${#md_files[@]} sector comparison report(s)"

send_epub() {
    local name="$1"
    local file="$2"
    local boundary="=====$(uuidgen)====="

    msmtp -t <<EOF
From: ${SMTP_EMAIL}
To: ${KINDLE_EMAIL}
Subject: Sector Comparison: ${name}
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="${boundary}"

--${boundary}
Content-Type: text/plain; charset=utf-8

Sector Comparison Report: ${name}

--${boundary}
Content-Type: application/epub+zip; name="${name}.epub"
Content-Disposition: attachment; filename="${name}.epub"
Content-Transfer-Encoding: base64

$(base64 "$file")
--${boundary}--
EOF
}

COUNT=0
for md_file in "${md_files[@]}"; do
    basename="$(basename "$md_file" .md)"
    name="${basename#sector_comparison_}"
    epub_file="${REPORTS_DIR}/${basename}.epub"

    echo "  [${name}] converting $md_file -> $epub_file"

    pandoc "$md_file" \
        -f markdown \
        -t epub \
        --metadata title="Sector Comparison: ${name}" \
        -o "$epub_file"

    echo "  [${name}] sending to $KINDLE_EMAIL ..."
    send_epub "$name" "$epub_file"

    COUNT=$((COUNT + 1))
    echo ""
done

echo "Done. Converted and sent $COUNT report(s) to $KINDLE_EMAIL."
