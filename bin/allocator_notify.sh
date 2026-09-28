#!/usr/bin/env bash
#
# Send a notification that the allocator needs human attention (e.g. a monthly
# rebalance proposal awaits approval, or stale reports must be refreshed).
#
# Transport is configurable via ALLOCATOR_NOTIFY_CMD, a command to which the
# title and body are appended as the last two arguments. Defaults to
# `notify-send`, falling back to stderr.
#
# Usage:
#   bin/allocator_notify.sh "title" "body"
#   ALLOCATOR_NOTIFY_CMD='curl -fsS -d @- https://example/hook' bin/allocator_notify.sh ...
#
set -uo pipefail

title="${1:-allocator}"
body="${2:-}"

if [ -n "${ALLOCATOR_NOTIFY_CMD:-}" ]; then
  read -r -a notify_cmd <<<"$ALLOCATOR_NOTIFY_CMD"
  "${notify_cmd[@]}" "$title" "$body"
  exit $?
fi

if command -v notify-send >/dev/null 2>&1; then
  notify-send "$title" "$body"
  exit 0
fi

printf '%s: %s\n' "$title" "$body" >&2
