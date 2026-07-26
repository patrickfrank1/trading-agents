#!/usr/bin/env bash
# Control a running ./bin/run_parallel.sh batch (which runs inside a detached
# screen session named ta_batch_<ts>).
#
# Usage:
#   ./bin/batch_control.sh status     # show running batch + worker states
#   ./bin/batch_control.sh pause      # freeze the batch (SIGSTOP whole tree)
#   ./bin/batch_control.sh resume     # unfreeze (SIGCONT)
#
# Pause stops the screen's bash launcher AND all worker processes (uv/python),
# but leaves the screen process itself alive so the session stays attached-able.
# New tickers won't launch while paused (the launcher is frozen in wait -n).

set -uo pipefail

action="${1:-status}"

# Find detached screen sessions named ta_batch_*  (format: <pid>.ta_batch_<ts>)
mapfile -t SESSIONS < <(screen -ls 2>/dev/null | grep -oE '[0-9]+\.ta_batch_[0-9_]+' || true)

if [ "${#SESSIONS[@]}" -eq 0 ]; then
    echo "No ta_batch screen sessions found."
    echo "(Is ./bin/run_parallel.sh running?)"
    [ "$action" = "status" ] && exit 0
    exit 1
fi

# Recursively list all descendant PIDs of a given PID (inclusive).
descendants() {
    local pid="$1"
    echo "$pid"
    local child
    while IFS= read -r child; do
        descendants "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
}

state_of() {
    # ps state: R=running, T=stopped, S=sleeping, D=uninterruptible, Z=zombie
    ps -o stat= -p "$1" 2>/dev/null | awk '{print $1}' | cut -c1
}

case "$action" in
    pause|stop)
        for s in "${SESSIONS[@]}"; do
            spid="${s%%.*}"
            # All descendants of the screen process (bash launcher + workers).
            mapfile -t pids < <(descendants "$spid")
            stopped=0
            for p in "${pids[@]}"; do
                [ "$p" = "$spid" ] && continue        # never stop screen itself
                kill -STOP "$p" 2>/dev/null || true
                stopped=$((stopped + 1))
            done
            echo "Paused: ${s}  (${stopped} processes frozen)"
        done
        echo "Resume with: ./bin/batch_control.sh resume"
        ;;

    resume|continue|unpause)
        for s in "${SESSIONS[@]}"; do
            spid="${s%%.*}"
            mapfile -t pids < <(descendants "$spid")
            resumed=0
            for p in "${pids[@]}"; do
                [ "$p" = "$spid" ] && continue
                kill -CONT "$p" 2>/dev/null || true
                resumed=$((resumed + 1))
            done
            echo "Resumed: ${s}  (${resumed} processes unfrozen)"
        done
        ;;

    status)
        for s in "${SESSIONS[@]}"; do
            spid="${s%%.*}"
            mapfile -t pids < <(descendants "$spid")
            running=0; stopped=0; other=0
            for p in "${pids[@]}"; do
                [ "$p" = "$spid" ] && continue
                case "$(state_of "$p")" in
                    T) stopped=$((stopped + 1)) ;;
                    R) running=$((running + 1)) ;;
                    *) other=$((other + 1)) ;;
                esac
            done
            echo "Session: ${s}"
            echo "  workers: running=${running} paused=${stopped} other=${other}"
        done
        # Per-ticker status files (OK/FAILED/in-progress)
        for d in logs/parallel_*; do
            [ -d "$d" ] || continue
            ok=$(grep -l '^OK' "$d"/*.status 2>/dev/null | wc -l)
            fail=$(grep -l '^FAILED' "$d"/*.status 2>/dev/null | wc -l)
            inprog=$(ls "$d"/*.log 2>/dev/null | grep -v '_summary\|_launch' | wc -l)
            inprog=$((inprog - ok - fail))
            [ "$inprog" -lt 0 ] && inprog=0
            echo "  $d: ok=${ok} failed=${fail} in-progress=${inprog}"
        done
        ;;

    *)
        echo "Usage: $0 {status|pause|resume}" >&2
        exit 2
        ;;
esac
