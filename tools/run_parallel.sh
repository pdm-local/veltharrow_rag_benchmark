#!/usr/bin/env bash
# Launch N parallel benchmark workers on the same machine. Each worker handles
# a round-robin slice of the tier's model specs. CSV writes are fcntl-locked
# inside run_benchmark.py so workers can append safely to the same file.
#
# Usage:
#   tools/run_parallel.sh --tier small               # 2 workers (default)
#   tools/run_parallel.sh --tier small --workers 3   # 3 workers
#   tools/run_parallel.sh --tier small --workers 2 -- --limit-questions 10
#
# Everything after `--` is forwarded to run_benchmark.py for each worker.
set -u

cd "$(dirname "$0")/.." || exit 1

WORKERS=2
TIER=small
PASSTHRU=()
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --tier)    TIER="$2";    PASSTHRU+=("--tier" "$2"); shift 2 ;;
        --)        shift; PASSTHRU+=("$@"); break ;;
        *)         PASSTHRU+=("$1"); shift ;;
    esac
done

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
    echo "--workers must be a positive integer" >&2
    exit 1
fi

LOG_DIR="results/$TIER"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
echo "Launching $WORKERS parallel workers on tier=$TIER  (logs: $LOG_DIR/worker_${TS}_*.log)"

PIDS=()
for ((i=0; i<WORKERS; i++)); do
    LOG="$LOG_DIR/worker_${TS}_${i}.log"
    echo "  worker $i -> $LOG"
    python3 scripts/run_benchmark.py --worker "$i" --total-workers "$WORKERS" "${PASSTHRU[@]}" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo "Tail any worker log with:  tail -f $LOG_DIR/worker_${TS}_0.log"

# Wait for all workers; surface the worst exit code.
rc=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || rc=$?
done
echo "all workers done; max_exit=$rc"
exit "$rc"
