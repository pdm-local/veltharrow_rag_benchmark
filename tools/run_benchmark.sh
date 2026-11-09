#!/usr/bin/env bash
# Portable launcher: forwards all args to scripts/run_benchmark.py and tees
# output to a timestamped log under results/<tier>/ so logs sit next to the CSVs.
#
# Examples:
#   tools/run_benchmark.sh --tier small --max-attempts 5
#   tools/run_benchmark.sh --tier medium --max-attempts 5
set -u

# Always run from project root (this script lives in tools/).
cd "$(dirname "$0")/.." || exit 1

# Parse --tier from args (default: medium). Does not consume: passes through.
TIER="medium"
prev=""
for a in "$@"; do
    if [ "$prev" = "--tier" ]; then TIER="$a"; fi
    prev="$a"
done

LOG_DIR="results/$TIER"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > "$LOG_DIR/latest_benchmark_log.txt"
echo "logging to $LOG"

python3 scripts/run_benchmark.py "$@" 2>&1 | tee "$LOG"
code=${PIPESTATUS[0]}
echo "benchmark_exit_code:$code" | tee -a "$LOG"
exit "$code"
