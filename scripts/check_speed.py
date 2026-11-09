#!/usr/bin/env python3
"""Generic benchmark speed monitor.

Usage:
  python3 check_speed.py                              # overview of all models
  python3 check_speed.py --model gemma4_e4b_thinking_off
  python3 check_speed.py --model qwen35_9b_thinking_off --file results/medium/data/raw_results.csv
"""
import argparse
import csv
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

csv.field_size_limit(sys.maxsize)


def gpu_stats() -> str:
    """One-line GPU summary (util/VRAM/temp/power) from ``nvidia-smi``, or a placeholder
    string if the query fails."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        util, mem_used, mem_total, temp, power = out.split(", ")
        return f"GPU {util}%  VRAM {mem_used}/{mem_total} MiB  {temp}°C  {power}W"
    except Exception:
        return "GPU stats unavailable"


def check(args: argparse.Namespace) -> None:
    """Print one progress snapshot: per-model row counts plus, for ``--model``, the recent
    throughput, ETA, and deadline margin estimated from row timestamps."""
    rows = list(csv.DictReader(open(args.file)))
    all_mc: Counter = Counter(r["model_label"] for r in rows)
    now = datetime.now(timezone.utc)

    print(f"\n{'='*58}")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*58}")
    print(f"  GPU:  {gpu_stats()}")
    print(f"{'='*58}")
    print(f"  All models in {args.file}:")
    for m, c in sorted(all_mc.items(), key=lambda x: -x[1]):
        status = "DONE" if c >= args.total else f"{c / args.total * 100:.0f}%"
        print(f"    {m:<42} {c:>5} ({status})")
    print(f"{'='*58}")

    if not args.model:
        return

    target = args.model
    on = [r for r in rows if r["model_label"] == target]
    trows = sorted([r for r in on if r.get("timestamp")], key=lambda r: r["timestamp"])

    if len(trows) < 10:
        print(f"  {target}: {len(on)} rows: not enough data yet")
        return

    last = trows[-50:] if len(trows) >= 50 else trows
    t0 = datetime.fromisoformat(last[0]["timestamp"])
    t1 = datetime.fromisoformat(last[-1]["timestamp"])
    elapsed = (t1 - t0).total_seconds()
    if elapsed == 0:
        print("  Waiting for new rows...")
        return

    speed = len(last) / elapsed * 3600
    remaining = args.total - len(on)
    eta_h = remaining / speed
    finish = (now + timedelta(hours=eta_h)).strftime("%b %d %H:%M UTC")
    margin = args.deadline - eta_h
    margin_str = f"+{margin:.1f}h OK" if margin > 0 else f"{margin:.1f}h OVER!"

    print(f"  {target}:")
    print(f"    Progress : {len(on)} / {args.total} ({len(on) / args.total * 100:.1f}%)")
    print(f"    Speed    : {speed:.0f} rows/hr  (~{3600 / speed:.0f}s per row)")
    print(f"    ETA      : {eta_h:.1f}h  ->  done {finish}")
    print(f"    Margin   : {margin_str}")
    print(f"{'='*58}")
    print(f"  Refreshing every {args.refresh}s  |  Ctrl+C to stop")


def main() -> None:
    """Re-print the speed snapshot every ``--refresh`` seconds until interrupted."""
    parser = argparse.ArgumentParser(description="Benchmark speed monitor.")
    parser.add_argument("--model", default="", help="Model label to track (optional)")
    parser.add_argument("--file", default="results/medium/data/raw_results.csv", help="CSV file to monitor")
    parser.add_argument("--total", type=int, default=1600,
                        help="Target rows per model label (standard grid: 400 questions x 4 conditions)")
    parser.add_argument("--deadline", type=float, default=47.0, help="Deadline in hours from now")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    args = parser.parse_args()

    while True:
        try:
            check(args)
        except FileNotFoundError:
            print(f"{args.file} not found: is benchmark running?")
        except Exception as exc:
            print(f"Error: {exc}")
        try:
            time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
