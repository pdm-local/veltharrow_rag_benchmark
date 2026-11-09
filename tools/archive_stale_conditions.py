#!/usr/bin/env python3
"""Move result rows for retrieval conditions no longer in the config out of the
canonical CSV.

An earlier config generation swept four embedders across k in {3, 6, 9}. The current
config keeps a single embedder at k in {1, 3, 5}, so those rows are orphans: nothing
regenerates them, no table or figure in the paper reports them, and the k=9 rows are
not even valid — roughly 45% of them hit the 8192-token context ceiling and lost part
of their retrieved evidence.

One cell group survives that rule: ``qwen3.5:9b`` (thinking off) completed the whole
sweep -- 4 embedders x k in {3, 6, 9}, 400 responses in every cell. That is the only
evidence in the benchmark about embedder choice and about depths beyond k=5, so it is
reported as a single-model ablation instead of archived. Its ``thinking_on`` twin got
27-28 rows per cell before the pilot was abandoned and is archived like the rest.

The rule is therefore *stale and short*, not *stale*: a cell is archived only if it is
outside the current config AND holds too few responses to report. That keeps the
decision derivable from the data rather than from a hand-maintained model name list.

They are archived rather than deleted, and the canonical CSV is backed up first.

    python3 tools/archive_stale_conditions.py --tier medium            # report only
    python3 tools/archive_stale_conditions.py --tier medium --write
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
csv.field_size_limit(sys.maxsize)

from run_benchmark import RAW_COLUMNS, raw_results_path
from veltharrow.config import resolve_tier, retrieval_conditions


def benchmark_running() -> bool:
    """True if a benchmark run is live. Rewriting the CSV under an appending writer
    would drop every row it writes between our read and our write."""
    try:
        out = subprocess.run(["pgrep", "-af", "run_benchmark.py"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out)


# A (model, condition) cell below this many responses cannot carry a claim -- the Wilson
# interval is far too wide -- so an out-of-config cell that short is a pilot fragment and
# is archived. At or above it the cell is a complete, reportable measurement and is kept
# even though nothing regenerates it. Mirrors MIN_REPORTABLE_N in tools/paper_tables.py.
MIN_REPORTABLE_N = 100


def archive_tier(tier: str, write: bool) -> None:
    path = raw_results_path(tier)
    current = {c["retrieval_condition"] for c in retrieval_conditions(tier)}

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    sizes = Counter((r["model_label"], r["retrieval_condition"]) for r in rows)

    def is_stale(row: dict) -> bool:
        """Out of the current config *and* too small to report on its own."""
        if row["retrieval_condition"] in current:
            return False
        return sizes[(row["model_label"], row["retrieval_condition"])] < MIN_REPORTABLE_N

    stale = [r for r in rows if is_stale(r)]
    keep = [r for r in rows if not is_stale(r)]

    kept_orphans = sorted({
        (m, c) for (m, c), n in sizes.items()
        if c not in current and n >= MIN_REPORTABLE_N
    })
    if kept_orphans:
        print(f"\n{tier}: keeping {len(kept_orphans)} out-of-config cells that are complete "
              f"enough to report (ablation data):")
        for model, cond in kept_orphans:
            print(f"  keep   {model:32s} {cond:32s} {sizes[(model, cond)]:6d}")

    print(f"\n{tier}: {len(rows)} rows, {len(current)} current conditions")
    if not stale:
        print("  no stale conditions")
        return
    for cond, n in Counter(r["retrieval_condition"] for r in stale).most_common():
        print(f"  stale  {cond:32s} {n:6d}")
    for model, n in Counter(r["model_label"] for r in stale).most_common():
        print(f"  from   {model:32s} {n:6d}")
    print(f"  -> archiving {len(stale)} rows, keeping {len(keep)}")

    if not write:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(path, path.with_name(f"{path.name}.bak_{stamp}_pre_stale_archive"))
    arch = path.with_name("archive_pilot_embedder_sweep.csv")
    with arch.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stale)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(keep)
    print(f"  archived -> {arch.name}; backup kept")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="all", help="small, medium, or all")
    parser.add_argument("--write", action="store_true", help="rewrite the CSV in place")
    args = parser.parse_args(argv)

    if args.write and benchmark_running():
        print("REFUSING: run_benchmark.py is running and appends to this CSV.\n"
              "Rewriting it now would discard rows written since it was read.\n"
              "Wait for the rerun to finish, then retry.", file=sys.stderr)
        return 1

    tiers = ["small", "medium"] if args.tier == "all" else [resolve_tier(args.tier)]
    for tier in tiers:
        archive_tier(tier, args.write)
    if not args.write:
        print("\n(dry run - pass --write to archive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
