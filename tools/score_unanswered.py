#!/usr/bin/env python3
"""Materialise unanswered questions as scored-wrong rows in ``raw_results.csv``.

A question the model never answered is a *result*, not a gap. The escalation protocol
gave every one of these ten attempts and, from the sixth, an 8192-token budget at
num_ctx=16384; the survivors ended at exactly the ceiling every time, exactly as they had
ended at 4096. Raising the ceiling again would relabel the same behaviour, so the honest
disposition is to score them wrong.

Writing them as real rows -- rather than adjusting a denominator in one report -- is what
keeps the figures, the tables, the statistical tests and the released dataset telling the
same story. ``plot_results.py`` divides by ``len(rows)``; a denominator fixed only in
``paper_tables.py`` would leave the figures reading up to 2.7 points higher than the
tables beside them.

Backend faults are scored wrong too. Ollama returns HTTP 500 from its native output
parser when it cannot reduce a model's token stream, and in this roster that happens for
exactly one model -- the one that reasons at greatest length in mathematical notation --
and never once for the other seven. The model's own output is the necessary condition, so
the failure is charged to it. ``--exclude-backend`` keeps the older behaviour of leaving
them out of the denominator.

    python3 tools/score_unanswered.py --tier all           # report only
    python3 tools/score_unanswered.py --tier all --write
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
csv.field_size_limit(sys.maxsize)

from run_benchmark import (
    NO_ANSWER_LETTER,
    RAW_COLUMNS,
    SCORED_UNANSWERED,
    failure_results_path,
    raw_results_path,
)
from veltharrow.config import (
    MEDIUM_MODEL_SPECS,
    SMALL_MODEL_SPECS,
    resolve_tier,
    retrieval_conditions,
)

import paper_tables as pt

# NO_ANSWER_LETTER and SCORED_UNANSWERED are imported from run_benchmark above rather
# than defined here. run_benchmark.completed_keys has to recognise the rows this tool
# writes so --resume treats them as done; a second copy of the strings here would let
# the writer and the reader drift apart silently.


def benchmark_running() -> bool:
    """True if a benchmark run is live; rewriting under an appending writer loses rows."""
    try:
        out = subprocess.run(["pgrep", "-af", "run_benchmark.py"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out)


def score_tier(tier: str, write: bool, exclude_backend: bool = False) -> None:
    raw_path = raw_results_path(tier)
    fail_path = failure_results_path(tier)
    current = {c["retrieval_condition"] for c in retrieval_conditions(tier)}
    # failed_attempts.csv carries rows for models that were once run under this tier's
    # results directory but are not in its roster now (small-tier models appear in the
    # medium file). Injecting those would invent cells for models the tier never had.
    roster = {m.label for m in
              (MEDIUM_MODEL_SPECS if resolve_tier(tier) == "medium" else SMALL_MODEL_SPECS)}

    rows = list(csv.DictReader(raw_path.open(encoding="utf-8", newline="")))
    scored = {(r["model_label"], r["retrieval_condition"], r["question_id"]) for r in rows}

    # failed_attempts.csv is append-only, so the last record for a question is its
    # current disposition.
    latest: dict[tuple[str, str, str], dict] = {}
    if fail_path.exists():
        for r in csv.DictReader(fail_path.open(encoding="utf-8", newline="")):
            if r["retrieval_condition"] not in current or r["model_label"] not in roster:
                continue
            key = (r["model_label"], r["retrieval_condition"], r["question_id"])
            if key not in scored:
                latest[key] = r

    new_rows, skipped = [], Counter()
    for row in latest.values():
        cls = pt.classify_failure(row)
        if cls == "backend" and exclude_backend:
            skipped[row["model_label"]] += 1
            continue
        new_rows.append({
            **{c: row.get(c, "") for c in RAW_COLUMNS},
            "predicted_letter": NO_ANSWER_LETTER,
            "is_correct": "0",
            "error": f"{SCORED_UNANSWERED} ({cls})",
        })

    print(f"\n{tier}: {len(rows)} scored rows; {len(latest)} questions unanswered")
    print(f"  -> scoring {len(new_rows)} as wrong; "
          f"{sum(skipped.values())} left out of the denominator")
    for model, n in Counter(r["model_label"] for r in new_rows).most_common():
        print(f"     wrong    {model:34s} {n:4d}")
    for model, n in skipped.most_common():
        print(f"     excluded {model:34s} {n:4d}  (backend fault)")
    if not write or not new_rows:
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(raw_path, raw_path.with_name(f"{raw_path.name}.bak_{stamp}_pre_score_unanswered"))
    with raw_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writerows(new_rows)
    print(f"  appended {len(new_rows)} rows to {raw_path.name}; backup kept")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="all", help="small, medium, or all")
    parser.add_argument("--write", action="store_true", help="append the rows")
    parser.add_argument("--exclude-backend", action="store_true",
                        help="leave backend faults out of the denominator instead of "
                             "scoring them wrong")
    args = parser.parse_args(argv)

    if args.write and benchmark_running():
        print("REFUSING: run_benchmark.py is running and appends to this CSV.",
              file=sys.stderr)
        return 1

    for tier in (["small", "medium"] if args.tier == "all" else [resolve_tier(args.tier)]):
        score_tier(tier, args.write, args.exclude_backend)
    if not args.write:
        print("\n(dry run - pass --write to append)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
