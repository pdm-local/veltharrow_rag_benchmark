#!/usr/bin/env python3
"""
Merge an incoming raw_results.csv into a primary one, deduped by composite key.

Usage:
    python tools/merge_results.py PRIMARY INCOMING

Key: (model_label, document_id, question_id, retrieval_condition)
Rows in INCOMING that are not already in PRIMARY are appended. PRIMARY is
rewritten in place.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


KEY = ("model_label", "document_id", "question_id", "retrieval_condition")


def main() -> int:
    """Append rows from INCOMING not already in PRIMARY (deduped on the composite KEY) and
    rewrite PRIMARY in place; returns a process exit code."""
    p = argparse.ArgumentParser()
    p.add_argument("primary", type=Path)
    p.add_argument("incoming", type=Path)
    args = p.parse_args()

    if not args.incoming.exists():
        print(f"incoming not found: {args.incoming}")
        return 1

    rows: list[dict] = []
    if args.primary.exists():
        with args.primary.open(newline="") as f:
            rows = list(csv.DictReader(f))
    seen = {tuple(r[k] for k in KEY) for r in rows}
    added = 0
    with args.incoming.open(newline="") as f:
        for r in csv.DictReader(f):
            k = tuple(r[col] for col in KEY)
            if k in seen:
                continue
            rows.append(r)
            seen.add(k)
            added += 1

    if not rows:
        print("nothing to write")
        return 0

    args.primary.parent.mkdir(parents=True, exist_ok=True)
    with args.primary.open("w", newline="") as f:
        # Header comes from the primary's first row; extra columns in incoming
        # rows are dropped rather than aborting the rewrite mid-file.
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"added {added} rows; total {len(rows)} in {args.primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
