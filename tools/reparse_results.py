#!/usr/bin/env python3
"""Re-score an existing ``raw_results.csv`` with the current ``parse_answer``.

The MCQ harness stores every model's full raw response, so a parser fix can be applied
to completed runs without spending any inference: the stored text is re-parsed, the
letter and ``is_correct`` recomputed, and the row rewritten. Rows whose corrected parse
yields no letter are *unanswered*, not wrong — they are marked with an error so
``--resume`` re-runs them, and they are listed in the audit report.

The original CSV is copied to ``<name>.prereparse.bak`` before anything is written.

    python3 tools/reparse_results.py --tier medium          # report only
    python3 tools/reparse_results.py --tier all --write     # rewrite in place
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
csv.field_size_limit(sys.maxsize)

from run_benchmark import (
    FAILURE_COLUMNS,
    RAW_COLUMNS,
    failure_results_path,
    load_questions,
    parse_answer,
    raw_results_path,
)
from veltharrow.config import DOCUMENTS, resolve_tier

UNANSWERED_ERROR = "No parseable A-F answer returned (reparse)"
# Rows written by tools/score_unanswered.py: the question was already adjudicated as
# unanswered and scored wrong, and carries a sentinel letter rather than a real response.
# Re-parsing them would strip the sentinel and reopen a decision that is already made.
SCORED_UNANSWERED_PREFIX = "scored-wrong:"


def question_index() -> dict[tuple[str, str], dict]:
    """(document_id, question_id) -> question dict, for the option-text fallback."""
    index = {}
    for spec in DOCUMENTS:
        for question in load_questions(spec.questions_path):
            index[(spec.document_id, question["QuestionID"])] = question
    return index


def reparse_tier(tier: str, write: bool) -> dict:
    path = raw_results_path(tier)
    if not path.exists():
        raise FileNotFoundError(path)
    questions = question_index()

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    per_model = defaultdict(lambda: {"n": 0, "old": 0, "new": 0, "changed": 0, "blank": 0})
    transitions = defaultdict(int)
    unanswered = []

    for row in rows:
        if (row.get("error") or "").startswith(SCORED_UNANSWERED_PREFIX):
            # Already adjudicated as unanswered and scored wrong; its stored response is
            # the failed generation, so re-parsing would only strip the sentinel letter.
            per_model[row["model_label"]]["n"] += 1
            continue
        gold = (row["correct_letter"] or "").strip().upper()
        old_letter = (row["predicted_letter"] or "").strip().upper()
        question = questions.get((row["document_id"], row["question_id"]))
        new_letter = parse_answer(row["raw_response"] or "", question=question)

        stats = per_model[row["model_label"]]
        stats["n"] += 1
        stats["old"] += old_letter == gold
        stats["new"] += new_letter == gold
        if new_letter != old_letter:
            stats["changed"] += 1
            transitions[(old_letter or "-", new_letter or "-")] += 1
            if not new_letter:
                stats["blank"] += 1
                unanswered.append(row)

        row["predicted_letter"] = new_letter
        row["is_correct"] = str(bool(new_letter) and new_letter == gold)
        if not new_letter:
            row["error"] = UNANSWERED_ERROR

    if write:
        # Never clobber an existing pre-fix backup: a second --write run would
        # otherwise overwrite the only copy of the original parses.
        backup = path.with_suffix(".csv.prereparse.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return {
        "tier": tier,
        "path": path,
        "rows": len(rows),
        "per_model": per_model,
        "transitions": transitions,
        "unanswered": unanswered,
    }


def report(result: dict) -> None:
    per_model = result["per_model"]
    n = result["rows"]
    old = sum(s["old"] for s in per_model.values())
    new = sum(s["new"] for s in per_model.values())
    changed = sum(s["changed"] for s in per_model.values())
    blank = sum(s["blank"] for s in per_model.values())

    print(f"\n=== {result['tier']}: {n} rows ({result['path']})")
    print(f"    reparsed differently: {changed} ({100 * changed / n:.2f}%), "
          f"of which now unanswered: {blank}")
    print(f"    accuracy {100 * old / n:.2f}% -> {100 * new / n:.2f}% "
          f"({100 * (new - old) / n:+.2f} pp)\n")
    print(f"    {'model':38s} {'n':>5} {'old':>8} {'new':>8} {'delta':>7} {'chg':>5} {'unans':>6}")
    for label, s in sorted(per_model.items(), key=lambda kv: kv[0]):
        delta = 100 * (s["new"] - s["old"]) / s["n"]
        flag = "  <<<" if abs(delta) >= 1.0 or s["blank"] else ""
        print(f"    {label:38s} {s['n']:5d} {100 * s['old'] / s['n']:7.2f}% "
              f"{100 * s['new'] / s['n']:7.2f}% {delta:+6.2f} {s['changed']:5d} "
              f"{s['blank']:6d}{flag}")

    if result["transitions"]:
        print("\n    top letter transitions (old -> new):")
        for (a, b), count in sorted(result["transitions"].items(), key=lambda kv: -kv[1])[:10]:
            print(f"      {a} -> {b}: {count}")

    if result["unanswered"]:
        cells = defaultdict(int)
        for row in result["unanswered"]:
            cells[(row["model_label"], row["retrieval_condition"])] += 1
        print(f"\n    unanswered rows needing rerun, by cell ({len(result['unanswered'])} total):")
        for (label, condition), count in sorted(cells.items(), key=lambda kv: -kv[1]):
            print(f"      {label:38s} {condition:24s} {count:4d}")


def quarantine_tier(tier: str) -> int:
    """Move reparse-unanswered rows out of raw_results.csv into failed_attempts.csv.

    ``run_benchmark.py`` appends, so leaving the dead rows in place would give every
    rerun a duplicate composite key. Moving them keeps the canonical CSV free of rows
    that carry no answer, preserves them for audit, and lets ``--resume`` refill the
    gaps cleanly.
    """
    raw_path = raw_results_path(tier)
    fail_path = failure_results_path(tier)
    stamp = datetime.now(timezone.utc).isoformat()

    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keep = [r for r in rows if r.get("error") != UNANSWERED_ERROR]
    moved = [r for r in rows if r.get("error") == UNANSWERED_ERROR]
    if not moved:
        print(f"{tier}: nothing to quarantine")
        return 0

    header_needed = not fail_path.exists() or fail_path.stat().st_size == 0
    with fail_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS, extrasaction="ignore")
        if header_needed:
            writer.writeheader()
        for row in moved:
            writer.writerow({**row, "failure_timestamp": stamp})

    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(keep)

    print(f"{tier}: quarantined {len(moved)} unanswered rows -> {fail_path.name} "
          f"({len(keep)} rows remain in raw_results.csv)")
    return len(moved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="medium", help="small, medium, or all")
    parser.add_argument("--write", action="store_true",
                        help="rewrite the CSV in place (a .prereparse.bak is kept)")
    parser.add_argument("--quarantine", action="store_true",
                        help="after writing, move unanswered rows to failed_attempts.csv "
                             "so --resume refills them without duplicate keys")
    args = parser.parse_args(argv)

    tiers = ["small", "medium"] if args.tier == "all" else [resolve_tier(args.tier)]
    for tier in tiers:
        report(reparse_tier(tier, args.write))
        if args.quarantine:
            if not args.write:
                parser.error("--quarantine requires --write")
            quarantine_tier(tier)
    if not args.write:
        print("\n(dry run - pass --write to rewrite the CSVs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
