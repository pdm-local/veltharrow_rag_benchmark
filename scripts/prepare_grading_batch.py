#!/usr/bin/env python3
"""
Prepare anonymized JSON batches to paste into Claude.ai for blind grading.

Usage:
  # split into 10 batches for 10 separate chats (your plan):
  python3 scripts/prepare_grading_batch.py \
    --input results/freeform/answers.csv \
    --batches 10 \
    --out-dir results/freeform/grading_batches

  # OR one batch with everything:
  python3 scripts/prepare_grading_batch.py --batches 1

After Claude returns the graded JSON, save each reply as
results/freeform/grading_batches/batch_NN.graded.json
then run:
  python3 scripts/prepare_grading_batch.py --merge --out results/freeform/graded.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def load_answers(path: Path) -> list[dict]:
    """Load the free-form answers CSV as a list of row dicts."""
    with path.open() as f:
        return list(csv.DictReader(f))


def make_batch(rows: list[dict]) -> list[dict]:
    """Build an anonymized grading batch (id, bucket, question, gold, context, answer)
    from answer rows — no model identity, for blind grading."""
    out = []
    for r in rows:
        out.append({
            "id": r["_row_id"],
            "bucket": r["bucket"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "context_chunks": r["retrieved_chunk_ids"],
            "answer": r["raw_answer"],
        })
    return out


def cmd_split(args) -> None:
    """Split gradable answers into N anonymized JSON batches (plus a row-id keymap for
    later merging) for blind grading in separate Claude.ai chats."""
    rows = load_answers(Path(args.input))
    # Skip failed/empty rows: nothing to grade.
    rows = [r for r in rows if r.get("raw_answer") and not r.get("error")]
    for i, r in enumerate(rows, 1):
        r["_row_id"] = f"row-{i:04d}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the id->original-row map so we can merge later.
    keymap_path = out_dir / "row_keymap.json"
    with keymap_path.open("w") as f:
        json.dump(
            [{"_row_id": r["_row_id"],
              "model_label": r["model_label"],
              "question_id": r["question_id"],
              "retrieval_mode": r["retrieval_mode"]} for r in rows],
            f, indent=2,
        )

    batch_size = (len(rows) + args.batches - 1) // args.batches
    for b in range(args.batches):
        chunk = rows[b * batch_size : (b + 1) * batch_size]
        if not chunk:
            continue
        batch_path = out_dir / f"batch_{b+1:02d}.json"
        with batch_path.open("w") as f:
            json.dump(make_batch(chunk), f, indent=2, ensure_ascii=False)
        print(f"  {batch_path.name}  {len(chunk)} rows")

    print(f"\nSaved {args.batches} batches under {out_dir}/")
    print(f"Row-id map: {keymap_path}")
    print()
    print("Workflow for each batch:")
    print("  1. Open scripts/grade_freeform_prompt.txt: copy its text.")
    print("  2. Open a NEW Claude.ai chat.")
    print("  3. Paste the prompt; replace <<<PASTE_JSON_HERE>>> with batch_NN.json contents.")
    print("  4. Save Claude's JSON reply as batch_NN.graded.json in the same dir.")
    print("  5. Repeat for each batch (10 chats per your plan).")
    print(f"  6. When done: python3 scripts/prepare_grading_batch.py --merge --input {args.input} --out results/freeform/graded.csv")


def cmd_merge(args) -> None:
    """Merge the returned ``batch_*.graded.json`` files back into a graded CSV, averaging
    the three metrics across any repeated gradings via the row-id keymap."""
    in_dir = Path(args.out_dir)
    keymap_path = in_dir / "row_keymap.json"
    if not keymap_path.exists():
        raise SystemExit(f"Missing {keymap_path}: run --split first")
    keymap = {entry["_row_id"]: entry for entry in json.loads(keymap_path.read_text())}

    grades_by_id: dict[str, list[dict]] = {}
    for graded_file in sorted(in_dir.glob("batch_*.graded.json")):
        try:
            graded = json.loads(graded_file.read_text())
        except json.JSONDecodeError as exc:
            print(f"  skip {graded_file.name}: bad JSON ({exc})")
            continue
        for entry in graded:
            grades_by_id.setdefault(entry["id"], []).append(entry)
        print(f"  loaded {graded_file.name}: {len(graded)} grades")

    rows = load_answers(Path(args.input))
    rows_by_key = {(r["model_label"], r["question_id"], r["retrieval_mode"]): r for r in rows}

    out_rows = []
    for row_id, runs in grades_by_id.items():
        meta = keymap.get(row_id)
        if not meta:
            continue
        src = rows_by_key.get((meta["model_label"], meta["question_id"], meta["retrieval_mode"]))
        if not src:
            continue
        # Average across multiple chats if the same row was graded more than once
        n = len(runs)
        avg = {
            "fact_correct":       sum(int(g.get("fact_correct", 0))       for g in runs) / n,
            "hallucination_free": sum(int(g.get("hallucination_free", 0)) for g in runs) / n,
            "source_grounded":    sum(int(g.get("source_grounded", 0))    for g in runs) / n,
        }
        notes = " | ".join(g.get("note", "") for g in runs if g.get("note"))
        out_rows.append({
            **src,
            "n_grades": n,
            "fact_correct":       f"{avg['fact_correct']:.3f}",
            "hallucination_free": f"{avg['hallucination_free']:.3f}",
            "source_grounded":    f"{avg['source_grounded']:.3f}",
            "grader_notes": notes,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        print("No grades merged: did you save batch_NN.graded.json files?")
        return
    fieldnames = list(out_rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nMerged {len(out_rows)} graded rows → {out_path}")


def main():
    """Dispatch to the split (default) or ``--merge`` sub-command for the manual blind-grading
    workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/freeform/answers.csv")
    parser.add_argument("--out-dir", default="results/freeform/grading_batches")
    parser.add_argument("--out", default="results/freeform/graded.csv")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--merge", action="store_true", help="Merge graded JSON files into final CSV")
    args = parser.parse_args()
    if args.merge:
        cmd_merge(args)
    else:
        cmd_split(args)


if __name__ == "__main__":
    main()
