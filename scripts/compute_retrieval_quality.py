#!/usr/bin/env python3
"""
Post-hoc retrieval-quality computation.

Maps each MCQ question's `evidence` text to the chunk(s) that contain it (the
"gold chunks"), then scans raw_results.csv to label every row with:
 - gold_in_retrieved : did the retrieved set contain at least one gold chunk?
 - gold_rank         : 1-indexed position of the first gold chunk in the
                        retrieved list, or empty if not retrieved.

Outputs:
  data/mcq/gold_chunks.json                  : {question_id: [chunk_id, ...]} mapping
  results/<tier>/data/retrieval_quality.csv  : one row per source raw_results row

Usage:
  python3 scripts/compute_retrieval_quality.py --tier medium
  python3 scripts/compute_retrieval_quality.py --tier small
  python3 scripts/compute_retrieval_quality.py --tier medium --audit-sample 5
  python3 scripts/compute_retrieval_quality.py --rebuild-mapping     # ignore cached gold_chunks.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veltharrow.config import DOCUMENTS, results_dir_for  # noqa: E402
from veltharrow.rag import build_chunks  # noqa: E402


GOLD_PATH = ROOT / "data" / "mcq" / "gold_chunks.json"

OUT_COLUMNS = [
    "run_id", "timestamp",
    "document_id", "question_id",
    "model_label", "retrieval_condition", "top_k",
    "is_correct",
    "gold_chunk_count",
    "n_retrieved",
    "gold_in_retrieved",
    "gold_rank",
]


def _normalize_text(text: str) -> str:
    """Whitespace-collapse + lowercase. Keeps markdown punctuation so we still
    match `**bold**` evidence against chunks that preserve the formatting."""
    return re.sub(r"\s+", " ", text).strip().lower()


def build_gold_mapping(rebuild: bool = False) -> dict[str, list[str]]:
    """Map question_id → [chunk_id, ...]. Cached on disk; rebuild=True ignores."""
    if GOLD_PATH.exists() and not rebuild:
        return json.loads(GOLD_PATH.read_text())

    mapping: dict[str, list[str]] = {}
    audit = {"direct": 0, "markdown_stripped": 0, "no_match": 0}

    for spec in DOCUMENTS:
        chunks = build_chunks(spec.document_id, spec.source_path)
        chunk_norm = [(c.chunk_id, _normalize_text(c.text)) for c in chunks]
        chunk_clean = [(cid, re.sub(r"[*_`]", "", txt)) for cid, txt in chunk_norm]
        questions = json.loads(spec.questions_path.read_text())
        for q in questions:
            qid = q["id"]
            evidence = (q.get("evidence") or "").strip()
            if not evidence:
                mapping[qid] = []
                audit["no_match"] += 1
                continue
            ev_norm = _normalize_text(evidence)

            # Pass 1: direct (normalized) substring containment.
            hits = [cid for cid, txt in chunk_norm if ev_norm in txt]
            if hits:
                mapping[qid] = hits
                audit["direct"] += 1
                continue

            # Pass 2: strip markdown bold/italics/backticks and re-try.
            ev_clean = re.sub(r"[*_`]", "", ev_norm)
            hits = [cid for cid, txt in chunk_clean if ev_clean in txt]
            if hits:
                mapping[qid] = hits
                audit["markdown_stripped"] += 1
                continue

            # No fuzzy/longest-fragment fallback: risk of false positives is
            # higher than the marginal coverage gain. Leave it unmapped.
            mapping[qid] = []
            audit["no_match"] += 1

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.write_text(json.dumps(mapping, indent=2))
    total = sum(audit.values())
    print(
        f"Gold mapping: {audit['direct']} direct  "
        f"+ {audit['markdown_stripped']} markdown-stripped  "
        f"+ {audit['no_match']} unmapped  =  {total} questions",
        flush=True,
    )
    print(f"  written to {GOLD_PATH}", flush=True)
    return mapping


def audit_samples(mapping: dict[str, list[str]], n: int) -> None:
    """Print ``n`` random (question → gold chunk) pairs with their evidence/chunk text
    for a manual sanity-check of the gold mapping."""
    chunks_by_id: dict[str, str] = {}
    questions_by_id: dict[str, dict] = {}
    for spec in DOCUMENTS:
        for c in build_chunks(spec.document_id, spec.source_path):
            chunks_by_id[c.chunk_id] = c.text
        for q in json.loads(spec.questions_path.read_text()):
            questions_by_id[q["id"]] = q

    keys = [k for k, v in mapping.items() if v]
    random.seed(0)
    sample = random.sample(keys, min(n, len(keys)))
    print(f"\n--- audit: {n} random mapped questions ---")
    for qid in sample:
        q = questions_by_id[qid]
        evidence = (q.get("evidence") or "").strip()
        chunk_id = mapping[qid][0]
        chunk_text = chunks_by_id.get(chunk_id, "")
        has = _normalize_text(evidence).replace("*", "").replace("_", "") in \
              re.sub(r"[*_`]", "", _normalize_text(chunk_text))
        print(f"\n[{qid}] → {chunk_id}  evidence-in-chunk: {has}")
        print(f"  EVIDENCE: {evidence[:160]!r}")
        print(f"  CHUNK   : {chunk_text[:180]!r}")


def compute_for_tier(tier: str, mapping: dict[str, list[str]]) -> Path:
    """Label every RAG row of a tier's raw_results.csv with gold-chunk hit/rank and write
    ``retrieval_quality.csv``; prints any-k recall. Returns the output path (empty if the
    tier's results are missing)."""
    src = results_dir_for(tier) / "data" / "raw_results.csv"
    if not src.exists():
        print(f"  skip {tier}: {src} not found")
        return Path()
    out_path = results_dir_for(tier) / "data" / "retrieval_quality.csv"
    n_in, n_out, n_with_gold, n_hit = 0, 0, 0, 0

    with src.open() as f_in, out_path.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for row in reader:
            n_in += 1
            cond = row["retrieval_condition"]
            if cond == "no_rag":
                # No retrieval done; skip rows that have no retrieval to score.
                continue
            qid = row["question_id"]
            gold = mapping.get(qid, [])
            retrieved = [c for c in (row["retrieved_chunk_ids"] or "").split("|") if c]
            rank: str = ""
            for i, cid in enumerate(retrieved, 1):
                if cid in gold:
                    rank = str(i)
                    break
            in_retrieved = 1 if rank else 0
            if gold:
                n_with_gold += 1
                n_hit += in_retrieved
            writer.writerow({
                "run_id":               row.get("run_id", ""),
                "timestamp":            row.get("timestamp", ""),
                "document_id":          row["document_id"],
                "question_id":          qid,
                "model_label":          row["model_label"],
                "retrieval_condition":  cond,
                "top_k":                row.get("top_k", ""),
                "is_correct":           row.get("is_correct", ""),
                "gold_chunk_count":     len(gold),
                "n_retrieved":          len(retrieved),
                "gold_in_retrieved":    in_retrieved,
                "gold_rank":            rank,
            })
            n_out += 1

    recall = n_hit / n_with_gold * 100 if n_with_gold else float("nan")
    print(
        f"  {tier:8s}  in={n_in:>6d} rows  out={n_out:>6d} retrieval rows  "
        f"recall(any-k)={recall:.2f}% over {n_with_gold} mappable rows",
        flush=True,
    )
    print(f"    written: {out_path}", flush=True)
    return out_path


def main() -> None:
    """Build (or load) the gold-chunk mapping and compute retrieval quality for each
    requested tier, optionally auditing a sample of the mapping first."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", action="append", default=[],
                        help="Tier(s) to score. Default: small + medium.")
    parser.add_argument("--rebuild-mapping", action="store_true",
                        help="Ignore cached data/mcq/gold_chunks.json and rebuild.")
    parser.add_argument("--audit-sample", type=int, default=0,
                        help="Print N random mapped (question, chunk) pairs for sanity-check.")
    args = parser.parse_args()

    tiers = args.tier or ["small", "medium"]

    mapping = build_gold_mapping(rebuild=args.rebuild_mapping)
    mappable = sum(1 for v in mapping.values() if v)
    unmappable = sorted([k for k, v in mapping.items() if not v])
    print(f"  mappable: {mappable}/{len(mapping)} questions"
          f"  ({len(unmappable)} have no chunk match)", flush=True)
    if unmappable[:10]:
        print(f"  unmapped (first 10): {', '.join(unmappable[:10])}", flush=True)

    if args.audit_sample:
        audit_samples(mapping, args.audit_sample)

    print("\nScoring tiers:")
    for tier in tiers:
        compute_for_tier(tier, mapping)


if __name__ == "__main__":
    main()
