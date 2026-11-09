#!/usr/bin/env python3
"""
Recompute every headline number that appears on the plots, straight from the
raw CSV. Lets you cross-check the figures by eye.

Usage:
  python3 scripts/validate_plot_numbers.py --tier medium                   # full
  python3 scripts/validate_plot_numbers.py --tier medium --fair            # qwen3-embedding:0.6b @ k=1/3/5 only
  python3 scripts/validate_plot_numbers.py --tier small --model gemma4_e2b_thinking_on
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from veltharrow.config import results_dir_for  # noqa: E402

FAIR = {
    "no_rag",
    "rag_qwen3-embedding:0.6b_k1",
    "rag_qwen3-embedding:0.6b_k3",
    "rag_qwen3-embedding:0.6b_k5",
}


def is_correct(row: dict) -> bool:
    """True if a raw_results row is marked correct (handles ``1``/``True``/``true``)."""
    return str(row.get("is_correct", "")).strip() in ("1", "True", "true")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (as percentages) for ``k`` correct of ``n``. Kept local and
    stdlib-only so this validator runs without the src package."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, (centre - margin) * 100), min(100, (centre + margin) * 100)


def fmt_pct(k: int, n: int) -> str:
    """Format ``k/n`` as a percentage with its Wilson interval and the raw k/n counts."""
    if n == 0:
        return "    n/a (0)"
    p = k / n * 100
    lo, hi = wilson(k, n)
    return f"{p:6.2f}%  [{lo:5.1f}, {hi:5.1f}]  k={k}/n={n}"


def main() -> None:
    """Recompute and print every headline figure (per-model accuracy, the model×condition
    matrix, per-document and per-Bloom accuracy, throughput) from a tier's raw_results.csv,
    so the plots can be cross-checked by eye."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="medium")
    parser.add_argument("--input", default="", help="Override CSV path")
    parser.add_argument("--fair", action="store_true",
                        help="Restrict to qwen3-embedding:0.6b @ k=1/3/5")
    parser.add_argument("--model", default="", help="Only show this model_label")
    args = parser.parse_args()

    src = Path(args.input) if args.input else (results_dir_for(args.tier) / "data" / "raw_results.csv")
    if not src.exists():
        print(f"Input not found: {src}")
        return

    with src.open() as f:
        rows = list(csv.DictReader(f))
    if args.fair:
        rows = [r for r in rows if r["retrieval_condition"] in FAIR]
    if args.model:
        rows = [r for r in rows if r["model_label"] == args.model]

    print(f"\nSource    : {src}")
    print(f"Filter    : {'fair (qwen3-0.6b only)' if args.fair else 'full'}"
          + (f"  ·  model={args.model}" if args.model else ""))
    print(f"Rows      : {len(rows):,}")
    if not rows:
        return

    models = sorted({r["model_label"] for r in rows})
    conds = sorted({r["retrieval_condition"] for r in rows})
    docs = sorted({r["document_id"] for r in rows})

    print(f"\n{'='*112}")
    print("  HEADLINE: overall accuracy per model (matches plot top5_01 / per-model dashboard #6)")
    print("="*112)
    print(f"{'model':38s} {'overall':>32s} {'best_cond':35s} {'best':>6s}")
    print("-"*112)
    for m in models:
        mrows = [r for r in rows if r["model_label"] == m]
        k = sum(1 for r in mrows if is_correct(r))
        by_c = defaultdict(list)
        for r in mrows:
            by_c[r["retrieval_condition"]].append(r)
        best_c, best_a, best_n = "", -1.0, 0
        for c, lst in by_c.items():
            kc = sum(1 for r in lst if is_correct(r))
            if not lst:
                continue
            a = kc / len(lst) * 100
            if a > best_a:
                best_c, best_a, best_n = c, a, len(lst)
        print(f"{m:38s} {fmt_pct(k, len(mrows)):>32s} {best_c:35s} {best_a:5.2f}%")

    print(f"\n{'='*112}")
    print("  MATRIX: accuracy per (model × condition)  (matches plot 03 heatmap / per-model dashboard #1)")
    print("="*112)
    cell_w = max(20, max(len(c) for c in conds) + 2)
    header = f"{'model':38s}" + "".join(f"{c:>{cell_w}s}" for c in conds)
    print(header[:200])
    print("-"*min(len(header), 200))
    for m in models:
        cells = [f"{m:38s}"]
        for c in conds:
            sub = [r for r in rows if r["model_label"] == m and r["retrieval_condition"] == c]
            kc = sum(1 for r in sub if is_correct(r))
            txt = f"{kc/len(sub)*100:5.1f}% ({kc}/{len(sub)})" if sub else "    -"
            cells.append(f"{txt:>{cell_w}s}")
        line = "".join(cells)
        print(line[:200])

    print(f"\n{'='*112}")
    print("  PER-DOCUMENT: best-RAG accuracy per (model × doc) (matches per-model dashboard #2)")
    print("="*112)
    print(f"{'model':38s}" + "".join(f"{d:>14s}" for d in docs))
    print("-"*112)
    for m in models:
        mrows = [r for r in rows if r["model_label"] == m]
        rag = {}
        for r in mrows:
            if r["retrieval_condition"] != "no_rag":
                rag.setdefault(r["retrieval_condition"], []).append(r)
        best_c = max(rag, key=lambda c: sum(1 for r in rag[c] if is_correct(r)) / len(rag[c])) if rag else None
        cells = [f"{m:38s}"]
        for d in docs:
            sub = [r for r in mrows
                   if r["document_id"] == d and r["retrieval_condition"] == best_c]
            if sub:
                kc = sum(1 for r in sub if is_correct(r))
                cells.append(f"{kc/len(sub)*100:>9.1f}% ({kc:>2d})")
            else:
                cells.append(f"{'-':>14s}")
        print("".join(cells))

    print(f"\n{'='*112}")
    print("  PER-BLOOM: accuracy by Bloom level, all conditions pooled (matches dashboard #3, plot 08)")
    print("="*112)
    blooms_set = sorted({(r.get('bloom_level') or '').strip().lower() for r in rows} - {''})
    if blooms_set:
        print(f"{'model':38s}" + "".join(f"{b:>14s}" for b in blooms_set))
        print("-"*112)
        for m in models:
            cells = [f"{m:38s}"]
            for b in blooms_set:
                sub = [r for r in rows
                       if r["model_label"] == m
                       and (r.get("bloom_level") or "").strip().lower() == b]
                if sub:
                    kc = sum(1 for r in sub if is_correct(r))
                    cells.append(f"{kc/len(sub)*100:>9.1f}% ({kc:>2d})")
                else:
                    cells.append(f"{'-':>14s}")
            print("".join(cells))

    print(f"\n{'='*112}")
    print("  THROUGHPUT: median eval & prompt tok/s per model (matches per-model dashboard panel 4)")
    print("="*112)
    import math as _math
    print(f"{'model':38s} {'eval tok/s (med)':>18s} {'prompt tok/s (med)':>20s} {'n_tps_rows':>12s}")
    print("-"*112)
    for m in models:
        mrows_m = [r for r in rows if r["model_label"] == m]
        eval_tps, prompt_tps = [], []
        for r in mrows_m:
            try:
                lat = float(r.get("latency_seconds") or 0)
                if lat <= 0:
                    continue
                ec = r.get("eval_count") or ""
                pc = r.get("prompt_eval_count") or ""
                if ec:
                    eval_tps.append(float(ec) / lat)
                if pc:
                    prompt_tps.append(float(pc) / lat)
            except Exception:
                pass
        def _med(lst: list) -> str:
            """Format the median of a list of tokens/sec values (or n/a if empty)."""
            if not lst:
                return "      n/a"
            s = sorted(lst)
            mid = len(s) // 2
            v = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
            return f"{v:>8.1f}"
        print(f"{m:38s} {_med(eval_tps):>18s} {_med(prompt_tps):>20s} {len(eval_tps):>12d}")

    print(f"\nDone. Cross-check against the relevant plot file in {results_dir_for(args.tier)}/plots/")


if __name__ == "__main__":
    main()
