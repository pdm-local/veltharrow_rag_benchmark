#!/usr/bin/env python3
"""
Recognition-vs-production figure: for each free-form model, plot best-RAG MCQ
accuracy (recognition) against free-form source-grounding (grounded production).
Points fall well below the y=x line, visualizing the core thesis. Read-only inputs.

Output: figures/main/recognition_vs_production.png. The manuscript reads this through
paper/article/figures, a symlink to figures/, so no second copy is written: a duplicate
under paper/ is exactly how these plots went stale once before.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[1]
FAIR_RAG = {"rag_qwen3-embedding:0.6b_k1", "rag_qwen3-embedding:0.6b_k3",
            "rag_qwen3-embedding:0.6b_k5"}

DISPLAY = {
    "gemma4_e4b_thinking_on": "gemma4:e4b",
    "qwen35_9b_thinking_on": "qwen3.5:9b",
    "gemma4_12b_thinking_on": "gemma4:12b",
    "qwen3_8b_thinking_on": "qwen3:8b",
    "gemma4_e2b_thinking_on": "gemma4:e2b",
    "qwen3_1_7b_thinking_on": "qwen3:1.7b",
}

# (dx, dy, horizontal alignment) in offset points, per label. The four strongest models
# land within a point of each other on the x axis, so they need separate directions.
LABEL_OFFSETS = {
    "gemma4:e4b": (8, 2, "left"),
    "qwen3.5:9b": (-9, 2, "right"),
    "gemma4:12b": (8, -11, "left"),
    "qwen3:8b": (8, -3, "left"),
    "gemma4:e2b": (8, -3, "left"),
    "qwen3:1.7b": (8, 3, "left"),
}


def load(p):
    """Read a result CSV into a list of row dicts."""
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_rag_mcq(model: str) -> float | None:
    """Best fair-subset RAG accuracy for a model, searching both tiers."""
    for tier in ("medium", "small"):
        rows = [r for r in load(ROOT / "results" / tier / "data" / "raw_results.csv")
                if r["model_label"] == model and r["retrieval_condition"] in FAIR_RAG]
        if not rows:
            continue
        by = defaultdict(list)
        for r in rows:
            by[r["retrieval_condition"]].append(r["is_correct"] in ("1", "True", "true"))
        return 100 * max(sum(v) / len(v) for v in by.values())
    return None


def main() -> None:
    """Render the recognition-vs-production figure: each model's best-RAG MCQ accuracy
    against its free-form source-grounding, with the y=x line and dominated region, written
    to figures/ and paper/figures/; also prints the per-model gap."""
    g = load(ROOT / "results" / "freeform" / "graded.csv")
    ff = defaultdict(lambda: defaultdict(list))
    for r in g:
        for m in ("fact_correct", "hallucination_free", "source_grounded"):
            ff[r["model_label"]][m].append(float(r[m]))

    pts = []
    for model, disp in DISPLAY.items():
        mcq = best_rag_mcq(model)
        if mcq is None or not ff[model]["source_grounded"]:
            raise SystemExit(f"no MCQ or graded free-form rows found for {model}")
        grounded = 100 * sum(ff[model]["source_grounded"]) / len(ff[model]["source_grounded"])
        pts.append((mcq, grounded, disp))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lo, hi = 45, 100
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#9CA3AF", zorder=1,
            label="recognition = production")
    ax.fill_between([lo, hi], [lo, hi], lo, color="#FEE2E2", alpha=0.5, zorder=0)
    for mcq, grounded, disp in pts:
        ax.scatter(mcq, grounded, s=70, color="#2563EB", zorder=3, edgecolor="white", lw=0.8)
        dx, dy, ha = LABEL_OFFSETS.get(disp, (7, 5, "left"))
        ax.annotate(disp, (mcq, grounded), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, ha=ha)
    ax.set_xlabel("Recognition: best-RAG multiple-choice accuracy (%)")
    ax.set_ylabel("Production: free-form source-grounding (%)")
    # A little headroom on x so the right-hand labels are not clipped by the axis.
    ax.set_xlim(lo, hi + 5)
    ax.set_ylim(lo, hi)
    ax.set_title("Recognizing answers is easier than grounding them")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()

    out = ROOT / "figures" / "main" / "recognition_vs_production.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    for mcq, grounded, disp in sorted(pts, key=lambda t: -t[0]):
        print(f"  {disp:12s} MCQ={mcq:.1f}%  grounded={grounded:.1f}%  gap={mcq-grounded:+.1f}")


if __name__ == "__main__":
    main()
