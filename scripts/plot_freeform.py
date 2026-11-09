#!/usr/bin/env python3
"""
Plot free-form evaluation results from graded.csv.

Usage:
  python3 scripts/plot_freeform.py
  python3 scripts/plot_freeform.py --input results/freeform/graded.csv --out-dir results/freeform/plots
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "cache" / ".matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(sys.maxsize)

PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2", "#DB2777"]
GRAY    = "#6B7280"

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.facecolor": "#FAFAFA", "axes.edgecolor": "#D1D5DB",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
})

MODEL_SHORT = {
    "gemma4_e2b_thinking_on":   "Gemma4-E2B (think)",
    "gemma4_e4b_thinking_on":   "Gemma4-E4B (think)",
    "gemma4_12b_thinking_on":   "Gemma4-12B (think)",
    "qwen35_9b_thinking_on":    "Qwen3.5-9B (think)",
    "qwen3_1_7b_thinking_on":   "Qwen3-1.7B (think)",
    "qwen3_8b_thinking_on":     "Qwen3-8B (think)",
}

BUCKETS = ["answerable", "unanswerable", "typo", "math", "multichunk", "vague", "longcontext"]
METRICS = ["fact_correct", "hallucination_free", "source_grounded"]
METRIC_LABELS = {
    "fact_correct":       "Fact-correct",
    "hallucination_free": "Hallucination-free",
    "source_grounded":    "Source-grounded",
}


def load(path: Path) -> list[dict]:
    """Load the graded free-form CSV as a list of row dicts."""
    with path.open() as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    """Arithmetic mean, returning 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


# Output formats for save(): overridable via --formats (e.g. "png,pdf").
SAVE_FORMATS = ["png"]


def save(fig, out_dir: Path, name: str) -> None:
    """Write ``fig`` to ``out_dir`` in every configured SAVE_FORMATS extension and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    for ext in SAVE_FORMATS:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  {path.name}")
    plt.close(fig)


# Plot 1: per-model overall scores (3 metrics, grouped bars)
def plot_overall(rows, out_dir):
    """Grouped bar chart of each model's overall score on all three free-form metrics."""
    models = sorted({r["model_label"] for r in rows})
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 5))
    x = np.arange(len(models))
    w = 0.27
    for i, m in enumerate(METRICS):
        vals = [mean([float(r[m]) for r in rows if r["model_label"] == mod]) * 100 for mod in models]
        ax.bar(x + (i - 1) * w, vals, w, color=PALETTE[i], alpha=0.88, label=METRIC_LABELS[m])
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 1.0, f"{v:.0f}", ha="center", fontsize=7, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Free-form evaluation: overall per model", fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save(fig, out_dir, "ff_01_overall.png")


# Plot 2: heatmap (model × bucket × metric)
def plot_heatmap(rows, out_dir):
    """One model × bucket heatmap per metric, cells coloured by score."""
    models = sorted({r["model_label"] for r in rows})
    for metric in METRICS:
        data = np.full((len(models), len(BUCKETS)), np.nan)
        for i, m in enumerate(models):
            for j, b in enumerate(BUCKETS):
                vals = [float(r[metric]) for r in rows if r["model_label"] == m and r["bucket"] == b]
                if vals:
                    data[i, j] = mean(vals) * 100
        fig, ax = plt.subplots(figsize=(max(8, len(BUCKETS) * 1.2), max(3, len(models) * 0.7)))
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Score (%)")
        ax.set_xticks(range(len(BUCKETS)))
        ax.set_xticklabels(BUCKETS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=9)
        for i in range(len(models)):
            for j in range(len(BUCKETS)):
                v = data[i, j]
                if not np.isnan(v):
                    color = "white" if v < 35 or v > 75 else "#111827"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                            fontweight="bold", color=color)
        ax.set_title(f"{METRIC_LABELS[metric]}: model × bucket", fontweight="bold", pad=10)
        plt.tight_layout()
        save(fig, out_dir, f"ff_02_heatmap_{metric}.png")


# Plot 3: bucket-level grouped bars per metric
def plot_per_bucket(rows, out_dir):
    """One grouped bar chart per metric, comparing models across question buckets."""
    models = sorted({r["model_label"] for r in rows})
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(max(11, len(BUCKETS) * 1.5), 5))
        x = np.arange(len(BUCKETS))
        w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            vals = []
            for b in BUCKETS:
                sub = [float(r[metric]) for r in rows if r["model_label"] == m and r["bucket"] == b]
                vals.append(mean(sub) * 100)
            ax.bar(x + (i - len(models)/2 + 0.5) * w, vals, w,
                   color=PALETTE[i % len(PALETTE)], alpha=0.88,
                   label=MODEL_SHORT.get(m, m))
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKETS, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{METRIC_LABELS[metric]} by bucket", fontweight="bold", pad=12)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        plt.tight_layout()
        save(fig, out_dir, f"ff_03_buckets_{metric}.png")


# Plot 5: latency vs fact_correct
def plot_latency(rows, out_dir):
    """Scatter of each model's median latency vs fact-correct score."""
    sub = [r for r in rows if r.get("latency_seconds")]
    if not sub:
        return
    models = sorted({r["model_label"] for r in sub})
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(models):
        mrows = [r for r in sub if r["model_label"] == m]
        try:
            lats = [float(r["latency_seconds"]) for r in mrows]
            med = float(np.median(lats))
        except Exception:
            continue
        score = mean([float(r["fact_correct"]) for r in mrows]) * 100
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(med, score, s=120, color=color, alpha=0.88, edgecolors="white", linewidths=1.5)
        ax.annotate(MODEL_SHORT.get(m, m), xy=(med, score), xytext=(8, 5),
                    textcoords="offset points", fontsize=9, color=color, fontweight="bold")
    ax.set_xlabel("Median Latency per Question (s)")
    ax.set_ylabel("Fact-correct (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Free-form: Accuracy vs Latency", fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, out_dir, "ff_05_latency.png")


def main():
    """Load the graded free-form CSV and render all free-form figures to the output dir."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/freeform/graded.csv")
    parser.add_argument("--out-dir", default="results/freeform/plots")
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution (use 300 for publication)")
    parser.add_argument("--formats", default="png", help="Comma-separated output formats, e.g. 'png,pdf'")
    args = parser.parse_args()

    global SAVE_FORMATS
    SAVE_FORMATS = [f.strip() for f in args.formats.split(",") if f.strip()]
    plt.rcParams["savefig.dpi"] = args.dpi

    src = Path(args.input)
    if not src.exists():
        print(f"Input not found: {src}")
        return
    rows = [r for r in load(src) if r.get("fact_correct")]
    if not rows:
        print(f"No graded rows in {src}")
        return
    out_dir = Path(args.out_dir)
    print(f"Loaded {len(rows)} graded rows → {out_dir}/")
    plot_overall(rows, out_dir)
    plot_heatmap(rows, out_dir)
    plot_per_bucket(rows, out_dir)
    plot_latency(rows, out_dir)
    print(f"\nDone: plots in {out_dir}/")


if __name__ == "__main__":
    main()
