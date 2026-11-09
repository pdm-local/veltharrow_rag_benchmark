#!/usr/bin/env python3
"""
Benchmark visualization suite: generates publication-quality plots.

Usage:
  python3 plot_results.py
  python3 plot_results.py --input results/raw_results.csv --out-dir results/plots
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "cache" / ".matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from veltharrow.stats import wilson_ci

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Palette & style ────────────────────────────────────────────────────────────
# Curated palette: distinct, colour-blind-friendlier, softer than primaries.
PALETTE = ["#3B82F6", "#10B981", "#EF4444", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]
GRAY    = "#6B7280"
GRAY_D  = "#374151"
LIGHT   = "#F3F4F6"
BG      = "#FBFBFC"
EDGE    = "#D1D5DB"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 10.5,
    "axes.labelcolor": GRAY_D,
    "axes.facecolor": BG,
    "axes.edgecolor": EDGE,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.85,
    "xtick.color": GRAY_D,
    "ytick.color": GRAY_D,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor": EDGE,
    "legend.fontsize": 9,
    "legend.labelcolor": GRAY_D,
    "patch.linewidth": 0,
})

# ── Model / label mappings ─────────────────────────────────────────────────────
MODEL_SHORT = {
    # medium tier
    "qwen3_8b_thinking_off":        "Qwen3-8B",
    "qwen3_8b_thinking_on":         "Qwen3-8B\n(think)",
    "qwen35_9b_thinking_off":       "Qwen3.5-9B",
    "qwen35_9b_thinking_on":        "Qwen3.5-9B\n(think)",
    "gemma4_e4b_thinking_off":      "Gemma4-E4B",
    "gemma4_e4b_thinking_on":       "Gemma4-E4B\n(think)",
    "gemma4_12b_thinking_off":      "Gemma4-12B",
    "gemma4_12b_thinking_on":       "Gemma4-12B\n(think)",
    "qwen25_7b_thinking_off":       "Qwen2.5-7B",
    "qwen25_7b_thinking_on":        "Qwen2.5-7B\n(think)",
    "llama31_8b_thinking_off":      "Llama3.1-8B",
    "llama31_8b_thinking_on":       "Llama3.1-8B\n(think)",
    "mistral_7b_thinking_off":      "Mistral-7B",
    "deepseek_r1_7b_thinking_on":   "DS-R1-7B\n(think)",
    # small tier (13 models × {off, on})
    "qwen3_0_6b_thinking_off":          "Qwen3-0.6B",
    "qwen3_0_6b_thinking_on":           "Qwen3-0.6B\n(think)",
    "qwen35_0_8b_thinking_off":         "Qwen3.5-0.8B",
    "qwen35_0_8b_thinking_on":          "Qwen3.5-0.8B\n(think)",
    "llama32_1b_thinking_off":          "Llama3.2-1B",
    "llama32_1b_thinking_on":           "Llama3.2-1B\n(think)",
    "qwen25_1_5b_thinking_off":         "Qwen2.5-1.5B",
    "qwen25_1_5b_thinking_on":          "Qwen2.5-1.5B\n(think)",
    "deepseek_r1_1_5b_thinking_off":    "DS-R1-1.5B",
    "deepseek_r1_1_5b_thinking_on":     "DS-R1-1.5B\n(think)",
    "qwen3_1_7b_thinking_off":          "Qwen3-1.7B",
    "qwen3_1_7b_thinking_on":           "Qwen3-1.7B\n(think)",
    "gemma3_1b_thinking_off":           "Gemma3-1B",
    "gemma3_1b_thinking_on":            "Gemma3-1B\n(think)",
    "gemma4_e2b_thinking_off":          "Gemma4-E2B",
    "gemma4_e2b_thinking_on":           "Gemma4-E2B\n(think)",
    "qwen35_2b_thinking_off":           "Qwen3.5-2B",
    "qwen35_2b_thinking_on":            "Qwen3.5-2B\n(think)",
    "llama32_3b_thinking_off":          "Llama3.2-3B",
    "llama32_3b_thinking_on":           "Llama3.2-3B\n(think)",
    "phi4_mini_thinking_off":           "Phi4-Mini",
    "phi4_mini_thinking_on":            "Phi4-Mini\n(think)",
    "phi4_mini_reasoning_thinking_off": "Phi4-Mini-R",
    "phi4_mini_reasoning_thinking_on":  "Phi4-Mini-R\n(think)",
    "gemma3_4b_thinking_off":           "Gemma3-4B",
    "gemma3_4b_thinking_on":            "Gemma3-4B\n(think)",
}

EMBEDDER_SHORT = {
    "nomic-embed-text":      "nomic",
    "qwen3-embedding:0.6b":  "qwen3-0.6b",
    "qwen3-embedding:4b":    "qwen3-4b",
    "qwen3-embedding:8b":    "qwen3-8b",
}

EMBEDDER_COLORS = {
    "nomic-embed-text":      PALETTE[0],
    "qwen3-embedding:0.6b":  PALETTE[1],
    "qwen3-embedding:4b":    PALETTE[3],
    "qwen3-embedding:8b":    PALETTE[2],
}

DOC_LABELS = {
    "charter":    "Municipal\nCharter",
    "budget":     "Annual\nBudget",
    "zoning":     "Zoning\nOrdinance",
    "ordinances": "General\nOrdinances",
}

BLOOM_ORDER = ["remember", "understand", "apply", "analyze", "analyse", "evaluate", "create"]


# ── Stats helpers ──────────────────────────────────────────────────────────────
# wilson_ci lives in veltharrow.stats (shared with tools/paper_tables.py).


def acc(rows: list[dict]) -> float:
    """Accuracy (%) over ``rows``, counting the ``is_correct`` column as truthy
    for the strings ``1`` / ``True`` / ``true``. Empty input returns 0.0."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if str(r.get("is_correct", "")).strip() in ("1", "True", "true")) / len(rows) * 100


def acc_ci(rows: list[dict]) -> tuple[float, float, float]:
    """Returns (accuracy_pct, ci_low_pct, ci_high_pct)."""
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0.0
    correct = sum(1 for r in rows if str(r.get("is_correct", "")).strip() in ("1", "True", "true"))
    a = correct / n * 100
    lo, hi = wilson_ci(correct, n)
    return a, lo, hi


def group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    """Group rows into a dict keyed by column ``key``."""
    out: dict[str, list] = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return dict(out)


def ordered_models(rows: list[dict]) -> list[str]:
    """Model labels in first-seen order (preserves roster order for plots)."""
    seen: list[str] = []
    for r in rows:
        m = r["model_label"]
        if m not in seen:
            seen.append(m)
    return seen


def grid_shape(n: int, max_cols: int = 5) -> tuple[int, int]:
    """Choose (nrows, ncols) so panels wrap to multiple rows once n is large.

    Among column counts up to ``max_cols``, pick the one whose last row is
    fullest (fewest empty trailing cells), breaking ties toward fewer rows and
    then wider grids. This avoids a single orphan panel stranded on its own row
    (e.g. 11 panels become 4x3 rather than 5,5,1)."""
    if n <= max_cols:
        return 1, n
    best = None
    for ncols in range(max_cols, 1, -1):
        nrows = math.ceil(n / ncols)
        empties = nrows * ncols - n
        key = (empties, nrows, -ncols)
        if best is None or key < best[0]:
            best = (key, nrows, ncols)
    return best[1], best[2]


def discover_k_values(rows: list[dict]) -> list[int]:
    """Sorted retrieval depths ``k`` present in the rows."""
    ks = set()
    for r in rows:
        c = r.get("retrieval_condition", "")
        if "_k" in c:
            try:
                ks.add(int(c.rsplit("_k", 1)[1]))
            except ValueError:
                pass
    return sorted(ks)


def discover_embedders(rows: list[dict]) -> list[str]:
    """Embedder tags present in the rows (excluding the no-RAG baseline)."""
    embs = set()
    for r in rows:
        e = r.get("embedding_model") or ""
        if e:
            embs.add(e)
    return sorted(embs)


# Output formats for save(): overridable via --formats (e.g. "png,pdf").
SAVE_FORMATS = ["png"]


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save ``fig`` to ``out_dir/name`` (png), creating the directory, and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    for ext in SAVE_FORMATS:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  {path.name}")
    plt.close(fig)


# ── Plot 01: no-RAG vs best-RAG with CI ───────────────────────────────────────

def plot_norag_vs_bestrag(rows: list[dict], out_dir: Path) -> None:
    """Figure 01: no-RAG vs best-RAG accuracy per model with 95% CIs."""
    models = ordered_models(rows)
    by_model = group(rows, "model_label")

    labels, no_vals, no_lo, no_hi, best_vals, best_lo, best_hi, best_lbl = [], [], [], [], [], [], [], []
    for m in models:
        mrows = by_model[m]
        by_cond = group(mrows, "retrieval_condition")
        no_r = by_cond.get("no_rag", [])
        a, lo, hi = acc_ci(no_r)
        no_vals.append(a); no_lo.append(a - lo); no_hi.append(hi - a)

        rag = {c: r for c, r in by_cond.items() if c != "no_rag"}
        best_cond = max(rag, key=lambda c: acc(rag[c])) if rag else None
        br = rag[best_cond] if best_cond else []
        ba, blo, bhi = acc_ci(br)
        best_vals.append(ba); best_lo.append(ba - blo); best_hi.append(bhi - ba)
        best_lbl.append(best_cond.replace("rag_", "").replace("_k", " k=") if best_cond else "")
        labels.append(MODEL_SHORT.get(m, m))

    # If many models, split into two stacked rows so labels don't smear.
    if len(models) > 8:
        half = math.ceil(len(models) / 2)
        chunks = [list(range(half)), list(range(half, len(models)))]
    else:
        chunks = [list(range(len(models)))]

    fig, axes = plt.subplots(len(chunks), 1,
                             figsize=(max(9, max(len(c) for c in chunks) * 1.1),
                                      5.2 * len(chunks)),
                             squeeze=False)
    axes = axes[:, 0]
    w = 0.38
    for ax, idxs in zip(axes, chunks):
        x = np.arange(len(idxs))
        nv = [no_vals[i] for i in idxs]
        nl = [no_lo[i]  for i in idxs]
        nh = [no_hi[i]  for i in idxs]
        bv = [best_vals[i] for i in idxs]
        bl = [best_lo[i]   for i in idxs]
        bh = [best_hi[i]   for i in idxs]
        lab = [labels[i]   for i in idxs]
        b1 = ax.bar(x - w/2, nv, w, yerr=[nl, nh], capsize=4,
                    color=GRAY, alpha=0.85, label="No RAG",
                    error_kw={"linewidth": 1.2, "ecolor": "#374151"})
        b2 = ax.bar(x + w/2, bv, w, yerr=[bl, bh], capsize=4,
                    color=PALETTE[0], alpha=0.85, label="Best RAG",
                    error_kw={"linewidth": 1.2, "ecolor": "#1E3A8A"})
        for bar, v, ue in zip(b1, nv, nh):
            label_above(ax, bar.get_x() + bar.get_width()/2, v, ue, f"{v:.0f}%",
                        color=GRAY_D, fontsize=7.5)
        for bar, v, ue in zip(b2, bv, bh):
            label_above(ax, bar.get_x() + bar.get_width()/2, v, ue, f"{v:.0f}%",
                        color="#1E40AF", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MCQ Accuracy (%)")
        ax.set_ylim(0, 100)
    axes[0].legend(loc="lower right")
    axes[0].set_title("No-RAG vs Best RAG: MCQ Accuracy with 95% CI",
                      fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, out_dir, "01_norag_vs_bestrag.png")


# ── Plot 02: top-3 RAG per model ──────────────────────────────────────────────

def plot_top3_per_model(rows: list[dict], out_dir: Path) -> None:
    """Figure 02: each model's top-3 retrieval conditions, grouped."""
    models = ordered_models(rows)
    by_model = group(rows, "model_label")
    nrows, ncols = grid_shape(len(models), max_cols=5)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 4.4 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax, model in zip(axes_flat, models):
        mrows = by_model[model]
        by_cond = group(mrows, "retrieval_condition")
        no_r = by_cond.get("no_rag", [])
        na, nlo, nhi = acc_ci(no_r)

        rag = [(c, acc_ci(r)) for c, r in by_cond.items() if c != "no_rag"]
        top3 = sorted(rag, key=lambda x: x[1][0], reverse=True)[:3]

        labels = ["no_rag"] + [
            c.replace("rag_", "").replace("-embedding:", "-").replace("_k", "\nk=")
            for c, _ in top3
        ]
        vals   = [na] + [v for _, (v, _, _) in top3]
        los    = [na - nlo] + [v - lo for _, (v, lo, _) in top3]
        his_   = [nhi - na] + [hi - v for _, (v, _, hi) in top3]
        ns     = [len(no_r)] + [len(by_cond.get(c, [])) for c, _ in top3]
        colors = [GRAY] + PALETTE[:3]

        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.9, width=0.65,
                      yerr=[los, his_], capsize=4, error_kw={"linewidth": 1.1, "ecolor": GRAY_D})
        for bar, v, ue, n in zip(bars, vals, his_, ns):
            label_above(ax, bar.get_x() + bar.get_width()/2, v, ue,
                        f"{v:.1f}%\nn={n}", color="#1F2937", fontsize=7.5, pad=2.2)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(MODEL_SHORT.get(model, model), fontsize=8.5, fontweight="bold", pad=6)
        ax.set_ylim(0, 105)
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].axis("off")
    for r in range(nrows):
        axes[r, 0].set_ylabel("MCQ Accuracy (%)")

    fig.suptitle("No-RAG vs Top 3 RAG Conditions per Model (95% CI)", fontweight="bold", y=1.005)
    plt.tight_layout()
    save(fig, out_dir, "02_top3_per_model.png")


# ── Plot 03: full heatmap ─────────────────────────────────────────────────────

def plot_heatmap(rows: list[dict], out_dir: Path) -> None:
    """Figure 03: model x condition accuracy heatmap."""
    models = ordered_models(rows)
    all_conds = sorted({r["retrieval_condition"] for r in rows})
    conds = ["no_rag"] + [c for c in all_conds if c != "no_rag"]

    by_model = group(rows, "model_label")
    data = np.full((len(models), len(conds)), np.nan)
    for i, m in enumerate(models):
        by_cond = group(by_model[m], "retrieval_condition")
        for j, c in enumerate(conds):
            if c in by_cond:
                data[i, j] = acc(by_cond[c])

    short_conds = [c.replace("rag_", "").replace("-embedding:", "-").replace("_k", " k=") for c in conds]
    short_models = [MODEL_SHORT.get(m, m).replace("\n", " ") for m in models]

    fig, ax = plt.subplots(figsize=(max(13, len(conds) * 0.9), max(3.5, len(models) * 0.9)))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap="cividis", vmin=40, vmax=95)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("MCQ Accuracy (%)", fontsize=9)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(short_conds, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short_models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(conds)):
            v = data[i, j]
            if not np.isnan(v):
                color = "white" if v < 68 else "#111827"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color=color)

    # Separator after no_rag column
    ax.axvline(0.5, color="#374151", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.set_title("Accuracy Heatmap: Model × Retrieval Condition", fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, out_dir, "03_heatmap.png")


# ── Plot 04: k-value line plot ────────────────────────────────────────────────

def plot_k_effect(rows: list[dict], out_dir: Path) -> None:
    """Figure 04: accuracy vs retrieval depth k."""
    models = ordered_models(rows)
    embedders = discover_embedders(rows) or list(EMBEDDER_SHORT.keys())
    k_vals = discover_k_values(rows) or [3, 6, 9]
    nrows, ncols = grid_shape(len(models), max_cols=5)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 3.6 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax, model in zip(axes_flat, models):
        mrows = [r for r in rows if r["model_label"] == model]
        by_cond = group(mrows, "retrieval_condition")
        for emb in embedders:
            ys, los, his = [], [], []
            for k in k_vals:
                cond = f"rag_{emb}_k{k}"
                cond_rows = by_cond.get(cond, [])
                if not cond_rows:
                    a = lo = hi = float("nan")
                else:
                    a, lo, hi = acc_ci(cond_rows)
                ys.append(a); los.append(lo); his.append(hi)
            ax.plot(k_vals, ys, marker="o", label=EMBEDDER_SHORT.get(emb, emb),
                    color=EMBEDDER_COLORS.get(emb, PALETTE[embedders.index(emb) % len(PALETTE)]),
                    linewidth=2, markersize=6)
            ax.fill_between(k_vals, los, his, alpha=0.12,
                            color=EMBEDDER_COLORS.get(emb, PALETTE[embedders.index(emb) % len(PALETTE)]))

        no_r = acc(by_cond.get("no_rag", []))
        ax.axhline(no_r, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.8, label="no_rag")
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"k={k}" for k in k_vals])
        ax.set_title(MODEL_SHORT.get(model, model).replace("\n", " "), fontsize=8.5, fontweight="bold")
        ax.set_ylim(20, 100)
    # legend on first axes only
    if models:
        axes_flat[0].legend(fontsize=7.5, loc="lower right")
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].axis("off")
    for r in range(nrows):
        axes[r, 0].set_ylabel("MCQ Accuracy (%)")

    fig.suptitle("Effect of Retrieval Depth (k) on Accuracy per Model", fontweight="bold", y=1.005)
    plt.tight_layout()
    save(fig, out_dir, "04_k_effect.png")


# ── Plot 05: embedder comparison at each k ───────────────────────────────────

def plot_embedder_comparison(rows: list[dict], out_dir: Path) -> None:
    """Figure 05: embedder comparison across k values."""
    embedders = discover_embedders(rows) or list(EMBEDDER_SHORT.keys())
    k_vals = discover_k_values(rows) or [3, 6, 9]
    models = ordered_models(rows)
    # When there are many models, restrict to the top 8 by best-config accuracy
    # so bars are wide enough to read.
    if len(models) > 8:
        best = best_condition_per_model(rows)
        models = [m for m, _ in sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:8]]
    by_cond = group(rows, "retrieval_condition")

    fig, axes = plt.subplots(1, len(k_vals),
                             figsize=(max(5, 4.3 * len(k_vals)), max(5, 0.25 * len(models) + 4)),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, k in zip(axes, k_vals):
        x = np.arange(len(embedders))
        for i, model in enumerate(models):
            vals, los, his = [], [], []
            for emb in embedders:
                cond = f"rag_{emb}_k{k}"
                mrows = [r for r in by_cond.get(cond, []) if r["model_label"] == model]
                a, lo, hi = acc_ci(mrows)
                vals.append(a); los.append(a - lo); his.append(hi - a)
            offset = (i - len(models)/2 + 0.5) * (0.7 / len(models))
            ax.bar(x + offset, vals, 0.7 / len(models), label=MODEL_SHORT.get(model, model).replace("\n", " "),
                   color=PALETTE[i % len(PALETTE)], alpha=0.85,
                   yerr=[los, his], capsize=3, error_kw={"linewidth": 0.9})
        ax.set_xticks(x)
        ax.set_xticklabels([EMBEDDER_SHORT.get(e, e) for e in embedders], rotation=15, ha="right", fontsize=8)
        ax.set_title(f"k = {k}", fontweight="bold")
        ax.set_ylim(20, 100)
        if ax is axes[0]:
            ax.set_ylabel("MCQ Accuracy (%)")
    # Single legend below the figure when many models
    if len(models) > 8:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(6, len(models)), fontsize=7,
                   bbox_to_anchor=(0.5, -0.05))
    else:
        axes[0].legend(fontsize=7, loc="lower right")

    fig.suptitle("Embedding Model Comparison at each k (95% CI)", fontweight="bold", y=1.005)
    plt.tight_layout()
    save(fig, out_dir, "05_embedder_by_k.png")


# ── Plot 06: RAG gain (delta) per model ──────────────────────────────────────

def plot_rag_gain(rows: list[dict], out_dir: Path) -> None:
    """Figure 06: RAG gain over the no-RAG baseline (percentage points)."""
    models = ordered_models(rows)
    by_model = group(rows, "model_label")
    gains, colors_bar = [], []

    for model in models:
        by_cond = group(by_model[model], "retrieval_condition")
        no_r = acc(by_cond.get("no_rag", []))
        rag = {c: acc(r) for c, r in by_cond.items() if c != "no_rag"}
        best = max(rag.values()) if rag else no_r
        gain = best - no_r
        gains.append(gain)
        colors_bar.append(PALETTE[1] if gain >= 0 else PALETTE[2])

    # Many models → split into two rows for legibility.
    if len(models) > 8:
        half = math.ceil(len(models) / 2)
        chunks = [list(range(half)), list(range(half, len(models)))]
    else:
        chunks = [list(range(len(models)))]

    fig, axes = plt.subplots(len(chunks), 1,
                             figsize=(max(8, max(len(c) for c in chunks) * 1.1),
                                      4.4 * len(chunks)),
                             squeeze=False)
    axes = axes[:, 0]
    for ax, idxs in zip(axes, chunks):
        x = np.arange(len(idxs))
        gv = [gains[i] for i in idxs]
        cb = [colors_bar[i] for i in idxs]
        bars = ax.bar(x, gv, color=cb, alpha=0.87, width=0.6, zorder=3)
        ax.axhline(0, color="#374151", linewidth=1.0)
        for bar, g in zip(bars, gv):
            va = "bottom" if g >= 0 else "top"
            offset = 0.3 if g >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width()/2, g + offset, f"{g:+.1f}pp",
                    ha="center", va=va, fontsize=8, fontweight="bold",
                    color=PALETTE[1] if g >= 0 else PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT.get(models[i], models[i]).replace("\n", " ")
                            for i in idxs], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Gain (pp)")
    pos_patch = mpatches.Patch(color=PALETTE[1], alpha=0.87, label="RAG improves")
    neg_patch = mpatches.Patch(color=PALETTE[2], alpha=0.87, label="RAG hurts")
    axes[0].legend(handles=[pos_patch, neg_patch])
    axes[0].set_title("RAG Gain over No-RAG Baseline (best condition)",
                      fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, out_dir, "06_rag_gain.png")


# ── Plot 07: accuracy by document ────────────────────────────────────────────

def plot_by_document(rows: list[dict], out_dir: Path) -> None:
    """Figure 07: per-document accuracy."""
    models = ordered_models(rows)
    docs = sorted({r["document_id"] for r in rows})
    by_model = group(rows, "model_label")

    # 2×2 grid when many models (avoid one giant horizontal smear), else 1 row.
    if len(models) > 8 and len(docs) >= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 1, len(docs)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(max(7, 0.45 * len(models) + 2) * ncols / max(ncols, 1),
                                      4.5 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax, doc in zip(axes_flat, docs):
        x = np.arange(len(models))
        for idx, condition in enumerate(["no_rag", "best_rag"]):
            vals, los, his = [], [], []
            for model in models:
                mrows = [r for r in by_model.get(model, []) if r["document_id"] == doc]
                by_cond = group(mrows, "retrieval_condition")
                if condition == "no_rag":
                    a, lo, hi = acc_ci(by_cond.get("no_rag", []))
                else:
                    rag = {c: r for c, r in by_cond.items() if c != "no_rag"}
                    best_cond = max(rag, key=lambda c: acc(rag[c])) if rag else None
                    a, lo, hi = acc_ci(rag[best_cond]) if best_cond else (0.0, 0.0, 0.0)
                vals.append(a); los.append(a - lo); his.append(hi - a)

            w = 0.35
            offset = -w/2 if condition == "no_rag" else w/2
            color = GRAY if condition == "no_rag" else PALETTE[0]
            label = "No RAG" if condition == "no_rag" else "Best RAG"
            ax.bar(x + offset, vals, w, color=color, alpha=0.85, label=label,
                   yerr=[los, his], capsize=3, error_kw={"linewidth": 0.9})

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT.get(m, m).replace("\n", " ") for m in models],
                           rotation=55, ha="right",
                           fontsize=6.5 if len(models) > 8 else 7.5)
        ax.set_title(DOC_LABELS.get(doc, doc), fontweight="bold")
        ax.set_ylim(0, 105)
    for j in range(len(docs), len(axes_flat)):
        axes_flat[j].axis("off")
    for r in range(nrows):
        axes[r, 0].set_ylabel("MCQ Accuracy (%)")
    axes_flat[0].legend(fontsize=8)

    fig.suptitle("Accuracy by Document Domain", fontweight="bold", y=1.005)
    plt.tight_layout()
    save(fig, out_dir, "07_by_document.png")


# ── Plot 08: bloom level breakdown ────────────────────────────────────────────

def plot_bloom(rows: list[dict], out_dir: Path) -> None:
    # Normalize "Analyse"/"Analyze"/"Understand" etc to lowercase, treat UK & US
    # spellings as the same bucket.
    """Figure 08: accuracy by Bloom's-taxonomy level."""
    def norm(b: str) -> str:
        """Normalize a label/condition string for consistent grouping."""
        b = (b or "").strip().lower()
        return "analyze" if b == "analyse" else b

    rows = [dict(r, bloom_level=norm(r.get("bloom_level", ""))) for r in rows]
    bloom_vals = sorted({r["bloom_level"] for r in rows if r["bloom_level"]})
    if not bloom_vals:
        print("  (skipping bloom plot: no bloom_level data)")
        return

    order = [b for b in BLOOM_ORDER if b in bloom_vals] + [b for b in bloom_vals if b not in BLOOM_ORDER]
    models = ordered_models(rows)
    by_model = group(rows, "model_label")

    # When models are many, switch from grouped bars to faceted grid (one panel
    # per model) so each model's bloom profile is readable.
    if len(models) > 6:
        nrows, ncols = grid_shape(len(models), max_cols=5)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.0 * ncols, 2.8 * nrows),
                                 sharey=True, squeeze=False)
        axes_flat = axes.flatten()
        x = np.arange(len(order))
        for ax, model in zip(axes_flat, models):
            mrows = by_model.get(model, [])
            by_bloom = group(mrows, "bloom_level")
            vals = [acc(by_bloom.get(b, [])) for b in order]
            ax.bar(x, vals, 0.7, color=PALETTE[models.index(model) % len(PALETTE)], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([b[:4].capitalize() for b in order], rotation=30, ha="right", fontsize=7)
            ax.set_title(MODEL_SHORT.get(model, model).replace("\n", " "),
                         fontsize=8.5, fontweight="bold")
            ax.set_ylim(0, 100)
        for j in range(len(models), len(axes_flat)):
            axes_flat[j].axis("off")
        for r in range(nrows):
            axes[r, 0].set_ylabel("MCQ Accuracy (%)")
        fig.suptitle("Accuracy by Bloom's Taxonomy Level (per model)",
                     fontweight="bold", y=1.005)
        plt.tight_layout()
        save(fig, out_dir, "08_bloom_level.png")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.2), 5))
    x = np.arange(len(order))
    w = 0.7 / len(models)
    for i, model in enumerate(models):
        mrows = by_model.get(model, [])
        by_bloom = group(mrows, "bloom_level")
        vals = [acc(by_bloom.get(b, [])) for b in order]
        offset = (i - len(models)/2 + 0.5) * w
        ax.bar(x + offset, vals, w, color=PALETTE[i % len(PALETTE)], alpha=0.85,
               label=MODEL_SHORT.get(model, model).replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in order])
    ax.set_ylabel("MCQ Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy by Bloom's Taxonomy Level", fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out_dir, "08_bloom_level.png")


# ── Plot 09: latency vs accuracy scatter ──────────────────────────────────────

def plot_latency_vs_accuracy(rows: list[dict], out_dir: Path) -> None:
    """Latency vs accuracy scatter across models/conditions."""
    lat_rows = [r for r in rows if r.get("latency_seconds")]
    if not lat_rows:
        print("  (skipping latency plot: no latency data)")
        return

    models = ordered_models(lat_rows)
    by_model = group(lat_rows, "model_label")

    fig, ax = plt.subplots(figsize=(10, 7))
    points = []
    for i, model in enumerate(models):
        mrows = by_model[model]
        try:
            lats = [float(r["latency_seconds"]) for r in mrows if r["latency_seconds"]]
            avg_lat = float(np.median(lats)) if lats else None
        except Exception:
            avg_lat = None
        if avg_lat is None:
            continue
        a, lo, hi = acc_ci(mrows)
        points.append((model, avg_lat, a, lo, hi, i))

    # Annotate only frontier points (Pareto-optimal) + top-3 accuracy to avoid clutter.
    by_acc = sorted(points, key=lambda p: p[2], reverse=True)
    top_acc = {p[0] for p in by_acc[:3]}
    pareto = set()
    for p in sorted(points, key=lambda q: q[1]):
        if all(p[2] > q[2] or q[0] == p[0] for q in points if q[1] <= p[1]):
            pareto.add(p[0])
    annotate = top_acc | pareto

    for model, lat, a, lo, hi, i in points:
        color = PALETTE[i % len(PALETTE)]
        ax.errorbar(lat, a, yerr=[[a - lo], [hi - a]], fmt="o",
                    color=color, markersize=8, capsize=3, linewidth=1.2,
                    alpha=0.85)
        if model in annotate:
            ax.annotate(MODEL_SHORT.get(model, model).replace("\n", " "),
                        xy=(lat, a), xytext=(6, 4), textcoords="offset points",
                        fontsize=7.5, color=color, fontweight="bold")

    ax.set_xlabel("Median Latency per Query (s)")
    ax.set_ylabel("MCQ Accuracy (%)")
    ax.set_ylim(20, 100)
    ax.set_title("Accuracy vs Latency per Model (Pareto frontier + top-3 annotated)",
                 fontweight="bold", pad=12)
    save(fig, out_dir, "09_latency_vs_accuracy.png")

    # Sibling plot: throughput (eval tok/s) on x-axis.
    tps_points = []
    for model, avg_lat, a, lo, hi, i in points:
        mrows_m = [r for r in lat_rows if r["model_label"] == model]
        try:
            tps_vals = [float(r["eval_count"]) / float(r["latency_seconds"])
                        for r in mrows_m
                        if r.get("eval_count") and r.get("latency_seconds")
                        and float(r["latency_seconds"]) > 0]
        except Exception:
            tps_vals = []
        if tps_vals:
            tps_points.append((model, float(np.median(tps_vals)), a, lo, hi, i))
    if tps_points:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        top_acc2 = {p[0] for p in sorted(tps_points, key=lambda p: p[2], reverse=True)[:3]}
        pareto2 = set()
        for p in sorted(tps_points, key=lambda q: -q[1]):  # higher tps = "better" x-axis
            if all(p[2] > q[2] or q[0] == p[0] for q in tps_points if q[1] >= p[1]):
                pareto2.add(p[0])
        annotate2 = top_acc2 | pareto2
        for model, tps, a, lo, hi, i in tps_points:
            color = PALETTE[i % len(PALETTE)]
            ax2.errorbar(tps, a, yerr=[[a - lo], [hi - a]], fmt="o",
                         color=color, markersize=8, capsize=3, linewidth=1.2, alpha=0.85)
            if model in annotate2:
                ax2.annotate(MODEL_SHORT.get(model, model).replace("\n", " "),
                             xy=(tps, a), xytext=(6, 4), textcoords="offset points",
                             fontsize=7.5, color=color, fontweight="bold")
        ax2.set_xlabel("Median Eval Throughput (tokens / s)")
        ax2.set_ylabel("MCQ Accuracy (%)")
        ax2.set_ylim(20, 100)
        ax2.set_title("Accuracy vs Throughput per Model (Pareto frontier + top-3 annotated)",
                      fontweight="bold", pad=12)
        save(fig2, out_dir, "09_latency_vs_accuracy_tokens_per_sec.png")


# ── Plot 10: condition ranking (averaged over all models) ─────────────────────

def plot_condition_ranking(rows: list[dict], out_dir: Path) -> None:
    """Conditions ranked by mean accuracy across models."""
    by_cond = group(rows, "retrieval_condition")
    cond_stats = {c: acc_ci(r) for c, r in by_cond.items()}
    ranked = sorted(cond_stats.items(), key=lambda x: x[1][0], reverse=True)

    labels = [c.replace("rag_", "").replace("-embedding:", "-").replace("_k", " k=") for c, _ in ranked]
    vals   = [v for _, (v, _, _) in ranked]
    los    = [v - lo for _, (v, lo, _) in ranked]
    his    = [hi - v for _, (v, _, hi) in ranked]
    ns     = [len(by_cond[c]) for c, _ in ranked]
    colors_bar = [GRAY if c == "no_rag" else PALETTE[0] for c, _ in ranked]

    fig, ax = plt.subplots(figsize=(6, max(6, len(ranked) * 0.42)))
    y = np.arange(len(ranked))
    ax.barh(y, vals, xerr=[los, his], color=colors_bar, alpha=0.85,
            capsize=3, height=0.65, error_kw={"linewidth": 1.0})
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(v + 0.5, i, f"{v:.1f}% (n={n})", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("MCQ Accuracy (%): averaged over all models")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.set_title("Retrieval Condition Ranking", fontweight="bold", pad=12)
    ax.axvline(cond_stats.get("no_rag", (0,))[0], color=GRAY, linestyle="--",
               linewidth=1.2, alpha=0.7, label="no_rag baseline")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, out_dir, "10_condition_ranking.png")


# ── Per-model detail dashboard ───────────────────────────────────────────────

def plot_per_model_detail(rows: list[dict], out_dir: Path) -> None:
    """One PNG per model: a 6-panel dashboard with everything that matters."""
    sub_dir = out_dir / "per_model"
    sub_dir.mkdir(parents=True, exist_ok=True)
    by_model = group(rows, "model_label")

    def norm_bloom(b: str) -> str:
        """Normalize a Bloom-level string to its canonical lowercase form."""
        b = (b or "").strip().lower()
        return "analyze" if b == "analyse" else b

    for model in ordered_models(rows):
        mrows = by_model[model]
        n_total = len(mrows)
        if n_total == 0:
            continue
        overall = acc(mrows)
        title = MODEL_SHORT.get(model, model).replace("\n", " ")

        fig = plt.figure(figsize=(15.5, 9.5))
        gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.32,
                              left=0.06, right=0.98, top=0.88, bottom=0.08)

        # ── 1: accuracy by retrieval condition (with CI) ─────────────────────
        ax = fig.add_subplot(gs[0, 0])
        by_cond = group(mrows, "retrieval_condition")
        conds_order = ["no_rag"] + sorted([c for c in by_cond if c != "no_rag"])
        vals, los, his, ns = [], [], [], []
        for c in conds_order:
            a, lo, hi = acc_ci(by_cond.get(c, []))
            vals.append(a); los.append(a - lo); his.append(hi - a); ns.append(len(by_cond.get(c, [])))
        x = np.arange(len(conds_order))
        colors = [GRAY] + [PALETTE[i % len(PALETTE)] for i in range(len(conds_order) - 1)]
        bars = ax.bar(x, vals, 0.62, color=colors, alpha=0.92,
                      yerr=[los, his], capsize=4,
                      error_kw={"linewidth": 1.0, "ecolor": GRAY_D})
        for bar, v, ue, n in zip(bars, vals, his, ns):
            label_above(ax, bar.get_x() + bar.get_width() / 2, v, ue,
                        f"{v:.1f}%\nn={n}", color=GRAY_D, fontsize=7.5, pad=2.2)
        ax.set_xticks(x)
        ax.set_xticklabels([short_cond_label(c) for c in conds_order],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by retrieval condition")

        # ── 2: accuracy by document ──────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        docs = sorted({r["document_id"] for r in mrows})
        rag = {c: r for c, r in by_cond.items() if c != "no_rag"}
        best_c = max(rag, key=lambda c: acc(rag[c])) if rag else None
        x = np.arange(len(docs))
        w = 0.38
        nor_vals, nor_lo, nor_hi = [], [], []
        bst_vals, bst_lo, bst_hi = [], [], []
        for d in docs:
            d_rows = [r for r in mrows if r["document_id"] == d]
            no_r = [r for r in d_rows if r["retrieval_condition"] == "no_rag"]
            a, lo, hi = acc_ci(no_r)
            nor_vals.append(a); nor_lo.append(a - lo); nor_hi.append(hi - a)
            if best_c:
                br = [r for r in d_rows if r["retrieval_condition"] == best_c]
            else:
                br = []
            a, lo, hi = acc_ci(br)
            bst_vals.append(a); bst_lo.append(a - lo); bst_hi.append(hi - a)
        ax.bar(x - w/2, nor_vals, w, color=GRAY, alpha=0.9,
               yerr=[nor_lo, nor_hi], capsize=3, label="No RAG",
               error_kw={"linewidth": 0.9})
        ax.bar(x + w/2, bst_vals, w, color=PALETTE[0], alpha=0.92,
               yerr=[bst_lo, bst_hi], capsize=3,
               label=f"Best RAG ({short_cond_label(best_c)})" if best_c else "Best RAG",
               error_kw={"linewidth": 0.9})
        ax.set_xticks(x)
        ax.set_xticklabels([DOC_LABELS.get(d, d).replace("\n", " ") for d in docs], fontsize=8.5)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by document")
        ax.legend(fontsize=8, loc="lower right")

        # ── 3: accuracy by Bloom level ───────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        bloom_rows = [dict(r, _b=norm_bloom(r.get("bloom_level", ""))) for r in mrows]
        bloom_vals = sorted({r["_b"] for r in bloom_rows if r["_b"]})
        order_b = [b for b in BLOOM_ORDER if b in bloom_vals] + [b for b in bloom_vals if b not in BLOOM_ORDER]
        if order_b:
            x = np.arange(len(order_b))
            vals = []
            for b in order_b:
                sub = [r for r in bloom_rows if r["_b"] == b]
                vals.append(acc(sub))
            ax.bar(x, vals, 0.62, color=PALETTE[4], alpha=0.92)
            for xi, v in zip(x, vals):
                label_above(ax, xi, v, 0.0, f"{v:.0f}%", color=GRAY_D, fontsize=8.5, pad=1.6)
            ax.set_xticks(x)
            ax.set_xticklabels([b.capitalize() for b in order_b], fontsize=8.5, rotation=15, ha="right")
            ax.set_ylim(0, 110)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy by Bloom level")

        # ── 4: latency distribution (boxplot, per condition) ─────────────────
        ax = fig.add_subplot(gs[1, 0])
        lat_data = []
        lat_labels = []
        for c in conds_order:
            try:
                lats = [float(r["latency_seconds"]) for r in by_cond.get(c, [])
                        if r.get("latency_seconds")]
            except Exception:
                lats = []
            if lats:
                lat_data.append(lats)
                lat_labels.append(short_cond_label(c))
        if lat_data:
            bp = ax.boxplot(lat_data, tick_labels=lat_labels, patch_artist=True,
                            widths=0.5, medianprops={"color": "white", "linewidth": 1.6},
                            flierprops={"marker": "o", "markersize": 2.5,
                                        "markerfacecolor": GRAY, "markeredgecolor": "none",
                                        "alpha": 0.4})
            for patch, c in zip(bp["boxes"], conds_order):
                col = GRAY if c == "no_rag" else PALETTE[0]
                patch.set_facecolor(col); patch.set_alpha(0.78)
            ax.set_xticklabels(lat_labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Latency per query (s)")
            ax.set_title("Latency distribution per condition")
            ax.grid(axis="y", alpha=0.6)
            # Throughput annotation
            try:
                tps_eval = [float(r["eval_count"]) / float(r["latency_seconds"])
                            for r in mrows
                            if r.get("eval_count") and r.get("latency_seconds")
                            and float(r["latency_seconds"]) > 0]
                tps_prompt = [float(r["prompt_eval_count"]) / float(r["latency_seconds"])
                              for r in mrows
                              if r.get("prompt_eval_count") and r.get("latency_seconds")
                              and float(r["latency_seconds"]) > 0]
                if tps_eval:
                    note = f"median eval: {float(np.median(tps_eval)):.0f} tok/s"
                    if tps_prompt:
                        note += f"  |  prompt: {float(np.median(tps_prompt)):.0f} tok/s"
                    ax.set_xlabel(note, fontsize=8, color=GRAY, style="italic")
            except Exception:
                pass

        # ── 5: k-effect line (with no_rag baseline) ──────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        embedders_local = discover_embedders(mrows)
        ks = discover_k_values(mrows)
        if ks and embedders_local:
            for i, emb in enumerate(embedders_local):
                ys, los, his = [], [], []
                for k in ks:
                    cond = f"rag_{emb}_k{k}"
                    a, lo, hi = acc_ci(by_cond.get(cond, []))
                    ys.append(a); los.append(lo); his.append(hi)
                col = EMBEDDER_COLORS.get(emb, PALETTE[i % len(PALETTE)])
                ax.plot(ks, ys, marker="o", linewidth=2.2, markersize=7,
                        color=col, label=EMBEDDER_SHORT.get(emb, emb))
                ax.fill_between(ks, los, his, alpha=0.14, color=col, linewidth=0)
            no_r = acc(by_cond.get("no_rag", []))
            ax.axhline(no_r, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.8,
                       label=f"no_rag = {no_r:.1f}%")
            ax.set_xticks(ks)
            ax.set_xticklabels([f"k={k}" for k in ks])
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(max(0, min(no_r - 15, 30)), 105)
            ax.set_title("Effect of retrieval depth (k)")
            ax.legend(fontsize=8, loc="lower right")

        # ── 6: stats summary (text panel) ────────────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        no_r_acc = acc(by_cond.get("no_rag", []))
        best_acc = acc(rag[best_c]) if best_c else 0.0
        gain = best_acc - no_r_acc
        try:
            lats_all = [float(r["latency_seconds"]) for r in mrows if r.get("latency_seconds")]
            med_lat = float(np.median(lats_all)) if lats_all else float("nan")
        except Exception:
            med_lat = float("nan")
        # Doc with lowest accuracy at best config
        worst_doc, worst_acc = None, 100.0
        if best_c:
            for d in docs:
                d_rows = [r for r in mrows if r["document_id"] == d and r["retrieval_condition"] == best_c]
                if d_rows:
                    a = acc(d_rows)
                    if a < worst_acc:
                        worst_acc, worst_doc = a, d
        overall_a, overall_lo, overall_hi = acc_ci(mrows)
        lines = [
            f"Total questions evaluated : {n_total:,}",
            f"Overall accuracy           : {overall_a:.2f}%",
            f"Lower / upper 95% CI       : {overall_lo:.1f}% / {overall_hi:.1f}%",
            "",
            f"No-RAG accuracy            : {no_r_acc:.2f}%",
            f"Best RAG accuracy          : {best_acc:.2f}%",
            f"Best RAG condition         : {short_cond_label(best_c) if best_c else 'n/a'}",
            f"RAG gain vs no-RAG         : {gain:+.2f} pp",
            "",
            f"Median latency / query     : {med_lat:.2f} s" if med_lat == med_lat else "Median latency             : n/a",
            f"Weakest doc (best config)  : {worst_doc or 'n/a'} ({worst_acc:.1f}%)",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                fontsize=10.5, family="DejaVu Sans Mono", color=GRAY_D,
                linespacing=1.55)

        fig.suptitle(f"{title}  ·  detail dashboard",
                     fontsize=15, fontweight="bold", color="#111827", y=0.965)
        fig.text(0.5, 0.93, f"n = {n_total:,} questions  ·  "
                            f"conditions: {len(by_cond)}  ·  overall accuracy {overall:.1f}%",
                 ha="center", fontsize=10, color=GRAY)
        save(fig, sub_dir, f"{model}.png")


def short_cond_label(c: str | None) -> str:
    """Compact human label for a retrieval condition string."""
    if not c:
        return ""
    return c.replace("rag_", "").replace("-embedding:", "-").replace("_k", " k=")


# ── Top-5 helpers ─────────────────────────────────────────────────────────────

def best_condition_per_model(rows: list[dict]) -> dict[str, tuple[str, float, int]]:
    """For each model_label, return (best_condition, accuracy, n) across all
    retrieval conditions (including no_rag)."""
    out: dict[str, tuple[str, float, int]] = {}
    for m, mrows in group(rows, "model_label").items():
        best_c, best_a, best_n = "", -1.0, 0
        for c, crows in group(mrows, "retrieval_condition").items():
            if not crows:
                continue
            a = acc(crows)
            if a > best_a:
                best_c, best_a, best_n = c, a, len(crows)
        if best_c:
            out[m] = (best_c, best_a, best_n)
    return out


def top_n_models(rows: list[dict], n: int = 5) -> list[tuple[str, str]]:
    """Return [(model_label, best_condition), ...] ranked by best-condition
    accuracy, descending. Length min(n, available)."""
    best = best_condition_per_model(rows)
    ranked = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)
    return [(m, info[0]) for m, info in ranked[:n]]


def short_cond(c: str) -> str:
    """Short condition tag used on axis ticks."""
    return short_cond_label(c)


# Place a value label safely above a bar + its CI cap. Adds a subtle white halo
# so the text never visually collides with grid lines or error bars.
def label_above(ax, x, value, upper_err, text, color=GRAY_D, fontsize=7.5, pad=2.0):
    """Annotate a bar with its value just above the bar."""
    import matplotlib.patheffects as pe
    y = value + max(0.0, upper_err) + pad
    ax.text(x, y, text, ha="center", va="bottom", fontsize=fontsize, color=color,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    return y


# ── Top-5 Plot A: ranked bars (best config per model, top 5) ─────────────────

def plot_top5_ranked(rows: list[dict], out_dir: Path, n: int = 5) -> None:
    """Top-5 conditions per model, ranked."""
    best = best_condition_per_model(rows)
    if not best:
        return
    ranked = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:n]
    by_model_cond = {
        (m, c): r
        for m, mrows in group(rows, "model_label").items()
        for c, r in group(mrows, "retrieval_condition").items()
    }

    labels = [MODEL_SHORT.get(m, m).replace("\n", " ") for m, _ in ranked]
    conds  = [c for _, (c, _, _) in ranked]
    vals, los, his, ns = [], [], [], []
    for m, (c, _, _) in ranked:
        a, lo, hi = acc_ci(by_model_cond.get((m, c), []))
        vals.append(a); los.append(a - lo); his.append(hi - a)
        ns.append(len(by_model_cond.get((m, c), [])))

    fig, ax = plt.subplots(figsize=(max(7, len(ranked) * 1.6), 5.2))
    x = np.arange(len(ranked))

    # CI overlap banding: group consecutive entries whose CI overlaps the current group leader.
    ci_lo = [v - lo for v, lo in zip(vals, los)]
    ci_hi = [v + hi for v, hi in zip(vals, his)]
    tie_groups: list[list[int]] = []
    cur_group: list[int] = [0]
    cur_leader_lo = ci_lo[0]
    for j in range(1, len(vals)):
        if ci_hi[j] >= cur_leader_lo:   # overlaps current group leader's CI
            cur_group.append(j)
        else:
            tie_groups.append(cur_group)
            cur_group = [j]
            cur_leader_lo = ci_lo[j]
    tie_groups.append(cur_group)

    band_colors = ["#F3F4F6", "#E5E7EB"]  # alternating very-light-grey bands
    for gi, grp in enumerate(tie_groups):
        if len(grp) < 2:
            continue
        ax.axvspan(grp[0] - 0.45, grp[-1] + 0.45, alpha=0.55,
                   color=band_colors[gi % len(band_colors)], zorder=0)

    bars = ax.bar(x, vals, 0.6, color=PALETTE[:len(ranked)], alpha=0.9,
                  yerr=[los, his], capsize=5,
                  error_kw={"linewidth": 1.2, "ecolor": GRAY_D})
    for bar, v, ue, c, n_ in zip(bars, vals, his, conds, ns):
        label_above(ax, bar.get_x() + bar.get_width()/2, v, ue,
                    f"{v:.1f}%\n{short_cond(c)}\nn={n_}",
                    color="#1F2937", fontsize=8.5, pad=2.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("MCQ Accuracy (%): best config per model")
    ax.set_ylim(0, 105)
    ax.set_title(f"Top {len(ranked)} Models at Their Best Retrieval Config (95% CI)",
                 fontweight="bold", pad=12)
    # Legend note only when there are actual tie groups with 2+ members.
    if any(len(g) >= 2 for g in tie_groups):
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#E5E7EB", edgecolor="none", alpha=0.7,
                                 label="Shaded groups: 95% CIs overlap → ranking within group not significant")],
                  fontsize=8, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    save(fig, out_dir, "top5_01_ranked.png")


# ── Top-5 Plot B: by document ────────────────────────────────────────────────

def plot_top5_by_document(rows: list[dict], out_dir: Path, n: int = 5) -> None:
    """Top-5 conditions per model, broken out by document."""
    selection = top_n_models(rows, n)
    if not selection:
        return
    docs = sorted({r["document_id"] for r in rows})
    fig, ax = plt.subplots(figsize=(max(8, len(docs) * 2.0), 5.2))
    x = np.arange(len(docs))
    w = 0.8 / len(selection)
    for i, (m, c) in enumerate(selection):
        vals, los, his = [], [], []
        for d in docs:
            sub = [r for r in rows
                   if r["model_label"] == m
                   and r["retrieval_condition"] == c
                   and r["document_id"] == d]
            a, lo, hi = acc_ci(sub)
            vals.append(a); los.append(a - lo); his.append(hi - a)
        offset = (i - len(selection)/2 + 0.5) * w
        ax.bar(x + offset, vals, w, color=PALETTE[i % len(PALETTE)], alpha=0.88,
               label=f"{MODEL_SHORT.get(m, m).replace(chr(10), ' ')} ({short_cond(c)})",
               yerr=[los, his], capsize=3, error_kw={"linewidth": 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels([DOC_LABELS.get(d, d) for d in docs], fontsize=9)
    ax.set_ylabel("MCQ Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Top {len(selection)} Models: Accuracy by Document Domain",
                 fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="lower right", ncol=1)
    plt.tight_layout()
    save(fig, out_dir, "top5_02_by_document.png")


# ── Top-5 Plot C: by bloom level ─────────────────────────────────────────────

def plot_top5_by_bloom(rows: list[dict], out_dir: Path, n: int = 5) -> None:
    """Top-5 conditions per model, broken out by Bloom level."""
    def norm(b: str) -> str:
        """Normalize a label/condition string for consistent grouping."""
        b = (b or "").strip().lower()
        return "analyze" if b == "analyse" else b

    rows_n = [dict(r, bloom_level=norm(r.get("bloom_level", ""))) for r in rows]
    selection = top_n_models(rows_n, n)
    if not selection:
        return
    bloom_vals = sorted({r["bloom_level"] for r in rows_n if r["bloom_level"]})
    if not bloom_vals:
        return
    order = [b for b in BLOOM_ORDER if b in bloom_vals] + [b for b in bloom_vals if b not in BLOOM_ORDER]

    fig, ax = plt.subplots(figsize=(max(9, len(order) * 1.4), 5.2))
    x = np.arange(len(order))
    w = 0.8 / len(selection)
    for i, (m, c) in enumerate(selection):
        vals = []
        for b in order:
            sub = [r for r in rows_n
                   if r["model_label"] == m
                   and r["retrieval_condition"] == c
                   and r["bloom_level"] == b]
            vals.append(acc(sub))
        offset = (i - len(selection)/2 + 0.5) * w
        ax.bar(x + offset, vals, w, color=PALETTE[i % len(PALETTE)], alpha=0.88,
               label=f"{MODEL_SHORT.get(m, m).replace(chr(10), ' ')} ({short_cond(c)})")
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in order], fontsize=9)
    ax.set_ylabel("MCQ Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Top {len(selection)} Models: Accuracy by Bloom Level",
                 fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    save(fig, out_dir, "top5_03_by_bloom.png")


# ── Top-5 Plot D: latency vs accuracy ────────────────────────────────────────

def plot_top5_latency(rows: list[dict], out_dir: Path, n: int = 5) -> None:
    """Top-5 conditions per model against latency."""
    lat_rows = [r for r in rows if r.get("latency_seconds")]
    if not lat_rows:
        return
    selection = top_n_models(lat_rows, n)
    if not selection:
        return

    def _make_latency_fig(use_tps: bool) -> plt.Figure | None:
        """Shared builder for the latency figures (axes, styling, annotation)."""
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plotted = False
        for i, (m, c) in enumerate(selection):
            sub = [r for r in lat_rows
                   if r["model_label"] == m and r["retrieval_condition"] == c]
            try:
                if use_tps:
                    xs = [float(r["eval_count"]) / float(r["latency_seconds"])
                          for r in sub
                          if r.get("eval_count") and r.get("latency_seconds")
                          and float(r["latency_seconds"]) > 0]
                else:
                    xs = [float(r["latency_seconds"]) for r in sub if r["latency_seconds"]]
            except Exception:
                continue
            if not xs:
                continue
            med = float(np.median(xs))
            a, lo, hi = acc_ci(sub)
            color = PALETTE[i % len(PALETTE)]
            ax2.errorbar(med, a, yerr=[[a - lo], [hi - a]], fmt="o",
                         color=color, markersize=11, capsize=5, linewidth=1.5)
            ax2.annotate(f"{MODEL_SHORT.get(m, m).replace(chr(10), ' ')}\n{short_cond(c)}",
                         xy=(med, a), xytext=(8, 6), textcoords="offset points",
                         fontsize=8, color=color, fontweight="bold")
            plotted = True
        if not plotted:
            plt.close(fig2)
            return None
        if use_tps:
            ax2.set_xlabel("Median Eval Throughput (tokens / s)")
        else:
            ax2.set_xlabel("Median Latency per Query (s)")
        ax2.set_ylabel("MCQ Accuracy (%)")
        ax2.set_title(f"Top {len(selection)} Models: Accuracy vs {'Throughput' if use_tps else 'Latency'} (best config)",
                      fontweight="bold", pad=12)
        plt.tight_layout()
        return fig2

    fig_lat = _make_latency_fig(use_tps=False)
    if fig_lat:
        save(fig_lat, out_dir, "top5_04_latency.png")
    fig_tps = _make_latency_fig(use_tps=True)
    if fig_tps:
        save(fig_tps, out_dir, "top5_04_tokens_per_sec.png")


# ── Plot 11: thinking_on vs thinking_off comparison ───────────────────────────

def plot_thinking_comparison(rows: list[dict], out_dir: Path) -> None:
    """Thinking-on vs thinking-off accuracy per model."""
    thinking_rows = {
        "off": [r for r in rows if r.get("thinking_mode") == "off"],
        "on":  [r for r in rows if r.get("thinking_mode") == "on"],
    }
    # Only plot if we have both modes
    base_models = {r["model"] for r in thinking_rows["off"]} & {r["model"] for r in thinking_rows["on"]}
    if not base_models:
        print("  (skipping thinking comparison: no paired models)")
        return

    models_off = [r for r in ordered_models(thinking_rows["off"])
                  if any(rr["model"] in base_models for rr in rows if rr["model_label"] == r)]
    models_on  = [r for r in ordered_models(thinking_rows["on"])
                  if any(rr["model"] in base_models for rr in rows if rr["model_label"] == r)]

    fig, ax = plt.subplots(figsize=(9, 5))
    by_cond_off = group(thinking_rows["off"], "retrieval_condition")
    by_cond_on  = group(thinking_rows["on"],  "retrieval_condition")
    conds = sorted(by_cond_off.keys())

    x = np.arange(len(conds))
    w = 0.35
    vals_off = [acc(by_cond_off.get(c, [])) for c in conds]
    vals_on  = [acc(by_cond_on.get(c, [])) for c in conds]
    ax.bar(x - w/2, vals_off, w, label="Thinking OFF", color=PALETTE[0], alpha=0.85)
    ax.bar(x + w/2, vals_on,  w, label="Thinking ON",  color=PALETTE[3], alpha=0.85)

    short = [c.replace("rag_", "").replace("-embedding:", "-").replace("_k", " k=") for c in conds]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("MCQ Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Thinking ON vs OFF: Accuracy per Condition (all models)", fontweight="bold", pad=12)
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "11_thinking_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────────────

# A (model, condition) cell should hold one response per question. Responses that could
# not be parsed into an answer are quarantined out of raw_results.csv, so a cell can end
# up far short — and a handful of responses will still plot as a bar with a CI that
# spans most of the axis (Phi4-Mini-R once drew a 100% bar from nine responses).
MIN_CELL_N = 100


def drop_underpowered_cells(rows: list[dict]) -> list[dict]:
    """Remove (model, condition) cells with too few parsed responses to plot honestly."""
    counts = Counter((r["model_label"], r["retrieval_condition"]) for r in rows)
    dropped = {cell: n for cell, n in counts.items() if n < MIN_CELL_N}
    if dropped:
        print(f"  Dropping {len(dropped)} cell(s) with fewer than {MIN_CELL_N} parsed "
              f"responses (too few to plot):")
        for (model, condition), n in sorted(dropped.items(), key=lambda kv: kv[1]):
            print(f"    {model:38s} {condition:28s} n={n}")
    return [r for r in rows
            if (r["model_label"], r["retrieval_condition"]) not in dropped]


def main() -> None:
    """CLI entry point: load the results CSV and render every figure to the out dir."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",    default="", help="Tier name (small/medium): sets default input/output paths")
    parser.add_argument("--input",   default="", help="Override input CSV path")
    parser.add_argument("--out-dir", default="", help="Override output plots directory")
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution (use 300 for publication)")
    parser.add_argument("--formats", default="png", help="Comma-separated output formats, e.g. 'png,pdf'")
    args = parser.parse_args()

    global SAVE_FORMATS
    SAVE_FORMATS = [f.strip() for f in args.formats.split(",") if f.strip()]
    plt.rcParams["savefig.dpi"] = args.dpi
    if args.dpi >= 250:  # publication render: nudge the smallest fonts up to ≥10pt
        plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

    if args.tier:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from veltharrow.config import results_dir_for
        tier_dir = results_dir_for(args.tier)
        src     = Path(args.input)   if args.input   else tier_dir / "data" / "raw_results.csv"
        out_dir = Path(args.out_dir) if args.out_dir else tier_dir / "plots"
    else:
        src     = Path(args.input)   if args.input   else Path("results/medium/data/raw_results.csv")
        out_dir = Path(args.out_dir) if args.out_dir else Path("results/medium/plots")
    if not src.exists():
        print(f"Input not found: {src}")
        return

    print(f"Loading {src} ...")
    csv.field_size_limit(sys.maxsize)
    rows = [r for r in csv.DictReader(src.open(encoding="utf-8"))]
    rows = drop_underpowered_cells(rows)
    models = ordered_models(rows)
    print(f"  {len(rows):,} rows | {len(models)} models | {len({r['retrieval_condition'] for r in rows})} conditions\n")
    print(f"Saving plots to {out_dir}/")

    def render(rs: list[dict], od: Path, label: str) -> None:
        """Render the full figure suite for one already-loaded set of rows."""
        ov = od / "overview"   # numbered overview plots (01-11)
        t5 = od / "top5"       # top-5 comparison plots
        print(f"\n[{label}] {len(rs):,} rows → {od}/")
        plot_norag_vs_bestrag(rs, ov)
        plot_top3_per_model(rs, ov)
        plot_heatmap(rs, ov)
        plot_k_effect(rs, ov)
        plot_embedder_comparison(rs, ov)
        plot_rag_gain(rs, ov)
        plot_by_document(rs, ov)
        plot_bloom(rs, ov)
        plot_latency_vs_accuracy(rs, ov)
        plot_condition_ranking(rs, ov)
        plot_thinking_comparison(rs, ov)
        print(f"[{label}] Top-5 focused plots:")
        plot_top5_ranked(rs, t5)
        plot_top5_by_document(rs, t5)
        plot_top5_by_bloom(rs, t5)
        plot_top5_latency(rs, t5)
        print(f"[{label}] Per-model detail dashboards:")
        plot_per_model_detail(rs, od)

    # Fair-comparison subset: only no_rag + qwen3-embedding:0.6b at k=1/3/5.
    # This is the primary analysis folder: apples-to-apples across all tiers.
    FAIR_CONDS = {
        "no_rag",
        "rag_qwen3-embedding:0.6b_k1",
        "rag_qwen3-embedding:0.6b_k3",
        "rag_qwen3-embedding:0.6b_k5",
    }
    fair_rows = [r for r in rows if r["retrieval_condition"] in FAIR_CONDS]
    is_multi_embedder = (
        fair_rows and
        {r["retrieval_condition"] for r in fair_rows} != {r["retrieval_condition"] for r in rows}
    )

    # Drop models that have no RAG rows in the fair subset (only no_rag): they were
    # never run on the fair grid, so their best-RAG bars/k-effect panels would be empty.
    # Their full multi-embedder results are preserved in archive/.
    def drop_norag_only(rs: list[dict]) -> list[dict]:
        """Drop models that only have a no-RAG row (nothing to compare)."""
        conds_by_model: dict[str, set[str]] = {}
        for r in rs:
            conds_by_model.setdefault(r["model_label"], set()).add(r["retrieval_condition"])
        keep = {m for m, cs in conds_by_model.items() if cs - {"no_rag"}}
        dropped = sorted(set(conds_by_model) - keep)
        if dropped:
            print(f"  excluding (no fair RAG rows): {', '.join(dropped)}")
        return [r for r in rs if r["model_label"] in keep]

    qwen_dir = out_dir / "qwen3_0_6b"
    if is_multi_embedder:
        # Multi-embedder tier (medium): archive the full set, primary = fair subset.
        render(rows, out_dir / "archive", "archive (all conditions)")
        render(drop_norag_only(fair_rows), qwen_dir, "qwen3_0_6b (fair: qwen3-embedding:0.6b, k=1/3/5)")
    else:
        # Single-embedder tier (small): already qwen3-0.6b, everything goes to qwen3_0_6b/.
        render(drop_norag_only(rows), qwen_dir, "qwen3_0_6b")

    total = sum(1 for _ in out_dir.rglob("*.png"))
    print(f"\nDone: {total} plots under {out_dir}/")


if __name__ == "__main__":
    main()
