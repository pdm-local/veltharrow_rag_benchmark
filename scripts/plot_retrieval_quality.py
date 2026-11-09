#!/usr/bin/env python3
"""
Plot retrieval-quality metrics computed by compute_retrieval_quality.py.

  rq_01_recall_by_condition.png   recall@k per (embedder × k), pooled across docs
  rq_02_per_document.png          recall@k per document × best-k, by embedder
  rq_03_failure_decomposition.png per model: correct/wrong × retrieved/not stacks

Usage:
  python3 scripts/plot_retrieval_quality.py --tier medium
  python3 scripts/plot_retrieval_quality.py --tier small
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
import matplotlib.patheffects as pe
import numpy as np

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from veltharrow.config import results_dir_for  # noqa: E402

# Reuse the same palette and short-label maps as plot_results.py for visual consistency.
sys.path.insert(0, str(ROOT / "scripts"))
from plot_results import (  # noqa: E402
    PALETTE, GRAY, GRAY_D, EMBEDDER_SHORT, EMBEDDER_COLORS, MODEL_SHORT, DOC_LABELS,
    short_cond_label,
)


def load(path: Path) -> list[dict]:
    """Load a retrieval_quality.csv as a list of row dicts (empty list if missing)."""
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


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


def _embedder_k_from_cond(c: str) -> tuple[str, int] | None:
    """Parse a retrieval condition into (embedder, k), e.g.
    ``"rag_qwen3-embedding:0.6b_k5"`` → ``("qwen3-embedding:0.6b", 5)``; ``None`` for
    ``no_rag`` or anything unparseable."""
    if not c.startswith("rag_"):
        return None
    try:
        body = c[len("rag_"):]
        emb, k = body.rsplit("_k", 1)
        return emb, int(k)
    except Exception:
        return None


def short_model(m: str) -> str:
    """Return the short single-line display label for a model (falls back to the raw label)."""
    return MODEL_SHORT.get(m, m).replace("\n", " ")


# ── Plot 01: recall by condition (embedder × k), pooled across docs ───────────
def plot_recall_by_condition(rq_rows: list[dict], out_dir: Path) -> None:
    """Grouped bar chart of recall@k per embedder × k, pooled across models and documents."""
    by_cond: dict[str, list[int]] = defaultdict(list)
    for r in rq_rows:
        if int(r["gold_chunk_count"] or 0) == 0:
            continue
        by_cond[r["retrieval_condition"]].append(int(r["gold_in_retrieved"]))

    # Group by embedder, k.
    rows = []
    for cond, hits in by_cond.items():
        parsed = _embedder_k_from_cond(cond)
        if not parsed:
            continue
        emb, k = parsed
        n = len(hits)
        if n == 0:
            continue
        rows.append((emb, k, sum(hits) / n * 100, n))

    if not rows:
        print("  (skipping rq_01: no retrieval rows with gold)")
        return

    embedders = sorted({r[0] for r in rows})
    ks = sorted({r[1] for r in rows})

    fig, ax = plt.subplots(figsize=(max(8, len(ks) * 1.5 + 2), 5.4))
    x = np.arange(len(ks))
    w = 0.8 / len(embedders)
    for i, emb in enumerate(embedders):
        vals, ns = [], []
        for k in ks:
            hit = [r for r in rows if r[0] == emb and r[1] == k]
            vals.append(hit[0][2] if hit else 0)
            ns.append(hit[0][3] if hit else 0)
        offset = (i - len(embedders) / 2 + 0.5) * w
        color = EMBEDDER_COLORS.get(emb, PALETTE[i % len(PALETTE)])
        bars = ax.bar(x + offset, vals, w, color=color, alpha=0.9,
                      label=EMBEDDER_SHORT.get(emb, emb))
        for bar, v, n in zip(bars, vals, ns):
            if n == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                    f"{v:.0f}%\nn={n}", ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Recall: % of questions where gold chunk is in retrieved set")
    ax.set_ylim(0, 110)
    ax.set_title("Retrieval recall by embedder × k (pooled across models & docs)",
                 fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right", title="Embedder")
    plt.tight_layout()
    save(fig, out_dir, "rq_01_recall_by_condition.png")


# ── Plot 02: recall per document × embedder (at best k) ───────────────────────
def plot_recall_per_document(rq_rows: list[dict], out_dir: Path) -> None:
    """Grouped bar chart of best-k recall per document, one bar group per embedder."""
    # Aggregate to (doc, embedder, k) → recall
    agg: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for r in rq_rows:
        if int(r["gold_chunk_count"] or 0) == 0:
            continue
        parsed = _embedder_k_from_cond(r["retrieval_condition"])
        if not parsed:
            continue
        emb, k = parsed
        agg[(r["document_id"], emb, k)].append(int(r["gold_in_retrieved"]))
    if not agg:
        print("  (skipping rq_02)")
        return

    # For each (doc, embedder) pick the best k.
    docs = sorted({k[0] for k in agg})
    embs = sorted({k[1] for k in agg})

    fig, ax = plt.subplots(figsize=(max(8, len(docs) * 2.0), 5.4))
    x = np.arange(len(docs))
    w = 0.8 / len(embs)
    for i, emb in enumerate(embs):
        vals, ks_used, ns = [], [], []
        for d in docs:
            best_v, best_k, best_n = 0.0, None, 0
            for (dd, ee, k), hits in agg.items():
                if dd != d or ee != emb:
                    continue
                v = sum(hits) / len(hits) * 100
                if v > best_v:
                    best_v, best_k, best_n = v, k, len(hits)
            vals.append(best_v); ks_used.append(best_k); ns.append(best_n)
        offset = (i - len(embs) / 2 + 0.5) * w
        color = EMBEDDER_COLORS.get(emb, PALETTE[i % len(PALETTE)])
        bars = ax.bar(x + offset, vals, w, color=color, alpha=0.9,
                      label=EMBEDDER_SHORT.get(emb, emb))
        for bar, v, k_, n in zip(bars, vals, ks_used, ns):
            if n == 0 or k_ is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                    f"{v:.0f}%\nk={k_}", ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels([DOC_LABELS.get(d, d).replace("\n", " ") for d in docs], fontsize=9.5)
    ax.set_ylabel("Recall (%) at best k")
    ax.set_ylim(0, 110)
    ax.set_title("Retrieval recall per document, best-k per embedder",
                 fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right", title="Embedder")
    plt.tight_layout()
    save(fig, out_dir, "rq_02_per_document.png")


# ── Plot 03: failure decomposition per model ──────────────────────────────────
def plot_failure_decomposition(rq_rows: list[dict], out_dir: Path) -> None:
    """Stacked bars splitting each model's questions into correct/wrong × gold-retrieved/not,
    on the fair embedder at each model's best-k, to attribute errors to the model vs retrieval."""
    # Restrict to the fair subset for a clean per-model story.
    FAIR_EMB = "qwen3-embedding:0.6b"
    # Pick best-k per model on the FAIR embedder so we compare apples to apples.
    by_model_cond: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rq_rows:
        if int(r["gold_chunk_count"] or 0) == 0:
            continue
        parsed = _embedder_k_from_cond(r["retrieval_condition"])
        if not parsed or parsed[0] != FAIR_EMB:
            continue
        by_model_cond[(r["model_label"], r["retrieval_condition"])].append(r)

    if not by_model_cond:
        print("  (skipping rq_03: no rows with fair embedder + gold mapping)")
        return

    # For each model pick best-k by ACCURACY (so the decomposition matches how
    # we'd actually deploy that model).
    model_best: dict[str, tuple[str, list[dict]]] = {}
    for (m, c), rs in by_model_cond.items():
        if not rs:
            continue
        acc = sum(1 for r in rs if str(r["is_correct"]).strip() in ("1", "True", "true")) / len(rs)
        prev = model_best.get(m)
        if prev is None or acc > prev[2]:
            model_best[m] = (c, rs, acc)

    if not model_best:
        return

    # Build the four-segment count per model.
    rows = []  # (model, n, frac_cr, frac_cnr, frac_wr, frac_wnr)
    for m, (cond, rs, _) in sorted(model_best.items(), key=lambda kv: kv[1][2], reverse=True):
        n = len(rs)
        cr = cnr = wr = wnr = 0
        for r in rs:
            correct = str(r["is_correct"]).strip() in ("1", "True", "true")
            retrieved = int(r["gold_in_retrieved"]) == 1
            if correct and retrieved:   cr += 1
            elif correct and not retrieved: cnr += 1
            elif (not correct) and retrieved: wr += 1
            else: wnr += 1
        rows.append((m, cond, n, cr / n * 100, cnr / n * 100, wr / n * 100, wnr / n * 100))

    models = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 1.05 + 2), 6.3))
    x = np.arange(len(models))

    labels = [
        ("Correct & retrieved (model OK, retrieval OK)",          "#16A34A"),
        ("Correct but NOT retrieved (model rescued without context)", "#84CC16"),
        ("Wrong despite retrieved (model fault)",                 "#F59E0B"),
        ("Wrong and not retrieved (retrieval fault)",             "#DC2626"),
    ]
    series = list(zip(*[r[3:] for r in rows]))  # 4 lists, len(models)
    bottom = np.zeros(len(models))
    for (lbl, color), vals in zip(labels, series):
        ax.bar(x, vals, 0.66, bottom=bottom, color=color, alpha=0.92, label=lbl)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 5:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in models], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Share of questions (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Failure decomposition: model vs. retrieval (fair: qwen3-0.6b, best k per model)",
                 fontweight="bold", pad=12)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

    # Footnote: which k each model used.
    cond_note = "  ·  ".join(f"{short_model(m).split()[0]}: {short_cond_label(c)}"
                             for m, c, *_ in rows[:6])
    ax.text(0.5, -0.18, "k chosen per model by accuracy.  " + cond_note,
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY,
            style="italic", wrap=True)
    plt.tight_layout()
    save(fig, out_dir, "rq_03_failure_decomposition.png")


# ── Plot 04: precision@k and recall@k for qwen3-embedding:0.6b ────────────────
def plot_precision_recall_fair(rq_rows: list[dict], out_dir: Path) -> None:
    """Precision@k and Recall@k lines across k values for the fair embedder.

    Precision@k = fraction of retrieved chunks that are gold (gold_in_retrieved / n_retrieved).
    Recall@k    = fraction of mappable questions where gold chunk was retrieved.

    Both metrics are pooled across all models and documents, restricted to
    qwen3-embedding:0.6b so the comparison is apples-to-apples.
    """
    FAIR_EMB = "qwen3-embedding:0.6b"

    # Collect per-k data: {k: {"recall_hits": int, "recall_n": int, "prec_sum": float, "prec_n": int}}
    by_k: dict[int, dict] = defaultdict(lambda: {"recall_hits": 0, "recall_n": 0,
                                                   "prec_sum": 0.0, "prec_n": 0})
    docs = sorted({r["document_id"] for r in rq_rows})
    per_doc_k: dict[tuple[str, int], dict] = defaultdict(lambda: {"recall_hits": 0, "recall_n": 0,
                                                                    "prec_sum": 0.0, "prec_n": 0})

    for r in rq_rows:
        if int(r["gold_chunk_count"] or 0) == 0:
            continue
        parsed = _embedder_k_from_cond(r["retrieval_condition"])
        if not parsed or parsed[0] != FAIR_EMB:
            continue
        _, k = parsed
        n_ret = int(r["n_retrieved"] or k)
        gold_in = int(r["gold_in_retrieved"])

        by_k[k]["recall_hits"] += gold_in
        by_k[k]["recall_n"] += 1
        if n_ret > 0:
            by_k[k]["prec_sum"] += gold_in / n_ret
            by_k[k]["prec_n"] += 1

        doc = r["document_id"]
        per_doc_k[(doc, k)]["recall_hits"] += gold_in
        per_doc_k[(doc, k)]["recall_n"] += 1
        if n_ret > 0:
            per_doc_k[(doc, k)]["prec_sum"] += gold_in / n_ret
            per_doc_k[(doc, k)]["prec_n"] += 1

    if not by_k:
        print(f"  (skipping rq_04: no rows for {FAIR_EMB})")
        return

    ks = sorted(by_k)
    recalls = [by_k[k]["recall_hits"] / by_k[k]["recall_n"] * 100 if by_k[k]["recall_n"] else 0
               for k in ks]
    precs = [by_k[k]["prec_sum"] / by_k[k]["prec_n"] * 100 if by_k[k]["prec_n"] else 0
             for k in ks]

    # ── main panel: precision + recall lines ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(ks, recalls, marker="o", linewidth=2.5, markersize=9,
            color=PALETTE[0], label="Recall@k")
    ax.plot(ks, precs, marker="s", linewidth=2.5, markersize=9,
            color=PALETTE[2], label="Precision@k", linestyle="--")
    for k, r, p in zip(ks, recalls, precs):
        ax.text(k, r + 1.8, f"{r:.0f}%", ha="center", va="bottom",
                fontsize=9, color=PALETTE[0], fontweight="bold")
        ax.text(k, p - 3.5, f"{p:.0f}%", ha="center", va="top",
                fontsize=9, color=PALETTE[2], fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Precision@k vs Recall@k\n(qwen3-embedding:0.6b, pooled)",
                 fontweight="bold", pad=10)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(axis="y", alpha=0.4)

    # ── right panel: per-document recall at each k ───────────────────────────
    ax2 = axes[1]
    x = np.arange(len(ks))
    w = 0.8 / len(docs)
    for i, doc in enumerate(docs):
        doc_recalls = []
        for k in ks:
            d = per_doc_k.get((doc, k))
            if d and d["recall_n"]:
                doc_recalls.append(d["recall_hits"] / d["recall_n"] * 100)
            else:
                doc_recalls.append(0.0)
        offset = (i - len(docs) / 2 + 0.5) * w
        ax2.bar(x + offset, doc_recalls, w, alpha=0.88,
                color=PALETTE[i % len(PALETTE)], label=DOC_LABELS.get(doc, doc).replace("\n", " "))
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"k={k}" for k in ks])
    ax2.set_ylabel("Recall@k (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Recall@k per document\n(qwen3-embedding:0.6b)", fontweight="bold", pad=10)
    ax2.legend(fontsize=9, loc="lower right", title="Document")
    ax2.grid(axis="y", alpha=0.4)

    fig.suptitle("Retrieval Quality: qwen3-embedding:0.6b (fair embedder)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, out_dir, "rq_04_precision_recall_fair.png")


def main() -> None:
    """Load a tier's retrieval_quality.csv and render all retrieval-quality figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="medium")
    parser.add_argument("--input", default="", help="Override path to retrieval_quality.csv")
    parser.add_argument("--out-dir", default="", help="Override output dir")
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution (use 300 for publication)")
    parser.add_argument("--formats", default="png", help="Comma-separated output formats, e.g. 'png,pdf'")
    args = parser.parse_args()

    global SAVE_FORMATS
    SAVE_FORMATS = [f.strip() for f in args.formats.split(",") if f.strip()]
    plt.rcParams["savefig.dpi"] = args.dpi
    if args.dpi >= 250:
        plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

    tier_dir = results_dir_for(args.tier)
    src = Path(args.input) if args.input else tier_dir / "data" / "retrieval_quality.csv"
    out_dir = Path(args.out_dir) if args.out_dir else tier_dir / "plots" / "qwen3_0_6b" / "retrieval"
    if not src.exists():
        print(f"Input not found: {src}. Run compute_retrieval_quality.py first.")
        return
    rq_rows = load(src)
    if not rq_rows:
        print(f"No retrieval rows in {src}")
        return
    print(f"Loaded {len(rq_rows):,} retrieval rows from {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {out_dir}/")
    plot_recall_by_condition(rq_rows, out_dir)
    plot_recall_per_document(rq_rows, out_dir)
    plot_failure_decomposition(rq_rows, out_dir)
    plot_precision_recall_fair(rq_rows, out_dir)
    print(f"\nDone: {len(list(out_dir.glob('*.png')))} plots in {out_dir}/")


if __name__ == "__main__":
    main()
