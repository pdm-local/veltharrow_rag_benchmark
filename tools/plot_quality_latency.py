#!/usr/bin/env python3
"""
Quality-vs-latency figure: for each free-form model, plot free-form composite
quality against median free-form generation latency. Both axes come from the
same (free-form) task, so the comparison is apples-to-apples. The upper-left
envelope is the Pareto frontier; the two largest models (qwen3.5:9b, gemma4:12b)
fall in the dominated region, strictly beaten by the 4B gemma4:e4b. Read-only inputs.

Output: figures/main/quality_vs_latency.png. The manuscript reads this through
paper/article/figures, a symlink to figures/, so no second copy is written.
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[1]

DISPLAY = {
    "gemma4_e4b_thinking_on": "gemma4:e4b",
    "qwen35_9b_thinking_on": "qwen3.5:9b",
    "gemma4_12b_thinking_on": "gemma4:12b",
    "qwen3_8b_thinking_on": "qwen3:8b",
    "gemma4_e2b_thinking_on": "gemma4:e2b",
    "qwen3_1_7b_thinking_on": "qwen3:1.7b",
}
METRICS = ("fact_correct", "hallucination_free", "source_grounded")


def load(p):
    """Read a result CSV into a list of row dicts."""
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Render the quality-vs-latency Pareto figure: each free-form model's composite quality
    against its median generation latency, with the Pareto frontier and dominated region
    drawn, written to figures/; also prints the per-model table."""
    # composite quality: mean of the three metrics (matches Table tab:ff-overall)
    g = defaultdict(lambda: defaultdict(list))
    for r in load(ROOT / "results" / "freeform" / "graded.csv"):
        for m in METRICS:
            g[r["model_label"]][m].append(float(r[m]))
    # median free-form generation latency
    lat = defaultdict(list)
    for r in load(ROOT / "results" / "freeform" / "answers.csv"):
        try:
            lat[r["model_label"]].append(float(r["latency_seconds"]))
        except (KeyError, ValueError):
            pass

    pts = []  # (latency, quality, label, model_key)
    for model, disp in DISPLAY.items():
        quality = 100 * statistics.mean(statistics.mean(g[model][m]) for m in METRICS)
        median_lat = statistics.median(lat[model])
        pts.append((median_lat, quality, disp, model))

    # a point is dominated if another point has >= quality AND <= latency (strictly better somewhere)
    def dominated(p):
        """True if some other point has >= quality and <= latency (strictly better on one),
        i.e. ``p`` is Pareto-dominated."""
        x, y, _, _ = p
        return any((ox <= x and oy >= y) and (ox < x or oy > y)
                   for ox, oy, _, _ in pts if (ox, oy) != (x, y))

    frontier = sorted((p for p in pts if not dominated(p)), key=lambda t: t[0])
    losers = [p for p in pts if dominated(p)]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    # Pareto frontier staircase (upper-left envelope) + shaded dominated region beneath it.
    # Extend the top of the frontier horizontally to the right edge so the dominated
    # high-latency models visibly fall inside the shaded zone.
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, ls="--", lw=1.4, color="#16A34A", zorder=2, label="Pareto frontier")
    x_max = max(p[0] for p in pts) * 1.12
    y_floor = min(p[1] for p in pts) - 4
    ax.fill_between(fx + [x_max], fy + [fy[-1]], y_floor, color="#FEE2E2", alpha=0.5,
                    zorder=0, label="dominated region")

    for x, y, disp, _ in frontier:
        ax.scatter(x, y, s=80, color="#16A34A", zorder=4, edgecolor="white", lw=0.9)
    for x, y, disp, _ in losers:
        ax.scatter(x, y, s=80, marker="X", color="#DC2626", zorder=4, edgecolor="white", lw=0.9)

    for x, y, disp, _ in pts:
        dx = -8 if disp in ("qwen3.5:9b", "gemma4:12b") else 8
        ha = "right" if dx < 0 else "left"
        ax.annotate(disp, (x, y), textcoords="offset points", xytext=(dx, 6),
                    fontsize=9, ha=ha)

    ax.set_xlabel("Free-form generation latency (s, median; lower is better)")
    ax.set_ylabel("Free-form composite quality (%)")
    ax.set_title("A 4B model dominates the larger ones on quality per second")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, color="#E5E7EB", lw=0.6)
    ax.margins(0.12)
    fig.tight_layout()

    out = ROOT / "figures" / "main" / "quality_vs_latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}\n")

    print(f"{'model':12s} {'quality%':>8s} {'med_lat(s)':>11s}  {'frontier':>8s}")
    fk = {p[3] for p in frontier}
    for x, y, disp, model in sorted(pts, key=lambda t: -t[1]):
        print(f"{disp:12s} {y:8.1f} {x:11.1f}  {'yes' if model in fk else 'DOMINATED':>8s}")


if __name__ == "__main__":
    main()
