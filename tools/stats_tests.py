#!/usr/bin/env python3
"""
Inferential statistics for the retrieval claims, paired by question.

Exact McNemar tests (two-sided binomial on discordant pairs) on per-question
multiple-choice correctness, comparing no-RAG against all three fair RAG
depths for all eleven medium configurations, with Holm correction over 33 tests. Also
tests for a detectable k=3 vs k=5 difference for qwen3.5:9b. A non-significant
result is not interpreted as proof of equivalence. Stdlib + reads CSVs only;
no plotting, no new dependency.

Run: PYTHONPATH=src .venv/bin/python tools/stats_tests.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[1]
FAIR = ("rag_qwen3-embedding:0.6b_k1", "rag_qwen3-embedding:0.6b_k3",
        "rag_qwen3-embedding:0.6b_k5")


def load(tier):
    """Read a tier's raw_results.csv into a list of row dicts."""
    with open(ROOT / "results" / tier / "data" / "raw_results.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def correct_map(rows, model, cond):
    """Return {question_id: is_correct(bool)} for one (model, retrieval condition), the
    paired unit the McNemar tests operate on."""
    return {r["question_id"]: (r["is_correct"] in ("1", "True", "true"))
            for r in rows if r["model_label"] == model and r["retrieval_condition"] == cond}


def acc(m):
    """Percentage of True values in a correctness map (NaN if empty)."""
    return 100 * sum(m.values()) / len(m) if m else float("nan")


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired dicts a, b (question_id -> bool)."""
    q = sorted(set(a) & set(b))
    n01 = sum(1 for k in q if (not a[k]) and b[k])
    n10 = sum(1 for k in q if a[k] and (not b[k]))
    n = n01 + n10
    if n == 0:
        return len(q), n01, n10, 1.0
    lo = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n))
    return len(q), n01, n10, p


def holm(pvals):
    """Holm step-down adjusted p-values, preserving input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    """APA-style: italic p, no leading zero (handled in LaTeX); here plain text."""
    if p < .001:
        return "p < .001"
    return f"p = {('%.3f' % p).lstrip('0')}"


def main():
    """Run all 33 medium-tier retrieval tests (one Holm family), and the k=3 vs k=5 difference check for
    qwen3.5:9b, printing accuracies and p-values."""
    rows = load("medium")

    print("== no-RAG vs all fair RAG depths (paired exact McNemar, medium tier) ==")
    results = []
    models = sorted({r["model_label"] for r in rows if r["retrieval_condition"] == "no_rag"})
    for model in models:
        nr = correct_map(rows, model, "no_rag")
        for cond in FAIR:
            rag = correct_map(rows, model, cond)
            if len(nr) != 400 or set(nr) != set(rag):
                raise ValueError(f"Incomplete pairing: {model} {cond}")
            n, n01, n10, p = mcnemar_exact(nr, rag)
            results.append((model, cond, acc(nr), acc(rag), n01, n10, p))
    adj = holm([r[-1] for r in results])
    for (model, cond, nr, rag, n01, n10, p), pa in zip(results, adj):
        print(f"{model:32s} {cond[-2:]} {nr:5.1f} -> {rag:5.1f} "
              f"discordant={n01}/{n10} raw={p:.6g} Holm={pa:.6g}")
    print(f"  family size={len(results)}; max Holm-adjusted p={max(adj):.6g}")

    print("\n== k=3 vs k=5 difference check, qwen3.5:9b (paired exact McNemar) ==")
    a = correct_map(rows, "qwen35_9b_thinking_on", "rag_qwen3-embedding:0.6b_k3")
    b = correct_map(rows, "qwen35_9b_thinking_on", "rag_qwen3-embedding:0.6b_k5")
    n, n01, n10, p = mcnemar_exact(a, b)
    print(f"  n={n}  k3={acc(a):.1f}%  k5={acc(b):.1f}%  "
          f"discordant(k5+/k3-={n01}, k3+/k5-={n10})  {fmt_p(p)}")


if __name__ == "__main__":
    main()
