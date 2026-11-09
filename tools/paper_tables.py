#!/usr/bin/env python3
r"""
Emit LaTeX table bodies + exact numbers for the manuscript, computed directly from
the result CSVs so every number in the paper is reproducible and matches the data.
Read-only; no inference.

Usage:
  python3 tools/paper_tables.py            # all sections, to stdout
  python3 tools/paper_tables.py mcq        # just the MCQ accuracy table
  python3 tools/paper_tables.py retrieval latency errors
  python3 tools/paper_tables.py --latex    # write paper/article/generated/*.tex

Sections: mcq | mcq_small | retrieval | latency | compliance | ablation | errors

``--latex`` writes the table bodies that paper/article/main.tex \input's, so the
manuscript's tables cannot drift from the data the way hand-pasted rows did.
"""
from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veltharrow.stats import wilson_ci

MCQ = ROOT / "results" / "medium" / "data" / "raw_results.csv"
MCQ_SMALL = ROOT / "results" / "small" / "data" / "raw_results.csv"
RQ = ROOT / "results" / "medium" / "data" / "retrieval_quality.csv"
FF = ROOT / "results" / "freeform" / "graded.csv"

FAIR = {
    "no_rag": "No-RAG",
    "rag_qwen3-embedding:0.6b_k1": "k1",
    "rag_qwen3-embedding:0.6b_k3": "k3",
    "rag_qwen3-embedding:0.6b_k5": "k5",
}
KCONDS = ["no_rag", "rag_qwen3-embedding:0.6b_k1",
          "rag_qwen3-embedding:0.6b_k3", "rag_qwen3-embedding:0.6b_k5"]

# The 2026-08-30 audit found these medium-tier k=5 cells had been produced by legacy runs
# with num_ctx=4096, whose prompts can exceed that limit. Membership here is the audit's
# finding and is not re-derived: a cell cannot be identified as truncated from the stored
# token counts, because Ollama's sliding window lets prompt_eval+eval exceed num_ctx on
# rows that were never cut off.
AUDITED_BAD_K5 = {
    "deepseek_r1_7b_thinking_on",
    "gemma4_e4b_thinking_off",
    "gemma4_e4b_thinking_on",
    "llama31_8b_thinking_off",
    "qwen25_7b_thinking_off",
    "qwen35_9b_thinking_on",
    "qwen3_8b_thinking_off",
    "qwen3_8b_thinking_on",
}
# The audit moved those rows out of raw_results.csv into legacy_ctx4096_k5.csv, so a cell
# that has since been rerun carries only post-audit run_ids. Anything still holding a
# pre-audit row has not been refilled and stays excluded. A cell therefore graduates out
# of the exclusion on its own as the rerun lands, with no list to maintain by hand.
AUDIT_STAMP = "20260830T000000Z"


def invalid_medium_k5(rows: list[dict]) -> set[str]:
    """Audited-bad k=5 cells that still contain a pre-audit row."""
    stale = set()
    for r in rows:
        if (r["model_label"] in AUDITED_BAD_K5
                and r["retrieval_condition"] == "rag_qwen3-embedding:0.6b_k5"
                and r["run_id"] < AUDIT_STAMP):
            stale.add(r["model_label"])
    return stale

# Display names for the medium roster (mirrors plot_results.MODEL_SHORT, one-line form).
DISPLAY = {
    "qwen35_9b_thinking_off": "Qwen3.5-9B",
    "qwen35_9b_thinking_on": "Qwen3.5-9B (think)",
    "qwen3_8b_thinking_off": "Qwen3-8B",
    "qwen3_8b_thinking_on": "Qwen3-8B (think)",
    "gemma4_e4b_thinking_off": "Gemma4-E4B",
    "gemma4_e4b_thinking_on": "Gemma4-E4B (think)",
    "gemma4_12b_thinking_off": "Gemma4-12B",
    "gemma4_12b_thinking_on": "Gemma4-12B (think)",
    "qwen25_7b_thinking_off": "Qwen2.5-7B",
    "llama31_8b_thinking_off": "Llama3.1-8B",
    "deepseek_r1_7b_thinking_on": "DS-R1-7B (think)",
    # small tier
    "qwen3_0_6b_thinking_off": "Qwen3-0.6B",
    "qwen3_0_6b_thinking_on": "Qwen3-0.6B (think)",
    "qwen35_0_8b_thinking_off": "Qwen3.5-0.8B",
    "qwen35_0_8b_thinking_on": "Qwen3.5-0.8B (think)",
    "llama32_1b_thinking_off": "Llama3.2-1B",
    "llama32_1b_thinking_on": "Llama3.2-1B (think)",
    "llama32_3b_thinking_on": "Llama3.2-3B (think)",
    "qwen25_1_5b_thinking_off": "Qwen2.5-1.5B",
    "qwen25_1_5b_thinking_on": "Qwen2.5-1.5B (think)",
    "deepseek_r1_1_5b_thinking_off": "DS-R1-1.5B",
    "deepseek_r1_1_5b_thinking_on": "DS-R1-1.5B (think)",
    "qwen3_1_7b_thinking_off": "Qwen3-1.7B",
    "qwen3_1_7b_thinking_on": "Qwen3-1.7B (think)",
    "gemma3_1b_thinking_off": "Gemma3-1B",
    "gemma3_1b_thinking_on": "Gemma3-1B (think)",
    "gemma3_4b_thinking_on": "Gemma3-4B (think)",
    "gemma4_e2b_thinking_off": "Gemma4-E2B",
    "gemma4_e2b_thinking_on": "Gemma4-E2B (think)",
    "qwen35_2b_thinking_off": "Qwen3.5-2B",
    "qwen35_2b_thinking_on": "Qwen3.5-2B (think)",
    "phi4_mini_reasoning_thinking_off": "Phi4-Mini-R",
}


# A cell is one (model, condition) pair and should hold 400 responses. Rows whose
# response contained no parseable answer are quarantined into failed_attempts.csv, so a
# short cell means the model failed to answer, not that the run is incomplete.
QUESTIONS_PER_CELL = 400
# Below this the Wilson interval is too wide to carry a claim, so the cell is suppressed
# rather than printed as a real result (a 9-response cell once printed as "100.0%").
MIN_REPORTABLE_N = 100


def load(path: Path) -> list[dict]:
    """Read a result CSV into a list of row dicts."""
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_correct(r: dict) -> bool:
    """True if a raw_results row is marked correct (handles ``1``/``True``/``true``)."""
    return str(r.get("is_correct", "")).strip() in ("1", "True", "true")


# ----------------------------------------------------------------------------- MCQ
def unrecovered_by_cell(path: Path) -> tuple[dict, dict]:
    """Questions with no scored row, per (model, condition), split by who is at fault.

    Returns ``(unanswered, backend)``. ``unanswered`` counts model failures --- responses
    that never terminated on an answer, or in which the model declined --- and these are
    scored wrong. ``backend`` counts inference-server errors that produced no model output
    at all, and these leave the denominator entirely.

    Reads the tier's failed_attempts.csv, which is append-only across runs, so the *last*
    record for a question is its current disposition.
    """
    fail_path = FAILED if path == MCQ else FAILED_SMALL
    scored = {(r["model_label"], r["retrieval_condition"], r["question_id"])
              for r in load(path)}
    latest: dict[tuple[str, str, str], dict] = {}
    if fail_path.exists():
        for r in load(fail_path):
            if r["retrieval_condition"] not in FAIR:
                continue
            key = (r["model_label"], r["retrieval_condition"], r["question_id"])
            if key not in scored:
                latest[key] = r

    unanswered: dict[tuple[str, str], int] = defaultdict(int)
    backend: dict[tuple[str, str], int] = defaultdict(int)
    for (model, cond, _), row in latest.items():
        bucket = backend if classify_failure(row) == "backend" else unanswered
        bucket[(model, cond)] += 1
    return unanswered, backend


def mcq_table(path: Path = MCQ) -> None:
    """Print the LaTeX body for the MCQ accuracy table (per model: No-RAG, k=1/3/5 with the
    best k bolded, and Δ), plus the No-RAG floor range and headline best-RAG Wilson CIs."""
    rows = [r for r in load(path) if r["retrieval_condition"] in FAIR]
    if path == MCQ:
        stale = invalid_medium_k5(rows)
        rows = [r for r in rows if not (
            r["model_label"] in stale
            and r["retrieval_condition"] == "rag_qwen3-embedding:0.6b_k5"
        )]
    by = defaultdict(lambda: defaultdict(list))  # model -> cond -> [rows]
    for r in rows:
        by[r["model_label"]][r["retrieval_condition"]].append(r)

    def acc(rs):
        """Return (accuracy%, Wilson low, Wilson high, n) for a list of rows.

        Questions the model never answered are already present as scored-wrong rows
        (tools/score_unanswered.py), so plain ``len(rows)`` is the right denominator and
        every report -- tables, figures, statistical tests -- divides by the same number.
        """
        n = len(rs)
        c = sum(1 for r in rs if _is_correct(r))
        return (100 * c / n, *wilson_ci(c, n), n) if n else (0, 0, 0, 0)

    def reportable(cell):
        """A cell carries a claim only if enough responses actually parsed."""
        return cell[3] >= MIN_REPORTABLE_N

    short_cells = []
    table = []
    for m in by:
        cells = {c: acc(by[m][c]) for c in KCONDS}
        for c in KCONDS:
            if 0 < cells[c][3] < QUESTIONS_PER_CELL:
                short_cells.append((m, c, cells[c][3]))
        nr = cells["no_rag"][0] if reportable(cells["no_rag"]) else None
        rag_conds = [c for c in ("rag_qwen3-embedding:0.6b_k1", "rag_qwen3-embedding:0.6b_k3",
                                 "rag_qwen3-embedding:0.6b_k5") if reportable(cells[c])]
        best_c = max(rag_conds, key=lambda c: cells[c][0]) if rag_conds else None
        best = cells[best_c][0] if best_c else -1.0
        table.append((best, m, cells, best_c, nr))
    table.sort(key=lambda t: (t[0], t[1]), reverse=True)

    print("% MCQ accuracy (fair subset, Wilson 95% CI)")
    print(r"% cols: Model & No-RAG & k=1 & k=3 & k=5 & $\Delta$")
    for best, m, cells, best_c, nr in table:
        def cell(c, bold):
            r"""Format one condition's accuracy as a LaTeX cell, bolding the best k.

            ``n/a$^\dagger$`` marks a cell with no data at all (a run not yet done);
            ``n/a$^\ddagger$`` marks one the model answered too rarely to report. The
            paper's house style bans the en dash that ``--`` would render.
            """
            a, lo, hi, n = cells[c]
            if n == 0:
                return r"n/a$^\dagger$"
            if n < MIN_REPORTABLE_N:
                return r"n/a$^\ddagger$"
            s = f"{a:.1f}"
            return r"\textbf{" + s + "}" if bold else s
        delta_s = f"{best - nr:+.1f}" if (best_c and nr is not None) else r"n/a$^\ddagger$"
        print(f"    {DISPLAY.get(m, m):22s} & {cell('no_rag', False)} & "
              f"{cell('rag_qwen3-embedding:0.6b_k1', bool(best_c) and best_c.endswith('k1'))} & "
              f"{cell('rag_qwen3-embedding:0.6b_k3', bool(best_c) and best_c.endswith('k3'))} & "
              f"{cell('rag_qwen3-embedding:0.6b_k5', bool(best_c) and best_c.endswith('k5'))} & "
              f"{delta_s} \\\\")
    if short_cells:
        print("\n% WARNING - cells with fewer than "
              f"{QUESTIONS_PER_CELL} parsed responses (disclose these in the caption):")
        for m, c, n in sorted(short_cells, key=lambda t: t[2]):
            mark = "  SUPPRESSED" if n < MIN_REPORTABLE_N else ""
            print(f"%   {DISPLAY.get(m, m):22s} {FAIR.get(c, c):8s} n={n}{mark}")
    # No-RAG range for the contamination paragraph
    nrs = [nr for *_, nr in table if nr is not None]
    print(f"\n% No-RAG floor range: {min(nrs):.1f}% to {max(nrs):.1f}%  (chance = 16.7%)")
    # headline CIs for prose
    print("% headline best-RAG CIs:")
    for best, m, cells, best_c, nr in [t for t in table if t[3]][:4]:
        a, lo, hi, n = cells[best_c]
        print(f"%   {DISPLAY.get(m,m)}: {a:.1f}% [{lo:.1f}, {hi:.1f}] @ {FAIR[best_c]} (n={n})")


# ----------------------------------------------------------------- retrieval metrics
def retrieval_table() -> None:
    """Print the LaTeX body for the retrieval-quality table: Recall@k, Precision@k, MRR and
    n per k, over the canonical qwen3-0.6b conditions and mappable questions only (deduped,
    since retrieval is model-independent)."""
    rows = load(RQ)
    # The retriever is model-independent, so a given (question, condition) is identical
    # across models: dedupe on it. Restrict to the canonical qwen3-0.6b k in {1,3,5} and
    # to MAPPABLE questions (gold_chunk_count>0); non-mappable Qs have no gold chunk to
    # find and would otherwise deflate recall.
    canon = {"rag_qwen3-embedding:0.6b_k1", "rag_qwen3-embedding:0.6b_k3",
             "rag_qwen3-embedding:0.6b_k5"}
    seen = {}
    for r in rows:
        if r["retrieval_condition"] not in canon:
            continue
        if int(r.get("gold_chunk_count") or 0) <= 0:
            continue
        seen[(r["question_id"], r["retrieval_condition"])] = r
    by_k = defaultdict(list)
    for r in seen.values():
        by_k[r["top_k"]].append(r)

    print("% Retrieval quality by k (mappable questions only)")
    print(r"% cols: k & Recall@k & Precision@k & MRR & n")
    for k in sorted(by_k, key=lambda x: int(x)):
        rs = by_k[k]
        n = len(rs)
        hits = sum(1 for r in rs if r["gold_in_retrieved"] == "1")
        recall = 100 * hits / n
        # precision@k: gold chunks retrieved / total retrieved (1 gold per Q)
        prec = 100 * sum(int(r["gold_in_retrieved"]) for r in rs) / sum(int(r["n_retrieved"]) for r in rs)
        # MRR: 1/rank when retrieved, else 0
        mrr = sum((1.0 / int(r["gold_rank"])) if r["gold_in_retrieved"] == "1" and int(r["gold_rank"]) > 0 else 0.0
                  for r in rs) / n
        print(f"    {k} & {recall:.1f}\\% & {prec:.1f}\\% & {mrr:.3f} & {n} \\\\")


# --------------------------------------------------------------------- latency / tps
def latency_table() -> None:
    """Print the LaTeX body for the latency/generation-cost table (median end-to-end latency
    and median generated tokens per model), plus the thinking-mode decode tok/s range."""
    import statistics as st
    rows = [r for r in load(MCQ) if r["retrieval_condition"] == "rag_qwen3-embedding:0.6b_k3"
            and r.get("latency_seconds") and r.get("eval_count")]
    by = defaultdict(list)
    for r in rows:
        by[r["model_label"]].append(r)
    # Median end-to-end latency/MCQ is the practically meaningful number. Generated-token
    # count explains the cost of thinking; decode tok/s is only meaningful where the model
    # actually generates (>=20 tokens) -- a non-thinking model emits ~1 letter, so its
    # "tok/s" would be a 1-token artifact (reported as "--").
    print("% Latency & generation cost (median over the common k=3 MCQs)")
    print(r"% cols: Model & Median latency (s) & Median gen.\ tokens")
    out, think_tps = [], []
    for m, rs in by.items():
        lat = [float(r["latency_seconds"]) for r in rs if float(r["latency_seconds"]) > 0]
        gen = [float(r["eval_count"]) for r in rs]
        tps = [float(r["eval_count"]) / float(r["latency_seconds"])
               for r in rs if float(r["latency_seconds"]) > 0 and float(r["eval_count"]) >= 20]
        if "thinking_on" in m and len(tps) >= 20:
            think_tps.append(st.median(tps))
        out.append((st.median(lat), DISPLAY.get(m, m), int(st.median(gen))))
    out.sort()
    for med_lat, name, med_gen in out:
        print(f"    {name:22s} & {med_lat:.2f} & {med_gen} \\\\")
    if think_tps:
        print(f"\n% thinking-mode decode throughput (4060 Ti): "
              f"{min(think_tps):.0f} to {max(think_tps):.0f} tok/s")


# --------------------------------------------------------------------- error mining
# ------------------------------------------------------- output-format compliance
FAILED = ROOT / "results" / "medium" / "data" / "failed_attempts.csv"
FAILED_SMALL = ROOT / "results" / "small" / "data" / "failed_attempts.csv"

# The largest num_predict any attempt is granted (RETRY_GENERATION_OPTIONS). A response
# whose eval_count lands on a cap was cut off mid-generation rather than stopping.
RETRY_CAP = 4096
# Historical caps: the protocol was raised 256 -> 1024 -> 4096 as truncation was
# diagnosed, so archived failures carry whichever ceiling was in force at the time.
KNOWN_CAPS = (256, 1024, 2048, RETRY_CAP, 6144)
# A natural stop this short is the model saying it cannot answer, not running out of
# room -- the observed refusals ("I don't have access to the specific documents...")
# land around 60 tokens.
REFUSAL_MAX_TOKENS = 400

# Backend faults recorded before run_benchmark.py stamped BACKEND_FAULT_PREFIX. Matched
# on the raw message only for these legacy rows; new rows carry the prefix.
LEGACY_BACKEND_MARKERS = (
    "peg-native",           # Ollama 0.32.13 native output parser, HTTP 500
    "Connection refused",   # server down / restarting mid-run
    "timed out",
    "TimeoutError",
)
BACKEND_FAULT_PREFIX = "backend_fault: "
# Written by tools/score_unanswered.py into raw_results.csv when a question is
# adjudicated as unanswered; the failure class follows in parentheses.
SCORED_UNANSWERED_PREFIX = "scored-wrong:"


def classify_failure(row: dict) -> str:
    """Bucket one unrecovered failed_attempts row into the paper's failure taxonomy.

    ``backend`` means the inference server broke and returned no model output at all, so
    the row is evidence about the harness, not the model, and is excluded from the
    accuracy denominator. ``non-termination`` and ``refusal`` are model behaviour and
    score as wrong.
    """
    err = (row.get("error") or "").strip()
    if err.startswith(BACKEND_FAULT_PREFIX) or any(m in err for m in LEGACY_BACKEND_MARKERS):
        return "backend"
    try:
        n = int(row.get("eval_count") or 0)
    except ValueError:
        n = 0
    if n == 0:
        # No token count survives: the request produced nothing to measure.
        return "backend"
    if any(abs(n - cap) <= 2 for cap in KNOWN_CAPS):
        return "non-termination"
    if n <= REFUSAL_MAX_TOKENS:
        return "refusal"
    return "non-termination"


def compliance_table() -> None:
    """Print the LaTeX body for the output-format compliance table.

    Escalation rate is a per-model *result*, not a tuning knob: every model gets the
    identical escalation protocol (Section on Experimental Setup), so the share of
    questions a model could not answer in the requested format on the first attempt is
    directly comparable across the roster.
    """
    scored, esc, unrec = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(int))
    for path in (MCQ, MCQ_SMALL):
        for r in load(path):
            if r["retrieval_condition"] not in FAIR:
                continue
            scored[r["model_label"]] += 1
            if (r.get("attempts") or "1") not in ("", "1"):
                esc[r["model_label"]] += 1
    # Questions the model never answered are scored wrong and live in raw_results.csv,
    # carrying SCORED_UNANSWERED_PREFIX and their failure class in the error column. Count
    # them from there: reading failed_attempts.csv instead would report zero, because
    # nothing is missing any more -- the failures were adjudicated, not dropped.
    for path in (MCQ, MCQ_SMALL):
        for r in load(path):
            if r["retrieval_condition"] not in FAIR:
                continue
            err = (r.get("error") or "")
            if not err.startswith(SCORED_UNANSWERED_PREFIX):
                continue
            cls = err[err.rfind("(") + 1:err.rfind(")")] if "(" in err else "unknown"
            unrec[r["model_label"]][cls] += 1

    rows = []
    for m in scored:
        u = unrec[m]
        rows.append((m, scored[m], esc[m], u["non-termination"], u["refusal"], u["backend"]))
    # Fully compliant models carry no information here and would pad the table; report
    # them as a count in a comment instead.
    interesting = [r for r in rows if r[2] or r[3] or r[4] or r[5]]
    clean = len(rows) - len(interesting)
    interesting.sort(key=lambda r: (-(r[2] / r[1] if r[1] else 0), r[0]))

    print(f"% {clean} of {len(rows)} models answered every question in format on the first "
          f"attempt (no escalation, no unrecovered rows) and are omitted.")
    tot_esc = sum(r[2] for r in rows)
    tot_n = sum(r[1] for r in rows)
    print(f"% roster totals: n={tot_n}, escalated={tot_esc} "
          f"({100 * tot_esc / tot_n:.2f}\\%), non-termination={sum(r[3] for r in rows)}, "
          f"refusal={sum(r[4] for r in rows)}, backend={sum(r[5] for r in rows)}")
    for m, n, e, nt, rf, bf in interesting:
        print(f"    {DISPLAY.get(m, m)} & {n} & {100 * e / n:.1f} & {nt} & {rf} & {bf} \\\\")


# ------------------------------------------------------------- embedder ablation
# One model (qwen3.5:9b, thinking off) was swept across four embedders at k in {3,6,9}
# before the config narrowed to a single embedder at k in {1,3,5}. Those 4,400 rows are
# the only evidence in the benchmark about embedder choice and about depths beyond k=5,
# so they are reported as a single-model ablation rather than archived as orphans.
ABLATION_MODEL = "qwen35_9b_thinking_off"
ABLATION_EMBEDDERS = ["nomic-embed-text", "qwen3-embedding:0.6b",
                      "qwen3-embedding:4b", "qwen3-embedding:8b"]
ABLATION_KS = [3, 6, 9]
EMBEDDER_DISPLAY = {
    "nomic-embed-text": r"nomic-embed-text",
    "qwen3-embedding:0.6b": r"qwen3-emb 0.6B",
    "qwen3-embedding:4b": r"qwen3-emb 4B",
    "qwen3-embedding:8b": r"qwen3-emb 8B",
}


def embedder_ablation() -> None:
    """Print the LaTeX body for the embedder x depth ablation (accuracy %, one model)."""
    cells = defaultdict(lambda: [0, 0])
    for r in load(MCQ):
        if r["model_label"] != ABLATION_MODEL:
            continue
        cells[r["retrieval_condition"]][1] += 1
        if _is_correct(r):
            cells[r["retrieval_condition"]][0] += 1

    def acc(cond):
        hit, n = cells[cond]
        if n < MIN_REPORTABLE_N:
            return "n/a", n
        return f"{100 * hit / n:.1f}", n

    print(f"% ablation model: {DISPLAY.get(ABLATION_MODEL, ABLATION_MODEL)}; "
          f"k=9 prompts can approach the 8192-token serving context, so the k=9 column "
          f"is reported but not used to support a claim.")
    for emb in ABLATION_EMBEDDERS:
        vals, ns = [], []
        for k in ABLATION_KS:
            a, n = acc(f"rag_{emb}_k{k}")
            vals.append(a)
            ns.append(n)
        # Bold the best depth for this embedder so the row reads like the MCQ table.
        numeric = [(float(v), i) for i, v in enumerate(vals) if v != "n/a"]
        if numeric:
            best = max(numeric)[1]
            vals[best] = r"\textbf{" + vals[best] + "}"
        print(f"    {EMBEDDER_DISPLAY.get(emb, emb)} & " + " & ".join(vals)
              + f" & {min(ns)} \\\\")


def errors() -> None:
    """Print qualitative free-form error examples (ungrounded/fabricated, over-confident on
    unanswerable, smallest-model ungrounded) mined from the graded CSV for the paper."""
    rows = load(FF)
    def grab(pred, label, k=3):
        """Print the ``k`` worst-grounded answers matching ``pred`` under a heading."""
        print(f"\n### {label}")
        hits = [r for r in rows if pred(r)]
        hits.sort(key=lambda r: (float(r["source_grounded"]), float(r["fact_correct"])))
        for r in hits[:k]:
            ans = " ".join((r["raw_answer"] or "").split())
            print(f"  [{r['model_label']} {r['question_id']} {r['bucket']}] "
                  f"fc={r['fact_correct']} hf={r['hallucination_free']} sg={r['source_grounded']}")
            print(f"    note: {r['grade_note'][:180]}")
            print(f"    ans : {ans[:240]}")
    grab(lambda r: float(r["source_grounded"]) < 0.3 and r["bucket"] != "unanswerable",
         "Ungrounded / fabricated-citation answers")
    grab(lambda r: r["bucket"] == "unanswerable" and float(r["fact_correct"]) < 0.5,
         "Over-confident on unanswerable")
    grab(lambda r: r["model_label"] == "qwen3_1_7b_thinking_on" and float(r["source_grounded"]) < 0.5,
         "Smallest model, ungrounded")


SECTIONS = {"mcq": mcq_table, "mcq_small": lambda: mcq_table(MCQ_SMALL),
            "retrieval": retrieval_table, "latency": latency_table,
            "compliance": compliance_table, "ablation": embedder_ablation,
            "errors": errors}

# Sections whose output is a LaTeX table, the file main.tex \input's it from, the column
# spec (numbers right aligned, as the University template requires), and the header row. ``errors`` is qualitative prose for the author to read, not a
# table, so it is excluded. The generated file holds the whole ``tabular``: \input of a
# bare body inside an open alignment leaves TeX mid-cell when \bottomrule arrives.
KCOLS = r"Model & No-RAG & $k{=}1$ & $k{=}3$ & $k{=}5$ & $\Delta$"
LATEX_SECTIONS = {
    "mcq":       ("mcq_medium.tex", "lrrrrr", KCOLS),
    "mcq_small": ("mcq_small.tex",  "lrrrrr", KCOLS),
    "retrieval": ("retrieval.tex",  "rrrrr",
                  r"$k$ & Recall@$k$ & Precision@$k$ & MRR & $n$"),
    "latency":   ("latency.tex",    "lrr",
                  r"Model & Median latency (s) & Median gen.\ tokens"),
    "compliance": ("compliance.tex", "lrrrrr",
                   r"Model & $n$ & Escalated (\%) & Non-term. & Refusal & Backend"),
    "ablation":  ("ablation.tex",   "lrrrr",
                  r"Embedder & $k{=}3$ & $k{=}6$ & $k{=}9$ & $n$"),
}
GENERATED = ROOT / "paper" / "article" / "generated"

HEADER = ("% GENERATED by tools/paper_tables.py from results/, do not edit by hand.\n"
          "% Regenerate with: bash tools/regenerate_outputs.sh\n")


def write_latex(out_dir: Path) -> None:
    r"""Write each table body to ``out_dir`` so main.tex can \input it.

    Hand-copying these rows is how the manuscript went stale once already: the tables
    were pasted before a reparse and the figures were regenerated after it, leaving the
    PDF internally inconsistent. Generating them removes that failure mode.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for section, (filename, colspec, header) in LATEX_SECTIONS.items():
        buf = io.StringIO()
        with redirect_stdout(buf):
            SECTIONS[section]()
        # Blank lines are paragraph breaks inside a tabular, so drop them; the diagnostic
        # commentary is hoisted above the table where it stays readable to the author.
        comments, rows = [], []
        for line in buf.getvalue().splitlines():
            if not line.strip():
                continue
            (comments if line.lstrip().startswith("%") else rows).append(line)
        text = "\n".join([
            HEADER.rstrip("\n"), *comments,
            "  \\begin{tabular}{" + colspec + "}",
            "    \\toprule",
            "    " + header + " \\\\",
            "    \\midrule", *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
        ])
        (out_dir / filename).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out_dir / filename}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--latex":
        target = Path(argv[1]) if len(argv) > 1 else GENERATED
        write_latex(target)
        sys.exit(0)
    want = argv or list(SECTIONS)
    for s in want:
        if s not in SECTIONS:
            print(f"unknown section {s!r}; choose from {list(SECTIONS)}", file=sys.stderr)
            continue
        print(f"\n======== {s} ========")
        SECTIONS[s]()
