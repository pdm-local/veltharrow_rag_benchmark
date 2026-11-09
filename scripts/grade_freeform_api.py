#!/usr/bin/env python3
"""
Integrated free-form grader: replaces the manual "paste into the chat interface"
loop with direct OpenAI API calls, using the SAME rubric as the blind manual
protocol (scripts/grade_freeform_prompt.txt) and the SAME three binary metrics.

The grader scores the model's ANSWER TEXT ONLY (the `raw_answer` column). The
generator's thinking tokens are never sent to the judge.

Setup:
  echo 'OPENAI_API_KEY=sk-...' >> .env        # .env is gitignored

Usage:
  # grade the whole answers.csv once (default judge = gpt-5-mini):
  python3 scripts/grade_freeform_api.py

  # replicate the 10x cross-run agreement protocol:
  python3 scripts/grade_freeform_api.py --repeat 10

  # cheap smoke test on 8 rows:
  python3 scripts/grade_freeform_api.py --sample 8

  # override the judge if ever needed:
  python3 scripts/grade_freeform_api.py --judge-model gpt-4o

Output: results/freeform/graded.csv  (answers.csv columns + n_grades +
fact_correct, hallucination_free, source_grounded [0/1, or averaged over
--repeat] + grade_note). This is exactly what scripts/plot_freeform.py reads.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROMPT_FILE = ROOT / "scripts" / "grade_freeform_prompt.txt"
PLACEHOLDER = "<<<PASTE_JSON_HERE>>>"

METRICS = ("fact_correct", "hallucination_free", "source_grounded")

# Resolve the chunk IDs stored in answers.csv back to the actual paragraph text the
# answerer was shown, so the judge can verify grounding/hallucination against the real
# retrieved context (not just opaque IDs). Chunking is deterministic, so this exactly
# reproduces what the generator saw.
from veltharrow.config import DOCUMENTS  # noqa: E402
from veltharrow.rag import build_chunks  # noqa: E402

_DOC_SRC = {d.document_id: d.source_path for d in DOCUMENTS}
_CHUNK_TEXT_CACHE: dict[str, dict[str, str]] = {}


def chunk_text_map(document_id: str) -> dict[str, str]:
    """Return (memoized) a {chunk_id: text} map for a document, rebuilt deterministically
    from source so stored chunk ids resolve back to the exact paragraph text shown."""
    if document_id not in _CHUNK_TEXT_CACHE:
        chunks = build_chunks(document_id, _DOC_SRC[document_id])
        _CHUNK_TEXT_CACHE[document_id] = {c.chunk_id: c.text for c in chunks}
    return _CHUNK_TEXT_CACHE[document_id]


def resolve_context(row: dict) -> str:
    """Full text of the retrieved chunks, each prefixed by its [chunk_id] header."""
    ids = [i for i in (row.get("retrieved_chunk_ids", "") or "").split("|") if i]
    if not ids:
        return ""
    tmap = chunk_text_map(row["document_id"])
    parts = []
    for cid in ids:
        txt = tmap.get(cid, "")
        parts.append(f"[{cid}]\n{txt}" if txt else f"[{cid}] (text unavailable)")
    return "\n\n".join(parts)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (stdlib only). Does not overwrite existing env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_answers(path: Path) -> list[dict]:
    """Load the free-form answers CSV (from run_freeform_eval.py) as a list of row dicts."""
    with path.open() as f:
        return list(csv.DictReader(f))


# Defensive: strip any inline chain-of-thought a model might have embedded in its
# answer text. answers.csv already stores only the model's `content` (never the
# separate `thinking` field), so this is a safeguard, not the primary guard.
_THINK_RE = re.compile(
    r"<(think|thinking|reasoning|thought)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)


def strip_thinking(answer: str) -> str:
    """Defensively remove any inline <think>/<reasoning> blocks from an answer so only the
    final answer text reaches the judge."""
    return _THINK_RE.sub("", answer or "").strip()


def build_record(row: dict, row_id: str) -> dict:
    """Anonymized grading record: answer text only, no model identity, no thinking."""
    return {
        "id": row_id,
        "bucket": row["bucket"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "context": resolve_context(row),
        "answer": strip_thinking(row["raw_answer"]),
    }


def extract_json_array(text: str) -> list[dict]:
    """Pull a JSON array out of a model reply, tolerating ```json fences / prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in judge reply")
    return json.loads(t[start : end + 1])


def grade_batch(client, judge_model: str, batch: list[dict], prompt_template: str,
                temperature: float | None, retries: int = 4) -> dict[str, dict]:
    """One API call grading a batch; returns {row_id: grade_dict}."""
    payload = json.dumps(batch, ensure_ascii=False, indent=2)
    content = prompt_template.replace(PLACEHOLDER, payload)
    messages = [{"role": "user", "content": content}]

    last_exc = None
    for attempt in range(retries):
        try:
            kwargs = {"model": judge_model, "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            graded = extract_json_array(resp.choices[0].message.content)
            return {g["id"]: g for g in graded if "id" in g}
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "credit_balance_exhausted" in msg or "insufficient_quota" in msg:
                raise RuntimeError(f"OpenAI API credit exhausted: {exc}") from exc
            # Some newer models reject temperature!=1: retry without it.
            if temperature is not None and "temperature" in msg.lower():
                temperature = None
                continue
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"grading batch failed after {retries} tries: {last_exc}")


def to_binary(value) -> int:
    """Coerce a judge metric value to 0/1 (>=1 → 1), treating unparseable values as 0."""
    try:
        return 1 if int(round(float(value))) >= 1 else 0
    except (TypeError, ValueError):
        return 0


def main() -> None:
    """Grade every gradable free-form answer with the OpenAI judge (the blind rubric in
    grade_freeform_prompt.txt), optionally averaging over ``--repeat`` runs, and write the
    three binary metrics per row to the graded CSV that plot_freeform.py reads."""
    load_dotenv()
    p = argparse.ArgumentParser(description="Grade free-form answers via the OpenAI API.")
    p.add_argument("--input", default="results/freeform/answers.csv")
    p.add_argument("--output", default="results/freeform/graded.csv")
    p.add_argument("--judge-model", default="gpt-5-mini",
                   help="OpenAI model id, e.g. gpt-5-mini, gpt-5.5, gpt-4o.")
    p.add_argument("--batch-size", type=int, default=20,
                   help="Rows per API call (the rubric grades a JSON array).")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent independent API batches (default: 1).")
    p.add_argument("--repeat", type=int, default=1,
                   help="Grade every row N times and average (cross-run agreement).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sample", type=int, default=0, help="Grade only the first N gradable rows.")
    p.add_argument("--api-key", default="", help="Overrides OPENAI_API_KEY / .env.")
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: no OPENAI_API_KEY found (put it in .env or pass --api-key).")
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not found. Run: pip install openai")
    client = OpenAI(api_key=api_key)

    prompt_template = PROMPT_FILE.read_text()
    if PLACEHOLDER not in prompt_template:
        raise SystemExit(f"{PROMPT_FILE} is missing the {PLACEHOLDER} marker.")

    rows = load_answers(Path(args.input))
    gradable = [r for r in rows if r.get("raw_answer") and not r.get("error")]
    if args.sample:
        gradable = gradable[: args.sample]
    if not gradable:
        raise SystemExit(f"No gradable rows in {args.input} (need raw_answer, no error).")

    for i, r in enumerate(gradable, 1):
        r["_row_id"] = f"row-{i:04d}"
    by_id = {r["_row_id"]: r for r in gradable}

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    print(f"Judge: {args.judge_model}  rows: {len(gradable)}  "
          f"batch_size: {args.batch_size}  repeat: {args.repeat}  "
          f"workers: {args.workers}", flush=True)

    # accumulate scores across repeats: row_id -> {metric: [vals]}, plus notes
    acc: dict[str, dict] = {rid: {m: [] for m in METRICS} | {"notes": []} for rid in by_id}

    batches = [gradable[i : i + args.batch_size] for i in range(0, len(gradable), args.batch_size)]
    total_calls = len(batches) * args.repeat
    temp = args.temperature
    checkpoint_path = Path(str(args.output) + ".checkpoint.json")
    signature = {
        "input": str(Path(args.input).resolve()),
        "judge_model": args.judge_model,
        "rows": len(gradable),
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "temperature": temp,
    }
    completed: set[str] = set()
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text())
        if saved.get("signature") != signature:
            raise SystemExit(
                f"Checkpoint settings differ from this run: {checkpoint_path}. "
                "Use the same arguments or move the checkpoint aside."
            )
        completed = set(saved.get("completed", []))
        saved_acc = saved.get("acc", {})
        for rid in acc:
            if rid in saved_acc:
                acc[rid] = saved_acc[rid]
        print(f"Resuming checkpoint: {len(completed)}/{total_calls} batches complete", flush=True)

    def save_checkpoint() -> None:
        """Atomically persist completed batches and score accumulators."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(checkpoint_path) + ".tmp")
        tmp.write_text(json.dumps({
            "signature": signature,
            "completed": sorted(completed),
            "acc": acc,
        }))
        tmp.replace(checkpoint_path)

    call = len(completed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rep in range(args.repeat):
            pending = {}
            for batch_index, batch in enumerate(batches):
                task_key = f"{rep}:{batch_index}"
                if task_key in completed:
                    continue
                records = [build_record(r, r["_row_id"]) for r in batch]
                future = pool.submit(
                    grade_batch,
                    client,
                    args.judge_model,
                    records,
                    prompt_template,
                    temp,
                )
                pending[future] = (batch, task_key)
            for future in as_completed(pending):
                batch, task_key = pending[future]
                grades = future.result()
                call += 1
                for rid in (r["_row_id"] for r in batch):
                    g = grades.get(rid)
                    if not g:
                        print(f"  WARN: judge omitted {rid} (rep {rep+1})", flush=True)
                        continue
                    for m in METRICS:
                        acc[rid][m].append(to_binary(g.get(m)))
                    if g.get("note"):
                        acc[rid]["notes"].append(str(g["note"]))
                completed.add(task_key)
                save_checkpoint()
                print(
                    f"  [{call}/{total_calls}] rep {rep+1} batch graded ({len(batch)} rows)",
                    flush=True,
                )

    # Build output rows
    out_rows = []
    for rid, src in by_id.items():
        scored = acc[rid]
        n = max((len(scored[m]) for m in METRICS), default=0)
        if n == 0:
            continue
        row = {k: v for k, v in src.items() if k != "_row_id"}
        row["n_grades"] = n
        for m in METRICS:
            vals = scored[m]
            row[m] = f"{sum(vals) / len(vals):.3f}" if len(vals) > 1 else str(vals[0] if vals else 0)
        notes = list(dict.fromkeys(scored["notes"]))  # dedupe, keep order
        row["grade_note"] = " | ".join(notes)
        out_rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [c for c in rows[0].keys() if c != "_row_id"] + [
        "n_grades", *METRICS, "grade_note",
    ]
    # drop any duplicate metric columns if answers.csv somehow already had them
    seen, ordered = set(), []
    for c in fieldnames:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    # quick summary
    def overall(metric: str) -> float:
        """Mean of one metric across all graded rows, as a percentage."""
        vals = [float(r[metric]) for r in out_rows]
        return 100 * sum(vals) / len(vals) if vals else 0.0

    print(f"\nWrote {len(out_rows)} graded rows -> {out_path}")
    print("Overall:  " + "  ".join(f"{m}={overall(m):.1f}%" for m in METRICS))
    save_checkpoint()


if __name__ == "__main__":
    main()
