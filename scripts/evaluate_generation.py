#!/usr/bin/env python3
"""
Step 2 - Part B: Judge generated answers with a local Ollama model or a GPT model.

Usage:
  # Default: local judge via Ollama (no API cost)
  python3 evaluate_generation.py --judge-model qwen3.5:9b

  # OpenAI judge (costs money, use --sample to cap):
  export OPENAI_API_KEY=sk-...
  python3 evaluate_generation.py --judge-backend openai --judge-model gpt-5-mini --sample 5

The --sample flag picks N questions per (model, condition, document) stratum,
keeping total calls bounded. --sample 0 (default) runs the full file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veltharrow.config import RESULTS_DIR
from veltharrow.ollama_client import OllamaClient

SCORE_COLUMNS = [
    "run_id",
    "timestamp",
    "document_id",
    "question_id",
    "model_label",
    "thinking_mode",
    "retrieval_condition",
    "question_text",
    "correct_answer_text",
    "generated_answer",
    "judge_model",
    "judge_backend",
    "factual_score",
    "relevance_score",
    "overall_score",
    "judge_rationale",
    "judge_timestamp",
]

JUDGE_SYSTEM = (
    "You are an expert evaluator for local government document question-answering systems. "
    "You assess whether AI-generated answers are accurate and helpful. "
    "Be strict but fair. Always respond in valid JSON."
)

JUDGE_TEMPLATE = """\
Evaluate this AI-generated answer against the reference answer.

Question: {question}

Reference answer (correct): {correct}

AI-generated answer: {generated}

Score on three dimensions (1 = completely wrong/irrelevant, 10 = perfect):
- factual_score: Does the answer contain accurate information matching the reference?
- relevance_score: Does the answer actually address the question asked?
- overall_score: Holistic quality (accuracy + clarity + usefulness)

Respond ONLY with JSON in this exact format:
{{
  "factual_score": <int 1-10>,
  "relevance_score": <int 1-10>,
  "overall_score": <int 1-10>,
  "rationale": "<one sentence explaining the scores>"
}}"""


def _parse_scores(raw: str) -> dict:
    """Parse the judge's JSON reply into factual/relevance/overall ints plus a rationale,
    defaulting missing fields to a neutral 5."""
    result = json.loads(raw)
    return {
        "factual_score": int(result.get("factual_score", 5)),
        "relevance_score": int(result.get("relevance_score", 5)),
        "overall_score": int(result.get("overall_score", 5)),
        "rationale": str(result.get("rationale", "")),
    }


def _empty_score(rationale: str) -> dict:
    """Score assigned when the model produced no answer: 1 on every dimension."""
    return {
        "factual_score": 1,
        "relevance_score": 1,
        "overall_score": 1,
        "rationale": rationale,
    }


def _error_score(exc: Exception) -> dict:
    """Sentinel score for a judge failure: blank numeric scores and the error in the
    rationale, so the row is recorded but excluded from aggregates."""
    return {
        "factual_score": "",
        "relevance_score": "",
        "overall_score": "",
        "rationale": f"Judge error: {exc}",
    }


def judge_openai(question: str, correct: str, generated: str, judge_model: str, client) -> dict:
    """Score one generated answer with an OpenAI judge model, returning the parsed scores.

    Empty answers short-circuit to :func:`_empty_score`. Retries with exponential backoff,
    and adapts to models that reject ``temperature`` or ``response_format`` by dropping
    those kwargs; returns :func:`_error_score` if all retries fail.
    """
    if not generated.strip():
        return _empty_score("Empty answer")

    prompt = JUDGE_TEMPLATE.format(question=question, correct=correct, generated=generated[:1500])
    base_kwargs: dict[str, object] = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    if not judge_model.startswith(("gpt-5", "o1", "o3", "o4")):
        base_kwargs["temperature"] = 0

    for attempt in range(5):
        try:
            response = client.chat.completions.create(**base_kwargs)
            return _parse_scores(response.choices[0].message.content)
        except Exception as exc:
            msg = str(exc)
            if "temperature" in msg and "temperature" in base_kwargs:
                base_kwargs.pop("temperature", None)
                continue
            if "response_format" in msg and "response_format" in base_kwargs:
                base_kwargs.pop("response_format", None)
                continue
            if attempt == 4:
                return _error_score(exc)
            time.sleep(min(30.0, (2 ** attempt) + random.random()))
    return _error_score(RuntimeError("retries exhausted"))


def judge_local(question: str, correct: str, generated: str, judge_model: str, client: OllamaClient) -> dict:
    """Score one generated answer with a local Ollama judge model (JSON mode, deterministic),
    returning the parsed scores. Empty answers short-circuit to :func:`_empty_score`;
    retries a few times, then returns :func:`_error_score`."""
    if not generated.strip():
        return _empty_score("Empty answer")

    prompt = JUDGE_TEMPLATE.format(question=question, correct=correct, generated=generated[:1500])
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": 3600,
        "options": {"temperature": 0, "seed": 42, "num_predict": 256},
    }
    for attempt in range(3):
        try:
            data = client._post("/api/chat", payload, timeout=600)
            content = data.get("message", {}).get("content", "")
            return _parse_scores(content)
        except json.JSONDecodeError as exc:
            if attempt == 2:
                return _error_score(exc)
            time.sleep(1.0 + random.random())
        except Exception as exc:
            if attempt == 2:
                return _error_score(exc)
            time.sleep(2.0 + random.random())
    return _error_score(RuntimeError("retries exhausted"))


def completed_keys(path: Path) -> set[tuple]:
    """Set of (document, question, model_label, condition) keys already scored, so a
    resumed run skips them."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("overall_score"):
                keys.add((
                    row["document_id"],
                    row["question_id"],
                    row["model_label"],
                    row["retrieval_condition"],
                ))
    return keys


def append_score(path: Path, row: dict) -> None:
    """Append one score row to the output CSV, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in SCORE_COLUMNS})


def stratified_sample(rows: list[dict], n_per_stratum: int, seed: int = 42) -> list[dict]:
    """Pick up to N rows per (model_label, retrieval_condition, document_id) stratum."""
    if n_per_stratum <= 0:
        return rows
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model_label"], r["retrieval_condition"], r["document_id"])
        buckets[key].append(r)
    rng = random.Random(seed)
    picked: list[dict] = []
    for key, items in buckets.items():
        items_sorted = sorted(items, key=lambda x: x["question_id"])
        rng.shuffle(items_sorted)
        picked.extend(items_sorted[:n_per_stratum])
    return picked


def run(args: argparse.Namespace) -> None:
    """Judge each pending generated answer (optionally stratified-sampled) with the chosen
    backend and append the scores to the output CSV. Resumable; warns before large OpenAI
    runs."""
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        print("Run generate_answers.py first.")
        sys.exit(1)

    # Set up backend
    if args.judge_backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("openai package not found. Run: pip install openai")
            sys.exit(1)
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: --judge-backend openai requires OPENAI_API_KEY or --api-key")
            sys.exit(1)
        openai_client = OpenAI(api_key=api_key)
        ollama_client = None
    else:
        openai_client = None
        ollama_client = OllamaClient(host=args.ollama_host or None)

    gen_rows = list(csv.DictReader(in_path.open(encoding="utf-8")))
    if args.model_filter:
        gen_rows = [r for r in gen_rows if r["model_label"] in args.model_filter]
    gen_rows = [r for r in gen_rows if not r.get("error")]

    if args.sample > 0:
        before = len(gen_rows)
        gen_rows = stratified_sample(gen_rows, args.sample)
        print(f"Stratified sample: {before} -> {len(gen_rows)} rows (--sample {args.sample})", flush=True)

    done = completed_keys(out_path)
    pending = [
        r for r in gen_rows
        if (r["document_id"], r["question_id"], r["model_label"], r["retrieval_condition"]) not in done
    ]

    print(f"Backend: {args.judge_backend} | Judge model: {args.judge_model}", flush=True)
    print(f"Rows to evaluate: {len(pending)} (already done: {len(done)})", flush=True)

    if args.judge_backend == "openai" and len(pending) > 1000 and args.sample == 0:
        print(
            f"WARNING: {len(pending)} OpenAI calls will hit your bill. "
            f"Use --sample N (e.g. --sample 5) to cap. Sleeping 5s; Ctrl-C to abort.",
            flush=True,
        )
        time.sleep(5)

    for i, gen_row in enumerate(pending, 1):
        if args.judge_backend == "openai":
            scores = judge_openai(
                gen_row["question_text"], gen_row["correct_answer_text"],
                gen_row["generated_answer"], args.judge_model, openai_client,
            )
        else:
            scores = judge_local(
                gen_row["question_text"], gen_row["correct_answer_text"],
                gen_row["generated_answer"], args.judge_model, ollama_client,
            )

        score_row = {
            "run_id": gen_row.get("run_id", ""),
            "timestamp": gen_row.get("timestamp", ""),
            "document_id": gen_row["document_id"],
            "question_id": gen_row["question_id"],
            "model_label": gen_row["model_label"],
            "thinking_mode": gen_row.get("thinking_mode", ""),
            "retrieval_condition": gen_row["retrieval_condition"],
            "question_text": gen_row["question_text"],
            "correct_answer_text": gen_row["correct_answer_text"],
            "generated_answer": gen_row["generated_answer"],
            "judge_model": args.judge_model,
            "judge_backend": args.judge_backend,
            "factual_score": scores.get("factual_score", ""),
            "relevance_score": scores.get("relevance_score", ""),
            "overall_score": scores.get("overall_score", ""),
            "judge_rationale": scores.get("rationale", ""),
            "judge_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_score(out_path, score_row)

        overall = scores.get("overall_score", "?")
        print(
            f"[{i}/{len(pending)}] {gen_row['model_label']} | "
            f"{gen_row['retrieval_condition']:<35} | "
            f"{gen_row['document_id']} {gen_row['question_id']} | overall={overall}",
            flush=True,
        )

        if args.judge_backend == "openai" and i % args.batch_size == 0:
            time.sleep(1)

    print(f"\nDone. Scores written to {out_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the generation-judging step."""
    parser = argparse.ArgumentParser(description="Judge generated answers (local Ollama or OpenAI).")
    parser.add_argument(
        "--input",
        default=str(RESULTS_DIR / "generation_answers.csv"),
        help="Path to generation_answers.csv",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "generation_scores.csv"),
        help="Path to write scores CSV",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["local", "openai"],
        default="local",
        help="Where to run the judge. 'local' uses Ollama (free); 'openai' costs money.",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen3.5:9b",
        help="Judge model tag. For local: an Ollama tag (e.g. qwen3.5:9b, gemma4:e4b). "
             "For openai: an API model name (e.g. gpt-5-mini, gpt-4o-mini).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Pick at most N rows per (model, condition, document) stratum. 0 = full file.",
    )
    parser.add_argument("--api-key", default="", help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--ollama-host", default="", help="Override Ollama host URL")
    parser.add_argument(
        "--model-filter",
        action="append",
        metavar="LABEL",
        help="Only evaluate rows from this model label (repeatable)",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Rows between OpenAI rate-limit pauses")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
