#!/usr/bin/env python3
"""
Step 2 - Part A: Generate free-text answers for each model × condition × question.

Usage:
  python3 generate_answers.py --tier medium
  python3 generate_answers.py --tier small --model qwen35_0_8b_thinking_off
  python3 generate_answers.py --tier medium --condition no_rag --condition rag_qwen3-embedding:0.6b_k3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(_SCRIPT_DIR))

from veltharrow.config import DEFAULT_TIER, DOCUMENTS, RESULTS_DIR, resolve_tier
from veltharrow.ollama_client import OllamaClient
from veltharrow.rag import PrecomputedRetriever, retrieval_cache_path
from run_benchmark import (
    load_questions,
    format_context,
    selected_models,
    selected_conditions,
)

GEN_COLUMNS = [
    "run_id",
    "timestamp",
    "document_id",
    "question_id",
    "model_label",
    "model",
    "thinking_mode",
    "retrieval_condition",
    "embedding_model",
    "top_k",
    "question_text",
    "correct_answer_letter",
    "correct_answer_text",
    "generated_answer",
    "thinking",
    "context_used",
    "latency_seconds",
    "error",
]

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant for a local government office. "
    "Answer the question below clearly and concisely in 2-4 sentences. "
    "Base your answer only on the provided document excerpts when available. "
    "Do not mention that you are an AI."
)


def build_gen_prompt(question: dict[str, Any], context: str = "") -> str:
    """Build the free-text prompt: optional retrieved-excerpts block plus the question
    (no answer options, unlike the MCQ prompt)."""
    context_block = ""
    if context:
        context_block = (
            "Document excerpts:\n"
            f"{context}\n\n"
        )
    return f"{context_block}Question: {question['Question']}"


def output_path(args: argparse.Namespace) -> Path:
    """Resolve the generation output CSV path (``--output`` or the default under
    ``results/``)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return Path(args.output) if args.output else RESULTS_DIR / "generation_answers.csv"


def completed_keys(path: Path) -> set[tuple]:
    """Set of (document, question, model_label, condition) keys already generated
    without error, so a resumed run skips them."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("error"):
                keys.add((
                    row["document_id"],
                    row["question_id"],
                    row["model_label"],
                    row["retrieval_condition"],
                ))
    return keys


def append_row(path: Path, row: dict) -> None:
    """Append one generation row to ``path`` under an exclusive file lock (writing the
    header if the file is new), so parallel workers can share the CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            write_header = fh.seek(0, 2) == 0
            writer = csv.DictWriter(fh, fieldnames=GEN_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({col: row.get(col, "") for col in GEN_COLUMNS})
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> None:
    """Generate free-text answers for every selected model × condition × question,
    reusing the precomputed retrieval cache for RAG conditions and appending each
    answer (with latency and the context used) to the output CSV. Resumable."""
    tier = resolve_tier(args.tier)
    print(f"Running generation - tier: {tier}", flush=True)
    client = OllamaClient(host=args.ollama_host, timeout=args.timeout)
    models = selected_models(args.model, tier=tier)
    conditions = selected_conditions(args.condition, tier=tier)
    out = output_path(args)
    done = completed_keys(out)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retrievers: dict = {}

    total = sum(
        len(load_questions(doc.questions_path)) * len(models) * len(conditions)
        for doc in DOCUMENTS
    )
    print(f"Planned rows: {total}", flush=True)

    for model_spec in models:
        print(f"\n=== {model_spec.label} ===", flush=True)
        for doc_spec in DOCUMENTS:
            questions = load_questions(doc_spec.questions_path)
            for question in questions:
                for condition in conditions:
                    rc = str(condition["retrieval_condition"])
                    key = (doc_spec.document_id, question["QuestionID"], model_spec.label, rc)
                    if key in done:
                        continue

                    embedding_model = str(condition["embedding_model"])
                    top_k = int(condition["top_k"])
                    context = ""
                    retrieved_chunks = []
                    error = ""

                    try:
                        if rc != "no_rag":
                            pre_path = retrieval_cache_path(doc_spec.document_id, embedding_model, top_k)
                            if pre_path.exists():
                                pre_key = (doc_spec.document_id, embedding_model, top_k)
                                if pre_key not in retrievers:
                                    retrievers[pre_key] = PrecomputedRetriever(
                                        doc_spec.document_id, embedding_model, top_k
                                    )
                                retrieved_chunks = retrievers[pre_key].retrieve(question["QuestionID"])
                            context = format_context(retrieved_chunks)

                        prompt = build_gen_prompt(question, context=context)

                        # Temporarily override system prompt in client
                        payload = {
                            "model": model_spec.model,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            "stream": False,
                            "think": model_spec.thinking,
                            "keep_alive": 3600,
                            "options": {"temperature": 0, "seed": 42, "num_predict": 256},
                        }
                        import time as _time
                        t0 = _time.perf_counter()
                        data = client._post("/api/chat", payload, timeout=300)
                        latency = _time.perf_counter() - t0
                        msg = data.get("message", {})
                        generated = msg.get("content", "").strip()
                        thinking = msg.get("thinking", "") or ""

                    except Exception as exc:
                        generated = ""
                        thinking = ""
                        latency = 0.0
                        error = repr(exc)

                    correct_letter = question["CorrectOptionLetter"]
                    correct_text = question.get(f"Option{correct_letter}", "")

                    row = {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "document_id": doc_spec.document_id,
                        "question_id": question["QuestionID"],
                        "model_label": model_spec.label,
                        "model": model_spec.model,
                        "thinking_mode": "on" if model_spec.thinking else "off",
                        "retrieval_condition": rc,
                        "embedding_model": embedding_model,
                        "top_k": top_k,
                        "question_text": question["Question"],
                        "correct_answer_letter": correct_letter,
                        "correct_answer_text": correct_text,
                        "generated_answer": generated,
                        "thinking": thinking,
                        "context_used": context[:2000] if context else "",
                        "latency_seconds": f"{latency:.3f}",
                        "error": error,
                    }
                    append_row(out, row)
                    status = "ERR" if error else "OK"
                    print(
                        f"{status} {doc_spec.document_id} {question['QuestionID']} "
                        f"{model_spec.label} {rc}",
                        flush=True,
                    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the generation step."""
    parser = argparse.ArgumentParser(description="Generate free-text answers for Step 2 evaluation.")
    parser.add_argument("--tier", default=os.environ.get("BENCHMARK_TIER", DEFAULT_TIER))
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", default="")
    parser.add_argument("--model", action="append", help="Model label filter")
    parser.add_argument("--condition", action="append", help="Retrieval condition filter")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
