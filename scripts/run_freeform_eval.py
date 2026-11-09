#!/usr/bin/env python3
"""
Free-form evaluation runner: generates open-ended answers from selected models
on the free-form question pack, with optional RAG and optional full-doc context.

Usage:
  python3 scripts/run_freeform_eval.py \
    --questions data/freeform/questions.json \
    --output results/freeform/answers.csv \
    --model gemma4_e2b_thinking_on --model gemma4_e4b_thinking_on \
    --model qwen35_9b_thinking_off --model qwen3_8b_thinking_on \
    --model qwen25_7b_thinking_off

Each (model, question) row records: latency, retrieved chunks, raw answer, full
prompt fed to the model. Grading is a separate step: run scripts/grade_freeform_api.py
(OpenAI judge, rubric in grade_freeform_prompt.txt). This script does NOT score anything.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veltharrow.config import (
    DOCUMENTS,
    OLLAMA_HOST,
    SMALL_MODEL_SPECS,
    MEDIUM_MODEL_SPECS,
)
from veltharrow.ollama_client import OllamaClient
from veltharrow.rag import Retriever, load_markdown


ALL_SPECS = {s.label: s for s in (*SMALL_MODEL_SPECS, *MEDIUM_MODEL_SPECS)}

# Free-form generation uses its own system prompt (not the MCQ single-letter one)
# and a large token budget so verbose thinking models are not truncated mid-answer.
FREEFORM_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions for a Veltharrow municipal "
    "staff member. Answer in clear, complete prose using only the information in "
    "the provided context. If the answer is not in the context, say so explicitly "
    "instead of inventing facts. Cite the relevant section or article when you can."
)
FREEFORM_GENERATION_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 8192,
    # Context window must hold the RAG prompt (~2.5-3.5K tokens) PLUS the generated
    # thinking+answer (up to num_predict). Ollama defaults to 4096, which is smaller
    # than a verbose thinking model's output: once reasoning passes 4096 the question
    # scrolls out of the window and the model never emits an answer. 16384 holds the
    # prompt + a full 8192-token generation with margin, and fits in 12 GB of VRAM
    # (Q4 9B weights ~6 GB + GQA KV cache ~1 GB at 16K).
    "num_ctx": 16384,
}

DOC_BY_ID = {d.document_id: d for d in DOCUMENTS}

COLUMNS = [
    "run_id", "timestamp",
    "question_id", "bucket", "document_id",
    "model_label", "model", "thinking_mode",
    "retrieval_mode", "top_k", "embedding_model",
    "question", "gold_answer", "gold_keywords",
    "retrieved_chunk_ids",
    "raw_answer",
    "latency_seconds", "prompt_eval_count", "eval_count",
    "error",
]


PROMPT_TEMPLATE = """You are answering questions from a Veltharrow municipal staff member. Use ONLY the information in the CONTEXT below. If the answer is not in the context, say so explicitly: do not invent facts.

Write a complete sentence or paragraph answer in plain prose. Do NOT respond with a single letter or number: this is NOT a multiple choice question.

When you cite a fact, mention which section or article it came from when possible.

CONTEXT:
{context}

QUESTION:
{question}

Write your answer below in full sentences:"""


def load_questions(path: Path) -> list[dict]:
    """Load the free-form question pack (a JSON list of question dicts)."""
    with path.open() as f:
        return json.load(f)


def append_row(path: Path, row: dict) -> None:
    """Append one answer row to ``path`` under an exclusive file lock (writing the
    header if new), so parallel workers can share the CSV safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0, 2)
            write_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    """Set of (model_label, question_id, retrieval_mode) keys already answered without
    error, so a resumed run skips them."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("error"):
                continue
            keys.add((row["model_label"], row["question_id"], row["retrieval_mode"]))
    return keys


def build_retrievers(embedding_model: str, client: OllamaClient) -> dict:
    """One Retriever per document, reused across all questions."""
    out = {}
    for doc in DOCUMENTS:
        out[doc.document_id] = Retriever(
            document_id=doc.document_id,
            source_path=doc.source_path,
            embedding_model=embedding_model,
            client=client,
        )
    return out


def full_document_text(document_id: str) -> str:
    """Return the entire markdown text of a document (for the full-document context mode)."""
    doc = DOC_BY_ID[document_id]
    return load_markdown(document_id, doc.source_path)


def build_context_rag(question: dict, retrievers: dict, top_k: int) -> tuple[str, list[str]]:
    """Build RAG context for a question: retrieve the top-k chunks from its document and
    return the joined, id-labelled text plus the list of retrieved chunk ids."""
    retriever = retrievers[question["document_id"]]
    chunks = retriever.retrieve(question["question"], top_k=top_k)
    chunk_texts = []
    chunk_ids = []
    for c in chunks:
        chunk_ids.append(c.chunk_id)
        chunk_texts.append(f"[{c.chunk_id}]\n{c.text}")
    return "\n\n".join(chunk_texts), chunk_ids


def build_context_full(question: dict) -> tuple[str, list[str]]:
    """Build full-document context for a question: the whole source document plus the
    sentinel chunk id ``FULL_DOC`` (the upper-bound, no-retrieval baseline)."""
    return full_document_text(question["document_id"]), ["FULL_DOC"]


def run_one(client: OllamaClient, spec, prompt: str, think: bool) -> dict:
    """Run one free-form generation and return a row fragment (answer, latency, token
    counts, error); errors are captured into the ``error`` field rather than raised."""
    t0 = time.time()
    try:
        resp = client.chat(
            spec.model,
            prompt,
            think=think,
            system=FREEFORM_SYSTEM_PROMPT,
            options=FREEFORM_GENERATION_OPTIONS,
            finalize=False,
        )
        latency = time.time() - t0
        return {
            "raw_answer": resp.get("content", ""),
            "latency_seconds": f"{latency:.3f}",
            "prompt_eval_count": resp.get("prompt_eval_count", ""),
            "eval_count": resp.get("eval_count", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "raw_answer": "",
            "latency_seconds": f"{time.time() - t0:.3f}",
            "prompt_eval_count": "",
            "eval_count": "",
            "error": repr(exc),
        }


def main() -> None:
    """Generate free-form answers for every selected model × question × context mode
    (RAG and optional full-document), appending each to the output CSV. Resumable; does
    no grading (see grade_freeform_api.py)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/freeform/questions.json")
    parser.add_argument("--output", default="results/freeform/answers.csv")
    parser.add_argument("--model", action="append", required=True,
                        help="Model label. Repeatable.")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ollama-host", default=OLLAMA_HOST)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--num-predict", type=int, default=FREEFORM_GENERATION_OPTIONS["num_predict"],
                        help="Max generated tokens (thinking + answer share this budget).")
    parser.add_argument("--num-ctx", type=int, default=FREEFORM_GENERATION_OPTIONS["num_ctx"],
                        help="Context window. Must hold the prompt + generated tokens, or the "
                             "model loses the question mid-reasoning. Larger needs more VRAM.")
    parser.add_argument("--rag-only", action="store_true",
                        help="Only run RAG (retrieved-paragraph) context; skip full-document mode.")
    parser.add_argument("--answer-budget-hint", type=int, default=0,
                        help="If >0, append an instruction telling the model to keep its entire "
                             "response (reasoning + answer) within roughly this many tokens. "
                             "For verbose thinking models that otherwise never terminate.")
    parser.add_argument("--answer-hint-text", default="",
                        help="If set, append this exact instruction to every prompt (overrides "
                             "--answer-budget-hint). For models that loop and never terminate.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    # Allow overriding the generation budget and context window (e.g. for verbose
    # thinking models that exhaust the default before emitting a final answer).
    FREEFORM_GENERATION_OPTIONS["num_predict"] = args.num_predict
    FREEFORM_GENERATION_OPTIONS["num_ctx"] = args.num_ctx

    client = OllamaClient(host=args.ollama_host, timeout=args.timeout)

    unknown = [m for m in args.model if m not in ALL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model labels: {unknown}")
    specs = [ALL_SPECS[m] for m in args.model]

    print(f"Models: {[s.label for s in specs]}", flush=True)
    print(f"Embedding: {args.embedding_model}  top_k={args.top_k}", flush=True)

    questions = load_questions(Path(args.questions))
    print(f"Questions: {len(questions)}", flush=True)

    retrievers = build_retrievers(args.embedding_model, client)

    output_path = Path(args.output)
    done = completed_keys(output_path) if args.resume else set()
    print(f"Already completed: {len(done)} rows. Resuming." if done else "Fresh run.", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total = 0
    for spec in specs:
        print(f"\n=== {spec.label} ({spec.model}) ===", flush=True)
        for q in questions:
            modes = ["rag"]
            if q.get("run_full_doc") and not args.rag_only:
                modes.append("full_doc")
            for mode in modes:
                key = (spec.label, q["id"], mode)
                if key in done:
                    continue
                if mode == "rag":
                    context, chunk_ids = build_context_rag(q, retrievers, args.top_k)
                else:
                    context, chunk_ids = build_context_full(q)

                prompt = PROMPT_TEMPLATE.format(context=context, question=q["question"])
                if args.answer_hint_text:
                    prompt += f"\n\n{args.answer_hint_text}"
                elif args.answer_budget_hint > 0:
                    prompt += (
                        f"\n\nIMPORTANT: Be concise and decisive. Keep your entire response, "
                        f"including any private reasoning, within roughly {args.answer_budget_hint} "
                        f"tokens. Do not over-deliberate or loop: reach and state your final answer "
                        f"directly."
                    )
                result = run_one(client, spec, prompt, think=spec.thinking)

                row = {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": q["id"],
                    "bucket": q["bucket"],
                    "document_id": q["document_id"],
                    "model_label": spec.label,
                    "model": spec.model,
                    "thinking_mode": "on" if spec.thinking else "off",
                    "retrieval_mode": mode,
                    "top_k": args.top_k if mode == "rag" else 0,
                    "embedding_model": args.embedding_model if mode == "rag" else "",
                    "question": q["question"],
                    "gold_answer": q["gold_answer"],
                    "gold_keywords": "|".join(q.get("gold_keywords", [])),
                    "retrieved_chunk_ids": "|".join(chunk_ids),
                    **result,
                }
                append_row(output_path, row)
                total += 1
                tag = result["error"] or f"{len(result['raw_answer'])} chars in {result['latency_seconds']}s"
                print(f"  {q['id']:8s} {q['bucket']:12s} {mode:8s} → {tag}", flush=True)

    print(f"\nDone. {total} new rows in {output_path}", flush=True)


if __name__ == "__main__":
    main()
