"""Phase 2 of the pipeline: the LLM-only MCQ benchmark.

For every (model, document, question, retrieval condition) it builds a prompt — using
the precomputed retrieval cache where available so no embedder is loaded — asks the
model for a single A–F letter, parses and scores the answer, and appends a row to
``results/<tier>/data/raw_results.csv``. It is resumable (``--resume`` skips rows already
present), shardable across workers (``--worker`` / ``--total-workers``), and filterable
by ``--model`` / ``--condition`` / ``--tier``. Run ``precompute_retrievals.py`` first.

stdlib + numpy only (numpy arrives via ``veltharrow.rag``); imports the ``src/`` package
by inserting it on ``sys.path`` rather than requiring an install.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import socket
import sys
import time

csv.field_size_limit(sys.maxsize)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veltharrow.config import (
    DEFAULT_TIER,
    DOCUMENTS,
    ModelSpec,
    FINAL_RETRY_FROM_ATTEMPT,
    FINAL_RETRY_GENERATION_OPTIONS,
    RETRY_GENERATION_OPTIONS,
    RETRY_SEED_STRIDE,
    auto_pull_models_for,
    model_specs_for,
    required_models_for,
    resolve_tier,
    results_dir_for,
    retrieval_conditions,
)
from veltharrow.ollama_client import OllamaClient, OllamaError
from veltharrow.rag import PrecomputedRetriever, Retriever, build_chunks, load_markdown, retrieval_cache_path


RAW_COLUMNS = [
    "run_id",
    "timestamp",
    "document_id",
    "question_id",
    "question_type",
    "bloom_level",
    "model_label",
    "model",
    "thinking_mode",
    "retrieval_condition",
    "embedding_model",
    "top_k",
    "predicted_letter",
    "correct_letter",
    "is_correct",
    "raw_response",
    "thinking",
    "retrieved_chunk_ids",
    "retrieved_pages",
    "latency_seconds",
    "prompt_eval_count",
    "eval_count",
    "fallback_used",
    "fallback_prompt_eval_count",
    "fallback_eval_count",
    "attempts",
    "error",
]

FAILURE_COLUMNS = RAW_COLUMNS + ["failure_timestamp"]
VALID_LETTERS = set("ABCDEF")


# Markers for a question adjudicated as never answered. They live here, not in
# tools/score_unanswered.py, because --resume has to recognise them: that tool imports
# these back rather than defining its own copy, so the two can never drift apart.
#
# ``predicted_letter`` is deliberately outside A-F so it can never be mistaken for a
# choice, and non-empty so it passes the "no blank letters" guard on the canonical CSV.
NO_ANSWER_LETTER = "X"
# Stamped into ``error``. reparse_results.py keys off it too, so an adjudicated response
# is never re-parsed.
SCORED_UNANSWERED = "scored-wrong: no answer within the escalation protocol"


def scored_unanswered(row: dict) -> bool:
    """True if this row is a deliberate "never answered, scored wrong" verdict.

    Such a row is finished work, not a gap. Without this, --resume sees an invalid
    predicted_letter plus a non-empty error and re-runs the question, then *appends* the
    result: on 2026-09-07 an auto-started gapfill did exactly that and pushed a cell to
    402 rows with duplicate keys.
    """
    return str(row.get("error") or "").strip().startswith(SCORED_UNANSWERED)


def valid_letter(value: object) -> bool:
    """True if ``value`` is one of the six valid option letters (A–F)."""
    return str(value or "").strip().upper() in VALID_LETTERS


def has_error(value: object) -> bool:
    """True if ``value`` holds a non-empty error string (treating ``nan`` as none)."""
    text = str(value or "").strip()
    return bool(text) and text.lower() != "nan"


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate one document's 100 MCQs, returning rows with normalized
    keys (``QuestionID``, ``Question``, ``CorrectOptionLetter``, ``OptionA``..``F``,
    ``Type``, ``BloomLevel``). Raises ``ValueError`` on a malformed set (wrong count,
    missing fields/options, or an out-of-range answer letter)."""
    with path.open("r", encoding="utf-8") as handle:
        raw_questions = json.load(handle)
    if len(raw_questions) != 100:
        raise ValueError(f"{path} has {len(raw_questions)} questions, expected 100")
    normalized: list[dict[str, Any]] = []
    for item in raw_questions:
        for key in ["id", "question", "options", "answer"]:
            if key not in item:
                raise ValueError(f"{path} question is missing {key}: {item}")
        options = item["options"]
        for letter in "ABCDEF":
            if letter not in options:
                raise ValueError(f"{path} {item['id']} missing option {letter}")
        if item["answer"] not in set("ABCDEF"):
            raise ValueError(f"{path} {item['id']} has invalid answer letter")
        normalized.append(
            {
                "QuestionID": item["id"],
                "Question": item["question"],
                "CorrectOptionLetter": item["answer"],
                "Type": item.get("type", ""),
                "BloomLevel": item.get("bloom", ""),
                **{f"Option{letter}": options[letter] for letter in "ABCDEF"},
            }
        )
    return normalized


def option_block(question: dict[str, Any]) -> str:
    """Render the six options as ``A. ...`` lines for the prompt."""
    return "\n".join(f"{letter}. {question[f'Option{letter}']}" for letter in "ABCDEF")


def build_prompt(question: dict[str, Any], context: str = "") -> str:
    """Build the MCQ prompt: optional retrieved-context block, the question and its
    six options, and the single-letter answer instruction."""
    context_block = ""
    if context:
        context_block = (
            "Use the following retrieved document excerpts as evidence. "
            "If the excerpts are not sufficient, choose the best answer from the options.\n\n"
            f"{context}\n\n"
        )
    return (
        f"{context_block}"
        "Question:\n"
        f"{question['Question']}\n\n"
        "Options:\n"
        f"{option_block(question)}\n\n"
        "Return exactly one uppercase letter: A, B, C, D, E, or F."
    )


def format_context(chunks: list[Any]) -> str:
    """Format retrieved chunks into a numbered, id-labelled evidence block."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(f"[{index}] {chunk.chunk_id}\n{chunk.text}")
    return "\n\n".join(parts)


# A letter token: optionally wrapped in parentheses/markdown emphasis, not glued to
# another alphanumeric, and not the English article "a" (which is followed by a word).
_LETTER = r"\(?\*{0,2}([A-Fa-f])\*{0,2}\)?(?![A-Za-z0-9])(?![-/])(?=[^A-Za-z]|$)"

# "Answer: D", "the final answer is (D)", "correct option — B", "I choose C".
_DECLARATION = re.compile(
    r"\b(?:final\s+answer|answer|correct\s+(?:option|choice|answer)|option|choice|select)\b"
    r"[^A-Za-z0-9\n]{0,6}(?:is|=|:)?[^A-Za-z0-9\n]{0,6}" + _LETTER,
    re.IGNORECASE,
)

# A line whose entire content is one letter, e.g. "D", "**D**", "D.", "(D)".
_LONE_LINE = re.compile(r"^[^A-Za-z0-9\n]{0,4}([A-Fa-f])[^A-Za-z0-9\n]{0,4}$", re.MULTILINE)

# A bare capital A-F standing alone in prose. Case-sensitive on purpose: a lowercase
# "a" is the English article, never an answer. The dash guards keep district and
# ordinance codes ("R-3", "C-3", "M-1", "T-2009-003") from being read as letters.
_BARE_CAPITAL = re.compile(r"(?<![A-Za-z0-9\-/])([A-F])(?![A-Za-z0-9\-/])")

# The escalation prompt used by retries (see ``ask_until_answer``). It is defined here,
# beside the parser, because the parser has to be able to recognise it: a model that
# echoes the instruction back verbatim ends its response on "...A, B, C, D, E, or F.",
# and _BARE_CAPITAL would score that trailing F as the model's answer. Seven rows were
# decided that way -- every one of them predicted F -- before this was caught.
RETRY_PROMPT_PREFIX = (
    "CRITICAL OUTPUT RULE: Reply with exactly one character only: A, B, C, D, E, or F.\n"
    "Do not include words, punctuation, explanation, markdown, or reasoning."
)
RETRY_PROMPT_SUFFIX = "Final answer must be exactly one character only: A, B, C, D, E, or F."

# Matches either instruction sentence wherever a model repeats it, tolerating whitespace
# and emphasis drift ("**A, B, C, D, E, or F**") but not the letters themselves.
_SEP = r"[^A-Za-z0-9]{0,3}(?:or[^A-Za-z0-9]{0,3})?"
_INSTRUCTION_ECHO = re.compile(
    r"(?:CRITICAL OUTPUT RULE|Final answer must be|Reply with)[^\n]*?"
    r"A" + _SEP + r"B" + _SEP + r"C" + _SEP + r"D" + _SEP + r"E" + _SEP + r"F[^A-Za-z0-9]{0,2}",
    re.IGNORECASE,
)


def strip_instruction_echo(text: str) -> str:
    """Remove any verbatim echo of our own escalation instruction from ``text``.

    A response is evidence about the model only where it is the model's own words.
    Leaving our instruction in lets the parser score the prompt.
    """
    return _INSTRUCTION_ECHO.sub(" ", text)


_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def answer_region(raw_response: str) -> str | None:
    """The part of ``raw_response`` that can contain a final answer.

    Some reasoning models emit an inline ``<think>`` block even when ``think=False``.
    Only what follows the last ``</think>`` is the answer. If a block was opened and
    never closed the generation was cut off mid-reasoning and never reached an answer:
    that is an *unanswered* response, not a wrong one, so ``None`` is returned and the
    caller retries rather than scoring a letter scraped out of the reasoning.
    """
    closes = list(_THINK_CLOSE.finditer(raw_response))
    if closes:
        return raw_response[closes[-1].end():]
    if _THINK_OPEN.search(raw_response):
        return None
    return raw_response


def parse_answer(raw_response: str, question: dict[str, Any] | None = None) -> str:
    """Extract the model's chosen letter from free-form text, returning ``""`` if none
    can be found.

    Two rules drive the strategy. **Last wins**: a model that reasons out loud names
    several options before committing, so the final mention is the answer, not the
    first. **Case matters**: matching ``[A-F]`` case-insensitively makes the English
    article "a" look like option A, which silently rewrites verbose answers to A and
    hides the fact that nothing was answered at all.

    Strategies run from most to least explicit — a bare letter, an "answer: X"
    declaration, a letter alone on its own line, a bare capital in prose — and each
    takes its *last* occurrence. If a ``question`` is supplied there is a final fallback
    matching option *text* against the response, exact substring first, then a token
    overlap vote with a clear-margin threshold, so a model that wrote out the answer
    instead of its letter is still scored.
    """
    stripped = raw_response.strip()
    exact = re.fullmatch(r"[^A-Za-z0-9]{0,4}([A-Fa-f])[^A-Za-z0-9]{0,4}", stripped)
    if exact:
        return exact.group(1).upper()

    region = answer_region(stripped)
    if region is None:
        return ""
    region = strip_instruction_echo(region).strip()
    if not region:
        return ""

    for pattern in (_DECLARATION, _LONE_LINE, _BARE_CAPITAL):
        matches = list(pattern.finditer(region))
        if matches:
            return matches[-1].group(1).upper()

    if question:
        text = region.upper()
        stopwords = {
            "THE",
            "AND",
            "FOR",
            "WITH",
            "THAT",
            "THIS",
            "FROM",
            "WHERE",
            "WHAT",
            "WHICH",
            "WEBSITE",
            "FILES",
            "AVAILABLE",
            "VIEWING",
            "DOWNLOADING",
        }
        for letter in "ABCDEF":
            option_text = str(question.get(f"Option{letter}", "")).strip().upper()
            if option_text and option_text in text:
                return letter
        scores: list[tuple[str, int, int]] = []
        for letter in "ABCDEF":
            option_text = str(question.get(f"Option{letter}", "")).upper()
            tokens = [
                token
                for token in re.findall(r"[A-Z0-9]+", option_text)
                if len(token) >= 4 and token not in stopwords
            ]
            if not tokens:
                continue
            hits = sum(1 for token in set(tokens) if token in text)
            scores.append((letter, hits, len(set(tokens))))
        if scores:
            scores.sort(key=lambda item: item[1], reverse=True)
            best_letter, best_hits, best_total = scores[0]
            second_hits = scores[1][1] if len(scores) > 1 else 0
            if best_hits >= 2 and best_hits > second_hits and best_hits / best_total >= 0.5:
                return best_letter
    return ""


# Prefix stamped onto the ``error`` column when the inference server itself failed —
# an HTTP 5xx, an unreachable host, or a timeout — as opposed to the model returning
# text we could not parse. The distinction matters for scoring: a backend fault yields
# no model output at all (empty ``raw_response``, empty ``eval_count``), so it is not
# evidence about the model and is excluded from the accuracy denominator rather than
# scored wrong. Observed in the wild as Ollama 0.32.13 returning
# ``llama-server chat error: ... expected peg-native format`` for phi4-mini-reasoning.
BACKEND_FAULT_PREFIX = "backend_fault: "


def classify_exception(exc: BaseException) -> str:
    """Render ``exc`` for the ``error`` column, prefixing genuine backend faults.

    A 4xx is our bug (bad request) and stays unprefixed so it is not quietly excused;
    a 5xx or a request that never reached a response is the server's."""
    detail = repr(exc)
    status = getattr(exc, "status", None)
    if isinstance(exc, OllamaError) and (status is None or status >= 500):
        return f"{BACKEND_FAULT_PREFIX}{detail}"
    if isinstance(exc, (TimeoutError, ConnectionError, socket.timeout)):
        return f"{BACKEND_FAULT_PREFIX}{detail}"
    return detail


def ask_until_answer(
    client: OllamaClient,
    model: str,
    prompt: str,
    question: dict[str, Any],
    think: bool,
    max_attempts: int,
) -> tuple[dict[str, Any], str, str]:
    """Ask the model until it yields a parseable A–F answer or ``max_attempts`` is hit.

    The first attempt honours ``think``; retries force a terse single-character prompt
    and disable thinking. Returns ``(response, predicted_letter, error)`` — the letter
    is empty and ``error`` is set if every attempt failed.
    """
    last_response: dict[str, Any] = {}
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        attempt_think = think and attempt == 1
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt = (
                f"{RETRY_PROMPT_PREFIX}\n\n{prompt}\n\n{RETRY_PROMPT_SUFFIX}"
            )
        attempt_options = None
        if attempt > 1:
            # Every retry sends the same terse prompt, so a pinned seed would make each
            # one reproduce the last byte-for-byte. Offset it per attempt to draw an
            # independent sample; decoding stays deterministic *given* the attempt index,
            # so the run as a whole is still reproducible.
            base = (FINAL_RETRY_GENERATION_OPTIONS
                    if attempt >= FINAL_RETRY_FROM_ATTEMPT
                    else RETRY_GENERATION_OPTIONS)
            attempt_options = {
                **base,
                "seed": base["seed"] + (attempt - 1) * RETRY_SEED_STRIDE,
            }
        try:
            response = client.chat(
                model,
                attempt_prompt,
                think=attempt_think,
                options=attempt_options,
            )
            predicted = parse_answer(response["content"], question=question)
            response["attempts"] = attempt
            if predicted:
                return response, predicted, ""
            last_response = response
            last_error = "No parseable A-F answer returned"
        except Exception as exc:
            last_response = {"attempts": attempt}
            last_error = classify_exception(exc)
        print(
            f"Retrying blank/error answer for {model}; attempt {attempt}/{max_attempts}: {last_error}",
            flush=True,
        )
    last_response["attempts"] = max_attempts
    return last_response, "", last_error


def raw_results_path(tier: str = "medium") -> Path:
    """Path to the tier's main results CSV (creating the data dir if needed)."""
    d = results_dir_for(tier) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d / "raw_results.csv"


def failure_results_path(tier: str = "medium") -> Path:
    """Path to the tier's failed-attempts CSV (creating the data dir if needed)."""
    d = results_dir_for(tier) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d / "failed_attempts.csv"


def quarantined_keys(path: Path) -> set[tuple[str, str, str, str]]:
    """Set of (document, question, model_label, condition) keys already recorded as
    unanswerable in ``failed_attempts.csv``.

    ``--resume`` alone only skips rows scored successfully, so every resumed run
    re-attempts each permanently-failed row — at ``max_attempts`` generations apiece, for
    answers the model has already shown it will not produce. ``--skip-quarantined`` adds
    these keys to the skip set. It is opt-in: writing a row off is a judgement about the
    model, not a fact about the run."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                keys.add((row["document_id"], row["question_id"],
                          row["model_label"], row["retrieval_condition"]))
            except KeyError:
                continue
    return keys


def completed_keys(path: Path) -> set[tuple[str, str, str, str]]:
    """Set of (document, question, model_label, condition) keys already scored
    in ``path`` — drives ``--resume`` by skipping done work. Rows with an error or an
    invalid predicted letter are not counted as done, *except* rows adjudicated as
    never answered (see ``scored_unanswered``), which are a final verdict and must not
    be re-attempted: re-running them appends a second row for a key that already has
    one."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not scored_unanswered(row) and (
                    has_error(row.get("error"))
                    or not valid_letter(row.get("predicted_letter"))):
                continue
            keys.add(
                (
                    row["document_id"],
                    row["question_id"],
                    row["model_label"],
                    row["retrieval_condition"],
                )
            )
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one result row to ``path`` under an exclusive file lock (writing the
    header first if the file is new), so parallel workers can share a CSV safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0, 2)  # seek to end to check if empty
            write_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in RAW_COLUMNS})
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def append_failure(row: dict[str, Any], tier: str) -> None:
    """Append a row that never produced a valid answer to the tier's failed-attempts
    CSV, stamped with a failure timestamp (kept for diagnostics, separate from
    results)."""
    path = failure_results_path(tier)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
        if not exists:
            writer.writeheader()
        failure_row = dict(row)
        failure_row["failure_timestamp"] = datetime.now(timezone.utc).isoformat()
        writer.writerow({column: failure_row.get(column, "") for column in FAILURE_COLUMNS})


def repair_results(path: Path) -> int:
    """Rewrite ``path`` keeping only complete, valid rows (backing up the original
    first) and return how many incomplete/error rows were dropped. Used by
    ``--repair-results`` to clean a CSV before a resumed run."""
    if not path.exists():
        return 0
    backup_path = path.with_suffix(path.suffix + f".bak_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    rows: list[dict[str, str]] = []
    removed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if has_error(row.get("error")) or not valid_letter(row.get("predicted_letter")):
                removed += 1
                continue
            rows.append(row)
    path.replace(backup_path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Removed {removed} incomplete/error rows from {path}; backup at {backup_path}")
    return removed


def preflight(
    client: OllamaClient,
    auto_pull: bool,
    needed_embedding_models: set[str],
    required_models: tuple[str, ...],
    tier: str = "medium",
) -> None:
    """Fail fast before a run: verify required LLMs are installed (hard error if not),
    auto-pull any needed embedders, and confirm every document's sources, questions,
    and chunking load cleanly."""
    print(f"Using Ollama host: {client.host}")
    installed = client.list_models()
    installed_with_aliases = set(installed)
    installed_with_aliases.update(name.removesuffix(":latest") for name in installed)
    missing_required = sorted(set(required_models) - installed_with_aliases)
    if missing_required:
        raise RuntimeError(f"Missing required Ollama models: {', '.join(missing_required)}")
    for model in auto_pull_models_for(tier):
        if model in needed_embedding_models and model not in installed_with_aliases:
            if not auto_pull:
                raise RuntimeError(f"{model} is missing. Re-run without --no-auto-pull to pull it.")
            print(f"Pulling missing embedding model: {model}", flush=True)
            client.pull_model(model)
    for spec in DOCUMENTS:
        if not spec.source_path.exists():
            raise FileNotFoundError(spec.source_path)
        if not spec.questions_path.exists():
            raise FileNotFoundError(spec.questions_path)
        questions = load_questions(spec.questions_path)
        markdown_text = load_markdown(spec.document_id, spec.source_path)
        if not markdown_text.strip():
            raise RuntimeError(f"Empty markdown source: {spec.source_path}")
        chunks = build_chunks(spec.document_id, spec.source_path)
        print(
            f"{spec.document_id}: {len(questions)} questions, "
            f"{len(markdown_text.splitlines())} lines, {len(chunks)} chunks",
            flush=True,
        )


def selected_models(labels: list[str] | None, tier: str) -> list[ModelSpec]:
    """Resolve ``--model`` filters (matching either label or Ollama tag) to the tier's
    model specs; with no filter, return the whole roster. Raises on unknown names."""
    tier_specs = model_specs_for(tier)
    if not labels:
        return list(tier_specs)
    wanted = set(labels)
    models = [spec for spec in tier_specs if spec.label in wanted or spec.model in wanted]
    missing = wanted - {spec.label for spec in models} - {spec.model for spec in models}
    if missing:
        raise ValueError(f"Unknown model labels/names for tier {tier!r}: {', '.join(sorted(missing))}")
    return models


def selected_conditions(names: list[str] | None, tier: str = "medium") -> list[dict[str, object]]:
    """Resolve ``--condition`` filters to the tier's retrieval conditions; with no
    filter, return all of them. Raises on an unknown condition name."""
    conditions = retrieval_conditions(tier)
    if not names:
        return conditions
    wanted = set(names)
    selected = [item for item in conditions if str(item["retrieval_condition"]) in wanted]
    missing = wanted - {str(item["retrieval_condition"]) for item in selected}
    if missing:
        raise ValueError(f"Unknown retrieval conditions: {', '.join(sorted(missing))}")
    return selected


def expand_tiers(tier_arg: str) -> list[str]:
    """Resolve a --tier value to a list of concrete tiers. 'all' runs both."""
    if tier_arg.strip().lower() == "all":
        return ["small", "medium"]
    return [resolve_tier(tier_arg)]


def run(args: argparse.Namespace, tier: str) -> None:
    """Execute the benchmark for one tier.

    Selects this worker's share of models, runs preflight, then loops over every
    model × document × question × condition: retrieves context (precomputed cache if
    present, else an on-demand :class:`Retriever`), queries the model, scores the
    answer, and appends to the shared CSV (and a per-model CSV). Already-completed
    keys are skipped when resuming; rows that never yield a valid letter go to the
    failures CSV instead.
    """
    print(f"Running tier: {tier}")
    client = OllamaClient(host=args.ollama_host, timeout=args.timeout)
    all_models = selected_models(args.model, tier=tier)
    # Round-robin split across workers so load is balanced
    models = [m for i, m in enumerate(all_models) if i % args.total_workers == args.worker]
    if args.total_workers > 1:
        print(f"Worker {args.worker}/{args.total_workers}: running {len(models)}/{len(all_models)} models: {[m.label for m in models]}", flush=True)
    conditions = selected_conditions(args.condition, tier=tier)
    needed_embedding_models = {
        str(condition["embedding_model"]) for condition in conditions if condition["embedding_model"]
    }
    required_models = tuple({m.model for m in models} | needed_embedding_models)
    preflight(
        client,
        auto_pull=not args.no_auto_pull,
        needed_embedding_models=needed_embedding_models,
        required_models=required_models,
        tier=tier,
    )

    output_path = Path(args.output) if args.output else raw_results_path(tier)
    if args.per_model_dir.strip().lower() in ("", "auto"):
        per_model_dir = results_dir_for(tier) / "by_model"
    elif args.per_model_dir.strip().lower() in ("none", "off", "no"):
        per_model_dir = None
    else:
        per_model_dir = Path(args.per_model_dir)
    done = completed_keys(output_path) if args.resume else set()
    if args.skip_quarantined:
        quarantined = quarantined_keys(failure_results_path(tier))
        # Only rows for the models this invocation would actually run are worth reporting.
        relevant = {k for k in quarantined - done if k[2] in {m.label for m in models}}
        print(f"--skip-quarantined: skipping {len(relevant)} rows previously recorded "
              f"as unanswerable", flush=True)
        done |= quarantined
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retrievers: dict[tuple[str, str], Retriever] = {}

    total_planned = 0
    for doc_spec in DOCUMENTS:
        questions = load_questions(doc_spec.questions_path)
        question_count = len(questions[: args.limit_questions]) if args.limit_questions else len(questions)
        total_planned += question_count * len(models) * len(conditions)
    print(f"Planned rows for this invocation: {total_planned}", flush=True)

    for model_spec in models:
        print(f"Starting model {model_spec.label} ({model_spec.model})", flush=True)
        for doc_spec in DOCUMENTS:
            questions = load_questions(doc_spec.questions_path)
            if args.limit_questions:
                questions = questions[: args.limit_questions]
            for question in questions:
                query = question["Question"]
                for condition in conditions:
                    retrieval_condition = str(condition["retrieval_condition"])
                    key = (
                        doc_spec.document_id,
                        question["QuestionID"],
                        model_spec.label,
                        retrieval_condition,
                    )
                    if key in done:
                        continue

                    embedding_model = str(condition["embedding_model"])
                    top_k = int(condition["top_k"])
                    retrieved_chunks = []
                    context = ""
                    error = ""
                    try:
                        if retrieval_condition != "no_rag":
                            pre_path = retrieval_cache_path(doc_spec.document_id, embedding_model, top_k)
                            if pre_path.exists():
                                pre_key = (doc_spec.document_id, embedding_model, top_k)
                                if pre_key not in retrievers:
                                    retrievers[pre_key] = PrecomputedRetriever(
                                        doc_spec.document_id, embedding_model, top_k
                                    )
                                retrieved_chunks = retrievers[pre_key].retrieve(question["QuestionID"])
                            else:
                                retriever_key = (doc_spec.document_id, embedding_model)
                                if retriever_key not in retrievers:
                                    retrievers[retriever_key] = Retriever(
                                        doc_spec.document_id,
                                        doc_spec.source_path,
                                        embedding_model,
                                        client,
                                    )
                                retrieved_chunks = retrievers[retriever_key].retrieve(query, top_k)
                            context = format_context(retrieved_chunks)
                        prompt = build_prompt(question, context=context)
                        response, predicted, error = ask_until_answer(
                            client,
                            model_spec.model,
                            prompt,
                            question,
                            think=model_spec.thinking,
                            max_attempts=args.max_attempts,
                        )
                        raw_response = response.get("content", "")
                        latency_seconds = response.get("latency_seconds", "")
                        latency = f"{latency_seconds:.3f}" if latency_seconds != "" else ""
                        thinking = response.get("thinking", "")
                        prompt_eval_count = response.get("prompt_eval_count", "")
                        eval_count = response.get("eval_count", "")
                        fallback_used = response.get("fallback_used", False)
                        fallback_prompt_eval_count = response.get("fallback_prompt_eval_count", "")
                        fallback_eval_count = response.get("fallback_eval_count", "")
                        attempts = response.get("attempts", 1)
                    except Exception as exc:
                        raw_response = ""
                        predicted = ""
                        latency = ""
                        thinking = ""
                        prompt_eval_count = ""
                        eval_count = ""
                        fallback_used = False
                        fallback_prompt_eval_count = ""
                        fallback_eval_count = ""
                        attempts = args.max_attempts
                        error = repr(exc)

                    correct = question["CorrectOptionLetter"]
                    row = {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "document_id": doc_spec.document_id,
                        "question_id": question["QuestionID"],
                        "question_type": question.get("Type", ""),
                        "bloom_level": question.get("BloomLevel", ""),
                        "model_label": model_spec.label,
                        "model": model_spec.model,
                        "thinking_mode": "on" if model_spec.thinking else "off",
                        "retrieval_condition": retrieval_condition,
                        "embedding_model": embedding_model,
                        "top_k": top_k,
                        "predicted_letter": predicted,
                        "correct_letter": correct,
                        "is_correct": int(predicted == correct),
                        "raw_response": raw_response,
                        "thinking": thinking,
                        "retrieved_chunk_ids": "|".join(chunk.chunk_id for chunk in retrieved_chunks),
                        "retrieved_pages": "|".join(
                            f"{chunk.page_start}-{chunk.page_end}" for chunk in retrieved_chunks
                        ),
                        "latency_seconds": latency,
                        "prompt_eval_count": prompt_eval_count,
                        "eval_count": eval_count,
                        "fallback_used": int(bool(fallback_used)),
                        "fallback_prompt_eval_count": fallback_prompt_eval_count,
                        "fallback_eval_count": fallback_eval_count,
                        "attempts": attempts,
                        "error": error,
                    }
                    if error or not valid_letter(predicted):
                        append_failure(row, tier)
                        print(
                            f"SKIP {doc_spec.document_id} {question['QuestionID']} "
                            f"{model_spec.label} {retrieval_condition}: invalid after retries ({error})",
                            flush=True,
                        )
                        continue

                    append_row(output_path, row)
                    if per_model_dir:
                        append_row(per_model_dir / f"{model_spec.label}.csv", row)
                    status = "OK"
                    print(
                        f"{status} {doc_spec.document_id} {question['QuestionID']} "
                        f"{model_spec.label} {retrieval_condition}: {predicted or '-'} / {correct}",
                        flush=True,
                    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark run."""
    parser = argparse.ArgumentParser(description="Run the Ollama RAG MCQ benchmark.")
    parser.add_argument(
        "--tier",
        default=os.environ.get("BENCHMARK_TIER", DEFAULT_TIER),
        help=(
            "Model tier to run: 'small' (small-footprint, roughly 0.6B to 4B-class), "
            "'medium' (7-12B-class), or 'all' to run "
            f"both sequentially on one machine. Default: {DEFAULT_TIER}."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help=(
            "Ollama host:port. If omitted, uses the OLLAMA_HOST env var, then the "
            "config default (http://localhost:11434)."
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-auto-pull", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--repair-results",
        action="store_true",
        help="Remove incomplete/error rows from the output CSV before running; creates a backup.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--skip-quarantined",
        action="store_true",
        help="With --resume, also skip rows already in failed_attempts.csv instead of "
             "re-attempting them. Saves the retry cost of rows the model has repeatedly "
             "failed to answer; omit it to give those rows another try.",
    )
    parser.add_argument("--worker", type=int, default=0, help="Worker index (0-based) for parallel runs.")
    parser.add_argument("--total-workers", type=int, default=1, help="Total number of parallel workers.")
    parser.add_argument("--model", action="append", help="Model label or Ollama model name to include.")
    parser.add_argument(
        "--condition",
        action="append",
        help="Retrieval condition to include, e.g. no_rag or rag_qwen3-embedding:0.6b_k3.",
    )
    parser.add_argument(
        "--per-model-dir",
        default="",
        help="Directory to write one CSV per model label. Default: results/<tier>/by_model. "
             "Pass an explicit path to override; pass 'none' to disable.",
    )
    return parser.parse_args(argv)


class _TeeStream:
    """Duplicate every write to multiple underlying streams."""
    def __init__(self, *streams: Any) -> None:
        """Store the underlying streams to fan writes out to (e.g. console + log file)."""
        self.streams = streams

    def write(self, data: str) -> int:
        """Write ``data`` to every underlying stream (flushing each), ignoring per-stream
        failures, and return the number of characters written."""
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        """Flush every underlying stream, ignoring per-stream failures."""
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        """Report whether the primary (first) stream is a terminal."""
        primary = self.streams[0] if self.streams else None
        try:
            return bool(primary and primary.isatty())
        except Exception:
            return False


def _start_logging(tier: str) -> Path:
    """Append stdout/stderr to a single persistent log per tier."""
    from veltharrow.config import results_dir_for
    log_dir = results_dir_for(tier) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "benchmark.log"
    log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, log_fh)
    sys.stderr = _TeeStream(sys.__stderr__, log_fh)
    print(f"\n{'='*60}", flush=True)
    print(f"Session started: {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"Tier: {tier} | Logging to {log_path}", flush=True)
    return log_path


if __name__ == "__main__":
    _parsed = parse_args(sys.argv[1:])
    _tiers = expand_tiers(_parsed.tier)
    _multi = len(_tiers) > 1
    try:
        for _tier in _tiers:
            _start_logging(tier=_tier)
            try:
                if _parsed.repair_results:
                    repair_results(Path(_parsed.output) if _parsed.output else raw_results_path(_tier))
                run(_parsed, _tier)
            except ValueError as exc:
                if _multi:
                    print(f"Skipping tier {_tier}: {exc}", flush=True)
                    continue
                raise
    except KeyboardInterrupt:
        raise SystemExit(130)
