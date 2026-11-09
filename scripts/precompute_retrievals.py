"""
Pre-compute and cache all top-k retrievals for every (document, embedding_model, k, question).

Runs all embedding models IN PARALLEL: each embedder gets its own thread.
After this completes, run_benchmark.py never loads an embedding model during LLM inference.

Usage:
    python3 scripts/precompute_retrievals.py [--tier small|medium|all] [--ollama-host HOST]

Output:
    cache/retrievals/{document}__{embedder}__k{k}.json
    results/benchmark.log (appended)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veltharrow.config import (
    CACHE_DIR,
    DEFAULT_TIER,
    DOCUMENTS,
    RESULTS_DIR,
    embedding_models_for,
    resolve_tier,
    top_k_values_for,
)
from veltharrow.ollama_client import OllamaClient
from veltharrow.rag import Retriever, build_chunks, retrieval_cache_path


def _tier_settings(tier: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Embedders and k values for a tier. 'all' unions both tiers (deduped)."""
    tiers = ["small", "medium"] if tier.strip().lower() == "all" else [resolve_tier(tier)]
    embedders: list[str] = []
    ks: list[int] = []
    for t in tiers:
        for e in embedding_models_for(t):
            if e not in embedders:
                embedders.append(e)
        for k in top_k_values_for(t):
            if k not in ks:
                ks.append(k)
    return tuple(embedders), tuple(ks)


# ---------------------------------------------------------------------------
# Logging: single file + stdout, shared across all threads
# ---------------------------------------------------------------------------

def setup_logging() -> Path:
    """Configure logging to both ``results/benchmark.log`` and stdout (shared across
    all embedder threads) and return the log file path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "benchmark.log"

    fmt = "%(asctime)s [%(threadName)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, "a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


# ---------------------------------------------------------------------------
# Per-embedder worker: runs in its own thread
# ---------------------------------------------------------------------------

def precompute_embedder(
    embedding_model: str,
    client: OllamaClient,
    top_k_values: tuple[int, ...],
) -> str:
    """Precompute and cache top-k retrievals for one embedder across all documents.

    Runs in its own thread. Embeds each document's chunks once (model kept resident),
    then for every k writes a ``{question_id: [chunk, ...]}`` JSON cache, skipping any
    (document, embedder, k) already cached. Evicts the embedder from VRAM when done and
    returns the embedder tag.
    """
    log = logging.getLogger()
    out_dir = CACHE_DIR / "retrievals"
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc_spec in DOCUMENTS:
        # Build / load chunk embeddings: keep model alive during this phase
        retriever = Retriever(
            doc_spec.document_id,
            doc_spec.source_path,
            embedding_model,
            client,
            keep_alive=3600,  # stay loaded across all docs in this thread
        )
        questions = json.loads(doc_spec.questions_path.read_text(encoding="utf-8"))

        for top_k in top_k_values:
            cache_path = retrieval_cache_path(doc_spec.document_id, embedding_model, top_k)
            if cache_path.exists():
                log.info("SKIP  %s | %s | k=%d (already cached)", doc_spec.document_id, embedding_model, top_k)
                continue

            log.info("START %s | %s | k=%d (%d questions)", doc_spec.document_id, embedding_model, top_k, len(questions))
            result: dict[str, list[dict]] = {}
            for q in questions:
                qid = q.get("QuestionID") or q.get("id")
                query = q.get("Question") or q.get("question")
                chunks = retriever.retrieve(query, top_k)
                result[qid] = [
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "text": c.text,
                    }
                    for c in chunks
                ]
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("DONE  %s | %s | k=%d -> %s", doc_spec.document_id, embedding_model, top_k, cache_path.name)

    # Evict this embedder from VRAM when done
    try:
        client._post("/api/generate", {"model": embedding_model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass

    return embedding_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Precompute all retrievals for the selected tier, running one thread per embedder
    and logging a summary of cache files produced."""
    parser = argparse.ArgumentParser(description="Pre-compute RAG retrievals for all embedders in parallel.")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--tier", default=DEFAULT_TIER, help="small | medium | all")
    args = parser.parse_args()

    embedding_models, top_k_values = _tier_settings(args.tier)

    log_path = setup_logging()
    log = logging.getLogger()

    client = OllamaClient(host=args.ollama_host)

    log.info("=" * 60)
    log.info("Precompute RAG retrievals (tier=%s)", args.tier)
    log.info("Ollama: %s", client.host)
    log.info("Embedders (%d): %s", len(embedding_models), ", ".join(embedding_models))
    log.info("k values: %s", top_k_values)
    log.info("Documents: %s", [d.document_id for d in DOCUMENTS])
    log.info("Output dir: %s", CACHE_DIR / "retrievals")
    log.info("Log: %s", log_path)
    log.info("=" * 60)

    t0 = datetime.now(timezone.utc)

    # Run one thread per embedder: all in parallel
    with ThreadPoolExecutor(max_workers=len(embedding_models), thread_name_prefix="emb") as executor:
        futures = {
            executor.submit(precompute_embedder, emb, client, top_k_values): emb
            for emb in embedding_models
        }
        for future in as_completed(futures):
            emb = futures[future]
            try:
                future.result()
                log.info("FINISHED embedder: %s", emb)
            except Exception as exc:
                log.error("FAILED embedder %s: %s", emb, exc)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    total_files = len(list((CACHE_DIR / "retrievals").glob("*.json")))
    log.info("=" * 60)
    log.info("All done in %.1fs. Cache files: %d", elapsed, total_files)
    log.info("Run benchmark.py next: no embedder will load during LLM inference.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
