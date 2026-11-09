"""Retrieval: markdown chunking, embedding, and top-k lookup.

This module turns each source document into a list of :class:`Chunk`s (splitting on
markdown headings while keeping tables intact), embeds them once and caches the vectors
under ``cache/``, and serves nearest-neighbour retrieval by cosine similarity. Two
retrievers are provided: :class:`Retriever`, which embeds on demand, and
:class:`PrecomputedRetriever`, which replays a stored retrieval cache so MCQ inference
runs with no embedder resident in VRAM. Everything here is stdlib + numpy.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from veltharrow.config import CACHE_DIR, CHUNK_OVERLAP_WORDS, CHUNK_WORDS, EMBED_BATCH_SIZE
from veltharrow.ollama_client import OllamaClient


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage. ``chunk_id`` is stable and deterministic (e.g.
    ``charter_chunk_00001``); ``page_start``/``page_end`` carry the chunk's ordinal
    position (this corpus has no real pages). ``text`` is the chunk body."""

    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    text: str


def load_markdown(document_id: str, source_path: Path) -> str:
    """Read a document's markdown, mirror it into ``cache/text/`` for reference, and
    return the raw text."""
    text_path = CACHE_DIR / "text" / f"{document_id}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    raw = source_path.read_text(encoding="utf-8", errors="replace")
    text_path.write_text(raw, encoding="utf-8")
    return raw


def _split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections. Preserves tables intact."""
    lines = markdown_text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []
    heading_re = re.compile(r"^#{1,6}\s+\S")
    for line in lines:
        if heading_re.match(line):
            if current_heading or current_body:
                sections.append((current_heading, current_body))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading or current_body:
        sections.append((current_heading, current_body))
    return [(h, "\n".join(body).strip()) for h, body in sections]


def _is_table_line(line: str) -> bool:
    """True if ``line`` looks like a markdown table row (pipe-delimited)."""
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _split_block_preserving_tables(body: str) -> list[str]:
    """Break a section body into atomic blocks: paragraphs and full tables.
    Tables are kept as a single block so they're never split mid-row."""
    if not body.strip():
        return []
    lines = body.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    in_table = False
    for line in lines:
        table_line = _is_table_line(line)
        if table_line and not in_table:
            if current:
                blocks.append(current)
                current = []
            in_table = True
            current.append(line)
        elif table_line and in_table:
            current.append(line)
        elif not table_line and in_table:
            blocks.append(current)
            current = []
            in_table = False
            if line.strip():
                current.append(line)
        else:
            if not line.strip():
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(block).strip() for block in blocks if any(l.strip() for l in block)]


def _word_count(text: str) -> int:
    """Count whitespace-delimited tokens (the unit CHUNK_WORDS is measured in)."""
    return len(re.findall(r"\S+", text))


def _chunk_markdown(document_id: str, markdown_text: str) -> list[Chunk]:
    """Pack the document's heading sections into ~``CHUNK_WORDS``-sized chunks.

    Blocks (paragraphs and whole tables) are accumulated until the next one would
    exceed the target size, then flushed as a chunk; a ``CHUNK_OVERLAP_WORDS`` tail
    carries over to the next chunk for context continuity. The active heading is
    prepended so each chunk is self-describing. A single oversized block (e.g. a big
    table) becomes its own chunk.
    """
    sections = _split_into_sections(markdown_text)
    flat_blocks: list[tuple[str, str]] = []  # (heading_context, block_text)
    for heading, body in sections:
        for block in _split_block_preserving_tables(body):
            flat_blocks.append((heading, block))
    chunks: list[Chunk] = []
    buf_blocks: list[tuple[str, str]] = []
    buf_words = 0

    def flush(buffer: list[tuple[str, str]]) -> None:
        """Emit the buffered blocks as one Chunk, prepending each distinct heading so
        the chunk text is self-describing; no-op on an empty/blank buffer."""
        if not buffer:
            return
        headings_seen: list[str] = []
        for heading, _block in buffer:
            if heading and (not headings_seen or headings_seen[-1] != heading):
                headings_seen.append(heading)
        parts: list[str] = []
        last_heading = None
        for heading, block in buffer:
            if heading and heading != last_heading:
                parts.append(heading)
                last_heading = heading
            parts.append(block)
        text = "\n\n".join(parts).strip()
        if not text:
            return
        chunk_no = len(chunks) + 1
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}_chunk_{chunk_no:05d}",
                document_id=document_id,
                page_start=chunk_no,
                page_end=chunk_no,
                text=text,
            )
        )

    overlap_target = CHUNK_OVERLAP_WORDS
    for heading, block in flat_blocks:
        block_words = _word_count(block)
        if buf_words + block_words > CHUNK_WORDS and buf_blocks:
            flush(buf_blocks)
            tail: list[tuple[str, str]] = []
            tail_words = 0
            for item in reversed(buf_blocks):
                w = _word_count(item[1])
                if tail_words + w > overlap_target and tail:
                    break
                tail.insert(0, item)
                tail_words += w
            buf_blocks = tail
            buf_words = tail_words
        buf_blocks.append((heading, block))
        buf_words += block_words
        if block_words >= CHUNK_WORDS:
            flush(buf_blocks)
            buf_blocks = []
            buf_words = 0
    flush(buf_blocks)
    return chunks


def build_chunks(document_id: str, source_path: Path) -> list[Chunk]:
    """Return the document's chunks, reading the cached JSONL if present and
    otherwise chunking the source and writing the cache. Chunking is deterministic,
    so the cache is safe to reuse across runs."""
    chunks_path = CACHE_DIR / "chunks" / f"{document_id}.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    if chunks_path.exists():
        chunks = []
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
        return chunks

    markdown_text = load_markdown(document_id, source_path)
    chunks = _chunk_markdown(document_id, markdown_text)
    if not chunks:
        raise RuntimeError(f"No chunks produced for {source_path}")

    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return chunks


def _embedding_cache_path(document_id: str, embedding_model: str) -> Path:
    """Cache path for a document's embedding matrix, with the model tag sanitized
    into a filesystem-safe filename."""
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", embedding_model)
    return CACHE_DIR / "embeddings" / f"{document_id}__{safe_model}.npz"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot products become cosine similarities (zero rows
    are left as-is rather than dividing by zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


class Retriever:
    """On-demand retriever: chunks a document, embeds it (caching the vectors), and
    returns the top-k chunks for a query by cosine similarity. Use this to *build*
    retrievals; :class:`PrecomputedRetriever` replays them without an embedder."""

    def __init__(
        self,
        document_id: str,
        source_path: Path,
        embedding_model: str,
        client: OllamaClient,
        keep_alive: int = 0,
    ) -> None:
        """Build (or load cached) chunks and embeddings for ``document_id`` so the
        retriever is ready to serve queries."""
        self.document_id = document_id
        self.embedding_model = embedding_model
        self.client = client
        self.keep_alive = keep_alive
        self.chunks = build_chunks(document_id, source_path)
        self.embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self) -> np.ndarray:
        """Return the normalized embedding matrix for this document's chunks, using
        the on-disk cache when its chunk ids still match, otherwise embedding in
        batches and rewriting the cache."""
        cache_path = _embedding_cache_path(self.document_id, self.embedding_model)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            chunk_ids = data["chunk_ids"].tolist()
            expected = [chunk.chunk_id for chunk in self.chunks]
            if chunk_ids == expected:
                return data["embeddings"]

        vectors: list[list[float]] = []
        texts = [chunk.text for chunk in self.chunks]
        total_batches = math.ceil(len(texts) / EMBED_BATCH_SIZE)
        for batch_no, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            print(
                f"Embedding {self.document_id} with {self.embedding_model}: "
                f"batch {batch_no}/{total_batches}",
                flush=True,
            )
            vectors.extend(self.client.embed(self.embedding_model, batch, keep_alive=self.keep_alive))

        embeddings = _normalize(np.array(vectors, dtype=np.float32))
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            chunk_ids=np.array([chunk.chunk_id for chunk in self.chunks]),
        )
        return embeddings

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """Embed ``query`` and return the ``top_k`` most cosine-similar chunks, best
        first."""
        query_vector = np.array(self.client.embed(self.embedding_model, [query], keep_alive=self.keep_alive)[0], dtype=np.float32)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = self.embeddings @ query_vector
        top_indices = np.argsort(-scores)[:top_k]
        return [self.chunks[int(index)] for index in top_indices]


def retrieval_cache_path(document_id: str, embedding_model: str, top_k: int) -> Path:
    """Cache path for a precomputed (document, embedder, k) retrieval, keyed by a
    filesystem-safe model tag."""
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", embedding_model)
    return CACHE_DIR / "retrievals" / f"{document_id}__{safe_model}__k{top_k}.json"


class PrecomputedRetriever:
    """Serves pre-computed top-k chunks with no embedder in VRAM.

    Loads the retrieval cache written by ``precompute_retrievals.py`` and maps each
    question id to its stored chunks, so MCQ inference never has to run the embedder.
    Raises ``FileNotFoundError`` if the cache is missing.
    """

    def __init__(self, document_id: str, embedding_model: str, top_k: int) -> None:
        """Load the (document, embedder, k) retrieval cache and index it by question id."""
        path = retrieval_cache_path(document_id, embedding_model, top_k)
        if not path.exists():
            raise FileNotFoundError(
                f"Retrieval cache missing: {path}. Run precompute_retrievals.py first."
            )
        raw: dict[str, list[dict]] = json.loads(path.read_text(encoding="utf-8"))
        self._data: dict[str, list[Chunk]] = {
            qid: [Chunk(**c) for c in chunks] for qid, chunks in raw.items()
        }

    def retrieve(self, question_id: str) -> list[Chunk]:
        """Return the precomputed chunks for ``question_id`` (empty list if none)."""
        return self._data.get(question_id, [])
