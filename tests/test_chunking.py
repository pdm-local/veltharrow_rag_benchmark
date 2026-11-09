from veltharrow.config import CHUNK_WORDS, DOCUMENTS
from veltharrow.rag import build_chunks


def test_build_chunks_invariants():
    spec = DOCUMENTS[0]
    chunks = build_chunks(spec.document_id, spec.source_path)
    assert chunks, "expected at least one chunk"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
    # Chunks may exceed CHUNK_WORDS slightly because section boundaries are
    # preserved, but should stay within a generous slack (3x is well clear of
    # any reasonable section length we care about).
    for c in chunks:
        word_count = len(c.text.split())
        assert word_count <= CHUNK_WORDS * 3, (c.chunk_id, word_count)
        assert c.document_id == spec.document_id
