"""RAG baseline: chunking geometry, retrieval ranking, end-to-end accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from wicer_core.corpus import Document
from wicer_core.embedding import Embedder, cosine_top_k
from rag_benchmark.build_index import build_index, chunk_document, load_index
from rag_benchmark.run_rag_benchmark import format_retrieved, run_topic


def test_chunking_matches_the_papers_geometry():
    # Paper Section 3.4: 1,800-character chunks with 360-character overlap.
    doc = Document("d", "x" * 5_000)
    spans = chunk_document(doc, 1_800, 360)
    assert spans[0][0] == 0 and spans[0][1] == 1_800
    assert spans[1][0] == 1_800 - 360          # stride = size - overlap
    assert all(end - start <= 1_800 for start, end, _ in spans)
    assert spans[-1][1] == 5_000               # clipped, never past the end


def test_chunking_covers_every_character():
    doc = Document("d", "abcdefghij" * 50)
    spans = chunk_document(doc, 120, 30)
    covered = set()
    for start, end, _ in spans:
        covered.update(range(start, end))
    assert covered == set(range(len(doc.text)))


def test_short_documents_yield_one_chunk_and_empty_yield_none():
    assert len(chunk_document(Document("d", "tiny"), 1_800, 360)) == 1
    assert chunk_document(Document("d", ""), 1_800, 360) == []


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_document(Document("d", "x" * 100), 100, 100)


def test_fake_embedder_is_deterministic_and_normalised():
    e = Embedder("fake-model", backend="fake", dim=64)
    a = e.encode(["the kiln festival", "the kiln festival"])
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_cosine_top_k_returns_descending_similarity():
    matrix = np.eye(5, dtype=np.float32)
    query = np.array([0.1, 0.9, 0.0, 0.5, 0.0], dtype=np.float32)
    idx, sims = cosine_top_k(query, matrix, 3)
    assert list(idx) == [1, 3, 0]
    assert list(sims) == sorted(sims, reverse=True)


def test_cosine_top_k_clamps_k_to_corpus_size():
    matrix = np.eye(2, dtype=np.float32)
    idx, _ = cosine_top_k(np.array([1.0, 0.0], dtype=np.float32), matrix, 10)
    assert len(idx) == 2


def test_index_records_chunk_provenance(profile, topic):
    meta = build_index(profile, topic)
    vectors, chunks, loaded = load_index(profile, topic)
    assert meta == loaded
    assert vectors.shape[0] == len(chunks) == meta["n_chunks"]
    assert meta["n_documents"] == 12
    assert meta["reranking"] is False
    # Every chunk traces to exactly one source document -- required for the
    # retrieval-accuracy metric to be well defined.
    assert all(c["document_id"].startswith("mini") for c in chunks)


def test_retrieved_context_labels_sources(profile, topic):
    build_index(profile, topic)
    _, chunks, _ = load_index(profile, topic)
    rendered = format_retrieved(chunks[:3])
    assert rendered.count("=== PASSAGE") == 3
    assert f"source: {chunks[0]['document_id']}" in rendered


def test_rag_run_records_retrieval_hits_and_ranks(profile, topic):
    build_index(profile, topic)
    summary = run_topic(profile, topic, limit=8, progress=False)
    assert summary["n_generated"] == 8
    assert 0.0 <= summary["retrieval_accuracy_pct"] <= 100.0

    from wicer_core.records import answers_path, read_jsonl

    rows = read_jsonl(answers_path(profile, topic, "rag_top5"))
    assert len(rows) == 8
    for r in rows:
        assert len(r["retrieved_document_ids"]) == profile.rag.top_k
        if r["retrieval_hit"]:
            assert 1 <= r["retrieval_rank"] <= profile.rag.top_k
            assert r["retrieved_document_ids"][r["retrieval_rank"] - 1] == r["document_id"]
        else:
            assert r["retrieval_rank"] is None
