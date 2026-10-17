#!/usr/bin/env python3
"""Chunk, embed and persist a per-topic retrieval index.

Paper Section 3.4 / Appendix A.  Chunks never span documents, so every chunk
has an unambiguous source ``document_id`` -- which is what makes the
retrieval-accuracy metric (is the gold document in the top-5?) well defined.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve
from wicer_core.corpus import Document, load_topic
from wicer_core.embedding import Embedder


@dataclass
class Chunk:
    chunk_id: int
    document_id: str
    start: int
    end: int
    text: str


def chunk_document(doc: Document, chunk_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    """Fixed-size character chunks with overlap, clipped to the document."""
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk size")
    text = doc.text
    if not text:
        return []
    stride = chunk_chars - overlap_chars
    spans: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        spans.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += stride
    return spans


def build_chunks(profile: Profile, topic: str) -> list[Chunk]:
    corpus = load_topic(profile, topic)
    cfg = profile.rag
    chunks: list[Chunk] = []
    for doc in corpus.documents:
        for start, end, text in chunk_document(doc, cfg.chunk_chars, cfg.overlap_chars):
            chunks.append(Chunk(len(chunks), doc.document_id, start, end, text))
    return chunks


def index_dir(profile: Profile, topic: str) -> Path:
    return resolve(profile.rag.index_dir) / topic


def build_index(profile: Profile, topic: str, *, embedder: Embedder | None = None) -> dict:
    chunks = build_chunks(profile, topic)
    embedder = embedder or Embedder(profile.rag.embed_model)
    vectors = embedder.encode([c.text for c in chunks])

    out = index_dir(profile, topic)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors)
    with (out / "chunks.jsonl").open("w") as fh:
        for c in chunks:
            fh.write(json.dumps({
                "chunk_id": c.chunk_id, "document_id": c.document_id,
                "start": c.start, "end": c.end, "text": c.text,
            }, ensure_ascii=False) + "\n")

    meta = {
        "topic": topic,
        "n_chunks": len(chunks),
        "n_documents": len({c.document_id for c in chunks}),
        "chunk_chars": profile.rag.chunk_chars,
        "overlap_chars": profile.rag.overlap_chars,
        "top_k": profile.rag.top_k,
        "reranking": False,
        **embedder.describe(),
    }
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_index(profile: Profile, topic: str) -> tuple[np.ndarray, list[dict], dict]:
    out = index_dir(profile, topic)
    meta_path = out / "index_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no RAG index for {topic!r}; run rag_benchmark/build_index.py --topic {topic}"
        )
    vectors = np.load(out / "embeddings.npy")
    with (out / "chunks.jsonl").open() as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]
    return vectors, chunks, json.loads(meta_path.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    embedder = Embedder(profile.rag.embed_model)
    for topic in args.topic or list(TOPICS):
        meta = build_index(profile, topic, embedder=embedder)
        print(
            f"{topic:<26} chunks={meta['n_chunks']:>5} docs={meta['n_documents']:>3} "
            f"dim={meta['dim']} backend={meta['backend']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
