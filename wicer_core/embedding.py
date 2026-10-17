"""Sentence embeddings for the RAG baseline.

Paper Section 3.4 uses ``all-MiniLM-L6-v2`` (Reimers & Gurevych, 2019).  The
``fake`` backend is a deterministic hashing embedder used only by the offline
test harness so the pipeline can be exercised without pulling in torch.
"""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np


class Embedder:
    def __init__(self, model_name: str, backend: str | None = None, dim: int = 384):
        self.model_name = model_name
        self.backend = backend or os.environ.get("WICER_EMBED_BACKEND", "sentence-transformers")
        self.dim = dim
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        if self.backend == "fake":
            return _hash_encode(texts, self.dim)
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def describe(self) -> dict[str, object]:
        return {"embed_model": self.model_name, "backend": self.backend, "dim": self.dim}


def _hash_encode(texts: Sequence[str], dim: int) -> np.ndarray:
    """Bag-of-words hashing into a fixed-dim unit vector.

    Not a semantic embedder -- it only needs to be deterministic and to rank
    lexically-overlapping chunks above unrelated ones, which is enough to
    exercise the retrieval path end to end.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split():
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by cosine similarity.  Both inputs are expected L2-normalised.

    Brute force: a topic has ~1-2K chunks, so an index structure would add a
    dependency and a failure mode for no measurable gain.
    """
    sims = matrix @ query.astype(np.float32)
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]
