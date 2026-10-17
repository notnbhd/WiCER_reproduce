"""Chunk size, top-k and embedding model for the RAG baseline.

Paper Section 3.4: 1,800-character chunks with 360-character overlap,
all-MiniLM-L6-v2 embeddings, top-5 by cosine similarity, no reranking, and the
same chat template as full-context.
"""

from wicer_core.config import (  # noqa: F401
    PROFILES,
    TOPICS,
    Profile,
    RagConfig,
    add_profile_arg,
    get_profile,
    resolve,
)

__all__ = ["PROFILES", "TOPICS", "Profile", "RagConfig", "add_profile_arg", "get_profile", "resolve"]
