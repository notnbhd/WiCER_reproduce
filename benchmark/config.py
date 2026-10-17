"""Paths, model and server parameters for the full-context arm.

Kept as the paper's Appendix C entry point; the actual definitions live in
``wicer_core.config`` so the RAG and WiCER arms share one source of truth.
"""

from wicer_core.config import (  # noqa: F401
    PROFILES,
    TOPICS,
    Profile,
    ReaderConfig,
    add_profile_arg,
    get_profile,
    resolve,
)

__all__ = [
    "PROFILES",
    "TOPICS",
    "Profile",
    "ReaderConfig",
    "add_profile_arg",
    "get_profile",
    "resolve",
]
