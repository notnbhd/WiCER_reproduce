"""Minimal `.env` support.

The repo keeps its dependency list short, and the only thing we need from
`python-dotenv` is twelve lines of parsing.  The one rule that matters:

    the shell always wins.

A variable already present in the environment is never overwritten, so an
exported ``WICER_PROFILE`` or a key pasted into the shell for one run cannot be
silently clobbered by a stale file on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

DOTENV_NAME = ".env"


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines.  Comments, blanks and junk are skipped."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Load ``path`` into ``os.environ`` and return only what it actually set.

    Never raises: a malformed or unreadable file leaves the environment alone.
    This runs at import time, so a typo in `.env` must not abort a run.
    """
    path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / DOTENV_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}

    applied: dict[str, str] = {}
    for key, value in parse_dotenv(text).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
