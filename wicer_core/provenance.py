"""run_meta.json -- what was actually run, so a result can be trusted later.

spec.md Section 9 names judge drift and hardware differences as the two risks
that silently invalidate a comparison.  Both are recorded here on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wicer_core.config import Profile, resolve


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str | None:
    p = resolve(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(resolve(".")),
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_run_meta(
    profile: Profile,
    *,
    command: str,
    extra: dict[str, Any] | None = None,
    server_props: dict[str, Any] | None = None,
    hash_model: bool = True,
) -> dict[str, Any]:
    judge_backend = os.environ.get("WICER_JUDGE_BACKEND", "anthropic")
    compiler_backend = os.environ.get("WICER_COMPILER_BACKEND", "anthropic")
    meta: dict[str, Any] = {
        "command": command,
        "argv": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.to_dict(),
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "reader": {
            "base_url": profile.reader.base_url,
            "model_path": profile.reader.model_path,
            # Hashing a 5 GB GGUF is slow; skip it where the caller says so.
            "model_sha256": file_sha256(profile.reader.model_path) if hash_model else None,
            "n_ctx": profile.reader.n_ctx,
            "cache_type_k": profile.reader.cache_type_k,
            "cache_type_v": profile.reader.cache_type_v,
            "seed": profile.reader.seed,
            "temperature": profile.reader.temperature,
            "server_props": server_props,
        },
        "judge_model_configured": profile.judge.model,
        "compiler_model_configured": profile.compiler.model,
        # Which service answered, and the id it was actually handed.  Under the
        # openrouter backend an Anthropic-native id is rewritten to a slug, and
        # a run_meta naming the id that was never sent would be worse than none.
        "judge_backend": judge_backend,
        "compiler_backend": compiler_backend,
        "judge_model_resolved": _resolved(profile.judge.model, judge_backend),
        "compiler_model_resolved": _resolved(profile.compiler.model, compiler_backend),
    }
    if extra:
        meta.update(extra)
    return meta


def _resolved(model: str, backend: str) -> str:
    """The model id this backend will really see."""
    if backend != "openrouter":
        return model
    from wicer_core.config import resolve_openrouter_model

    return resolve_openrouter_model(model)


def write_run_meta(path: str | Path, meta: dict[str, Any]) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "finished_at": datetime.now(timezone.utc).isoformat()}
    p.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return p
