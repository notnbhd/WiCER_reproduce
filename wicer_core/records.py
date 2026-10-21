"""Result layout and per-question records shared by every arm.

    results/<profile>/<topic>/<condition>/
        answers.jsonl   one row per question: text + timing
        scores.jsonl    one row per question: judge score
        run_meta.json   provenance
        context.txt     the exact system prompt the reader saw (FC/wiki arms)

Conditions are the eight rows of spec.md Section 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterator

from wicer_core.config import Profile, resolve

CONDITIONS = (
    "fc_raw",
    "fc_raw_q4kv",
    "rag_top5",
    "wiki_light",
    "wiki_moderate",
    "wiki_aggressive",
    "wicer_iter1",
    "wicer_iter2",
    # max_iterations=3 lets the loop compile a third pinned wiki, and
    # --eval-full scores it under its own name when it is the best pinned.
    "wicer_iter3",
    "random_blind",
    "random_pin",
)


def run_dir(profile: Profile, topic: str, condition: str) -> Path:
    return resolve(profile.results_dir) / topic / condition


def answers_path(profile: Profile, topic: str, condition: str) -> Path:
    return run_dir(profile, topic, condition) / "answers.jsonl"


def scores_path(profile: Profile, topic: str, condition: str) -> Path:
    return run_dir(profile, topic, condition) / "scores.jsonl"


def skip_path(profile: Profile, topic: str, condition: str) -> Path:
    return run_dir(profile, topic, condition) / "skipped.json"


@dataclass
class AnswerRecord:
    question_id: str
    document_id: str
    question: str
    gold: str
    generated: str
    ttft_s: float
    total_s: float
    throughput_tok_s: float
    n_generated_tokens: int
    n_prompt_tokens: int | None = None
    cache_n: int | None = None
    finish_reason: str | None = None
    # RAG only
    retrieval_hit: bool | None = None
    retrieval_rank: int | None = None
    retrieved_document_ids: list[str] = field(default_factory=list)
    retrieved_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreRecord:
    question_id: str
    document_id: str
    score: int
    reasoning: str
    judge_model: str
    parse_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlWriter:
    """Append-only JSONL writer that flushes every row, so a killed run keeps data."""

    def __init__(self, path: str | Path, *, resume: bool = False):
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a" if resume else "w")

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = resolve(path)
    if not p.exists():
        return []
    with p.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = resolve(path)
    if not p.exists():
        return
    with p.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def completed_question_ids(path: str | Path) -> set[str]:
    """For --resume: which questions already have a row."""
    return {r["question_id"] for r in read_jsonl(path) if "question_id" in r}


def write_skip(profile: Profile, topic: str, condition: str, reason: str, **detail: Any) -> Path:
    p = skip_path(profile, topic, condition)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"skipped": reason, **detail}, indent=2) + "\n")
    return p


# -- compiled wikis -----------------------------------------------------------


def wiki_dir(profile: Profile, topic: str) -> Path:
    return resolve(profile.results_dir) / topic / "wikis"


def wiki_paths(profile: Profile, topic: str, name: str) -> tuple[Path, Path]:
    """(markdown, metadata) for a compiled wiki named e.g. 'blind_moderate'."""
    d = wiki_dir(profile, topic)
    return d / f"{name}.md", d / f"{name}.meta.json"


def load_wiki_metas(profile: Profile, topic: str) -> dict[str, dict[str, Any]]:
    d = wiki_dir(profile, topic)
    if not d.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(d.glob("*.meta.json")):
        try:
            out[p.name[: -len(".meta.json")]] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return out


def ledger_path(profile: Profile, topic: str, name: str = "wicer") -> Path:
    return resolve(profile.results_dir) / topic / f"{name}_ledger.json"


def load_ledger(profile: Profile, topic: str, name: str = "wicer") -> dict[str, Any] | None:
    p = ledger_path(profile, topic, name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
