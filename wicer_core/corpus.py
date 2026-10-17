"""RepLiQA corpus construction.

Paper Section 3.1 uses "17 topic domains with 80 documents and 400 QA pairs
per topic".  The published dataset is larger than that (split ``repliqa_0``
has 17,955 rows over the same 17 topics), so the 80/400 figure is a
*subsample* and the paper does not say which one.

Our rule -- deterministic, recorded in every manifest, and listed in
DEVIATIONS.md:

  1. Restrict to one split (``repliqa_0`` by default).
  2. Drop rows whose answer is marked unanswerable; count them.
  3. Sort the surviving ``document_id`` values lexicographically, keep the
     first ``n_docs_per_topic``.
  4. Within each document, sort rows by ``question_id`` and keep the first
     ``n_qa_per_doc``.

Steps 3 and 4 are pure functions of the dataset contents, so two machines
building the same split get byte-identical corpora.  The manifest carries a
content hash that proves it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve

UNANSWERABLE_MARKERS = {"unanswerable", "n/a", ""}


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class QAPair:
    question_id: str
    document_id: str
    question: str
    answer: str
    long_answer: str = ""


@dataclass(frozen=True)
class Corpus:
    topic: str
    documents: list[Document]
    qa: list[QAPair]
    manifest: dict

    def doc_by_id(self, document_id: str) -> Document:
        for d in self.documents:
            if d.document_id == document_id:
                return d
        raise KeyError(f"no document {document_id!r} in topic {self.topic!r}")

    @property
    def n_words(self) -> int:
        return sum(d.n_words for d in self.documents)


def format_documents(documents: list[Document]) -> str:
    """Render documents into the single blob the reader and compiler both see.

    Used for the full-context system prompt *and* as the denominator of the
    compression ratio, so both sides measure the same thing.
    """
    parts = [
        f"=== DOCUMENT {i + 1} (id: {d.document_id}) ===\n{d.text.strip()}"
        for i, d in enumerate(documents)
    ]
    return "\n\n".join(parts)


def _is_unanswerable(answer: str) -> bool:
    return answer.strip().lower() in UNANSWERABLE_MARKERS


def _content_hash(documents: list[Document], qa: list[QAPair]) -> str:
    h = hashlib.sha256()
    for d in documents:
        h.update(d.document_id.encode())
        h.update(b"\0")
        h.update(d.text.encode())
        h.update(b"\0")
    for q in qa:
        h.update(q.question_id.encode())
        h.update(b"\0")
        h.update(q.question.encode())
        h.update(b"\0")
        h.update(q.answer.encode())
        h.update(b"\0")
    return h.hexdigest()


def topic_dir(profile: Profile, topic: str) -> Path:
    return resolve(profile.corpus.data_dir) / topic


def _split_files(profile: Profile) -> list[str]:
    """Parquet shards for a split, discovered rather than assumed.

    RepLiQA currently ships one shard per split as
    ``data/repliqa_0-00000-of-00001.parquet``; the shard count is a publishing
    detail that can change, so match on the prefix.
    """
    from huggingface_hub import list_repo_files

    cfg = profile.corpus
    prefix = f"data/{cfg.split}-"
    files = sorted(
        f
        for f in list_repo_files(cfg.hf_repo, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(
            f"no parquet shards for split {cfg.split!r} in {cfg.hf_repo!r}"
        )
    return files


def _download_split(profile: Profile) -> "pandas.DataFrame":  # noqa: F821
    import pandas as pd
    from huggingface_hub import hf_hub_download

    cfg = profile.corpus
    cache_dir = resolve(cfg.hf_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        pd.read_parquet(
            hf_hub_download(
                repo_id=cfg.hf_repo,
                repo_type="dataset",
                filename=name,
                cache_dir=str(cache_dir),
            )
        )
        for name in _split_files(profile)
    ]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def build_topic(profile: Profile, topic: str, *, force: bool = False) -> Corpus:
    """Materialise one topic corpus to disk and return it."""
    out = topic_dir(profile, topic)
    if out.joinpath("manifest.json").exists() and not force:
        return load_topic(profile, topic)

    if topic not in TOPICS:
        raise KeyError(f"unknown topic {topic!r}; choose from {sorted(TOPICS)}")

    df = _download_split(profile)
    return _build_from_frame(profile, topic, df, out)


def _build_from_frame(profile: Profile, topic: str, df, out: Path) -> Corpus:
    cfg = profile.corpus
    hf_topic = TOPICS[topic]
    sub = df[df["document_topic"] == hf_topic]
    if sub.empty:
        raise ValueError(f"no rows for topic {hf_topic!r} in split {cfg.split!r}")

    n_before = len(sub)
    sub = sub[~sub["answer"].map(_is_unanswerable)]
    dropped_unanswerable = n_before - len(sub)

    doc_ids = sorted(sub["document_id"].unique())[: cfg.n_docs_per_topic]
    documents: list[Document] = []
    qa: list[QAPair] = []
    short_docs: list[str] = []

    for doc_id in doc_ids:
        rows = sub[sub["document_id"] == doc_id].sort_values("question_id")
        documents.append(
            Document(document_id=doc_id, text=str(rows.iloc[0]["document_extracted"]))
        )
        kept = rows.head(cfg.n_qa_per_doc)
        if len(kept) < cfg.n_qa_per_doc:
            short_docs.append(doc_id)
        for _, r in kept.iterrows():
            qa.append(
                QAPair(
                    question_id=str(r["question_id"]),
                    document_id=doc_id,
                    question=str(r["question"]),
                    answer=str(r["answer"]),
                    long_answer=str(r.get("long_answer") or ""),
                )
            )

    manifest = {
        "topic": topic,
        "hf_topic": hf_topic,
        "hf_repo": cfg.hf_repo,
        "split": cfg.split,
        "n_documents": len(documents),
        "n_qa": len(qa),
        "n_words": sum(d.n_words for d in documents),
        "document_ids": [d.document_id for d in documents],
        "dropped_unanswerable": int(dropped_unanswerable),
        "documents_with_fewer_qa": short_docs,
        "subsample_rule": (
            "split-restricted; unanswerable dropped; first N document_ids "
            "lexicographically; first K question_ids per document"
        ),
        "target_n_documents": cfg.n_docs_per_topic,
        "target_n_qa_per_doc": cfg.n_qa_per_doc,
        "content_sha256": _content_hash(documents, qa),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "docs.jsonl", [{"document_id": d.document_id, "text": d.text} for d in documents])
    _write_jsonl(
        out / "qa.jsonl",
        [
            {
                "question_id": q.question_id,
                "document_id": q.document_id,
                "question": q.question,
                "answer": q.answer,
                "long_answer": q.long_answer,
            }
            for q in qa
        ],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return Corpus(topic=topic, documents=documents, qa=qa, manifest=manifest)


def load_topic(profile: Profile, topic: str) -> Corpus:
    """Load a previously built topic corpus."""
    out = topic_dir(profile, topic)
    mf = out / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(
            f"topic {topic!r} not built at {out}; run "
            f"`python -m wicer_core.corpus --profile {profile.name} --topic {topic}`"
        )
    manifest = json.loads(mf.read_text())
    documents = [Document(**r) for r in _read_jsonl(out / "docs.jsonl")]
    qa = [QAPair(**r) for r in _read_jsonl(out / "qa.jsonl")]
    return Corpus(topic=topic, documents=documents, qa=qa, manifest=manifest)


def available_topics(profile: Profile) -> list[str]:
    root = resolve(profile.corpus.data_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build RepLiQA topic corpora.")
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all 17")
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    ap.add_argument(
        "--check",
        action="store_true",
        help="rebuild into a scratch dir and assert the content hash is unchanged",
    )
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    topics = args.topic or list(TOPICS)

    rc = 0
    for topic in topics:
        corpus = build_topic(profile, topic, force=args.force)
        m = corpus.manifest
        print(
            f"{topic:<26} docs={m['n_documents']:>3} qa={m['n_qa']:>4} "
            f"words={m['n_words']:>7} dropped_unanswerable={m['dropped_unanswerable']:>4} "
            f"sha={m['content_sha256'][:12]}"
        )
        if args.check:
            df = _download_split(profile)
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                again = _build_from_frame(profile, topic, df, Path(td) / topic)
            if again.manifest["content_sha256"] != m["content_sha256"]:
                print("  NONDETERMINISTIC: hash changed on rebuild", file=sys.stderr)
                rc = 1
            else:
                print("  deterministic: hash stable across rebuild")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
