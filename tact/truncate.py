"""Baseline (a) of draft section 3.9: truncation, with no model in the loop.

Each document is cut, at a sentence boundary, to its share of the budget --
the same B = ratio x (reader tokens of the collection) and the same shares
proportional to document length that TACT's renderer uses, but filled to B
itself rather than (1 - rho) B.  It is the ceiling on what filling the budget
buys by itself: if TACT's iteration 0 does not beat it, the typed layout is
adding nothing over keeping the start of every document.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

from wicer_core.config import Profile
from wicer_core.corpus import Document, format_documents, load_topic
from wicer_core.metrics import compression_ratio
from wicer_core.records import wiki_paths

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)|\n\s*\n")


def _header(i: int, doc: Document) -> str:
    return f"=== DOCUMENT {i + 1} (id: {doc.document_id}) ==="


def _render(documents: Sequence[Document], cuts: Sequence[int]) -> str:
    return "\n\n".join(
        f"{_header(i, d)}\n{d.text[:cut].strip()}" for i, (d, cut) in enumerate(zip(documents, cuts))
    )


def truncate(
    documents: Sequence[Document], *, ratio: float, count_tokens: Callable[[str], int]
) -> tuple[str, dict[str, Any]]:
    source_tokens = count_tokens(format_documents(list(documents)))
    budget = round(ratio * source_tokens)
    doc_tokens = [count_tokens(d.text) for d in documents]
    total = sum(doc_tokens) or 1

    cuts: list[int] = []
    taken: list[tuple[int, int]] = []          # (document index, previous cut), in take order
    for i, (d, n) in enumerate(zip(documents, doc_tokens)):
        share = budget * n / total
        used = count_tokens(_header(i, d))
        cut = 0
        for m in _SENTENCE_END.finditer(d.text):
            piece = count_tokens(d.text[cut:m.end()])
            if used + piece > share:
                break
            used += piece
            taken.append((i, cut))
            cut = m.end()
        cuts.append(cut)

    text = _render(documents, cuts)
    actual = count_tokens(text)
    while actual > budget and taken:
        i, previous = taken.pop()
        cuts[i] = previous
        text = _render(documents, cuts)
        actual = count_tokens(text)

    return text, {
        "method": "truncate",
        "ratio": ratio,
        "source_tokens": source_tokens,
        "budget_tokens": budget,
        "actual_tokens": actual,
        "actual_ratio_tokens": compression_ratio(actual, source_tokens),
        "kept_chars_share": sum(cuts) / max(1, sum(len(d.text) for d in documents)),
    }


def truncate_topic(
    profile: Profile,
    topic: str,
    *,
    name: str,
    count_tokens: Callable[[str], int],
    ratio: float | None = None,
) -> tuple[str, dict[str, Any]]:
    corpus = load_topic(profile, topic)
    ratio = profile.wicer.ratio if ratio is None else ratio
    text, meta = truncate(corpus.documents, ratio=ratio, count_tokens=count_tokens)
    actual_words = len(text.split())
    meta.update({
        "topic": topic,
        "name": name,
        "source_words": corpus.n_words,
        "actual_words": actual_words,
        "actual_ratio_words": compression_ratio(actual_words, corpus.n_words),
        "n_preserved_facts": 0,
    })
    md, meta_path = wiki_paths(profile, topic, name)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return text, meta
