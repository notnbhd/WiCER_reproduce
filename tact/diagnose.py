"""Mechanical failure diagnosis (draft section 3.5.2) and pin closure (3.5.1).

No model is asked why a probe failed; the attribution is set arithmetic over
the ledger and the rendering:

* MISS-EXTRACT   A(q) is empty: no claim covers the probe's passage.
* MISS-RENDER    some of A+(q) is not in the wiki.  The draft names only the
                 case where none of it is; partly rendered is the same loss
                 at a smaller size, and its remainder is what gets pinned.
* MISS-RELATION  the probe asks for a relation between facts (the question
                 writer flags it), all of A+(q) is rendered, and no relation
                 claim is among it: the facts are there, the connective is not.
* MISS-READ      everything is rendered, and the reader still failed.

A+(q) is A(q) plus, for a relational probe, the ledger's relation claims that
point into A(q) -- a connective the ledger has but the probe's passage does
not cover, so that an unrendered relation reads as MISS-RENDER, not as missing.
"""

from __future__ import annotations

import re
from typing import Sequence

from tact.ledger import Claim
from tact.probes import Probe

MISS_EXTRACT = "MISS-EXTRACT"
MISS_RENDER = "MISS-RENDER"
MISS_RELATION = "MISS-RELATION"
MISS_READ = "MISS-READ"
REPAIRABLE = (MISS_EXTRACT, MISS_RENDER, MISS_RELATION)

_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")
# local_economy's longest paragraph is 1,217 characters and its 90th
# percentile 560; a longer block is a document written without paragraph
# breaks, and closing over it would pin the whole document.
MAX_PARAGRAPH = 1_500
CHUNK = 600


def paragraphs(text: str) -> list[tuple[int, int]]:
    """(start, end) of each blank-line-separated block, long ones cut at sentences."""
    blocks, start = [], 0
    for m in _BLANK_LINE.finditer(text):
        if text[start:m.start()].strip():
            blocks.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        blocks.append((start, len(text)))
    out = []
    for s, e in blocks:
        if e - s <= MAX_PARAGRAPH:
            out.append((s, e))
            continue
        cut = last = s
        for m in _SENTENCE_END.finditer(text, s, e):
            # Close the chunk at the previous sentence end once this one would
            # take it past CHUNK.
            if m.end() - cut > CHUNK and last > cut:
                out.append((cut, last))
                cut = last
            last = m.end()
        out.append((cut, e))
    return out


def neighbourhood(ids: Sequence[str], doc_claims: Sequence[Claim], text: str) -> list[str]:
    """N(A): ``ids`` plus every claim of the document starting in a paragraph they touch."""
    if not ids:
        return []
    wanted = set(ids)
    paras = paragraphs(text)
    touched = {
        i for c in doc_claims if c.id in wanted for s, e in c.spans
        for i, (ps, pe) in enumerate(paras) if s < pe and e > ps
    }
    inside = {
        c.id for c in doc_claims
        if any(paras[i][0] <= c.pos < paras[i][1] for i in touched)
    }
    return [c.id for c in sorted(doc_claims, key=lambda c: (c.pos, c.id))
            if c.id in wanted or c.id in inside]


def extended_answer(probe: Probe, answer: Sequence[str], doc_claims: Sequence[Claim]) -> list[str]:
    if not probe.relational:
        return list(answer)
    inside = set(answer)
    return list(answer) + [c.id for c in doc_claims
                           if c.ref and c.v in inside and c.id not in inside]


def diagnose(
    probe: Probe,
    answer: Sequence[str],
    extended: Sequence[str],
    realized: set[str],
    by_id: dict[str, Claim],
) -> str:
    if not answer:
        return MISS_EXTRACT
    if any(i not in realized for i in extended):
        return MISS_RENDER
    if probe.relational and not any(by_id[i].ref for i in extended):
        return MISS_RELATION
    return MISS_READ
