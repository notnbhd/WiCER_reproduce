"""Typed rendering under an enforced budget (draft section 3.4): (L, B, rho) -> (W, sigma).

The rendering is a deterministic function of the ledger and the selected
claims; no model is involved (draft open question 6).  Every claim routes to
exactly one block type, so cl(W) -- the claims a sidecar row points at -- is
the whole of what the wiki asserts:

* T2  timeline      claims with a time, per document, in rough date order
* T3a entity block  an entity with >= 2 distinct relations among its untimed
                    claims; rows are ``relation: value``
* T3b fact list     every other claim, as ``entity — relation: value``

Rows are lines, not markdown table cells: the pipes and padding cost tokens and
the line keeps the same (entity, relation) key.  Routing is computed over the
whole ledger, not over what was selected, and block and row order come from
source position, so adding a claim only ever adds lines -- the property the
append-only repair of section 3.5.1 relies on.

Budget: B = ratio x (reader tokens of the source collection), the same
denominator wicer's compile reports.  Iteration 0 fills to (1 - rho) B.  Each
document first fills its own share, proportional to its token length, in
entity round-robin order; what a document cannot use is then offered to every
document, round by round.  The final wiki is re-counted as a whole and rows are
dropped, last-taken first, until it fits.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from wicer_core.config import Profile
from wicer_core.corpus import Document, format_documents, load_topic
from wicer_core.metrics import compression_ratio
from wicer_core.records import wiki_dir, wiki_paths
from tact.ledger import Claim, load_ledger

ROUTES = ("T2", "T3a", "T3b")
_YEAR = re.compile(r"\b(\d{4})\b")


@dataclass
class RenderResult:
    text: str
    sidecar: list[dict[str, Any]]
    realized: set[str]
    meta: dict[str, Any] = field(default_factory=dict)


# -- routing and ranking ---------------------------------------------------------------


def route(claims: Sequence[Claim]) -> dict[str, str]:
    """claim id -> T2 | T3a | T3b.  Exhaustive by construction."""
    relations: dict[tuple[str, str], set[str]] = {}
    for c in claims:
        if c.tau is None:
            relations.setdefault((c.doc, c.e), set()).add(c.r)
    out = {}
    for c in claims:
        if c.tau is not None:
            out[c.id] = "T2"
        else:
            out[c.id] = "T3a" if len(relations[(c.doc, c.e)]) >= 2 else "T3b"
    return out


def round_robin(claims: Sequence[Claim]) -> list[Claim]:
    """Each entity's first claim, then each entity's second, ... by source position."""
    threads: dict[str, list[Claim]] = {}
    for c in sorted(claims, key=lambda c: (c.pos, c.id)):
        threads.setdefault(c.e, []).append(c)
    out: list[Claim] = []
    for k in range(max((len(t) for t in threads.values()), default=0)):
        out += [t[k] for t in threads.values() if k < len(t)]
    return out


# -- layout -------------------------------------------------------------------------------


def _rel(r: str) -> str:
    return r.replace("_", " ")


def _value(c: Claim, by_id: dict[str, Claim]) -> str:
    if not c.ref:
        return c.v
    t = by_id.get(c.v)
    if t is None:
        return " … ".join(c.verbatim)
    tv = " … ".join(t.verbatim) if t.ref else t.v
    return f"→ {t.e} — {_rel(t.r)}: {tv}"


def _line(c: Claim, kind: str, by_id: dict[str, Claim]) -> str:
    src = f" (per {c.src})" if c.src else ""
    v = _value(c, by_id)
    if kind == "T2":
        return f"- {c.tau}: {c.e} — {_rel(c.r)}: {v}{src}"
    if kind == "T3a":
        return f"- {_rel(c.r)}: {v}{src}"
    return f"- {c.e} — {_rel(c.r)}: {v}{src}"


def _block(c: Claim, kind: str) -> str:
    return {"T2": "timeline", "T3a": f"entity:{c.e}", "T3b": "facts"}[kind]


def _heading(block: str) -> str:
    if block == "timeline":
        return "### Timeline"
    if block == "facts":
        return "### Other facts"
    return f"### {block[len('entity:'):]}"


def _block_rank(block: str) -> int:
    return 0 if block == "timeline" else 2 if block == "facts" else 1


def _row_key(c: Claim, kind: str) -> tuple:
    if kind == "T2":
        year = _YEAR.search(c.tau or "")
        return (int(year.group()) if year else 9999, c.pos, c.id)
    return (c.pos, c.id)


def layout(
    selected: Sequence[Claim],
    by_id: dict[str, Claim],
    *,
    routes: dict[str, str] | None = None,
    titles: dict[str, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """(wiki text, sidecar) for the selected claims.  ``by_id`` is the ledger."""
    routes = routes or route(list(by_id.values()))
    titles = titles or {}
    doc_order: dict[str, int] = {}
    entity_first: dict[tuple[str, str], int] = {}
    for c in by_id.values():
        doc_order.setdefault(c.doc, len(doc_order))
        key = (c.doc, c.e)
        entity_first[key] = min(entity_first.get(key, c.pos), c.pos)

    blocks: dict[tuple[str, str], list[Claim]] = {}
    for c in selected:
        blocks.setdefault((c.doc, _block(c, routes[c.id])), []).append(c)

    def block_key(k: tuple[str, str]) -> tuple:
        doc, block = k
        first = entity_first.get((doc, block[len("entity:"):]), 0) if block.startswith("entity:") else 0
        return (doc_order.get(doc, len(doc_order)), _block_rank(block), first, block)

    lines: list[str] = []
    sidecar: list[dict[str, Any]] = []
    current_doc = None
    for doc, block in sorted(blocks, key=block_key):
        if doc != current_doc:
            if lines:
                lines.append("")
            lines.append(f"## {titles.get(doc) or doc}")
            current_doc = doc
        lines += ["", _heading(block)]
        rows = sorted(blocks[(doc, block)], key=lambda c: _row_key(c, routes[c.id]))
        for i, c in enumerate(rows):
            sidecar.append({"doc": doc, "block": block, "row": i, "line": len(lines),
                            "claims": [c.id]})
            lines.append(_line(c, routes[c.id], by_id))
    return "\n".join(lines) + ("\n" if lines else ""), sidecar


# -- budget enforcement ------------------------------------------------------------------


def doc_title(doc: Document) -> str | None:
    """A document's first line, when it reads as a title (RepLiQA's do)."""
    first = doc.text.strip().split("\n", 1)[0].strip()
    return first if first and len(first.split()) <= 16 and first[-1] not in ".!?" else None


@dataclass
class Budget:
    source_tokens: int
    budget: int                 # B
    target: int                 # (1 - rho) B, what iteration 0 fills to
    share: dict[str, float]     # each document's part of the target


def budget_shares(
    documents: Sequence[Document], *, ratio: float, rho: float,
    count_tokens: Callable[[str], int],
) -> Budget:
    source_tokens = count_tokens(format_documents(list(documents)))
    budget = round(ratio * source_tokens)
    target = round((1 - rho) * budget)
    doc_tokens = {d.document_id: count_tokens(d.text) for d in documents}
    total = sum(doc_tokens.values()) or 1
    return Budget(source_tokens, budget, target,
                  {d: target * n / total for d, n in doc_tokens.items()})


def fill(
    claims: Sequence[Claim],
    documents: Sequence[Document],
    *,
    ratio: float,
    rho: float,
    count_tokens: Callable[[str], int],
) -> RenderResult:
    by_id = {c.id: c for c in claims}
    routes = route(claims)
    titles = {d.document_id: t for d in documents if (t := doc_title(d))}
    b = budget_shares(documents, ratio=ratio, rho=rho, count_tokens=count_tokens)
    source_tokens, budget, target, share = b.source_tokens, b.budget, b.target, b.share

    queues: dict[str, list[Claim]] = {d.document_id: [] for d in documents}
    for c in claims:
        queues.setdefault(c.doc, []).append(c)
    queues = {d: round_robin(q) for d, q in queues.items()}

    cache: dict[str, int] = {}

    def cost(text: str) -> int:
        if text not in cache:
            cache[text] = count_tokens(text)
        return cache[text]

    row_cost = {c.id: cost(_line(c, routes[c.id], by_id)) for c in claims}
    taken: list[Claim] = []
    used: Counter[str] = Counter()
    opened: set[tuple[str, str]] = set()

    def marginal(c: Claim) -> int:
        n = row_cost[c.id]
        block = (c.doc, _block(c, routes[c.id]))
        if block not in opened:
            n += cost(_heading(block[1]))
        if used[c.doc] == 0:
            n += cost(f"## {titles.get(c.doc) or c.doc}")
        return n

    def take(c: Claim, n: int) -> None:
        taken.append(c)
        used[c.doc] += n
        opened.add((c.doc, _block(c, routes[c.id])))

    for d, queue in queues.items():
        for c in queue:
            n = marginal(c)
            if used[d] + n <= share.get(d, 0):
                take(c, n)
    chosen = {c.id for c in taken}
    for k in range(max((len(q) for q in queues.values()), default=0)):
        for queue in queues.values():
            if k < len(queue) and queue[k].id not in chosen:
                n = marginal(queue[k])
                if sum(used.values()) + n <= target:
                    take(queue[k], n)
                    chosen.add(queue[k].id)

    # The row costs are an estimate (tokenization is not quite additive);
    # the whole wiki is the measurement.
    text, sidecar = layout(taken, by_id, routes=routes, titles=titles)
    actual = count_tokens(text)
    while actual > target and taken:
        taken.pop()
        text, sidecar = layout(taken, by_id, routes=routes, titles=titles)
        actual = count_tokens(text)

    realized = {c.id for c in taken}
    routing = {}
    for kind in ROUTES:
        ids = [c.id for c in claims if routes[c.id] == kind]
        got = [i for i in ids if i in realized]
        tok = sum(row_cost[i] for i in got)
        routing[kind] = {"n_claims": len(ids), "n_realized": len(got), "row_tokens": tok,
                         "tokens_per_claim": round(tok / len(got), 2) if got else None}
    meta = {
        "method": "tact",
        "ratio": ratio,
        "rho": rho,
        "source_tokens": source_tokens,
        "budget_tokens": budget,
        "target_tokens": target,
        "actual_tokens": actual,
        "actual_ratio_tokens": compression_ratio(actual, source_tokens),
        "n_claims": len(claims),
        "n_realized": len(realized),
        "coverage": len(realized) / len(claims) if claims else 0.0,
        "n_documents_rendered": len({c.doc for c in taken}),
        "routing": routing,
    }
    return RenderResult(text=text, sidecar=sidecar, realized=realized, meta=meta)


# -- the wiki on disk --------------------------------------------------------------------


def sidecar_path(profile: Profile, topic: str, name: str) -> Path:
    return wiki_dir(profile, topic) / f"{name}.sidecar.json"


def render_topic(
    profile: Profile,
    topic: str,
    *,
    name: str,
    rho: float,
    count_tokens: Callable[[str], int],
    ratio: float | None = None,
) -> RenderResult:
    """Render a topic's ledger to a wiki the benchmark runner can serve."""
    ledger = load_ledger(profile, topic)
    if ledger is None:
        raise FileNotFoundError(f"no ledger for {topic!r}: run `python -m tact.run extract` first")
    corpus = load_topic(profile, topic)
    ratio = profile.wicer.ratio if ratio is None else ratio
    res = fill(ledger.claims, corpus.documents, ratio=ratio, rho=rho, count_tokens=count_tokens)
    source_words = corpus.n_words
    actual_words = len(res.text.split())
    res.meta.update({
        "topic": topic,
        "name": name,
        "source_words": source_words,
        "actual_words": actual_words,
        "actual_ratio_words": compression_ratio(actual_words, source_words),
        "n_preserved_facts": 0,
        "ledger": {k: ledger.meta.get(k) for k in (
            "n_claims", "rejection_rate", "n_multispan", "n_relational",
            "n_contradictions", "entity_collision_rate")},
    })
    md, meta = wiki_paths(profile, topic, name)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(res.text)
    meta.write_text(json.dumps(res.meta, indent=2) + "\n")
    sidecar_path(profile, topic, name).write_text(json.dumps(res.sidecar, indent=1) + "\n")
    return res
