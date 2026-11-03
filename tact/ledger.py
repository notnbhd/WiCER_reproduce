"""The claim ledger (draft section 3.3): D -> L, once, offline.

Extraction runs two passes per document: an entity inventory, then claims
whose entity is drawn from it.  Three code steps follow, and none of them
trusts the model:

* Span validation.  Reported offsets are never asked for; each verbatim piece
  is located in the document by exact, then word-normalised, then fuzzy match,
  and the claim's verbatim is replaced by the source text it matched.  A claim
  with any piece below the threshold is rejected, and so is a relation that
  points at a rejected claim.
* Entity canonicalization.  Names linked by whole-word containment collapse to
  their most-claimed member: a head with trailing modifiers always ("Joe
  Harris company"), a name qualified on the left only when one qualifier
  exists ("Harris" in "Joe Harris" and "Mary Harris" links nothing).  An
  optional similarity function adds near-duplicate names.
* Provenance-aware merging, keyed on (entity, relation, time).  Equal values
  merge and keep every source; unequal values across documents are both kept
  and recorded as a contradiction.  Nothing is deleted for looking similar.

Each document's raw extraction is cached under ``ledger/extract/``, so a
crashed run resumes without paying for the documents it already did.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from wicer_core.config import CompilerConfig, Profile, resolve
from wicer_core.corpus import Document, load_topic

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
EXTRACT_SYSTEM = (
    "You turn a document into structured claims. You copy the document's own "
    "words exactly and you reply with JSON only."
)
# Word-level similarity a verbatim piece needs to count as found.  One slip in
# a nine-word passage scores 0.89; a paraphrase scores well under 0.6.
MATCH_THRESHOLD = 0.85

_WORD = re.compile(r"\S+")
_PUNCT = re.compile(r"[^\w%$.]+|(?<!\d)\.|\.(?!\d)")
_SENTENCE = re.compile(r"[^.!?]+[.!?]+")
_TRUNCATED = frozenset({"length", "max_tokens"})


@dataclass
class Claim:
    id: str
    doc: str
    e: str
    r: str
    v: str                      # a literal, or a claim id when ``ref``
    verbatim: list[str]
    spans: list[list[int]]      # [start, end) per verbatim piece, in source order
    ref: bool = False
    tau: str | None = None
    src: str | None = None      # speaker the document attributes the claim to
    sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = [self.doc]

    @property
    def multi_span(self) -> bool:
        return len(self.spans) > 1

    @property
    def pos(self) -> int:
        return self.spans[0][0] if self.spans else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ledger:
    claims: list[Claim]
    contradictions: list[dict[str, Any]]
    meta: dict[str, Any]

    @property
    def by_id(self) -> dict[str, Claim]:
        return {c.id: c for c in self.claims}


# -- span validation ----------------------------------------------------------


def _words(text: str) -> list[tuple[str, int, int]]:
    return [(m.group().lower(), m.start(), m.end()) for m in _WORD.finditer(text)]


def locate(verbatim: str, text: str, threshold: float = MATCH_THRESHOLD
           ) -> tuple[int, int, float] | None:
    """(start, end, score) of ``verbatim`` in ``text``, or None below threshold."""
    i = text.find(verbatim)
    if i >= 0 and verbatim:
        return i, i + len(verbatim), 1.0
    want = [w for w, _, _ in _words(verbatim)]
    have = _words(text)
    n = len(want)
    if not n or n > len(have):
        return None
    tokens = [w for w, _, _ in have]
    best: tuple[float, int, int] | None = None
    # Windows a little shorter and longer than the piece absorb a dropped or
    # added word without letting a long window score on overlap alone.
    for size in sorted({max(1, n - 1), n, n + 1}):
        for s in range(0, len(have) - size + 1):
            window = tokens[s:s + size]
            if window == want:
                return have[s][1], have[s + size - 1][2], 1.0
            score = difflib.SequenceMatcher(None, want, window, autojunk=False).ratio()
            if best is None or score > best[0]:
                best = (score, s, size)
    score, s, size = best
    if score < threshold:
        return None
    return have[s][1], have[s + size - 1][2], score


# -- canonicalization -----------------------------------------------------------


def _name_words(name: str) -> tuple[str, ...]:
    return tuple(w for w in re.split(r"[^\w]+", name.lower()) if w)


def _inside(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whole-word contiguous containment of ``a`` in ``b``."""
    n = len(a)
    return 0 < n <= len(b) and any(b[i:i + n] == a for i in range(len(b) - n + 1))


def canonical_map(
    counts: dict[str, int],
    similar: Callable[[list[str]], list[tuple[int, int]]] | None = None,
) -> dict[str, str]:
    """Map every name to its canonical form.

    ``counts`` is claims per name; the canonical member of a group is the most
    claimed, then the shortest.  ``similar`` returns index pairs of names to
    link on top of containment (e.g. by embedding similarity).
    """
    names = sorted(counts)
    words = {n: _name_words(n) for n in names}
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for a in names:
        supers = [b for b in names if b != a and _inside(words[a], words[b])]
        # A name extended on the right is its head plus a modifier ("Joe
        # Harris company"); extended on the left it is being qualified ("Joe
        # Harris" for "Harris"), and two different qualifiers mean two
        # different entities, so the short name links to neither.
        modified = [b for b in supers if words[b][:len(words[a])] == words[a]]
        qualified = [b for b in supers if b not in modified]
        for b in modified + (qualified if len(qualified) == 1 else []):
            union(a, b)
    if similar is not None:
        for i, j in similar(names):
            union(names[i], names[j])

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out: dict[str, str] = {}
    for members in groups.values():
        head = min(members, key=lambda n: (-counts[n], len(words[n]), n))
        out.update({n: head for n in members})
    return out


def embedding_similar(threshold: float = 0.92) -> Callable[[list[str]], list[tuple[int, int]]]:
    """Name pairs whose MiniLM embeddings exceed ``threshold`` (needs the rag extra)."""
    from wicer_core.embedding import Embedder

    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")

    def pairs(names: list[str]) -> list[tuple[int, int]]:
        if len(names) < 2:
            return []
        m = embedder.encode(names)
        sims = m @ m.T
        return [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
                if sims[i, j] >= threshold]

    return pairs


def collision_rate(entities: Sequence[str]) -> float:
    """Share of distinct names still in a containment pair with another name."""
    names = sorted(set(entities))
    words = [_name_words(n) for n in names]
    hit = {
        i for i in range(len(names)) for j in range(len(names))
        if i != j and (_inside(words[i], words[j]) or words[i] == words[j])
    }
    return len(hit) / len(names) if names else 0.0


# -- extraction output -> claims ---------------------------------------------------


def claims_from_extraction(
    doc_id: str, text: str, raw: dict[str, Any], cmap: dict[str, str]
) -> tuple[list[Claim], dict[str, int]]:
    """Validate one document's raw claims and give them ledger ids."""
    rows = [r for r in raw.get("claims", []) if isinstance(r, dict)]
    local_ids: dict[str, str] = {}
    located: list[tuple[dict, str, list[list[int]], list[str]] | None] = []
    for i, row in enumerate(rows):
        cid = f"{doc_id}#{i + 1}"
        local_ids[str(row.get("id", ""))] = cid
        pieces = row.get("verbatim") or []
        if isinstance(pieces, str):
            pieces = [pieces]
        hits = [locate(str(p), text) for p in pieces if str(p).strip()]
        if not hits or any(h is None for h in hits):
            located.append(None)
            continue
        hits.sort(key=lambda h: h[0])
        located.append((row, cid, [[h[0], h[1]] for h in hits], [text[h[0]:h[1]] for h in hits]))

    kept_ids = {x[1] for x in located if x is not None}
    claims: list[Claim] = []
    n_multispan = n_relational = 0
    for x in located:
        if x is None:
            continue
        row, cid, spans, verbatim = x
        target = row.get("value_ref")
        if target is not None:
            v = local_ids.get(str(target))
            if v is None or v not in kept_ids:
                kept_ids.discard(cid)
                continue
        else:
            v = str(row.get("value", "")).strip()
        entity = str(row.get("entity", "")).strip() or "(document)"
        claims.append(Claim(
            id=cid, doc=doc_id, e=cmap.get(entity, entity),
            r=str(row.get("relation", "")).strip() or "states", v=v,
            verbatim=verbatim, spans=spans, ref=target is not None,
            tau=_opt(row.get("time")), src=_opt(row.get("speaker")),
        ))
        n_multispan += len(spans) > 1
        n_relational += target is not None
    stats = {"n_raw": len(rows), "n_rejected": len(rows) - len(claims),
             "n_multispan": n_multispan, "n_relational": n_relational}
    return claims, stats


def _opt(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in {"null", "none", "n/a"} else None


# -- provenance-aware merging -----------------------------------------------------


def _norm(text: str | None) -> str:
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def merge(claims: Sequence[Claim]) -> tuple[list[Claim], list[dict[str, Any]]]:
    """Apply the merge table of section 3.3; returns (claims, contradictions).

    Literal claims merge first so that a relation pointing at a merged-away
    duplicate is rewritten to the survivor before relations are compared.
    """
    alias: dict[str, str] = {}
    kept: dict[str, Claim] = {}
    groups: dict[tuple[str, str, str, bool], list[Claim]] = {}

    def absorb(batch: Sequence[Claim]) -> None:
        for c in batch:
            key = (_norm(c.e), _norm(c.r), _norm(c.tau), c.ref)
            group = groups.setdefault(key, [])
            same = next((k for k in group if _norm(k.v) == _norm(c.v)), None)
            if same is None:
                group.append(c)
                kept[c.id] = c
            else:
                alias[c.id] = same.id
                same.sources += [d for d in c.sources if d not in same.sources]

    absorb([c for c in claims if not c.ref])
    for c in claims:
        if c.ref:
            c.v = alias.get(c.v, c.v)
    absorb([c for c in claims if c.ref])

    contradictions = []
    for (e, r, tau, _), group in groups.items():
        docs = sorted({d for c in group for d in c.sources})
        if len(group) > 1 and len(docs) > 1:
            contradictions.append({"key": [e, r, tau], "claims": [c.id for c in group],
                                   "docs": docs})
    order = {c.id: i for i, c in enumerate(claims)}
    return sorted(kept.values(), key=lambda c: order[c.id]), contradictions


# -- the extractor -------------------------------------------------------------------


class ExtractError(RuntimeError):
    pass


class Extractor:
    """Two-pass claim extraction with the same backends as wicer's compiler."""

    def __init__(self, cfg: CompilerConfig, backend: str | None = None,
                 openrouter_client: Any = None):
        self.model = cfg.model
        self.backend = backend or os.environ.get("WICER_COMPILER_BACKEND", "anthropic")
        # Effort sets how exhaustive the ledger is.  Measured on two
        # local_economy documents with Haiku 5.5: low extracted 55 claims
        # covering 36% of the text -- the whole ledger rendered to 36% of the
        # source, under a 45% target, so the budget never binds; medium 87
        # claims, 55%; high 138 claims, 69-96%, and 7 relational claims against
        # 1.  Only high gives the ledger section 3.3 assumes, at ~$1 a topic.
        # High spent ~25K output tokens on a 1,000-word document, so the cap
        # leaves room for longer ones.
        self.effort = os.environ.get("TACT_EXTRACT_EFFORT", "high")
        self.max_tokens = int(os.environ.get("TACT_EXTRACT_MAX_TOKENS", "64000"))
        self._or_client = openrouter_client
        self._client = None
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.measured_cost_usd = 0.0

    @property
    def openrouter(self):
        if self._or_client is None:
            from wicer_core.openrouter import OpenRouterClient

            self._or_client = OpenRouterClient()
        return self._or_client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def extract(self, doc: Document) -> dict[str, Any]:
        """{"inventory": [...], "raw": {"claims": [...]}} for one document."""
        if self.backend == "fake":
            self.n_calls += 2
            return {**_fake_extract(doc), "extracted_with": "fake"}
        inv = self._json((PROMPT_DIR / "inventory.md").read_text().format(document=doc.text))
        entities = [str(e) for e in inv.get("entities", []) if str(e).strip()]
        raw = self._json((PROMPT_DIR / "claims.md").read_text().format(
            document=doc.text, entities="\n".join(f"- {e}" for e in entities)))
        return {"inventory": entities, "raw": raw,
                "extracted_with": f"{self.model} effort={self.effort}"}

    def _json(self, prompt: str) -> dict[str, Any]:
        last: Exception | None = None
        for _ in range(2):
            text = self._call(prompt)
            try:
                return _parse_json(text)
            except ValueError as e:
                last = e
        raise ExtractError(f"{self.model} returned no parseable JSON: {last}")

    def _call(self, prompt: str) -> str:
        self.n_calls += 1
        if self.backend == "openrouter":
            from wicer_core.config import resolve_openrouter_model

            reply = self.openrouter.chat(
                model=resolve_openrouter_model(self.model), system=EXTRACT_SYSTEM,
                user=prompt, max_tokens=self.max_tokens, effort=self.effort,
            )
            self.input_tokens += reply.input_tokens
            self.output_tokens += reply.output_tokens
            self.measured_cost_usd += reply.cost_usd
            text, finish = reply.text, reply.finish_reason
        else:
            with self.client.messages.stream(
                model=self.model, max_tokens=self.max_tokens, system=EXTRACT_SYSTEM,
                thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                msg = stream.get_final_message()
            self.input_tokens += msg.usage.input_tokens
            self.output_tokens += msg.usage.output_tokens
            text = "".join(b.text for b in msg.content if b.type == "text")
            finish = msg.stop_reason or ""
        if finish in _TRUNCATED:
            raise ExtractError(
                f"{self.model} hit max_tokens={self.max_tokens} mid-extraction; raise "
                f"TACT_EXTRACT_MAX_TOKENS or lower TACT_EXTRACT_EFFORT ({self.effort!r})"
            )
        return text

    def usage_summary(self) -> dict[str, Any]:
        return {"extract_model": self.model, "backend": self.backend, "effort": self.effort,
                "calls": self.n_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "measured_cost_usd": round(self.measured_cost_usd, 6)}


def _parse_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("reply is not a JSON object")
    return value


def _fake_extract(doc: Document) -> dict[str, Any]:
    """One claim per sentence, keyed on its first capitalised name.

    Deterministic and offline.  Sentences with a year carry it as the time, and
    an entity named in several sentences gets several relations, so the fixture
    exercises every route of the renderer.
    """
    rows, seen = [], Counter()
    for i, m in enumerate(_SENTENCE.finditer(doc.text)):
        sentence = m.group().strip()
        name = re.search(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", sentence)
        entity = name.group() if name else "(document)"
        seen[entity] += 1
        year = re.search(r"\b(1[89]\d\d|20\d\d)\b", sentence)
        rows.append({"id": f"c{i + 1}", "entity": entity, "relation": f"states_{seen[entity]}",
                     "value": sentence.rstrip("."), "time": year.group() if year else None,
                     "speaker": None, "verbatim": [sentence]})
    return {"inventory": sorted(seen), "raw": {"claims": rows}}


# -- the ledger on disk -----------------------------------------------------------


def ledger_dir(profile: Profile, topic: str) -> Path:
    return resolve(profile.results_dir) / topic / "ledger"


def load_ledger(profile: Profile, topic: str) -> Ledger | None:
    d = ledger_dir(profile, topic)
    if not (d / "ledger.jsonl").exists():
        return None
    claims = [Claim(**json.loads(line))
              for line in (d / "ledger.jsonl").read_text().splitlines() if line.strip()]
    return Ledger(
        claims=claims,
        contradictions=json.loads((d / "contradictions.json").read_text()),
        meta=json.loads((d / "ledger_meta.json").read_text()),
    )


def build_ledger(
    profile: Profile,
    topic: str,
    extractor: Extractor,
    *,
    workers: int = 8,
    similar: Callable[[list[str]], list[tuple[int, int]]] | None = None,
    documents: Sequence[Document] | None = None,
) -> Ledger:
    """Extract (or reuse cached extractions), validate, canonicalize, merge, write.

    The corpus fingerprint goes in the meta so a ledger cannot be rendered
    against a rebuilt corpus it was not extracted from -- the laptop's data
    was once a different 80 documents under the same topic name.  An explicit
    ``documents`` subset has no fingerprint.
    """
    corpus_sha = None
    if documents is None:
        corpus = load_topic(profile, topic)
        documents, corpus_sha = corpus.documents, corpus.manifest.get("content_sha256")
    documents = list(documents)
    d = ledger_dir(profile, topic)
    cache = d / "extract"
    cache.mkdir(parents=True, exist_ok=True)

    def one(doc: Document) -> dict[str, Any]:
        path = cache / f"{doc.document_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        out = extractor.extract(doc)
        path.write_text(json.dumps(out, indent=1) + "\n")
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        extractions = list(pool.map(one, documents))

    claims: list[Claim] = []
    totals = Counter()
    inventory_sizes = []
    for doc, ex in zip(documents, extractions):
        rows = [r for r in ex["raw"].get("claims", []) if isinstance(r, dict)]
        counts = Counter(str(r.get("entity", "")).strip() for r in rows)
        for name in ex.get("inventory", []):
            counts.setdefault(str(name).strip(), 0)
        counts.pop("", None)
        cmap = canonical_map(dict(counts), similar)
        doc_claims, stats = claims_from_extraction(doc.document_id, doc.text, ex["raw"], cmap)
        claims += doc_claims
        totals.update(stats)
        inventory_sizes.append(len(counts))

    merged, contradictions = merge(claims)
    meta = {
        "topic": topic,
        "corpus_sha256": corpus_sha,
        "n_documents": len(documents),
        "n_claims": len(merged),
        "n_claims_before_merge": len(claims),
        **dict(totals),
        "rejection_rate": totals["n_rejected"] / totals["n_raw"] if totals["n_raw"] else 0.0,
        "n_contradictions": len(contradictions),
        "inventory_size": sum(inventory_sizes),
        "n_canonical_entities": len({(c.doc, c.e) for c in merged}),
        # Per document: the same name in two documents is one string, not a
        # collision.
        "entity_collision_rate": _per_doc_collision(merged),
        "embedding_canonicalization": similar is not None,
        # What the cached extractions came from, per document -- not what this
        # invocation spent, which is zero on a fully cached re-run.
        "extracted_with": dict(Counter(ex.get("extracted_with", "unknown") for ex in extractions)),
        "usage_this_run": extractor.usage_summary(),
    }
    (d / "ledger.jsonl").write_text("".join(json.dumps(c.to_dict()) + "\n" for c in merged))
    (d / "contradictions.json").write_text(json.dumps(contradictions, indent=1) + "\n")
    (d / "ledger_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return Ledger(claims=merged, contradictions=contradictions, meta=meta)


def _per_doc_collision(claims: Sequence[Claim]) -> float:
    by_doc: dict[str, set[str]] = {}
    for c in claims:
        by_doc.setdefault(c.doc, set()).add(c.e)
    names = sum(len(s) for s in by_doc.values())
    hit = sum(collision_rate(sorted(s)) * len(s) for s in by_doc.values())
    return hit / names if names else 0.0
