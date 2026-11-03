"""Self-generated probes (draft section 3.2).

A question writer reads each document and writes m questions spread over it,
each grounded in verbatim passages that are validated like claim spans.
Grounding in the source, not in the ledger, is what lets a probe expose a fact
the ledger never caught (MISS-EXTRACT).  A(q) is then computed mechanically:
the document's claims whose spans overlap the probe's.

Benchmark questions never reach the writer, so every benchmark question stays
held out.  Probes are cached per document under ``self_probes/write/``.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from wicer_core.config import CompilerConfig, Profile, resolve
from wicer_core.corpus import Document, QAPair, load_topic
from tact.ledger import PROMPT_DIR, Claim, locate
from tact.llm import JsonLLM

KINDS = ("date", "quantity", "person", "place", "attribution", "reason", "process", "other")
_SENTENCE_END = ".!?"


@dataclass
class Probe:
    id: str
    doc: str
    question: str
    answer: str
    verbatim: list[str]
    spans: list[list[int]]
    kind: str
    relational: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_qa(self) -> QAPair:
        return QAPair(question_id=self.id, document_id=self.doc, question=self.question,
                      answer=self.answer, long_answer=" ".join(self.verbatim))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def probes_from_raw(doc_id: str, text: str, raw: dict[str, Any]
                    ) -> tuple[list[Probe], dict[str, int]]:
    rows = [r for r in raw.get("probes", []) if isinstance(r, dict)]
    probes = []
    for i, row in enumerate(rows):
        pieces = row.get("verbatim") or []
        if isinstance(pieces, str):
            pieces = [pieces]
        hits = [locate(str(p), text) for p in pieces if str(p).strip()]
        question = str(row.get("question", "")).strip()
        if not hits or any(h is None for h in hits) or not question:
            continue
        hits.sort(key=lambda h: h[0])
        kind = str(row.get("kind", "")).strip().lower()
        probes.append(Probe(
            id=f"{doc_id}#p{i + 1}", doc=doc_id, question=question,
            answer=str(row.get("answer", "")).strip(),
            verbatim=[text[h[0]:h[1]] for h in hits], spans=[[h[0], h[1]] for h in hits],
            kind=kind if kind in KINDS else "other", relational=_truthy(row.get("relational")),
        ))
    return probes, {"n_raw": len(rows), "n_rejected": len(rows) - len(probes)}


def answer_set(probe: Probe, doc_claims: Sequence[Claim]) -> list[str]:
    """A(q): the claims whose spans overlap any of the probe's spans."""
    return [
        c.id for c in doc_claims
        if any(min(a[1], b[1]) > max(a[0], b[0]) for a in c.spans for b in probe.spans)
    ]


class ProbeWriter(JsonLLM):
    system = ("You write reading-comprehension questions about a document. You copy "
              "the document's own words exactly and you reply with JSON only.")

    def __init__(self, cfg: CompilerConfig, backend: str | None = None,
                 openrouter_client: Any = None):
        # Writing five questions needs far less reasoning than exhaustive
        # extraction; medium keeps the passages careful at a fraction of the cost.
        super().__init__(cfg.model, env="TACT_PROBE", effort="medium", max_tokens=32_000,
                         backend=backend, openrouter_client=openrouter_client)

    def write(self, doc: Document, m: int) -> dict[str, Any]:
        if self.backend == "fake":
            self.n_calls += 1
            return {"raw": _fake_write(doc, m), "m": m, "extracted_with": "fake"}
        raw = self.ask((PROMPT_DIR / "probes.md").read_text().format(m=m, document=doc.text))
        return {"raw": raw, "m": m, "extracted_with": self.provenance}


def _fake_write(doc: Document, m: int) -> dict[str, Any]:
    """m sentences spread evenly over the document, each asked about verbatim.

    The question repeats the sentence's opening words, so the fake reader --
    which answers with the best-overlapping sentence in its context -- answers
    it exactly when the sentence survived compilation.  Only sentences that
    occur once are used: the fixture repeats filler, and a repeated passage
    anchors at its first occurrence (real documents repeat none -- 0 of 3,230
    sentences in local_economy).
    """
    import re

    found = [s.group().strip() for s in re.finditer(r"[^.!?]+[.!?]+", doc.text)]
    sentences = [s for s in found if found.count(s) == 1]
    n = len(sentences)
    picks = [sentences[min(n - 1, int((i + 0.5) * n / m))] for i in range(m)] if n else []
    return {"probes": [
        {"question": "What does the document say about " + " ".join(s.split()[:6]) + "?",
         "answer": s, "verbatim": [s], "kind": "other",
         "relational": any(w in s.lower() for w in ("because", "so the", "result"))}
        for s in picks
    ]}


# -- on disk ---------------------------------------------------------------------------


def probe_dir(profile: Profile, topic: str) -> Path:
    return resolve(profile.results_dir) / topic / "self_probes"


def load_probes(profile: Profile, topic: str) -> list[Probe] | None:
    path = probe_dir(profile, topic) / "probes.jsonl"
    if not path.exists():
        return None
    return [Probe(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def build_probes(
    profile: Profile,
    topic: str,
    writer: ProbeWriter,
    *,
    m: int = 5,
    workers: int = 8,
) -> tuple[list[Probe], dict[str, Any]]:
    documents = load_topic(profile, topic).documents
    d = probe_dir(profile, topic)
    cache = d / "write"
    cache.mkdir(parents=True, exist_ok=True)

    def one(doc: Document) -> dict[str, Any]:
        path = cache / f"{doc.document_id}.m{m}.json"
        if path.exists():
            return json.loads(path.read_text())
        out = writer.write(doc, m)
        path.write_text(json.dumps(out, indent=1) + "\n")
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        written = list(pool.map(one, documents))

    probes: list[Probe] = []
    totals: Counter[str] = Counter()
    quartiles = [0, 0, 0, 0]
    for doc, w in zip(documents, written):
        got, stats = probes_from_raw(doc.document_id, doc.text, w["raw"])
        probes += got
        totals.update(stats)
        for p in got:
            quartiles[min(3, int(4 * p.spans[0][0] / max(1, len(doc.text))))] += 1
    meta = {
        "topic": topic,
        "m": m,
        "n_probes": len(probes),
        **dict(totals),
        "rejection_rate": totals["n_rejected"] / totals["n_raw"] if totals["n_raw"] else 0.0,
        "n_relational": sum(p.relational for p in probes),
        "kinds": dict(sorted(Counter(p.kind for p in probes).items())),
        # Where in its document each probe's passage starts, by quarter: a
        # writer that ignores "spread them" piles probes into the first one.
        "position_quartiles": quartiles,
        "extracted_with": dict(Counter(w.get("extracted_with", "unknown") for w in written)),
        "usage_this_run": writer.usage_summary(),
    }
    (d / "probes.jsonl").write_text("".join(json.dumps(p.to_dict()) + "\n" for p in probes))
    (d / "probes_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return probes, meta
