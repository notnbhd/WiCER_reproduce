"""TACT self-generated probes (draft section 3.2): span-grounded, A(q) by overlap."""

from __future__ import annotations

import json

from tact.ledger import Claim
from tact.probes import (
    ProbeWriter,
    answer_set,
    build_probes,
    load_probes,
    probe_dir,
    probes_from_raw,
)

TEXT = ("The mill opened in 1904. It employed 120 workers. "
        "A flood in 1921 damaged 17 bridges, so the mill closed. The festival began in 1950.")


def _claim(id, s, e):
    return Claim(id=id, doc="d", e="X", r="r", v="v", verbatim=[TEXT[s:e]], spans=[[s, e]])


def test_probes_from_raw_validates_spans_and_normalises_fields():
    raw = {"probes": [
        {"question": "When did the mill open?", "answer": "1904",
         "verbatim": ["The mill opened in 1904."], "kind": "date", "relational": False},
        {"question": "Why did the mill close?", "answer": "A flood damaged 17 bridges",
         "verbatim": ["A flood in 1921 damaged 17 bridges, so the mill closed."],
         "kind": "reason", "relational": True},
        {"question": "Who owned it?", "answer": "Nobody",
         "verbatim": ["The owner was a reclusive banker from out of town."], "kind": "person"},
        {"question": "How big?", "answer": "big", "verbatim": ["It employed 120 workers."],
         "kind": "size-ish", "relational": "yes"},
    ]}
    probes, stats = probes_from_raw("d", TEXT, raw)
    assert stats == {"n_raw": 4, "n_rejected": 1}
    assert [p.id for p in probes] == ["d#p1", "d#p2", "d#p4"]
    assert probes[1].relational is True and probes[1].kind == "reason"
    assert probes[2].kind == "other" and probes[2].relational is True
    for p in probes:
        for (s, e), v in zip(p.spans, p.verbatim):
            assert TEXT[s:e] == v


def test_answer_set_is_the_claims_overlapping_the_probe_span():
    probes, _ = probes_from_raw("d", TEXT, {"probes": [
        {"question": "Why?", "answer": "flood", "kind": "reason", "relational": True,
         "verbatim": ["A flood in 1921 damaged 17 bridges, so the mill closed."]}]})
    flood = TEXT.index("A flood")
    claims = [_claim("d#1", 0, 24), _claim("d#2", flood, flood + 34),
              _claim("d#3", flood + 36, flood + 56)]
    assert answer_set(probes[0], claims) == ["d#2", "d#3"]


def test_build_probes_writes_caches_and_spreads(profile, topic):
    writer = ProbeWriter(profile.compiler, backend="fake")
    probes, meta = build_probes(profile, topic, writer, m=4)
    docs = {p.doc for p in probes}
    assert all(sum(p.doc == d for p in probes) == 4 for d in docs)
    assert meta["n_probes"] == len(probes) and meta["extracted_with"] == {"fake": len(docs)}
    assert sum(meta["position_quartiles"]) == len(probes) and min(meta["position_quartiles"]) > 0
    rows = (probe_dir(profile, topic) / "probes.jsonl").read_text().splitlines()
    assert len(rows) == len(probes)

    again = ProbeWriter(profile.compiler, backend="fake")
    reloaded, _ = build_probes(profile, topic, again, m=4)
    assert again.n_calls == 0 and [p.id for p in reloaded] == [p.id for p in probes]
    assert [p.id for p in load_probes(profile, topic)] == [p.id for p in probes]
    assert json.loads(rows[0])["question"]
