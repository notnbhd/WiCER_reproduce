"""TACT typed rendering (draft section 3.4): routing, ranking, budget, sidecar."""

from __future__ import annotations

import difflib
import json

from wicer_core.corpus import Document
from tact.ledger import Claim, Extractor, build_ledger
from tact.render import fill, layout, render_topic, round_robin, route, sidecar_path


def words(text: str) -> int:
    return len(text.split())


def _claim(id, doc, e, r, v, *, tau=None, pos=0, ref=False, src=None):
    return Claim(id=id, doc=doc, e=e, r=r, v=v, ref=ref, tau=tau, src=src,
                 verbatim=[v], spans=[[pos, pos + 1]])


# -- routing -----------------------------------------------------------------------


def test_routing_is_exhaustive_and_follows_the_rule():
    claims = [
        _claim("d#1", "d", "Mill", "opened", "1904", tau="1904"),
        _claim("d#2", "d", "Mill", "workers", "120"),
        _claim("d#3", "d", "Mill", "owner", "Aldwin"),
        _claim("d#4", "d", "Festival", "held", "every autumn"),
        _claim("d#5", "d", "Bridge", "count", "17"),
        _claim("d#6", "d", "Bridge", "count", "18"),
    ]
    r = route(claims)
    assert set(r) == {c.id for c in claims}
    assert r["d#1"] == "T2"
    assert r["d#2"] == r["d#3"] == "T3a"          # two distinct relations
    assert r["d#4"] == "T3b"                       # entity seen once
    assert r["d#5"] == r["d#6"] == "T3b"           # twice, but one relation


# -- ranking -------------------------------------------------------------------------


def test_round_robin_takes_one_claim_per_entity_per_round():
    claims = [
        _claim("d#1", "d", "A", "r1", "x", pos=0),
        _claim("d#2", "d", "A", "r2", "x", pos=10),
        _claim("d#3", "d", "A", "r3", "x", pos=20),
        _claim("d#4", "d", "B", "r1", "x", pos=5),
        _claim("d#5", "d", "C", "r1", "x", pos=30),
        _claim("d#6", "d", "B", "r2", "x", pos=40),
    ]
    assert [c.id for c in round_robin(claims)] == ["d#1", "d#4", "d#5", "d#2", "d#6", "d#3"]


# -- layout and sidecar -----------------------------------------------------------


def test_layout_renders_every_selected_claim_and_maps_it_back():
    claims = [
        _claim("d#1", "d", "Mill", "opened", "1904", tau="1904"),
        _claim("d#2", "d", "Mill", "workers", "120"),
        _claim("d#3", "d", "Mill", "owner", "Aldwin", src="the archive"),
        _claim("d#4", "d", "Festival", "held", "every autumn"),
        _claim("d#5", "d", "Festival", "results_in", "d#2", ref=True),
    ]
    text, sidecar = layout(claims, {c.id: c for c in claims})
    assert {i for row in sidecar for i in row["claims"]} == {c.id for c in claims}
    lines = text.splitlines()
    for row in sidecar:
        assert lines[row["line"]].startswith("- ")
    assert "Aldwin (per the archive)" in text
    assert "Mill — workers: 120" in text          # a relation names its target
    assert "d#" not in text                        # no identifiers reach the reader


def test_adding_a_claim_only_adds_lines():
    base = [
        _claim("d#1", "d", "Mill", "opened", "1904", tau="1904", pos=0),
        _claim("d#2", "d", "Mill", "workers", "120", pos=1),
        _claim("d#3", "d", "Mill", "owner", "Aldwin", pos=2),
        _claim("d#4", "d", "Festival", "held", "every autumn", pos=3),
        _claim("e#1", "e", "Port", "depth", "9 m", pos=0),
    ]
    by_id = {c.id: c for c in base}
    before, _ = layout(base[:-2] + base[-1:], by_id)
    after, _ = layout(base, by_id)
    ops = [l for l in difflib.ndiff(before.splitlines(), after.splitlines()) if l[:2] in ("- ", "+ ")]
    assert ops and all(op.startswith("+ ") for op in ops)


# -- budget ----------------------------------------------------------------------------


def _docs_and_claims():
    small = Document("a", " ".join(["word"] * 100))
    big = Document("b", " ".join(["word"] * 300))
    claims = [_claim(f"a#{i}", "a", f"E{i}", "has", "one two three four five six", pos=i)
              for i in range(30)]
    claims += [_claim(f"b#{i}", "b", f"F{i}", "has", "one two three four five six", pos=i)
               for i in range(5)]
    return [small, big], claims


def test_fill_lands_just_under_the_target():
    docs, claims = _docs_and_claims()
    res = fill(claims, docs, ratio=0.5, rho=0.0, count_tokens=words)
    target = res.meta["target_tokens"]
    assert res.meta["actual_tokens"] == words(res.text) <= target
    assert res.meta["actual_tokens"] >= target - 15   # within one row plus a header
    assert 0 < res.meta["coverage"] < 1


def test_shares_protect_the_sparse_document_and_leftovers_are_redistributed():
    docs, claims = _docs_and_claims()
    res = fill(claims, docs, ratio=0.5, rho=0.0, count_tokens=words)
    realized = res.realized
    assert {f"b#{i}" for i in range(5)} <= realized      # its share fits all of it
    n_a = len([i for i in realized if i.startswith("a#")])
    # a's own share (~1/4 of the budget) holds ~5 rows; b's unused share must go to a
    assert n_a > 8


def test_headroom_holds_back_rho_of_the_budget():
    docs, claims = _docs_and_claims()
    full = fill(claims, docs, ratio=0.5, rho=0.0, count_tokens=words)
    held = fill(claims, docs, ratio=0.5, rho=0.2, count_tokens=words)
    assert held.meta["target_tokens"] == round(0.8 * full.meta["budget_tokens"])
    assert held.meta["actual_tokens"] <= held.meta["target_tokens"]
    assert held.realized < full.realized


def test_routing_mix_is_reported():
    docs, claims = _docs_and_claims()
    res = fill(claims, docs, ratio=0.5, rho=0.0, count_tokens=words)
    mix = res.meta["routing"]
    assert sum(m["n_claims"] for m in mix.values()) == len(claims)
    assert sum(m["n_realized"] for m in mix.values()) == len(res.realized)


# -- end to end ------------------------------------------------------------------------


def test_render_topic_writes_wiki_meta_and_sidecar(profile, topic):
    build_ledger(profile, topic, Extractor(profile.compiler, backend="fake"))
    res = render_topic(profile, topic, name="tact_iter0", rho=0.1, count_tokens=words)
    from wicer_core.records import wiki_paths

    md, meta = wiki_paths(profile, topic, "tact_iter0")
    assert md.read_text() == res.text
    m = json.loads(meta.read_text())
    assert m["rho"] == 0.1 and m["actual_tokens"] <= m["target_tokens"]
    side = json.loads(sidecar_path(profile, topic, "tact_iter0").read_text())
    assert {i for row in side for i in row["claims"]} == res.realized


# -- truncation baseline -------------------------------------------------------------


def test_truncation_keeps_each_document_start_within_its_share():
    from tact.truncate import truncate

    docs = [Document("a", "One two three. Four five six. Seven eight nine. " * 4),
            Document("b", "Alpha beta gamma delta. Epsilon zeta eta theta. " * 12)]
    text, meta = truncate(docs, ratio=0.5, count_tokens=words)
    assert words(text) == meta["actual_tokens"] <= meta["budget_tokens"]
    assert meta["actual_tokens"] >= meta["budget_tokens"] - 8
    parts = text.split("=== DOCUMENT ")[1:]
    assert len(parts) == 2
    for doc, part in zip(docs, parts):
        body = part.split("===\n", 1)[1].strip()
        assert body and doc.text.startswith(body)
