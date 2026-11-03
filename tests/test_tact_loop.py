"""TACT contract-enforced refinement loop (draft sections 3.5.1 and 3.5.3)."""

from __future__ import annotations

import difflib
import json

import pytest

from wicer_core.judge import Judge
from wicer_core.reader import ReaderClient
from wicer_core.records import wiki_paths
from tact.ledger import Claim, Extractor, build_ledger
from tact.loop import eviction_order, loop_dir, run_loop
from tact.probes import ProbeWriter, build_probes
from tact.render import render_topic, sidecar_path


def _c(id, doc, pos):
    return Claim(id=id, doc=doc, e="E", r="r", v="v", verbatim=["."], spans=[[pos, pos + 1]])


def test_eviction_takes_the_pinned_documents_lowest_ranked_rows_first_and_never_a_pin():
    selected = [_c("a#1", "a", 0), _c("a#2", "a", 1), _c("a#3", "a", 2),
                _c("b#1", "b", 0), _c("b#2", "b", 1)]
    rank = {"a#1": 0, "a#2": 1, "a#3": 2, "b#1": 0, "b#2": 1}
    order = eviction_order(selected, pinned={"a#3"}, touched={"a"}, rank=rank)
    assert order == ["a#2", "a#1", "b#2", "b#1"]


@pytest.fixture
def prepared(profile, topic):
    reader = ReaderClient(profile.reader)
    build_ledger(profile, topic, Extractor(profile.compiler, backend="fake"))
    build_probes(profile, topic, ProbeWriter(profile.compiler, backend="fake"), m=4)
    render_topic(profile, topic, name="tact_iter0", rho=0.1, count_tokens=reader.count_tokens)
    render_topic(profile, topic, name="tact_full", rho=0.0, count_tokens=reader.count_tokens)
    render_topic(profile, topic, name="tact_sparse", rho=0.7, count_tokens=reader.count_tokens)
    return reader


def _run(profile, topic, reader, start="tact_iter0", **kw):
    return run_loop(profile, topic, reader=reader, judge=Judge(profile.judge),
                    extractor=Extractor(profile.compiler, backend="fake"), start=start, **kw)


def test_loop_repairs_append_only_and_keeps_the_contract(profile, topic, prepared):
    # The fixture's documents are one paragraph each, so paragraph closure
    # would pin whole documents and exhaust the headroom at once; without
    # closure the pins fit, and append-only repair is what is under test.
    res = _run(profile, topic, prepared, start="tact_sparse", max_iter=3, closure="none")
    recs = res.iterations
    assert any(r["rows_added"] > 0 and r["rows_evicted"] == 0 for r in recs[:-1])
    assert recs[0]["n_failures"] > 0
    assert recs[0]["diagnoses"].get("MISS-RENDER", 0) > 0 and recs[0]["rows_added"] > 0
    assert all(r["contract_ok"] for r in recs)

    texts = [wiki_paths(profile, topic, r["wiki_name"])[0].read_text() for r in recs]
    for r, before, after in zip(recs, texts, texts[1:]):
        if r["rows_evicted"] == 0:
            removed = [l for l in difflib.ndiff(before.splitlines(), after.splitlines())
                       if l.startswith("- ")]
            assert removed == []                       # append-only while headroom lasts
        assert r["tokens_after_repair"] <= res.budget_tokens

    final = wiki_paths(profile, topic, "tact_loop")[0].read_text()
    assert final == texts[-1]                          # the last evaluated wiki, not a best-of
    saved = json.loads((loop_dir(profile, topic) / "loop.json").read_text())
    assert saved["final_wiki"] == recs[-1]["wiki_name"]
    assert set(saved["pinned_docs"]) == {i.split("#")[0] for i in saved["pinned"]}
    assert (loop_dir(profile, topic) / "iter0_probes.jsonl").exists()


def test_without_headroom_pins_evict_unpinned_rows_and_stay_under_budget(profile, topic, prepared):
    res = _run(profile, topic, prepared, start="tact_full", max_iter=2)
    first = res.iterations[0]
    assert first["rows_added"] > 0 and first["rows_evicted"] > 0
    assert first["tokens_after_repair"] <= res.budget_tokens
    assert first["contract_ok"]
    # every claim that left the wiki was unpinned or refused, never an admitted pin
    final_side = json.loads(sidecar_path(profile, topic, res.final_wiki).read_text())
    assert set(res.pinned) <= {i for row in final_side for i in row["claims"]}


def test_loop_refuses_a_start_wiki_that_does_not_match_its_sidecar(profile, topic, prepared):
    md = wiki_paths(profile, topic, "tact_iter0")[0]
    md.write_text(md.read_text() + "- an extra row nobody rendered\n")
    with pytest.raises(ValueError, match="sidecar"):
        _run(profile, topic, prepared, max_iter=1)


def test_pins_beyond_the_budget_are_refused_newest_extras_first():
    from tact.loop import admit

    cost = {f"x{i}": 10 for i in range(10)}
    selected = {f"x{i}" for i in range(10)}
    kept, refused = admit(selected, old_pins={"x0", "x1"},
                          new_order=["x2", "x3", "x4", "x5"], tokens=100, budget=70,
                          cost=cost)
    assert refused == ["x5", "x4", "x3"]
    assert {"x0", "x1", "x2"} <= kept and not kept & set(refused)


def test_a_repair_claim_the_ledger_already_has_pins_the_existing_one():
    from tact.loop import _Ledger
    from wicer_core.corpus import Document

    text = "The mill opened in 1904. It employed 120 workers."
    old = Claim(id="d#1", doc="d", e="Mill", r="workers", v="120",
                verbatim=["It employed 120 workers."], spans=[[25, 49]])
    led = _Ledger([old], [Document("d", text)])
    raw = {"claims": [
        {"id": "c1", "entity": "Mill", "relation": "workers", "value": "120",
         "verbatim": ["It employed 120 workers."]},
        {"id": "c2", "entity": "Mill", "relation": "opened_in", "value": "1904",
         "verbatim": ["The mill opened in 1904."]},
    ]}
    new, matched = led.add("d", raw, prefix="i0p1.")
    assert [c.id for c in new] == ["d#i0p1.2"] and matched == ["d#1"]
    again_new, again_matched = led.add("d", raw, prefix="i1p1.")
    assert again_new == [] and set(again_matched) == {"d#1", "d#i0p1.2"}
