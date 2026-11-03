"""TACT claim ledger (draft section 3.3): span validation, canonicalization, merging."""

from __future__ import annotations

import json

from tact.ledger import (
    Claim,
    Extractor,
    build_ledger,
    canonical_map,
    claims_from_extraction,
    ledger_dir,
    load_ledger,
    locate,
    merge,
)

DOC = (
    "Joe Harris opened the Riverside Market in March 2021. "
    "The market employs 40 people.  According to Quinn, AR eliminates misunderstandings. "
    "Sales rose 12% in 2022, resulting in a second location."
)


def _claim(id, doc, e, r, v, *, tau=None, ref=False, pos=0):
    return Claim(id=id, doc=doc, e=e, r=r, v=v, ref=ref, tau=tau,
                 verbatim=[v], spans=[[pos, pos + len(v)]])


# -- span validation -----------------------------------------------------------


def test_locate_exact():
    v = "The market employs 40 people."
    start, end, score = locate(v, DOC)
    assert DOC[start:end] == v and score == 1.0


def test_locate_tolerates_case_and_whitespace():
    start, end, _ = locate("the market  EMPLOYS 40 people.", DOC)
    assert DOC[start:end] == "The market employs 40 people."


def test_locate_accepts_a_near_copy():
    # one word of nine differs: a copy slip, not a paraphrase
    hit = locate("Joe Harris opened the Riverside Market in May 2021.", DOC)
    assert hit is not None
    assert DOC[hit[0]:hit[1]].startswith("Joe Harris opened")


def test_locate_rejects_a_paraphrase():
    assert locate("Harris launched a waterfront shop two years ago.", DOC) is None


# -- canonicalization ----------------------------------------------------------


def test_containment_variants_collapse_to_the_head_term():
    m = canonical_map({"Joe Harris": 5, "Joe Harris company": 1, "Joe Harris work": 1})
    assert m == {"Joe Harris": "Joe Harris", "Joe Harris company": "Joe Harris",
                 "Joe Harris work": "Joe Harris"}


def test_ambiguous_short_name_links_nothing():
    m = canonical_map({"Harris": 2, "Joe Harris": 1, "Mary Harris": 1})
    assert m["Joe Harris"] != m["Mary Harris"]
    assert m["Harris"] == "Harris"


def test_containment_is_on_whole_words():
    m = canonical_map({"Art": 1, "Artisan Guild": 1})
    assert m["Artisan Guild"] == "Artisan Guild"


# -- provenance-aware merging (the table in 3.3) ---------------------------------


def test_cross_document_agreement_merges_and_keeps_both_sources():
    a = _claim("d1#1", "d1", "Market", "employees", "40")
    b = _claim("d2#1", "d2", "market", "employees", "40.")
    claims, contradictions = merge([a, b])
    assert len(claims) == 1 and claims[0].sources == ["d1", "d2"]
    assert contradictions == []


def test_cross_document_disagreement_keeps_both_and_records_it():
    a = _claim("d1#1", "d1", "Market", "employees", "40")
    b = _claim("d2#1", "d2", "Market", "employees", "55")
    claims, contradictions = merge([a, b])
    assert {c.id for c in claims} == {"d1#1", "d2#1"}
    assert len(contradictions) == 1 and set(contradictions[0]["claims"]) == {"d1#1", "d2#1"}


def test_same_document_paraphrase_duplicate_merges():
    a = _claim("d1#1", "d1", "Market", "employees", "40 people")
    b = _claim("d1#2", "d1", "Market", "employees", "40 people")
    claims, _ = merge([a, b])
    assert [c.id for c in claims] == ["d1#1"]


def test_same_document_disagreement_is_kept_without_a_record():
    a = _claim("d1#1", "d1", "Market", "employees", "40")
    b = _claim("d1#2", "d1", "Market", "employees", "55")
    claims, contradictions = merge([a, b])
    assert len(claims) == 2 and contradictions == []


def test_different_relations_or_times_never_merge():
    a = _claim("d1#1", "d1", "Market", "employees", "40", tau="2021")
    b = _claim("d2#1", "d2", "Market", "employees", "40", tau="2022")
    c = _claim("d2#2", "d2", "Market", "branches", "40", tau="2021")
    claims, _ = merge([a, b, c])
    assert len(claims) == 3


def test_references_to_a_merged_claim_are_rewritten():
    a = _claim("d1#1", "d1", "Market", "employees", "40")
    b = _claim("d1#2", "d1", "Market", "employees", "40")
    rel = _claim("d1#3", "d1", "Market", "results_in", "d1#2", ref=True)
    claims, _ = merge([a, b, rel])
    assert next(c for c in claims if c.id == "d1#3").v == "d1#1"


# -- extraction output -> claims -------------------------------------------------


def test_claims_from_extraction_validates_spans_and_maps_entities():
    raw = {"claims": [
        {"id": "c1", "entity": "Joe Harris company", "relation": "opened",
         "value": "Riverside Market", "time": "March 2021",
         "verbatim": ["Joe Harris opened the Riverside Market in March 2021."]},
        {"id": "c2", "entity": "AR", "relation": "eliminates", "value": "misunderstandings",
         "speaker": "Quinn", "verbatim": ["According to Quinn, AR eliminates misunderstandings."]},
        {"id": "c3", "entity": "Market", "relation": "revenue", "value": "doubled",
         "verbatim": ["Revenue doubled after the merger with a rival chain."]},
        {"id": "c4", "entity": "Sales", "relation": "results_in", "value_ref": "c1",
         "verbatim": ["Sales rose 12% in 2022,", "resulting in a second location."]},
    ]}
    claims, stats = claims_from_extraction("d1", DOC, raw, {"Joe Harris company": "Joe Harris"})
    by = {c.id: c for c in claims}
    assert stats == {"n_raw": 4, "n_rejected": 1, "n_multispan": 1, "n_relational": 1}
    assert by["d1#1"].e == "Joe Harris" and by["d1#1"].tau == "March 2021"
    assert by["d1#2"].src == "Quinn"
    assert "d1#3" not in by
    rel = by["d1#4"]
    assert rel.ref and rel.v == "d1#1" and rel.multi_span and len(rel.spans) == 2
    for c in claims:
        for (s, e), v in zip(c.spans, c.verbatim):
            assert DOC[s:e].lower() == v.lower()


def test_reference_to_a_rejected_claim_drops_the_relation():
    raw = {"claims": [
        {"id": "c1", "entity": "X", "relation": "r", "value": "v",
         "verbatim": ["nowhere in the document at all, not even close"]},
        {"id": "c2", "entity": "Sales", "relation": "results_in", "value_ref": "c1",
         "verbatim": ["resulting in a second location."]},
    ]}
    claims, stats = claims_from_extraction("d1", DOC, raw, {})
    assert claims == [] and stats["n_rejected"] == 2


# -- end to end on the fixture ---------------------------------------------------


def test_build_ledger_writes_and_caches(profile, topic):
    ex = Extractor(profile.compiler, backend="fake")
    ledger = build_ledger(profile, topic, ex)
    assert ledger.claims and ex.n_calls > 0
    d = ledger_dir(profile, topic)
    rows = [json.loads(l) for l in (d / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == len(ledger.claims)
    meta = json.loads((d / "ledger_meta.json").read_text())
    assert meta["n_claims"] == len(ledger.claims) and 0 <= meta["rejection_rate"] <= 1

    n_docs = meta["n_documents"]
    assert meta["extracted_with"] == {"fake": n_docs}

    again = Extractor(profile.compiler, backend="fake")
    reloaded = build_ledger(profile, topic, again)
    assert again.n_calls == 0
    # a fully cached run still says what the ledger was extracted with
    assert reloaded.meta["extracted_with"] == {"fake": n_docs}
    assert reloaded.meta["usage_this_run"]["calls"] == 0
    assert [c.id for c in reloaded.claims] == [c.id for c in ledger.claims]
    assert [c.id for c in load_ledger(profile, topic).claims] == [c.id for c in ledger.claims]
