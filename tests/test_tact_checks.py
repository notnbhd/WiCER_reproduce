"""TACT ledger inspection: judge-free checks run between extraction and scoring."""

from __future__ import annotations

import json

from wicer_core.corpus import Document, QAPair
from tact import run
from tact.checks import answer_coverage, budget_fit, ledger_shape
from tact.ledger import Claim, ledger_dir

TEXT = ("The mill opened in 1904. It employed 120 workers. "
        "A flood in 1921 damaged 17 bridges. The festival moved to spring in 1950.")


def words(text: str) -> int:
    return len(text.split())


def _claim(id, doc, e, r, v, span, *, tau=None, ref=False):
    s, e_ = span
    return Claim(id=id, doc=doc, e=e, r=r, v=v, ref=ref, tau=tau,
                 verbatim=[TEXT[s:e_]], spans=[[s, e_]])


def _q(qid, passage, doc="d"):
    return QAPair(question_id=qid, document_id=doc, question="?", answer="x", long_answer=passage)


def test_answer_coverage_finds_claims_behind_each_gold_passage():
    mill = TEXT.index("It employed")
    end = mill + len("It employed 120 workers.")
    claims = [_claim("d#1", "d", "Mill", "workers", "120", (mill, end))]
    docs = {"d": Document("d", TEXT)}
    qs = [
        _q("covered", "It employed 120 workers."),
        _q("partly", "The mill opened in 1904. It employed 120 workers."),
        _q("missed", "A flood in 1921 damaged 17 bridges."),
        _q("unlocated", "Nothing like this sentence is anywhere in the source text."),
    ]
    out = answer_coverage(claims, docs, qs)
    rows = {r["question_id"]: r for r in out["questions"]}
    assert rows["covered"]["claims"] == ["d#1"] and rows["covered"]["passage_covered"] == 1.0
    assert 0.4 < rows["partly"]["passage_covered"] < 0.6
    assert rows["missed"]["claims"] == [] and rows["missed"]["passage_covered"] == 0.0
    assert rows["unlocated"]["located"] is False
    assert out["n_questions"] == 4 and out["n_located"] == 3
    assert out["share_with_any_claim"] == 2 / 3          # over located passages
    assert out["mean_k"] == 1.0                          # |A(q)| over covered questions


def test_budget_fit_flags_documents_whose_whole_ledger_fits_its_share():
    small = Document("a", " ".join(["w"] * 200))
    big = Document("b", " ".join(["w"] * 200))
    claims = [_claim("a#1", "a", "A", "r", "v", (0, 1))]
    claims += [Claim(id=f"b#{i}", doc="b", e=f"B{i}", r="has", v=" ".join(["x"] * 20),
                     verbatim=["x"], spans=[[i, i + 1]]) for i in range(20)]
    out = budget_fit(claims, [small, big], ratio=0.5, rho=0.1, count_tokens=words)
    rows = {r["doc"]: r for r in out["documents"]}
    assert rows["a"]["fits"] and not rows["b"]["fits"]
    assert rows["b"]["ledger_tokens"] > rows["b"]["share_tokens"]
    assert out["n_fit"] == 1 and out["ledger_tokens"] == sum(r["ledger_tokens"] for r in rows.values())


def test_ledger_shape_counts_relations_and_single_relation_entities():
    claims = [
        _claim("d#1", "d", "Mill", "opened", "1904", (0, 1), tau="1904"),
        _claim("d#2", "d", "Mill", "workers", "120", (1, 2)),
        _claim("d#3", "d", "Flood", "damaged", "17 bridges", (2, 3)),
        _claim("d#4", "d", "Flood", "results_in", "d#2", (3, 4), ref=True),
        _claim("d#5", "d", "Festival", "moved_to", "spring", (4, 5)),
    ]
    s = ledger_shape(claims)
    assert s["n_relations"] == 5 and s["n_entities"] == 3
    assert s["single_relation_entity_share"] == 1 / 3
    assert s["routing"] == {"T2": 1, "T3a": 2, "T3b": 2}
    assert s["n_relational"] == 1


def test_inspect_command_writes_a_report(profile, topic):
    argv = ["--profile", "test", "--topic", topic]
    assert run.main(["extract", *argv], profile=profile) == 0
    assert run.main(["inspect", *argv], profile=profile) == 0
    report = json.loads((ledger_dir(profile, topic) / "inspect.json").read_text())
    assert {"answers", "budget", "shape", "tokens"} <= set(report)
    assert report["answers"]["n_questions"] > 0
