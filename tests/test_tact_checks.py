"""TACT ledger inspection: judge-free checks run between extraction and scoring."""

from __future__ import annotations

import json

import pytest

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


def test_context_attribution_splits_questions_by_where_their_gold_went():
    from tact.checks import context_attribution

    text = ("The mill opened in 1904. It employed 120 workers. "
            "A flood in 1921 damaged 17 bridges. The festival moved to spring in 1950. "
            "Nobody wrote this down.")
    docs = {"d": Document("d", text)}

    def at(sentence):
        s = text.index(sentence)
        return Claim(id=f"d#{s}", doc="d", e="E", r="r", v="v",
                     verbatim=[sentence], spans=[[s, s + len(sentence)]])

    claims = [at("The mill opened in 1904."), at("It employed 120 workers."),
              at("A flood in 1921 damaged 17 bridges.")]
    context = "# Wiki\n\nThe mill  opened in 1904.\nIt employed 120\nworkers.\n"
    qs = [
        _q("full", "It employed 120 workers."),
        _q("partly", "It employed 120 workers. A flood in 1921 damaged 17 bridges."),
        _q("dropped", "A flood in 1921 damaged 17 bridges."),
        _q("unclaimed", "Nobody wrote this down."),
        _q("unlocated", "Nothing like this sentence is anywhere in the source text."),
        _q("unscored", "The mill opened in 1904."),
    ]
    scores = {"full": 5, "partly": 4, "dropped": 1, "unclaimed": 2, "unlocated": 3}
    baseline = {"full": 5, "partly": 5, "dropped": 4, "unclaimed": 4, "unlocated": 3}

    out = context_attribution(claims, docs, qs, context, scores, baseline)
    rows = {r["question_id"]: r["bucket"] for r in out["questions"]}
    assert rows == {"full": "in_context", "partly": "partly_in_context",
                    "dropped": "extracted_not_selected", "unclaimed": "not_extracted",
                    "unlocated": "unlocated"}
    assert out["n_questions"] == 5 and out["n_unscored"] == 1
    b = out["buckets"]
    assert b["extracted_not_selected"]["n"] == 1 and b["extracted_not_selected"]["mean_score"] == 1
    # gap = baseline - score: 0 + 1 + 3 + 2 + 0 = 6
    assert b["extracted_not_selected"]["gap_share"] == 0.5
    assert b["not_extracted"]["gap_share"] == pytest.approx(2 / 6)
    # Missing gold sentences: the flood twice (claimed), "Nobody" once (unclaimed).
    assert out["missing_gold_sentences"] == {"extracted": 2, "not_extracted": 1}
    assert out["sentences_in_context"] == 2 / 5
    assert out["sentences_extracted"] == 3 / 5


def test_context_attribution_without_a_baseline_reports_no_gap():
    from tact.checks import context_attribution

    docs = {"d": Document("d", TEXT)}
    out = context_attribution([], docs, [_q("q", "It employed 120 workers.")],
                              "nothing", {"q": 3})
    assert out["buckets"]["not_extracted"]["gap_share"] is None
    assert out["buckets"]["not_extracted"]["mean_baseline"] is None


def test_inspect_with_a_condition_attributes_its_scores(profile, topic):
    argv = ["--profile", "test", "--topic", topic]
    assert run.main(["extract", *argv], profile=profile) == 0
    assert run.main(["truncate", *argv], profile=profile) == 0
    assert run.main(["score", *argv, "--condition", "truncate"], profile=profile) == 0
    assert run.main(["inspect", *argv, "--condition", "truncate"], profile=profile) == 0
    report = json.loads((ledger_dir(profile, topic) / "inspect.json").read_text())
    att = report["attribution"]
    assert att["condition"] == "truncate" and att["n_questions"] > 0
    assert sum(b["n"] for b in att["buckets"].values()) == att["n_questions"]
