"""RepLiQA subsampling: determinism, the unanswerable filter, rendering."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from wicer_core.config import TOPICS
from wicer_core.corpus import (
    Document,
    _build_from_frame,
    _is_unanswerable,
    format_documents,
    load_topic,
)


def _frame(
    n_docs: int = 5, n_qa: int = 6, topic: str = "Company Policies",
    length_of=lambda d: 20,
) -> pd.DataFrame:
    rows = []
    for d in range(n_docs):
        doc_id = f"doc{d:03d}"
        for q in range(n_qa):
            rows.append({
                "document_id": doc_id,
                "document_topic": topic,
                "document_extracted": f"Body of {doc_id}. " * length_of(d),
                "question_id": f"{doc_id}-q{q}",
                "question": f"question {q} about {doc_id}?",
                "answer": f"answer {q} for {doc_id}",
                "long_answer": "",
            })
    return pd.DataFrame(rows)


def test_subsample_takes_first_n_docs_and_k_qa(profile, tmp_path):
    df = _frame(n_docs=20, n_qa=6)
    corpus = _build_from_frame(profile, "company_policies", df, tmp_path / "cp")
    assert len(corpus.documents) == profile.corpus.n_docs_per_topic
    assert len(corpus.qa) == profile.corpus.n_docs_per_topic * profile.corpus.n_qa_per_doc
    # Lexicographically first document ids, and lowest question ids within each.
    assert [d.document_id for d in corpus.documents] == [
        f"doc{i:03d}" for i in range(profile.corpus.n_docs_per_topic)
    ]
    assert {q.question_id for q in corpus.qa if q.document_id == "doc000"} == {
        f"doc000-q{i}" for i in range(profile.corpus.n_qa_per_doc)
    }


def test_subsample_is_deterministic_across_builds(profile, tmp_path):
    df = _frame(n_docs=20, n_qa=6)
    a = _build_from_frame(profile, "company_policies", df, tmp_path / "a")
    b = _build_from_frame(profile, "company_policies", df.sample(frac=1, random_state=7), tmp_path / "b")
    # Row order in the source frame must not change the corpus.
    assert a.manifest["content_sha256"] == b.manifest["content_sha256"]


def test_unanswerable_rows_are_dropped_and_counted(profile, tmp_path):
    df = _frame(n_docs=3, n_qa=6)
    df.loc[df["question_id"].str.endswith("-q0"), "answer"] = "UNANSWERABLE"
    corpus = _build_from_frame(profile, "company_policies", df, tmp_path / "u")
    assert corpus.manifest["dropped_unanswerable"] == 3
    assert all("q0" not in q.question_id for q in corpus.qa)


def test_documents_with_too_few_qa_are_recorded_not_silently_short(profile, tmp_path):
    df = _frame(n_docs=2, n_qa=3)  # fewer QA than n_qa_per_doc
    corpus = _build_from_frame(profile, "company_policies", df, tmp_path / "s")
    assert set(corpus.manifest["documents_with_fewer_qa"]) == {"doc000", "doc001"}


@pytest.mark.parametrize("answer,expected", [
    ("UNANSWERABLE", True), ("unanswerable", True), ("", True), ("  ", True),
    ("n/a", True), ("1890", False), ("The mayor was Ilse Hallam.", False),
])
def test_unanswerable_detection(answer, expected):
    assert _is_unanswerable(answer) is expected


def test_format_documents_marks_every_document():
    docs = [Document("a", "first"), Document("b", "second")]
    rendered = format_documents(docs)
    assert "DOCUMENT 1 (id: a)" in rendered
    assert "DOCUMENT 2 (id: b)" in rendered
    assert rendered.count("=== DOCUMENT") == 2


def test_topic_slugs_cover_the_papers_seventeen_domains():
    assert len(TOPICS) == 17
    assert TOPICS["regional_cuisine"] == "Regional Cuisine and Recipes"
    assert TOPICS["local_education"] == "Local Education Systems"


def test_fixture_topic_loads_with_one_document_per_id(profile, topic):
    corpus = load_topic(profile, topic)
    assert len(corpus.documents) == 12
    assert len(corpus.qa) == 60
    assert len({d.document_id for d in corpus.documents}) == 12
    assert corpus.doc_by_id("mini00").text.startswith("Ardenmoor")
    with pytest.raises(KeyError):
        corpus.doc_by_id("nope")


# -- document selection ------------------------------------------------------
#
# The paper subsamples 80 of a topic's ~200 documents and never says which 80.
# Its reported corpus sizes are only reachable by a length-biased selection --
# see DEVIATIONS.md 2 -- so the rule is explicit and recorded, not implied.


def _selection(profile, rule):
    from dataclasses import replace
    return replace(profile, corpus=replace(profile.corpus, doc_selection=rule))


def test_shortest_selection_takes_the_smallest_documents(profile, tmp_path):
    """doc000 is longest, doc019 shortest; the keepers are the short tail."""
    df = _frame(n_docs=20, n_qa=6, length_of=lambda d: 100 - d)
    n = profile.corpus.n_docs_per_topic
    corpus = _build_from_frame(_selection(profile, "shortest"), "company_policies", df, tmp_path / "s")
    assert [d.document_id for d in corpus.documents] == [
        f"doc{i:03d}" for i in range(20 - n, 20)
    ]


def test_selected_documents_are_ordered_lexicographically(profile, tmp_path):
    """Selection picks WHICH documents; order stays id-sorted so the rendered
    context -- and the reader's KV prefix -- does not depend on the rule."""
    df = _frame(n_docs=20, n_qa=6, length_of=lambda d: 100 - d)
    corpus = _build_from_frame(_selection(profile, "shortest"), "company_policies", df, tmp_path / "s")
    ids = [d.document_id for d in corpus.documents]
    assert ids == sorted(ids)


def test_equal_length_documents_fall_back_to_document_id(profile, tmp_path):
    """Ties must break deterministically or two machines build different corpora."""
    df = _frame(n_docs=20, n_qa=6)  # every document identical in length
    n = profile.corpus.n_docs_per_topic
    corpus = _build_from_frame(_selection(profile, "shortest"), "company_policies", df, tmp_path / "s")
    assert [d.document_id for d in corpus.documents] == [f"doc{i:03d}" for i in range(n)]


def test_lexicographic_selection_is_still_available(profile, tmp_path):
    df = _frame(n_docs=20, n_qa=6, length_of=lambda d: 100 - d)
    n = profile.corpus.n_docs_per_topic
    corpus = _build_from_frame(_selection(profile, "lexicographic"), "company_policies", df, tmp_path / "l")
    assert [d.document_id for d in corpus.documents] == [f"doc{i:03d}" for i in range(n)]


def test_an_unknown_selection_rule_is_rejected(profile, tmp_path):
    df = _frame(n_docs=20, n_qa=6)
    with pytest.raises(ValueError, match="doc_selection"):
        _build_from_frame(_selection(profile, "longest"), "company_policies", df, tmp_path / "x")


def test_manifest_records_which_rule_built_it(profile, tmp_path):
    """A corpus that does not say how it was subsampled cannot be compared to
    another one."""
    df = _frame(n_docs=20, n_qa=6, length_of=lambda d: 100 - d)
    _build_from_frame(_selection(profile, "shortest"), "company_policies", df, tmp_path / "s")
    manifest = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert manifest["doc_selection"] == "shortest"
    assert "shortest" in manifest["subsample_rule"]
