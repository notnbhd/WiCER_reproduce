"""Paged cache: cuts marked in a verbatim context, and a reader that may ask for one back."""

from __future__ import annotations

import json

import pytest

from wicer_core.corpus import Document, QAPair, format_documents, load_topic
from wicer_core.reader import Generation, build_user_prompt
from wicer_core.records import answers_path, read_jsonl, scores_path
from tact import run
from tact.ledger import Claim, Extractor, build_ledger
from tact.paged import (
    build_doc_restored_prompt,
    build_fetch_prompt,
    build_restored_prompt,
    classify,
    load_cuts,
    page_report,
    pages_path,
    parse_fetch,
    parse_fetch_docs,
    run_paged,
)
from tact.render import fill_verbatim, layout_verbatim, render_topic, sentences, sidecar_path


def words(text: str) -> int:
    return len(text.split())


SENTS = ["The mill opened in 1904.", "It employed 120 workers.", "Aldwin owned it.",
         "The roof was red.", "The festival is held every autumn.", "Seventeen bridges cross it."]


def _doc(doc_id: str = "a", sents=SENTS) -> Document:
    return Document(doc_id, " ".join(sents))


def _claim_on(doc: Document, n: int, sentence: str, e: str = "Mill", r: str = "fact") -> Claim:
    a = doc.text.index(sentence)
    return Claim(id=f"{doc.document_id}#{n}", doc=doc.document_id, e=e, r=r, v="x",
                 verbatim=[sentence], spans=[[a, a + len(sentence)]])


def _layout(doc: Document, keep: set[int], claims=(), stubs: str = "bare"):
    sents = {doc.document_id: sentences(doc.text)}
    owners: dict[tuple[str, int], list[str]] = {}
    for c in claims:
        for i, (a, b) in enumerate(sents[c.doc]):
            if any(s < b and e > a for s, e in c.spans):
                owners.setdefault((c.doc, i), []).append(c.id)
    return layout_verbatim({doc.document_id: keep}, sents, [doc], {doc.document_id: "Mill Town"},
                           owners, stubs=stubs, claims={c.id: c for c in claims})


# -- stubs in the rendered context ------------------------------------------------------


def test_every_cut_is_marked_and_maps_back_to_the_source_text():
    doc = _doc()
    text, sidecar = _layout(doc, {0, 4})
    assert text.splitlines() == ["## Mill Town", "", SENTS[0], "[cut 1.1]", SENTS[4], "[cut 1.2]"]
    cuts = [r for r in sidecar if r["block"] == "cut"]
    assert [c["id"] for c in cuts] == ["1.1", "1.2"]
    assert doc.text[slice(*cuts[0]["span"])] == " ".join(SENTS[1:4])
    assert doc.text[slice(*cuts[1]["span"])] == SENTS[5]
    assert [c["n_sentences"] for c in cuts] == [3, 1]
    # A stub's line number points at the stub, as a text row's points at its text.
    assert all(text.splitlines()[c["line"]] == f"[cut {c['id']}]" for c in cuts)


def test_without_stubs_the_layout_is_unchanged():
    doc = _doc()
    text, sidecar = _layout(doc, {0, 4}, stubs="none")
    assert text.splitlines() == ["## Mill Town", "", SENTS[0], SENTS[4]]
    assert all(r["block"] == "text" for r in sidecar)


def test_a_described_stub_names_what_was_cut_but_not_its_value():
    doc = _doc()
    claims = [_claim_on(doc, 1, SENTS[1], e="Mill", r="employed_workers"),
              _claim_on(doc, 2, SENTS[2], e="Aldwin", r="owned"),
              _claim_on(doc, 3, SENTS[3], e="roof", r="colour")]
    for c in claims:
        c.v = "SECRET"
    text, sidecar = _layout(doc, {0, 4, 5}, claims, stubs="described")
    stub = text.splitlines()[3]
    assert stub == "[cut 1.1: Mill employed workers; Aldwin owned]"     # two topics at most
    assert "SECRET" not in text
    assert next(r for r in sidecar if r["block"] == "cut")["claims"] == ["a#1", "a#2", "a#3"]


def test_a_cut_no_claim_describes_gets_a_bare_stub():
    text, _ = _layout(_doc(), {0}, stubs="described")
    assert text.splitlines()[-1] == "[cut 1.1]"


def test_stubs_are_paid_for_from_the_same_budget_and_every_document_keeps_text():
    docs = [Document(f"d{k}", " ".join(f"Fact {k}-{i} is about item {i} here." for i in range(12)))
            for k in range(5)]
    claims = [_claim_on(d, i, f"Fact {k}-{i} is about item {i} here.", e=f"item{i}")
              for k, d in enumerate(docs) for i in range(12)]
    plain = fill_verbatim(claims, docs, ratio=0.5, rho=0.0, count_tokens=words)
    paged = fill_verbatim(claims, docs, ratio=0.5, rho=0.0, count_tokens=words, stubs="bare")
    assert paged.meta["actual_tokens"] <= paged.meta["target_tokens"] == plain.meta["target_tokens"]
    assert 0 < paged.meta["n_sentences"] < plain.meta["n_sentences"]
    assert paged.meta["n_cuts"] > 0 and paged.meta["stub_tokens"] > 0
    assert paged.meta["n_documents_rendered"] == 5
    # The stubs' cost is spread over the documents, not taken from the last ones.
    kept = [sum(r["block"] == "text" for r in paged.sidecar if r["doc"] == d.document_id)
            for d in docs]
    assert min(kept) >= 1 and max(kept) - min(kept) <= 2


# -- the fetch round ---------------------------------------------------------------------


def test_parse_fetch_reads_the_requested_cuts():
    assert parse_fetch("FETCH 12.3") == ["12.3"]
    assert parse_fetch("FETCH 12.3, 12.4 and [cut 2.1]") == ["12.3", "12.4", "2.1"]
    assert parse_fetch("I need FETCH 3.10") == ["3.10"]
    assert parse_fetch("FETCH 1.1 1.1 1.2 1.3 1.4", limit=3) == ["1.1", "1.2", "1.3"]
    assert parse_fetch("Revenue rose 12.3 percent.") == []
    assert parse_fetch("FETCH") == []


def test_prompts_keep_the_standard_instruction_and_put_restored_text_before_the_question():
    fetch = build_fetch_prompt("Who owned the mill?")
    assert "FETCH" in fetch and fetch.endswith("Question: Who owned the mill?\nAnswer:")
    restored = build_restored_prompt("Who owned the mill?", [("1.1", "Aldwin owned it.")])
    assert restored.endswith(build_user_prompt("Who owned the mill?"))
    assert restored.index("Aldwin owned it.") < restored.index("Question:")
    assert "[cut 1.1]" in restored


class ScriptedReader:
    """Replies in order; a callable reply sees the user prompt."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, system, user, *, max_tokens=None):
        self.calls.append(user)
        reply = self.replies.pop(0)
        text = reply(user) if callable(reply) else reply
        return Generation(text=text, ttft_s=0.1, total_s=0.2, n_generated_tokens=3,
                          throughput_tok_s=10.0)

    def prime(self, system):
        return Generation("ready", 0.1, 0.1, 1, 0.0)

    def wait_for_health(self):
        pass

    def check_fits(self, context):
        return len(context.split())

    def props(self):
        return {}


def _paged_wiki(profile, topic, stubs="bare"):
    build_ledger(profile, topic, Extractor(profile.compiler, backend="fake"))
    render_topic(profile, topic, name="paged", rho=0.0, count_tokens=words, style="verbatim",
                 stubs=stubs)
    return load_cuts(profile, topic, "paged")


def test_a_fetch_restores_the_cut_and_the_second_answer_is_the_one_recorded(profile, topic):
    cuts = _paged_wiki(profile, topic)
    sid, cut = next(iter(cuts.items()))
    qa = load_topic(profile, topic).qa[:3]
    reader = ScriptedReader([f"FETCH {sid}", "after the fetch", "direct answer",
                             "FETCH 99.9", "asked again"])
    run_paged(profile, topic, "paged_reader", wiki="paged", mode="reader", reader=reader,
              questions=qa, progress=False)

    # q1: fetch, then answer with the cut's text ahead of the question.
    assert "FETCH" in reader.calls[0]
    assert cut["text"] in reader.calls[1] and reader.calls[1].endswith(build_user_prompt(qa[0].question))
    # q2: answered at once.  q3: an unknown cut is asked again without the fetch option.
    assert len(reader.calls) == 5
    assert reader.calls[4] == build_user_prompt(qa[2].question)

    answers = read_jsonl(answers_path(profile, topic, "paged_reader"))
    assert [a["generated"] for a in answers] == ["after the fetch", "direct answer", "asked again"]
    assert answers[0]["total_s"] == 0.4 and answers[1]["total_s"] == 0.2
    pages = read_jsonl(pages_path(profile, topic, "paged_reader"))
    assert pages[0]["requested"] == [sid] and pages[0]["fetched"] == [sid]
    assert pages[1]["requested"] == [] and pages[1]["fetched"] == []
    assert pages[2]["requested"] == ["99.9"] and pages[2]["fetched"] == []
    assert pages[0]["first_reply"] == f"FETCH {sid}"


def test_classify_says_where_the_gold_passage_went(profile, topic):
    cuts = _paged_wiki(profile, topic)
    corpus = load_topic(profile, topic)
    docs = {d.document_id: d for d in corpus.documents}
    seen = {}
    for q in corpus.qa:
        klass, ids = classify(q, docs[q.document_id], cuts)
        seen.setdefault(klass, []).append((q, ids))
        text = docs[q.document_id].text
        a = text.find(q.long_answer)
        inside = [i for i in ids if cuts[i]["span"][0] < a + len(q.long_answer)
                  and cuts[i]["span"][1] > a]
        assert inside == ids                       # every id overlaps the gold passage
        assert bool(ids) == (klass in ("partly_cut", "cut"))
    assert seen.get("in_context") and (seen.get("cut") or seen.get("partly_cut"))

    gone = QAPair("x-q1", corpus.documents[0].document_id, "Q?",
                  "The answer is not found in the document.", "NA")
    assert classify(gone, corpus.documents[0], cuts) == ("unanswerable", [])


def test_oracle_fetches_the_cuts_that_hold_the_gold_passage(profile, topic):
    cuts = _paged_wiki(profile, topic)
    corpus = load_topic(profile, topic)
    docs = {d.document_id: d for d in corpus.documents}
    labelled = [(q, *classify(q, docs[q.document_id], cuts)) for q in corpus.qa]
    lost = next(t for t in labelled if t[1] == "cut")
    kept = next(t for t in labelled if t[1] == "in_context")
    reader = ScriptedReader(["restored answer", "plain answer"])
    run_paged(profile, topic, "paged_oracle", wiki="paged", mode="oracle", reader=reader,
              questions=[lost[0], kept[0]], progress=False)

    assert len(reader.calls) == 2                  # no fetch round: the pointer is given
    assert all(cuts[i]["text"] in reader.calls[0] for i in lost[2])
    assert reader.calls[1] == build_user_prompt(kept[0].question)
    pages = read_jsonl(pages_path(profile, topic, "paged_oracle"))
    assert pages[0]["klass"] == "cut" and pages[0]["fetched"] == lost[2] == pages[0]["gold_cuts"]
    assert pages[1]["klass"] == "in_context" and pages[1]["fetched"] == []


def test_report_splits_fetching_and_scores_by_where_the_gold_went(profile, topic):
    cuts = _paged_wiki(profile, topic)
    corpus = load_topic(profile, topic)
    docs = {d.document_id: d for d in corpus.documents}
    labelled = [(q, *classify(q, docs[q.document_id], cuts)) for q in corpus.qa]
    lost = [t for t in labelled if t[1] == "cut"][:2]
    kept = [t for t in labelled if t[1] == "in_context"][:2]
    # One lost question fetches its gold cut, the other answers blind; one kept
    # question fetches something it did not need.
    wrong = next(i for i in cuts if i not in lost[0][2])
    reader = ScriptedReader([f"FETCH {lost[0][2][0]}", "a", "b", f"FETCH {wrong}", "c", "d"])
    qs = [lost[0][0], lost[1][0], kept[0][0], kept[1][0]]
    run_paged(profile, topic, "paged_reader", wiki="paged", mode="reader", reader=reader,
              questions=qs, progress=False)
    scores_path(profile, topic, "paged_reader").write_text("".join(
        json.dumps({"question_id": q.question_id, "score": s}) + "\n"
        for q, s in zip(qs, (5, 1, 4, 2))))
    scores_path(profile, topic, "verb").parent.mkdir(parents=True)
    scores_path(profile, topic, "verb").write_text("".join(
        json.dumps({"question_id": q.question_id, "score": 3}) + "\n" for q in qs))

    rep = page_report(profile, topic, "paged_reader", against=["verb"], baseline_dir=None)
    cut, inc = rep["classes"]["cut"], rep["classes"]["in_context"]
    assert cut["n"] == 2 and cut["fetch_rate"] == 0.5 and cut["hit_rate"] == 1.0
    assert cut["mean_score"] == 3.0 and cut["against"]["verb"] == 3.0
    assert inc["n"] == 2 and inc["fetch_rate"] == 0.5 and inc["hit_rate"] is None
    assert inc["mean_score"] == 3.0
    assert rep["n_questions"] == 4 and rep["mode"] == "reader"


# -- whole-document fetch ----------------------------------------------------------------


def test_parse_fetch_docs_reads_document_numbers():
    assert parse_fetch_docs("FETCH 12") == ["12"]
    assert parse_fetch_docs("FETCH 12.3") == ["12"]               # a cut id names its document
    assert parse_fetch_docs("FETCH 12.3, 12.4 and 7") == ["12", "7"]
    assert parse_fetch_docs("FETCH 1 2 3", limit=2) == ["1", "2"]
    assert parse_fetch_docs("Revenue rose 12 percent.") == []


def test_doc_prompts_offer_the_document_and_restore_it_before_the_question():
    fetch = build_fetch_prompt("Who owned the mill?", "doc")
    assert "FETCH 12" in fetch and "number of its document" in fetch
    restored = build_doc_restored_prompt("Who owned the mill?", [("3", "The mill. Aldwin owned it.")])
    assert restored.index("Full text of document 3:") < restored.index("Aldwin") < restored.index("Question:")
    assert restored.endswith(build_user_prompt("Who owned the mill?"))


def test_a_document_fetch_restores_the_whole_document(profile, topic):
    cuts = _paged_wiki(profile, topic)
    corpus = load_topic(profile, topic)
    docs = {d.document_id: d for d in corpus.documents}
    labelled = [(q, *classify(q, docs[q.document_id], cuts)) for q in corpus.qa]
    lost = next(t for t in labelled if t[1] == "cut")
    n = str(next(i for i, d in enumerate(corpus.documents, 1) if d.document_id == lost[0].document_id))
    other = "1" if n != "1" else "2"
    reader = ScriptedReader([f"FETCH {n}.1", "after", f"FETCH {other}", "wrong doc", "FETCH 999", "again"])
    run_paged(profile, topic, "paged_reader_doc", wiki="paged", mode="reader", fetch="doc",
              reader=reader, questions=[lost[0]] * 3, progress=False)

    assert "number of its document" in reader.calls[0]
    assert docs[lost[0].document_id].text.strip() in reader.calls[1]
    assert reader.calls[5] == build_user_prompt(lost[0].question)   # no such document
    pages = read_jsonl(pages_path(profile, topic, "paged_reader_doc"))
    assert [p["gold_cuts"] for p in pages] == [[n]] * 3
    assert [p["fetched"] for p in pages] == [[n], [other], []]


def test_document_oracle_restores_the_gold_document_only_when_gold_was_cut(profile, topic):
    cuts = _paged_wiki(profile, topic)
    corpus = load_topic(profile, topic)
    docs = {d.document_id: d for d in corpus.documents}
    labelled = [(q, *classify(q, docs[q.document_id], cuts)) for q in corpus.qa]
    lost = next(t for t in labelled if t[1] in ("cut", "partly_cut"))
    kept = next(t for t in labelled if t[1] == "in_context")
    reader = ScriptedReader(["restored", "plain"])
    run_paged(profile, topic, "paged_oracle_doc", wiki="paged", mode="oracle", fetch="doc",
              reader=reader, questions=[lost[0], kept[0]], progress=False)
    assert docs[lost[0].document_id].text.strip() in reader.calls[0]
    assert reader.calls[1] == build_user_prompt(kept[0].question)


# -- command line ------------------------------------------------------------------------


def test_offline_paged_pipeline_renders_pages_and_scores(profile, topic):
    argv = ["--profile", "test", "--topic", topic]
    assert run.main(["extract", *argv], profile=profile) == 0
    assert run.main(["render", *argv, "--name", "paged", "--rho", "0", "--style", "verbatim",
                     "--stubs", "described"], profile=profile) == 0
    side = json.loads(sidecar_path(profile, topic, "paged").read_text())
    assert any(r["block"] == "cut" for r in side)
    n = len(load_topic(profile, topic).qa)
    for mode in ("reader", "oracle"):
        assert run.main(["page", *argv, "--wiki", "paged", "--mode", mode], profile=profile) == 0
        assert len(read_jsonl(scores_path(profile, topic, f"paged_{mode}"))) == n
        assert len(read_jsonl(pages_path(profile, topic, f"paged_{mode}"))) == n
    assert run.main(["page", *argv, "--wiki", "paged", "--fetch", "doc"], profile=profile) == 0
    assert len(read_jsonl(scores_path(profile, topic, "paged_reader_doc"))) == n


def test_limit_pages_only_the_first_questions(profile, topic):
    argv = ["--profile", "test", "--topic", topic]
    run.main(["extract", *argv], profile=profile)
    run.main(["render", *argv, "--name", "paged", "--rho", "0", "--style", "verbatim",
              "--stubs", "bare"], profile=profile)
    assert run.main(["page", *argv, "--wiki", "paged", "--limit", "5"], profile=profile) == 0
    assert len(read_jsonl(pages_path(profile, topic, "paged_reader"))) == 5


# -- a corpus larger than the window -------------------------------------------------------


def test_full_profile_results_go_to_their_own_directory():
    from tact import get_profile

    assert get_profile("full").results_dir == "result-full"
    assert get_profile("paper").results_dir == "result-tact"


def test_render_can_fix_the_budget_in_tokens(profile, topic):
    build_ledger(profile, topic, Extractor(profile.compiler, backend="fake"))
    source = words(format_documents(load_topic(profile, topic).documents))
    res = render_topic(profile, topic, name="fixed", rho=0.0, count_tokens=words,
                       style="verbatim", stubs="bare", budget_tokens=source // 3)
    assert abs(res.meta["target_tokens"] - source // 3) <= 1
    assert res.meta["actual_tokens"] <= res.meta["target_tokens"]
    with pytest.raises(ValueError):
        render_topic(profile, topic, name="x", rho=0.0, count_tokens=words, ratio=0.5,
                     budget_tokens=100)
