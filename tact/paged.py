"""Paged cache: a verbatim context that marks its cuts, read by a reader that may
ask for one back.

``render --style verbatim --stubs bare|described`` leaves a stub where each run
of sentences was dropped.  This module serves that context in two modes:

* ``reader``: the question goes out with one extra sentence offering the fetch.
  If the reply is ``FETCH <ids>``, the cut text is put in the user turn, ahead
  of the question, and the reader answers again; that second answer is the one
  recorded.  The cached context never changes, so the KV prefix is reused.
* ``oracle``: no fetch round.  The cuts that hold the question's gold passage
  are restored up front -- what a perfect request would have bought.

``fetch`` sets what one request brings back: a ``cut`` (the default), or a whole
``doc`` -- the full text of the document a cut belongs to (the 12 of
``[cut 12.3]``).  A document request only has to find the right document, and
it brings every cut of a gold passage at once, as the oracle does.

Serving the same context with the ordinary prompt (``tact.run score``) is the
third arm: what the stubs cost when nothing is fetched.

Every question is labelled by where its gold passage went -- ``in_context``,
``partly_cut``, ``cut``, or ``unanswerable`` (never in the source) -- so fetch
behaviour and scores can be read per class (``page_report``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from wicer_core.config import Profile, resolve
from wicer_core.corpus import Document, QAPair, is_not_in_document, load_topic
from wicer_core.metrics import mean
from wicer_core.provenance import build_run_meta, write_run_meta
from wicer_core.reader import ANSWER_INSTRUCTION, Generation, ReaderClient, build_user_prompt
from wicer_core.records import (
    AnswerRecord,
    JsonlWriter,
    answers_path,
    completed_question_ids,
    read_jsonl,
    run_dir,
    wiki_paths,
)
from tact.ledger import locate
from tact.render import sentences, sidecar_path

MODES = ("reader", "oracle")
FETCH_UNITS = ("cut", "doc")
CLASSES = ("in_context", "partly_cut", "cut", "unanswerable", "unlocated")
MAX_FETCH = 3          # cuts one reply may ask for
MAX_ORACLE_FETCH = 6   # a gold passage rarely spans more
MAX_DOC_FETCH = 2      # documents one reply may ask for (~1.1K tokens each)

FETCH_INSTRUCTION = (
    "Some passages were cut from the material above; each cut is marked like [cut 12.3]. "
    "If the answer is in a cut passage, reply with only the word FETCH and the number of "
    "that cut, for example: FETCH 12.3"
)


DOC_FETCH_INSTRUCTION = (
    "Some passages were cut from the material above; each cut is marked like [cut 12.3], "
    "where 12 is the number of its document. If the answer is in a cut passage, reply with "
    "only the word FETCH and the number of that document, for example: FETCH 12"
)


INDEX_FETCH_INSTRUCTION = (
    "Some sentences were cut from the documents above; a document that lost sentences ends "
    "with a note like [cut from document 12: ...] naming what they were about. If the answer "
    "is in a cut part, reply with only the word FETCH and the number of that document, for "
    "example: FETCH 12"
)


def build_fetch_prompt(question: str, fetch: str = "cut", stubs: str = "bare") -> str:
    """The ordinary suffix plus the offer to fetch; nothing else differs."""
    if stubs == "index":
        offer = INDEX_FETCH_INSTRUCTION
    else:
        offer = DOC_FETCH_INSTRUCTION if fetch == "doc" else FETCH_INSTRUCTION
    return f"{ANSWER_INSTRUCTION}\n{offer}\n\nQuestion: {question}\nAnswer:"


def build_restored_prompt(question: str, passages: Sequence[tuple[str, str]]) -> str:
    """Restored cuts, then the ordinary suffix: the text sits next to the question."""
    blocks = "\n\n".join(f"Restored passage [cut {sid}]:\n{text}" for sid, text in passages)
    return f"{blocks}\n\n{build_user_prompt(question)}"


def build_doc_restored_prompt(question: str, documents: Sequence[tuple[str, str]]) -> str:
    """Whole documents, then the ordinary suffix."""
    blocks = "\n\n".join(f"Full text of document {n}:\n{text}" for n, text in documents)
    return f"{blocks}\n\n{build_user_prompt(question)}"


def parse_fetch_docs(text: str, limit: int = MAX_DOC_FETCH) -> list[str]:
    """The document numbers a reply asks for; ``FETCH 12.3`` counts as document 12."""
    m = re.search(r"\bFETCH\b", text)
    if not m:
        return []
    ids: list[str] = []
    for n in re.findall(r"(\d+)(?:\.\d+)?", text[m.end():]):
        if n not in ids:
            ids.append(n)
    return ids[:limit]


def doc_numbers(cuts: dict[str, dict[str, Any]], documents: Sequence[Document]) -> dict[str, str]:
    """Document number (the 12 of ``[cut 12.3]``) -> document id.

    The render numbers documents by their place in the corpus; the cuts confirm it.
    """
    numbers = {str(i): d.document_id for i, d in enumerate(documents, start=1)}
    for sid, c in cuts.items():
        if numbers.get(sid.split(".")[0]) != c["doc"]:
            raise ValueError(f"cut {sid} is in {c['doc']}, not document {sid.split('.')[0]}")
    return numbers


def parse_fetch(text: str, limit: int = MAX_FETCH) -> list[str]:
    """The cut ids a reply asks for; empty when the reply is an answer."""
    m = re.search(r"\bFETCH\b", text)
    if not m:
        return []
    ids: list[str] = []
    for sid in re.findall(r"\d+\.\d+", text[m.end():]):
        if sid not in ids:
            ids.append(sid)
    return ids[:limit]


def pages_path(profile: Profile, topic: str, condition: str) -> Path:
    return run_dir(profile, topic, condition) / "pages.jsonl"


def load_cuts(profile: Profile, topic: str, wiki: str) -> dict[str, dict[str, Any]]:
    """cut id -> {doc, span, text} for a wiki rendered with stubs."""
    side = sidecar_path(profile, topic, wiki)
    if not side.exists():
        raise FileNotFoundError(f"no sidecar for wiki {wiki!r} in {topic!r}: render it first")
    docs = {d.document_id: d for d in load_topic(profile, topic).documents}
    cuts = {}
    for row in json.loads(side.read_text()):
        if row.get("block") == "cut":
            a, b = row["span"]
            cuts[row["id"]] = {"doc": row["doc"], "span": [a, b],
                               "text": " ".join(docs[row["doc"]].text[a:b].split())}
    if not cuts:
        raise ValueError(f"wiki {wiki!r} has no stubs: render it with --stubs bare|described")
    return cuts


def classify(q: QAPair, doc: Document, cuts: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Where the question's gold passage went, and the cuts that hold part of it."""
    if is_not_in_document(q.answer):
        return "unanswerable", []
    hit = locate(q.long_answer, doc.text) if q.long_answer else None
    if not hit:
        return "unlocated", []
    gold = [(a, b) for a, b in sentences(doc.text) if a < hit[1] and b > hit[0]]
    mine = {sid: c["span"] for sid, c in cuts.items() if c["doc"] == doc.document_id}
    ids = [sid for sid, (a, b) in mine.items() if any(a < gb and b > ga for ga, gb in gold)]
    missing = sum(any(a <= ga and gb <= b for a, b in mine.values()) for ga, gb in gold)
    if not missing:
        return "in_context", []
    return ("cut" if missing == len(gold) else "partly_cut"), ids


def _record(q: QAPair, gens: Sequence[Generation]) -> dict[str, Any]:
    first, last = gens[0], gens[-1]
    return AnswerRecord(
        question_id=q.question_id, document_id=q.document_id, question=q.question,
        gold=q.answer, generated=last.text.strip(), ttft_s=first.ttft_s,
        total_s=sum(g.total_s for g in gens), throughput_tok_s=last.throughput_tok_s,
        n_generated_tokens=sum(g.n_generated_tokens for g in gens),
        n_prompt_tokens=last.n_prompt_tokens, cache_n=last.cache_n,
        finish_reason=last.finish_reason,
    ).to_dict()


def run_paged(
    profile: Profile,
    topic: str,
    condition: str,
    *,
    wiki: str,
    mode: str,
    fetch: str = "cut",
    reader: ReaderClient | None = None,
    questions: Sequence[QAPair] | None = None,
    resume: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Answer every question from a stubbed wiki, fetching cuts as ``mode`` says."""
    from benchmark.run_benchmark import build_wiki_context

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if fetch not in FETCH_UNITS:
        raise ValueError(f"fetch must be one of {FETCH_UNITS}, not {fetch!r}")
    corpus = load_topic(profile, topic)
    docs = {d.document_id: d for d in corpus.documents}
    cuts = load_cuts(profile, topic, wiki)
    numbers = doc_numbers(cuts, corpus.documents)
    number_of = {doc: n for n, doc in numbers.items()}
    md, meta_path = wiki_paths(profile, topic, wiki)
    context = build_wiki_context(md.read_text())
    stubs = json.loads(meta_path.read_text()).get("stubs", "bare") if meta_path.exists() else "bare"
    if stubs == "index" and fetch != "doc":
        raise ValueError(f"wiki {wiki!r} names documents, not cuts: page it with fetch='doc'")
    questions = corpus.qa if questions is None else questions

    reader = reader or ReaderClient(profile.reader)
    reader.wait_for_health()
    n_context_tokens = reader.check_fits(context)
    out = run_dir(profile, topic, condition)
    out.mkdir(parents=True, exist_ok=True)
    (out / "context.txt").write_text(context)

    path = answers_path(profile, topic, condition)
    done = completed_question_ids(path) if resume else set()
    pending = [q for q in questions if q.question_id not in done]
    reader.prime(context)

    iterator: Sequence[QAPair] = pending
    if progress and pending:
        try:
            from tqdm import tqdm

            iterator = tqdm(pending, desc=f"{topic}/{condition}", unit="q")
        except ImportError:
            pass

    with JsonlWriter(path, resume=resume) as answers, \
            JsonlWriter(pages_path(profile, topic, condition), resume=resume) as pages:
        for q in iterator:
            klass, gold_cuts = classify(q, docs[q.document_id], cuts)
            if fetch == "doc":
                # The unit of a hit is the document: the gold one, when any of
                # its gold passage was cut.
                gold_cuts = [number_of[q.document_id]] if gold_cuts else []
            plain = build_user_prompt(q.question)
            if mode == "oracle":
                requested = gold_cuts[:MAX_ORACLE_FETCH]
                gens: list[Generation] = []
            else:
                gens = [reader.chat(context, build_fetch_prompt(q.question, fetch, stubs))]
                requested = (parse_fetch_docs if fetch == "doc" else parse_fetch)(gens[0].text)
            fetched = [i for i in requested if i in (numbers if fetch == "doc" else cuts)]
            if mode == "oracle" or requested:
                # A reply that asked for a cut is not an answer: ask again, with
                # the text if the cut exists and without the offer either way.
                if fetch == "doc":
                    prompt = build_doc_restored_prompt(
                        q.question, [(n, docs[numbers[n]].text.strip()) for n in fetched])
                else:
                    prompt = build_restored_prompt(
                        q.question, [(sid, cuts[sid]["text"]) for sid in fetched])
                gens.append(reader.chat(context, prompt if fetched else plain))
            answers.write(_record(q, gens))
            pages.write({
                "question_id": q.question_id, "klass": klass, "gold_cuts": gold_cuts,
                "requested": requested, "fetched": fetched,
                "first_reply": gens[0].text.strip() if mode == "reader" else None,
                "n_calls": len(gens),
            })

    summary = {
        "topic": topic, "condition": condition, "wiki": wiki, "mode": mode, "fetch": fetch,
        "stubs": stubs,
        "n_context_tokens": n_context_tokens, "n_cuts": len(cuts),
        "n_questions": len(questions), "n_generated": len(pending), "n_resumed": len(done),
    }
    try:
        props = reader.props()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        props = None
    write_run_meta(out / "run_meta.json", build_run_meta(
        profile, command="tact.run page", extra=summary, server_props=props, hash_model=False))
    return summary


# -- report --------------------------------------------------------------------------------


def page_report(
    profile: Profile,
    topic: str,
    condition: str,
    *,
    against: Sequence[str] = (),
    baseline_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fetching and scores per class, on the questions the sweep held out.

    ``against`` names other conditions in this results directory to put beside
    it on the same questions; with ``baseline_dir`` the sweep's probes are
    dropped and its ``fc_raw`` is added as one more column.
    """
    from dataclasses import replace

    from tact.compare import _scores

    rows = read_jsonl(pages_path(profile, topic, condition))
    meta_path = run_dir(profile, topic, condition) / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    columns = {name: _scores(profile, topic, name) for name in against}
    if baseline_dir is not None:
        probes = resolve(baseline_dir) / topic / "probes.json"
        if probes.exists():
            held_back = {p["question_id"] for p in json.loads(probes.read_text())}
            rows = [r for r in rows if r["question_id"] not in held_back]
        fc = _scores(replace(profile, results_dir=str(baseline_dir)), topic, "fc_raw")
        if fc:
            columns["fc_raw"] = fc
    scores = _scores(profile, topic, condition)

    def avg(table: dict[str, int], ids: Sequence[str]) -> float | None:
        vals = [float(table[i]) for i in ids if i in table]
        return mean(vals) if vals else None

    classes = {}
    for klass in CLASSES:
        mine = [r for r in rows if r["klass"] == klass]
        if not mine:
            continue
        ids = [r["question_id"] for r in mine]
        asked = [r for r in mine if r["requested"]]
        with_gold = [r for r in asked if r["gold_cuts"]]
        classes[klass] = {
            "n": len(mine),
            "fetch_rate": len(asked) / len(mine),
            # Of the requests that had a right answer, the share that named it.
            "hit_rate": (sum(bool(set(r["fetched"]) & set(r["gold_cuts"])) for r in with_gold)
                         / len(with_gold)) if with_gold else None,
            "mean_score": avg(scores, ids),
            "mean_score_fetched": avg(scores, [r["question_id"] for r in asked]),
            "against": {name: avg(table, ids) for name, table in columns.items()},
        }
    return {"topic": topic, "condition": condition, "mode": meta.get("mode"),
            "wiki": meta.get("wiki"), "n_questions": len(rows), "classes": classes}


def render_report(rep: dict[str, Any]) -> str:
    def f(x: float | None, fmt: str = "{:.2f}") -> str:
        return "--" if x is None else fmt.format(x)

    names = sorted({n for c in rep["classes"].values() for n in c["against"]})
    head = ["Gold passage", "n", "Fetched", "Named a gold cut", "Score", "Score when fetched", *names]
    out = [f"### {rep['topic']}: {rep['condition']} ({rep['mode']} fetch on {rep['wiki']})\n",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for klass, c in rep["classes"].items():
        cells = [klass, str(c["n"]), f(c["fetch_rate"], "{:.0%}"), f(c["hit_rate"], "{:.0%}"),
                 f(c["mean_score"]), f(c["mean_score_fetched"]),
                 *(f(c["against"].get(n)) for n in names)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"
