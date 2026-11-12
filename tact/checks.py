"""Judge-free checks on an extracted ledger, run before paying for scoring.

* Answer coverage: for each answerable question, RepLiQA's ``long_answer`` is
  the source passage that supports it.  The claims whose spans overlap that
  passage are the draft's A(q) by span overlap (section 3.2), so a question
  with none is MISS-EXTRACT: no budget, ranking or pin can serve it.  The
  share of questions with a claim behind them is TACT's ceiling.
* Budget fit: each document's whole ledger, rendered, against its share of
  the (1 - rho) B target.  A document whose ledger fits gains nothing from
  budget enforcement -- iteration 0 just renders all of it.
* Shape: the pilot statistics the draft argues from (relations per claim,
  single-relation entities, routing mix), recomputed on the full topic.

Questions are the sweep's held-out set when its probes are available, so the
ceiling is for the same questions ``compare`` scores.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from wicer_core.config import Profile, resolve
from wicer_core.corpus import Document, QAPair, is_not_in_document, load_topic
from wicer_core.metrics import mean
from tact.ledger import Claim, ledger_dir, load_ledger, locate
from tact.render import _covering, budget_shares, headings, layout, route, sentences

# Llama 3.1 tokens per word, measured on the local_economy pilot.  Used only
# when no reader is up to count exactly.
TOKENS_PER_WORD = 1.288


def estimate_tokens(text: str) -> int:
    return round(len(text.split()) * TOKENS_PER_WORD)


def _overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def answer_coverage(
    claims: Sequence[Claim], documents: dict[str, Document], questions: Sequence[QAPair]
) -> dict[str, Any]:
    by_doc: dict[str, list[Claim]] = {}
    for c in claims:
        by_doc.setdefault(c.doc, []).append(c)
    rows = []
    for q in questions:
        hit = locate(q.long_answer, documents[q.document_id].text) if q.long_answer else None
        if hit is None:
            rows.append({"question_id": q.question_id, "located": False, "claims": [],
                         "passage_covered": None})
            continue
        passage = (hit[0], hit[1])
        behind = [c for c in by_doc.get(q.document_id, [])
                  if any(_overlap(s, passage) for s in c.spans)]
        covered = set()
        for c in behind:
            for s, e in c.spans:
                covered.update(range(max(s, passage[0]), min(e, passage[1])))
        rows.append({"question_id": q.question_id, "located": True,
                     "claims": [c.id for c in behind],
                     "passage_covered": len(covered) / max(1, passage[1] - passage[0])})
    located = [r for r in rows if r["located"]]
    with_claim = [r for r in located if r["claims"]]
    return {
        "n_questions": len(rows),
        "n_located": len(located),
        "share_with_any_claim": len(with_claim) / len(located) if located else None,
        "mean_passage_covered": mean([r["passage_covered"] for r in located]) if located else None,
        "share_half_covered": (sum(r["passage_covered"] >= 0.5 for r in located) / len(located)
                               if located else None),
        "mean_k": mean([float(len(r["claims"])) for r in with_claim]) if with_claim else None,
        "questions": rows,
    }


ATTRIBUTION_BUCKETS = ("in_context", "partly_in_context", "extracted_not_selected",
                       "not_extracted", "unlocated")


def _flat(text: str) -> str:
    return " ".join(text.split())


def context_attribution(
    claims: Sequence[Claim],
    documents: dict[str, Document],
    questions: Sequence[QAPair],
    context: str,
    scores: dict[str, float],
    baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Where each scored question's gold passage went: into the context, or lost
    at extraction (no claim covers any of its sentences) or at selection (claimed,
    but none of it rendered).

    A gold sentence counts as in the context when its text appears there verbatim,
    whitespace aside, so any condition's context.txt works -- a verbatim render,
    truncation, or a compiled wiki (where it measures copying, not paraphrase).
    With ``baseline`` scores (full context), each bucket also gets its share of the
    total baseline-minus-score gap: which loss the condition's shortfall lives in.
    """
    flat = _flat(context)
    sents = {d: sentences(doc.text) for d, doc in documents.items()}
    claimed: dict[str, set[int]] = {d: set() for d in documents}
    for c in claims:
        if c.doc in sents:
            claimed[c.doc].update(_covering(c.spans, sents[c.doc]))
    present = {d: {i for i, (a, b) in enumerate(s) if _flat(documents[d].text[a:b]) in flat}
               for d, s in sents.items()}

    rows, missing, unscored = [], {"extracted": 0, "not_extracted": 0}, 0
    gold_total = gold_in = gold_claimed = 0
    for q in questions:
        if q.question_id not in scores:
            unscored += 1
            continue
        text = documents[q.document_id].text
        hit = locate(q.long_answer, text) if q.long_answer else None
        gold = ([i for i, (a, b) in enumerate(sents[q.document_id]) if a < hit[1] and b > hit[0]]
                if hit else [])
        inside = [i for i in gold if i in present[q.document_id]]
        out = [i for i in gold if i not in present[q.document_id]]
        unclaimed = [i for i in out if i not in claimed[q.document_id]]
        missing["extracted"] += len(out) - len(unclaimed)
        missing["not_extracted"] += len(unclaimed)
        gold_total += len(gold)
        gold_in += len(inside)
        gold_claimed += sum(i in claimed[q.document_id] for i in gold)
        if not gold:
            bucket = "unlocated"
        elif not out:
            bucket = "in_context"
        elif inside:
            bucket = "partly_in_context"
        elif len(unclaimed) < len(out):
            bucket = "extracted_not_selected"
        else:
            bucket = "not_extracted"
        rows.append({"question_id": q.question_id, "bucket": bucket, "n_gold": len(gold),
                     "n_in_context": len(inside), "n_missing_unclaimed": len(unclaimed),
                     "score": scores[q.question_id],
                     "baseline": baseline.get(q.question_id) if baseline else None})

    gaps = [r["baseline"] - r["score"] for r in rows if r["baseline"] is not None]
    total_gap = sum(gaps)
    buckets = {}
    for name in ATTRIBUTION_BUCKETS:
        mine = [r for r in rows if r["bucket"] == name]
        base = [r for r in mine if r["baseline"] is not None]
        gap = sum(r["baseline"] - r["score"] for r in base)
        buckets[name] = {
            "n": len(mine),
            "share": len(mine) / len(rows) if rows else None,
            "mean_score": mean([float(r["score"]) for r in mine]) if mine else None,
            "mean_baseline": mean([float(r["baseline"]) for r in base]) if base else None,
            "gap_share": gap / total_gap if gaps and total_gap else None,
        }
    n_sents = sum(len(s) for s in sents.values())
    return {
        "n_questions": len(rows),
        "n_unscored": unscored,
        "buckets": buckets,
        "gap_per_question": total_gap / len(gaps) if gaps else None,
        "missing_gold_sentences": missing,
        "gold_sentences_in_context": gold_in / gold_total if gold_total else None,
        "gold_sentences_extracted": gold_claimed / gold_total if gold_total else None,
        "sentences_in_context": sum(len(p) for p in present.values()) / n_sents if n_sents else None,
        "sentences_extracted": sum(len(c) for c in claimed.values()) / n_sents if n_sents else None,
        "questions": rows,
    }


def budget_fit(
    claims: Sequence[Claim],
    documents: Sequence[Document],
    *,
    ratio: float,
    rho: float,
    count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    b = budget_shares(documents, ratio=ratio, rho=rho, count_tokens=count_tokens)
    by_id = {c.id: c for c in claims}
    routes = route(claims)
    titles = headings(documents)
    rows = []
    for d in documents:
        own = [c for c in claims if c.doc == d.document_id]
        text, _ = layout(own, by_id, routes=routes, titles=titles)
        n = count_tokens(text) if own else 0
        share = b.share[d.document_id]
        rows.append({"doc": d.document_id, "n_claims": len(own), "ledger_tokens": n,
                     "share_tokens": round(share), "fits": n <= share})
    total = sum(r["ledger_tokens"] for r in rows)
    return {
        "source_tokens": b.source_tokens,
        "target_tokens": b.target,
        "ledger_tokens": total,
        "ledger_ratio": total / b.source_tokens if b.source_tokens else None,
        "n_documents": len(rows),
        "n_fit": sum(r["fits"] for r in rows),
        "documents": rows,
    }


def ledger_shape(claims: Sequence[Claim]) -> dict[str, Any]:
    relations_of: dict[tuple[str, str], set[str]] = {}
    for c in claims:
        relations_of.setdefault((c.doc, c.e), set()).add(c.r)
    n_docs = len({c.doc for c in claims})
    n_relations = len({c.r for c in claims})
    return {
        "n_claims": len(claims),
        "claims_per_doc": len(claims) / n_docs if n_docs else 0.0,
        "n_relations": n_relations,
        "relations_per_claim": n_relations / len(claims) if claims else 0.0,
        "n_entities": len(relations_of),
        "single_relation_entity_share": (
            sum(len(r) == 1 for r in relations_of.values()) / len(relations_of)
            if relations_of else 0.0),
        "routing": dict(sorted(Counter(route(claims).values()).items())),
        "n_relational": sum(c.ref for c in claims),
        "n_multispan": sum(c.multi_span for c in claims),
    }


def inspect_topic(
    profile: Profile,
    topic: str,
    *,
    baseline_dir: str | Path,
    rho: float,
    count_tokens: Callable[[str], int],
    token_source: str,
    condition: str | None = None,
) -> dict[str, Any]:
    ledger = load_ledger(profile, topic)
    if ledger is None:
        raise FileNotFoundError(f"no ledger for {topic!r}: run `python -m tact.run extract` first")
    corpus = load_topic(profile, topic)
    questions = [q for q in corpus.qa if not is_not_in_document(q.answer)]
    probes = resolve(baseline_dir) / topic / "probes.json"
    if probes.exists():
        probe_ids = {q["question_id"] for q in json.loads(probes.read_text())}
        questions = [q for q in questions if q.question_id not in probe_ids]
    report = {
        "topic": topic,
        "questions": "held-out answerable" if probes.exists() else "all answerable",
        "tokens": token_source,
        "rho": rho,
        "ledger": {k: v for k, v in ledger.meta.items() if k != "usage_this_run"},
        "shape": ledger_shape(ledger.claims),
        "budget": budget_fit(ledger.claims, corpus.documents, ratio=profile.wicer.ratio,
                             rho=rho, count_tokens=count_tokens),
        "answers": answer_coverage(ledger.claims,
                                   {d.document_id: d for d in corpus.documents}, questions),
    }
    if condition:
        report["attribution"] = attribute_condition(
            profile, topic, condition, ledger.claims, corpus.documents, questions, baseline_dir)
    (ledger_dir(profile, topic) / "inspect.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def attribute_condition(
    profile: Profile,
    topic: str,
    condition: str,
    claims: Sequence[Claim],
    documents: Sequence[Document],
    questions: Sequence[QAPair],
    baseline_dir: str | Path,
) -> dict[str, Any]:
    """context_attribution for a scored condition, against the sweep's fc_raw if present."""
    from dataclasses import replace

    from tact.compare import _scores
    from wicer_core.records import run_dir

    context = run_dir(profile, topic, condition) / "context.txt"
    if not context.exists():
        raise FileNotFoundError(f"no context for {condition!r} in {topic!r}: score it first")
    scores = _scores(profile, topic, condition)
    baseline = _scores(replace(profile, results_dir=str(baseline_dir)), topic, "fc_raw") or None
    out = context_attribution(claims, {d.document_id: d for d in documents}, questions,
                              context.read_text(), scores, baseline)
    return {"condition": condition, "baseline": "fc_raw" if baseline else None, **out}


def summary(report: dict[str, Any]) -> str:
    a, b, s, led = report["answers"], report["budget"], report["shape"], report["ledger"]

    def pct(x: float | None) -> str:
        return "--" if x is None else f"{x:.0%}"

    return "\n".join([
        f"{report['topic']}  ({report['questions']} questions; tokens: {report['tokens']})",
        f"  answers   {a['n_located']}/{a['n_questions']} gold passages located; "
        f"{pct(a['share_with_any_claim'])} have a claim behind them, "
        f"{pct(a['share_half_covered'])} are at least half covered "
        f"(mean {pct(a['mean_passage_covered'])}); |A(q)| = "
        + (f"{a['mean_k']:.2f}" if a["mean_k"] is not None else "--"),
        f"  budget    ledger {b['ledger_tokens']:,} tokens = {pct(b['ledger_ratio'])} of source, "
        f"target {b['target_tokens']:,} (rho {report['rho']}); "
        f"{b['n_fit']}/{b['n_documents']} documents fit whole in their share",
        f"  shape     {s['n_claims']} claims ({s['claims_per_doc']:.1f}/doc), "
        f"{s['n_relations']} relations, {pct(s['single_relation_entity_share'])} of entities "
        f"have one relation; routing {s['routing']}; relational {s['n_relational']}, "
        f"multi-span {s['n_multispan']}",
        f"  ledger    rejected {pct(led.get('rejection_rate'))}, "
        f"contradictions {led.get('n_contradictions')}, "
        f"entity collisions {pct(led.get('entity_collision_rate'))}",
    ] + (_attribution_lines(report["attribution"], pct) if "attribution" in report else []))


def _attribution_lines(att: dict[str, Any], pct: Callable[[float | None], str]) -> list[str]:
    def num(x: float | None) -> str:
        return "--" if x is None else f"{x:.2f}"

    m = att["missing_gold_sentences"]
    lines = [
        f"  context   {att['condition']}: {att['n_questions']} scored questions; holds "
        f"{pct(att['sentences_in_context'])} of sentences and "
        f"{pct(att['gold_sentences_in_context'])} of gold ones; claims cover "
        f"{pct(att['sentences_extracted'])} of sentences and "
        f"{pct(att['gold_sentences_extracted'])} of gold ones",
        f"            missing gold sentences: {m['extracted']} extracted, "
        f"{m['not_extracted']} never extracted"
        + (f"; {att['baseline']} gap {att['gap_per_question']:+.2f}/question"
           if att["gap_per_question"] is not None else ""),
    ]
    for name, b in att["buckets"].items():
        if b["n"]:
            lines.append(f"            {name:<24} {b['n']:>4} ({pct(b['share'])})  "
                         f"score {num(b['mean_score'])}  {att['baseline'] or 'baseline'} "
                         f"{num(b['mean_baseline'])}  gap share {pct(b['gap_share'])}")
    return lines
