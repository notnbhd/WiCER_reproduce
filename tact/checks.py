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
from tact.render import budget_shares, headings, layout, route

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
    (ledger_dir(profile, topic) / "inspect.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


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
    ])
