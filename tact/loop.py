"""Contract-enforced refinement (draft sections 3.5.1 and 3.5.3).

    S_0      <- the claims of an iteration-0 wiki (tact_iter0), read from its sidecar
    for t in 0..T-1:
        W_t   <- layout(S_t)                           a pure function of the claim set
        score every self-generated probe against W_t   (reader + judge, gold = probe's own)
        stop if no failures, or t > 0 and relative improvement < 10%, or t = T-1
        for each failed probe q:  diagnose, then
            MISS-RENDER    pin N(A+(q))
            MISS-EXTRACT   targeted extraction on d_q; pin N(new claims)
            MISS-RELATION  relation extraction on d_q; pin N(A+(q) + new claims)
            MISS-READ      record only
        S_{t+1} <- S_t + P; evict unpinned rows only if W exceeds B
    return the last evaluated W                       (no best-of on the probes)

N is paragraph closure.  The rendering adds lines when it gains claims, so
while the (1 - rho) B start leaves headroom, every block a pin does not reach
is byte-identical across iterations.  Past B, eviction takes the lowest-ranked
unpinned rows of the documents pinned this iteration first, then of the rest;
a pinned claim is never evicted, so P is a subset of cl(W) at every iteration.
That holds by construction (draft open question 6) and is asserted anyway.

Claims extracted during the loop are added to the loop's copy of the ledger,
never to ledger.jsonl, so the bulk ledger stays what milestone A rendered.
Existing claims keep their route; a new claim is routed on the extended
ledger.  Promotion (3.5.4) is deferred: each failure's (diagnosis, probe kind)
signature is recorded so a later run can tell whether signatures recur.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from wicer_core.config import Profile, resolve
from wicer_core.corpus import Document, load_topic
from wicer_core.judge import Judge, JudgeRequest
from wicer_core.metrics import compression_ratio, mean
from wicer_core.reader import ReaderClient
from wicer_core.records import wiki_paths
from wicer.loop import answer_probes
from tact.diagnose import (
    MISS_EXTRACT, MISS_READ, MISS_RELATION, MISS_RENDER,
    diagnose, extended_answer, neighbourhood,
)
from tact.ledger import (
    Claim, Extractor, canonical_map, claim_key, claims_from_extraction, ledger_dir, load_ledger,
)
from tact.probes import Probe, answer_set, load_probes
from tact.render import (
    budget_shares, check_corpus, headings, layout, round_robin, route, sidecar_path,
)

FINAL_WIKI = "tact_loop"


def loop_dir(profile: Profile, topic: str) -> Path:
    return resolve(profile.results_dir) / topic / "tact_loop"


@dataclass
class LoopResult:
    topic: str
    start: str
    budget_tokens: int
    final_wiki: str
    stop_reason: str
    iterations: list[dict[str, Any]] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    pinned_docs: list[str] = field(default_factory=list)
    added_claims: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


def admit(
    selected: set[str], *, old_pins: set[str], new_order: Sequence[str], tokens: int,
    budget: int, cost: dict[str, int],
) -> tuple[set[str], list[str]]:
    """Refuse this iteration's pins, newest first, until the estimate fits B.

    ``new_order`` lists the new pins answers-first, neighbourhoods last, so the
    neighbourhood is given up before any probe's answer.  Earlier pins are
    never refused: the contract covers every pin that was admitted.
    """
    kept, refused = set(selected), []
    for i in reversed(new_order):
        if tokens <= budget:
            break
        if i in old_pins or i not in kept:
            continue
        kept.discard(i)
        refused.append(i)
        tokens -= cost.get(i, 0)
    return kept, refused


def eviction_order(
    selected: Sequence[Claim], *, pinned: set[str], touched: set[str], rank: dict[str, int]
) -> list[str]:
    """Unpinned claims, most evictable first: pinned-this-round documents first,
    then lowest-ranked (latest in round-robin order) first."""
    free = [c for c in selected if c.id not in pinned]
    return [c.id for c in sorted(free, key=lambda c: (c.doc not in touched, -rank[c.id], c.id))]


class _Ledger:
    """The loop's working copy of the ledger: bulk claims plus repair extractions."""

    def __init__(self, claims: Sequence[Claim], documents: Sequence[Document]):
        self.claims = list(claims)
        self.by_id = {c.id: c for c in self.claims}
        self.by_doc: dict[str, list[Claim]] = {}
        for c in self.claims:
            self.by_doc.setdefault(c.doc, []).append(c)
        self.routes = route(self.claims)
        self.text = {d.document_id: d.text for d in documents}
        self.added: list[Claim] = []
        self.rank = {c.id: i for doc in self.by_doc.values()
                     for i, c in enumerate(round_robin(doc))}

    def add(self, doc: str, raw: dict[str, Any], prefix: str) -> tuple[list[Claim], list[str]]:
        """Validate a repair extraction; (claims added, existing claims it repeated).

        A repair claim the document already has elsewhere is the same fact, so
        its existing claim is what gets pinned -- otherwise a fact stated twice
        could never be repaired.
        """
        own = self.by_doc.get(doc, [])
        counts = Counter(c.e for c in own)
        for row in raw.get("claims", []):
            if isinstance(row, dict):
                counts.setdefault(str(row.get("entity", "")).strip(), 0)
        counts.pop("", None)
        new, _ = claims_from_extraction(doc, self.text[doc], raw, canonical_map(dict(counts)),
                                        prefix=prefix, known=set(self.by_id))
        have = {claim_key(c): c.id for c in own}
        matched = list(dict.fromkeys(have[claim_key(c)] for c in new if claim_key(c) in have))
        new = [c for c in new if claim_key(c) not in have and c.id not in self.by_id]
        if not new:
            return [], matched
        extended = route(self.claims + new)
        last = max((self.rank[c.id] for c in own), default=-1)
        for i, c in enumerate(new):
            self.claims.append(c)
            self.by_id[c.id] = c
            self.by_doc.setdefault(doc, []).append(c)
            self.routes[c.id] = extended[c.id]
            self.rank[c.id] = last + 1 + i
        self.added += new
        return new, matched


def _cached(path: Path, call: Callable[[], dict[str, Any]], provenance: str) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())["raw"]
    raw = call()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"raw": raw, "extracted_with": provenance}, indent=1) + "\n")
    return raw


def run_loop(
    profile: Profile,
    topic: str,
    *,
    reader: ReaderClient,
    judge: Judge,
    extractor: Extractor,
    start: str = "tact_iter0",
    max_iter: int | None = None,
    improvement_threshold: float | None = None,
    closure: str = "paragraph",
) -> LoopResult:
    """``closure`` is N in N(A(q)): ``paragraph`` (the default) or ``none``."""
    if closure not in ("paragraph", "none"):
        raise ValueError(f"closure must be 'paragraph' or 'none', not {closure!r}")
    cfg = profile.wicer
    max_iter = cfg.max_iterations if max_iter is None else max_iter
    threshold = cfg.improvement_threshold if improvement_threshold is None else improvement_threshold
    count_tokens = reader.count_tokens

    bulk = load_ledger(profile, topic)
    if bulk is None:
        raise FileNotFoundError(f"no ledger for {topic!r}: run `python -m tact.run extract`")
    probes = load_probes(profile, topic)
    if not probes:
        raise FileNotFoundError(f"no probes for {topic!r}: run `python -m tact.run probes`")
    corpus = load_topic(profile, topic)
    check_corpus(bulk, corpus)
    docs = {d.document_id: d for d in corpus.documents}
    led = _Ledger(bulk.claims, corpus.documents)
    titles = headings(corpus.documents)
    shares = budget_shares(corpus.documents, ratio=cfg.ratio, rho=0.0, count_tokens=count_tokens)
    budget = shares.budget

    def render(ids: set[str]) -> tuple[str, list[dict[str, Any]]]:
        chosen = [led.by_id[i] for i in sorted(ids, key=lambda i: (led.by_id[i].pos, i))]
        return layout(chosen, led.by_id, routes=led.routes, titles=titles)

    # S_0 is read from the start wiki's sidecar, and the wiki must be exactly
    # what that set renders to: a loop built on a wiki it cannot reproduce
    # would compare iterations that differ by more than their pins.
    side = json.loads(sidecar_path(profile, topic, start).read_text())
    selected = {i for row in side for i in row["claims"]}
    start_text = wiki_paths(profile, topic, start)[0].read_text()
    if render(selected)[0] != start_text:
        raise ValueError(f"{start}.md is not what its sidecar renders to -- re-render it")

    out = loop_dir(profile, topic)
    out.mkdir(parents=True, exist_ok=True)
    qas = [p.as_qa() for p in probes]
    pinned: set[str] = set()
    records: list[dict[str, Any]] = []
    prev_mean: float | None = None
    stop = "max_iterations"
    line_cost: dict[str, int] = {}

    for t in range(max_iter):
        name = f"tact_loop_iter{t}"
        text, sidecar = render(selected)
        tokens = count_tokens(text)
        _write_wiki(profile, topic, name, text, sidecar, {
            "method": "tact_loop", "iteration": t, "start": start,
            "budget_tokens": budget, "actual_tokens": tokens,
            "source_tokens": shares.source_tokens,
            "actual_ratio_tokens": compression_ratio(tokens, shares.source_tokens),
            "n_selected": len(selected),
            "n_pinned_facts": len(pinned), "coverage": len(selected) / len(led.claims),
            "source_words": corpus.n_words,
            "actual_ratio_words": compression_ratio(len(text.split()), corpus.n_words),
        })

        answers = answer_probes(reader, text, qas, desc=f"{topic}/{name}")
        judged = judge.score_many([JudgeRequest(question=q.question, gold=q.answer, generated=a)
                                   for q, a in zip(qas, answers)])
        scores = [j.score for j in judged]
        failed = [p for p, s in zip(probes, scores) if s <= cfg.failure_threshold]
        probe_mean = mean([float(s) for s in scores])
        coverage = []
        for p in probes:
            a = answer_set(p, led.by_doc.get(p.doc, []))
            if a:
                coverage.append(sum(i in selected for i in a) / len(a))
        rec: dict[str, Any] = {
            "iteration": t, "wiki_name": name, "probe_mean": probe_mean,
            "probe_score_1_pct": 100.0 * len(failed) / len(probes),
            "n_failures": len(failed), "tokens": tokens, "headroom_tokens": budget - tokens,
            "partial_coverage_mean": mean(coverage) if coverage else None,
            "n_pinned": len(pinned), "improvement": None,
            "diagnoses": {}, "signatures": {}, "rows_added": 0, "rows_evicted": 0,
            "n_claims_added": 0, "tokens_after_repair": tokens,
            "contract_ok": pinned <= selected, "stop": None,
        }
        if prev_mean is not None:
            rec["improvement"] = (probe_mean - prev_mean) / prev_mean if prev_mean else None
        prev_mean = probe_mean

        if not failed:
            stop = "no_failures"
        elif rec["improvement"] is not None and rec["improvement"] < threshold:
            stop = "insufficient_improvement"
        elif t == max_iter - 1:
            stop = "max_iterations"
        else:
            stop = ""
        if stop:
            rec["stop"] = stop
            records.append(rec)
            _write_probe_rows(out / f"iter{t}_probes.jsonl", probes, answers, scores, {})
            break

        diag: dict[str, dict[str, Any]] = {}
        cores: list[str] = []          # what each failed probe needs, in probe order
        extras: list[str] = []         # its paragraph neighbourhood
        touched: set[str] = set()
        n_added = 0
        for q in failed:
            doc = q.doc
            own = led.by_doc.get(doc, [])
            a = answer_set(q, own)
            a_plus = extended_answer(q, a, own)
            label = diagnose(q, a, a_plus, selected, led.by_id)
            tag = q.id.replace("#", "_")
            core: list[str] = []
            if label == MISS_RENDER:
                core = a_plus
            elif label == MISS_EXTRACT:
                raw = _cached(ledger_dir(profile, topic) / "repair" / f"{tag}.targeted.json",
                              lambda: extractor.extract_targeted(
                                  docs[doc], q, sorted({c.e for c in own})),
                              extractor.provenance)
                new, matched = led.add(doc, raw, prefix=f"i{t}{tag.split('_')[-1]}.")
                n_added += len(new)
                core = [c.id for c in new] + matched
            elif label == MISS_RELATION:
                raw = _cached(ledger_dir(profile, topic) / "repair" / f"{tag}.relation.json",
                              lambda: extractor.extract_relation(
                                  docs[doc], q, [led.by_id[i] for i in a_plus]),
                              extractor.provenance)
                new, matched = led.add(doc, raw, prefix=f"i{t}{tag.split('_')[-1]}r.")
                n_added += len(new)
                core = list(dict.fromkeys(a_plus + [c.id for c in new] + matched))
            near = [] if closure == "none" else [
                i for i in neighbourhood(core, led.by_doc.get(doc, []), docs[doc].text)
                if i not in core]
            if core:
                touched.add(doc)
            cores += core
            extras += near
            diag[q.id] = {"diagnosis": label, "n_answer": len(a), "n_pins": len(core) + len(near)}

        old_pins = set(pinned)
        new_order = list(dict.fromkeys(i for i in cores + extras if i not in old_pins))
        pinned |= set(new_order)
        before = set(selected)
        selected |= pinned

        # Past B, evict unpinned rows; if the pins alone still exceed B, refuse
        # this iteration's newest pins, neighbourhood before answers.  Each step
        # estimates from per-line costs and is confirmed on the whole wiki.
        text, _ = render(selected)
        after = count_tokens(text)
        if after > budget:
            for i in selected:
                if i not in line_cost:
                    line_cost[i] = count_tokens(render({i})[0].splitlines()[-1])
            order = eviction_order([led.by_id[i] for i in selected], pinned=pinned,
                                   touched=touched, rank=led.rank)
            for i in order:
                if after <= budget:
                    break
                selected.discard(i)
                after -= line_cost[i]
            text, _ = render(selected)
            after = count_tokens(text)
            if after > budget:
                selected, _ = admit(selected, old_pins=old_pins, new_order=new_order,
                                    tokens=after, budget=budget, cost=line_cost)
                text, _ = render(selected)
                after = count_tokens(text)
            # The estimates can undershoot by a few tokens; drop one more at a time.
            spare = [i for i in order if i in selected] + [
                i for i in reversed(new_order) if i in selected and i not in old_pins]
            for i in spare:
                if after <= budget:
                    break
                selected.discard(i)
                text, _ = render(selected)
                after = count_tokens(text)
        # Counted from the sets, not along the way: a refused pin that was
        # already a row of W_t is an eviction, not a refusal.
        pinned &= selected | old_pins
        added = selected - before
        evicted = before - selected
        refused = [i for i in new_order if i not in selected and i not in before]

        # A repairable failure's signature, recorded for deferred promotion
        # (3.5.4): the same signature across many probes is what a ledger-wide
        # pass would target.
        signatures = Counter(f"{diag[q.id]['diagnosis']}:{q.kind}" for q in failed
                             if diag[q.id]["diagnosis"] != MISS_READ)
        rec.update({
            "diagnoses": dict(Counter(d["diagnosis"] for d in diag.values())),
            "signatures": dict(signatures),
            "rows_added": len(added), "rows_evicted": len(evicted),
            "pins_refused": len(refused),
            "n_claims_added": n_added, "n_pinned": len(pinned),
            "tokens_after_repair": after, "contract_ok": pinned <= selected,
        })
        if not rec["contract_ok"]:
            raise AssertionError(f"pins evicted at iteration {t}: the contract is broken")
        records.append(rec)
        _write_probe_rows(out / f"iter{t}_probes.jsonl", probes, answers, scores, diag)

    final = records[-1]["wiki_name"]
    md, meta = wiki_paths(profile, topic, final)
    f_md, f_meta = wiki_paths(profile, topic, FINAL_WIKI)
    f_md.write_text(md.read_text())
    f_meta.write_text(json.dumps({**json.loads(meta.read_text()), "final_of_loop": final},
                                 indent=2) + "\n")
    sidecar_path(profile, topic, FINAL_WIKI).write_text(
        sidecar_path(profile, topic, final).read_text())

    # Pins made after the last evaluation never reached a scored wiki; the
    # final wiki's pins are the ones its held-out split is about.
    final_pins = sorted(pinned & {i for row in json.loads(
        sidecar_path(profile, topic, final).read_text()) for i in row["claims"]})
    result = LoopResult(
        topic=topic, start=start, budget_tokens=budget, final_wiki=final, stop_reason=stop,
        iterations=records, pinned=final_pins,
        pinned_docs=sorted({led.by_id[i].doc for i in final_pins}),
        added_claims=[c.to_dict() for c in led.added],
        usage={"extractor": extractor.usage_summary(), "judge": judge.usage_summary()},
    )
    (out / "loop.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def _write_wiki(profile: Profile, topic: str, name: str, text: str,
                sidecar: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    md, meta_path = wiki_paths(profile, topic, name)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    sidecar_path(profile, topic, name).write_text(json.dumps(sidecar, indent=1) + "\n")


def _write_probe_rows(path: Path, probes: Sequence[Probe], answers: Sequence[str],
                      scores: Sequence[int], diag: dict[str, dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps({"probe_id": p.id, "doc": p.doc, "kind": p.kind, "relational": p.relational,
                    "score": s, "generated": a, **diag.get(p.id, {})}) + "\n"
        for p, a, s in zip(probes, answers, scores)))
