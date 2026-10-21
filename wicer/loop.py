#!/usr/bin/env python3
"""WiCER: the compile-evaluate-refine loop (paper Algorithm 1).

    Q_probe <- one QA pair per source document              (80 probes/topic)
    W_0     <- Compile(D, r)                                 blind compilation
    F_cum   <- {}
    for t in 0..T-1:
        answers    <- generate each q in Q_probe against W_t  (local reader, free)
        Failures_t <- { q : judge_score(q) <= 1 }
        if |Failures_t| == 0 or (t > 0 and improvement < 10%): break
        for q in Failures_t:
            F_cum |= extract_critical_facts(source_doc(q))    ~50-100 words each
        W_{t+1} <- Compile(D, r, preserve = F_cum)
    return best W_t

Notes that matter for reproduction:

* The compiler never sees the probes or the evaluation questions.  Only the
  diagnosed facts cross the boundary.
* Convergence is empirical, not formal (paper Appendix I): pinned facts cannot
  be lost again, but pinning consumes budget and can displace other content --
  "random knowledge displacement".  That is why gains plateau after one or two
  iterations rather than accumulating, and why we do not raise T.
* The loop scores probes.  Paper Table 4 reports on those same 80 probes, so we
  record probe means for comparability -- and, with --eval-full, score the
  blind and the best pinned wiki over the full question set, so the report can
  measure both on questions the loop never saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve
from wicer_core.corpus import Corpus, QAPair, load_topic
from wicer_core.judge import Judge, JudgeRequest
from wicer_core.metrics import score_stats
from wicer_core.provenance import build_run_meta, write_run_meta
from wicer_core.reader import ContextOverflow, ReaderClient, build_user_prompt
from wicer_core.records import ledger_path, wiki_paths
from wicer import probe as probe_mod
from wicer.compile import Compiler, PreservedFact, compile_topic
from wicer.diagnose import Diagnoser, diagnose_failures
from benchmark.run_benchmark import build_wiki_context, run_condition

PinFn = Callable[[list[tuple[QAPair, Any, str]], int], list["PreservedFact"]]

BLIND_WIKI = "blind_moderate"
BLIND_CONDITION = "wiki_moderate"


@dataclass
class IterationRecord:
    iteration: int
    wiki_name: str
    probe_mean: float
    probe_score_1_pct: float
    n_failures: int
    n_pinned_facts: int
    pinned_words: int
    wiki_words: int
    wiki_tokens: int | None
    actual_ratio_words: float
    actual_ratio_tokens: float | None
    # Share of pinned facts that came back verbatim in the wiki this iteration
    # evaluated.  None on iteration 0, which pins nothing.  A rate well below
    # 1.0 means the compiler ignored the preservation instruction, and a flat
    # trajectory under it is an artefact of that, not of the method.
    n_preserved_found: int = 0
    preservation_rate: float | None = None
    improvement: float | None = None
    stop: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopResult:
    topic: str
    best_iteration: int
    best_wiki_name: str
    best_probe_mean: float
    blind_probe_mean: float
    stop_reason: str
    iterations: list[dict[str, Any]] = field(default_factory=list)
    preserved_facts: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


def answer_probes(
    reader: ReaderClient, wiki_text: str, probes: Sequence[QAPair], *, desc: str = ""
) -> list[str]:
    """Generate one answer per probe against a compiled wiki.

    Local inference -- free, which is why the probe set can be re-run on every
    iteration (paper Section 7.4: "the 80 local inference probes run at zero
    API cost").
    """
    context = build_wiki_context(wiki_text)
    reader.check_fits(context)
    reader.prime(context)
    iterator: Sequence[QAPair] = probes
    if desc:
        try:
            from tqdm import tqdm

            iterator = tqdm(probes, desc=desc, unit="probe", leave=False)
        except ImportError:
            pass
    return [reader.chat(context, build_user_prompt(q.question)).text.strip() for q in iterator]


def _wiki_stats(profile: Profile, topic: str, name: str) -> dict[str, Any]:
    _, meta_path = wiki_paths(profile, topic, name)
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def run_loop(
    profile: Profile,
    topic: str,
    *,
    reader: ReaderClient | None = None,
    judge: Judge | None = None,
    compiler: Compiler | None = None,
    diagnoser: Diagnoser | None = None,
    seed: int | None = None,
    force_blind: bool = False,
    pin_fn: "PinFn | None" = None,
    blind_wiki: str = BLIND_WIKI,
    iter_prefix: str = "wicer_iter",
    ledger_name: str = "wicer",
) -> LoopResult:
    cfg = profile.wicer
    corpus: Corpus = load_topic(profile, topic)
    probes, _heldout = probe_mod.load(profile, topic)

    reader = reader or ReaderClient(profile.reader)
    reader.wait_for_health()
    judge = judge or Judge(profile.judge)
    compiler = compiler or Compiler(
        profile.compiler,
        price_in=profile.judge.price_in_per_mtok,
        price_out=profile.judge.price_out_per_mtok,
    )
    diagnoser = diagnoser or Diagnoser(
        profile.compiler.model, cfg,
        backend=compiler.backend,
        price_in=profile.judge.price_in_per_mtok,
        price_out=profile.judge.price_out_per_mtok,
    )

    # W_0: blind compilation.  Reuse the existing one so the WiCER arm and the
    # Table 3 wiki-moderate arm share a baseline instead of drifting apart on
    # compiler non-determinism.
    blind_md, _ = wiki_paths(profile, topic, blind_wiki)
    if force_blind or not blind_md.exists():
        compile_topic(
            profile, topic, "moderate", compiler=compiler, seed=seed,
            name=blind_wiki, count_tokens=reader.count_tokens,
        )

    wikis: list[str] = [blind_wiki]
    records: list[IterationRecord] = []
    f_cum: list[PreservedFact] = []
    stop_reason = "max_iterations"
    prev_mean: float | None = None

    def evaluate(t: int, name: str) -> tuple[IterationRecord, list]:
        """Generate against W_t, judge, and identify catastrophic failures."""
        wiki_text = wiki_paths(profile, topic, name)[0].read_text()
        answers = answer_probes(reader, wiki_text, probes, desc=f"{topic}/{name}")
        judgements = judge.score_many([
            JudgeRequest(question=q.question, gold=q.answer, generated=a)
            for q, a in zip(probes, answers)
        ])
        stats = score_stats([j.score for j in judgements])
        failures = [
            (q, corpus.doc_by_id(q.document_id), a)
            for q, a, j in zip(probes, answers, judgements)
            if j.score <= cfg.failure_threshold
        ]
        _persist_probe_run(profile, topic, name, probes, answers, judgements)
        meta = _wiki_stats(profile, topic, name)
        return (
            IterationRecord(
                iteration=t,
                wiki_name=name,
                probe_mean=stats.mean,
                probe_score_1_pct=stats.pct_eq_1,
                n_failures=len(failures),
                n_pinned_facts=len(f_cum),
                pinned_words=sum(f.n_words for f in f_cum),
                wiki_words=meta.get("actual_words", len(wiki_text.split())),
                wiki_tokens=meta.get("actual_tokens"),
                actual_ratio_words=meta.get("actual_ratio_words", float("nan")),
                actual_ratio_tokens=meta.get("actual_ratio_tokens"),
                n_preserved_found=meta.get("n_preserved_found", 0),
                preservation_rate=meta.get("preservation_rate"),
            ),
            failures,
        )

    for t in range(cfg.max_iterations):
        record, failures = evaluate(t, wikis[t])
        if prev_mean is not None and prev_mean > 0:
            record.improvement = (record.probe_mean - prev_mean) / prev_mean

        if not failures:
            record.stop = stop_reason = "no_failures"
            records.append(record)
            break
        if (
            t > 0
            and record.improvement is not None
            and record.improvement < cfg.improvement_threshold
        ):
            record.stop = stop_reason = "insufficient_improvement"
            records.append(record)
            break
        records.append(record)
        prev_mean = record.probe_mean

        # Diagnose: what did the compiler drop, for each catastrophic failure?
        pin = pin_fn or (lambda fs, iteration: diagnose_failures(diagnoser, fs, iteration=iteration))
        f_cum.extend(pin(failures, t))

        next_name = f"{iter_prefix}{t + 1}"
        try:
            compile_topic(
                profile, topic, "moderate", compiler=compiler, preserve=f_cum,
                name=next_name, seed=seed, count_tokens=reader.count_tokens,
            )
        except ContextOverflow:
            stop_reason = "context_overflow"
            break
        wikis.append(next_name)

    # A wiki compiled on the final pass is never evaluated inside the loop.
    # Evaluate it here so "best W_t" ranges over every wiki actually produced.
    evaluated = {r.wiki_name for r in records}
    for t, name in enumerate(wikis):
        if name not in evaluated:
            record, _ = evaluate(t, name)
            if records:
                prev = records[-1].probe_mean
                if prev > 0:
                    record.improvement = (record.probe_mean - prev) / prev
            records.append(record)

    best = max(records, key=lambda r: r.probe_mean)
    result = LoopResult(
        topic=topic,
        best_iteration=best.iteration,
        best_wiki_name=best.wiki_name,
        best_probe_mean=best.probe_mean,
        blind_probe_mean=records[0].probe_mean if records else float("nan"),
        stop_reason=stop_reason,
        iterations=[r.to_dict() for r in records],
        preserved_facts=[f.to_dict() for f in f_cum],
        usage={
            "judge": judge.usage_summary(),
            "compiler": compiler.usage_summary(),
            "diagnosis": diagnoser.usage_summary(),
        },
    )
    lp = ledger_path(profile, topic, ledger_name)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def _persist_probe_run(
    profile: Profile, topic: str, name: str, probes, answers, judgements
) -> None:
    d = resolve(profile.results_dir) / topic / "probe_runs"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{name}.jsonl").open("w") as fh:
        for q, a, j in zip(probes, answers, judgements):
            fh.write(json.dumps({
                "question_id": q.question_id, "document_id": q.document_id,
                "question": q.question, "gold": q.answer, "generated": a,
                "score": j.score, "judge_model": j.model,
            }, ensure_ascii=False) + "\n")


def best_pinned_iteration(iterations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The iteration with the best probe mean among those that pinned something.

    The paper's "best W_t" ranges over every iteration, iteration 0 included --
    the unpinned blind -- so it can name a wiki no pinning produced.  This is
    the best wiki pinning did produce.  None when nothing was pinned.
    """
    pinned = [r for r in iterations if r.get("n_pinned_facts")]
    return max(pinned, key=lambda r: r["probe_mean"]) if pinned else None


def full_set_wikis(
    result: LoopResult, blind_condition: str, pinned_condition: str | None = None
) -> list[tuple[str, str]]:
    """The (wiki, condition) pairs ``--eval-full`` scores on every question.

    An arm's blind and its best pinned wiki: the pair that says what pinning
    did on questions the loop never saw.  The pinned wiki's condition defaults
    to the wiki's own name.
    """
    if not result.iterations:
        return []
    pairs = [(result.iterations[0]["wiki_name"], blind_condition)]
    best = best_pinned_iteration(result.iterations)
    if best is not None:
        pairs.append((best["wiki_name"], pinned_condition or best["wiki_name"]))
    return pairs


def evaluate_wikis_full(
    profile: Profile,
    topic: str,
    wikis: Sequence[tuple[str, str]],
    *,
    reader: ReaderClient | None = None,
) -> list[dict[str, Any]]:
    """Run compiled wikis over the whole question set, each as a normal condition.

    The loop itself scores only the probes.  ``wikis`` holds (wiki, condition)
    pairs, as ``full_set_wikis`` builds them.
    """
    reader = reader or ReaderClient(profile.reader)
    corpus = load_topic(profile, topic)
    out = []
    for name, condition in wikis:
        text = wiki_paths(profile, topic, name)[0].read_text()
        out.append(
            run_condition(
                profile, topic, condition, build_wiki_context(text), corpus.qa, reader=reader
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    ap.add_argument(
        "--seed", type=int,
        help="compiler seed; reaches only the offline fake compiler -- the API takes none",
    )
    ap.add_argument("--force-blind", action="store_true", help="recompile W_0 even if present")
    ap.add_argument(
        "--eval-full",
        action="store_true",
        help="after the loop, run the blind wiki and the best pinned iteration "
        "over the full question set",
    )
    ap.add_argument("--dry-run", action="store_true", help="estimate API cost and exit")
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    topics = args.topic or list(TOPICS)

    if args.dry_run:
        return _dry_run(profile, topics)

    reader = ReaderClient(profile.reader)
    judge = Judge(profile.judge)
    compiler = Compiler(
        profile.compiler,
        price_in=profile.judge.price_in_per_mtok,
        price_out=profile.judge.price_out_per_mtok,
    )
    diagnoser = Diagnoser(
        profile.compiler.model, profile.wicer, backend=compiler.backend,
        price_in=profile.judge.price_in_per_mtok, price_out=profile.judge.price_out_per_mtok,
    )

    for topic in topics:
        res = run_loop(
            profile, topic, reader=reader, judge=judge, compiler=compiler,
            diagnoser=diagnoser, seed=args.seed, force_blind=args.force_blind,
        )
        print(f"\n{topic}")
        for r in res.iterations:
            imp = f"{r['improvement']:+.1%}" if r["improvement"] is not None else "   --"
            kept = (
                f"kept={r['preservation_rate']:.0%} "
                if r["preservation_rate"] is not None
                else ""
            )
            print(
                f"  iter {r['iteration']}  {r['wiki_name']:<14} "
                f"probe_mean={r['probe_mean']:.2f}  score1={r['probe_score_1_pct']:.1f}%  "
                f"failures={r['n_failures']:>3}  pinned={r['n_pinned_facts']:>3} "
                f"{kept}({r['pinned_words']}w)  ratio={r['actual_ratio_words']:.1%}  imp={imp}"
            )
        print(
            f"  best: {res.best_wiki_name} (iter {res.best_iteration}, "
            f"probe_mean {res.best_probe_mean:.2f} vs blind {res.blind_probe_mean:.2f})  "
            f"stop={res.stop_reason}"
        )

        if args.eval_full:
            evaluate_wikis_full(profile, topic, full_set_wikis(res, BLIND_CONDITION), reader=reader)

        write_run_meta(
            resolve(profile.results_dir) / topic / "wicer_run_meta.json",
            build_run_meta(profile, command="wicer/loop.py", extra=res.usage, hash_model=False),
        )
    return 0


def _dry_run(profile: Profile, topics: Sequence[str]) -> int:
    compiler = Compiler(
        profile.compiler, price_in=profile.judge.price_in_per_mtok,
        price_out=profile.judge.price_out_per_mtok,
    )
    total = 0.0
    for topic in topics:
        corpus = load_topic(profile, topic)
        probes, _ = probe_mod.split_probes(corpus.qa, profile.wicer.probe_seed)
        iters = profile.wicer.max_iterations
        # Per iteration: 1 compilation, |probes| judge calls, <=|probes| diagnoses.
        compile_cost = compiler.estimate_cost_usd(corpus.n_words, iters)
        judge_cost = iters * len(probes) * (
            600 / 1e6 * profile.judge.price_in_per_mtok
            + 120 / 1e6 * profile.judge.price_out_per_mtok
        )
        diag_cost = iters * (len(probes) * 0.5) * (
            1000 / 1e6 * profile.judge.price_in_per_mtok
            + 150 / 1e6 * profile.judge.price_out_per_mtok
        )
        est = compile_cost + judge_cost + diag_cost
        total += est
        print(
            f"{topic:<26} words={corpus.n_words:>7} probes={len(probes):>3} "
            f"iters<={iters}  compile=${compile_cost:.2f} judge=${judge_cost:.2f} "
            f"diagnose=${diag_cost:.2f}  est=${est:.2f}"
        )
    print(f"\nestimated total: ${total:.2f} (upper bound: ignores early stopping, "
          f"prompt-cache savings and the judge cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
