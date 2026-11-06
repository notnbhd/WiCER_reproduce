#!/usr/bin/env python3
"""TACT: ledger, typed wikis, self-generated probes, and the contract loop.

    python -m tact.run extract  --topic local_economy           # API: two calls per document
    python -m tact.run inspect  --topic local_economy           # free: ceiling, budget fit, shape
    python -m tact.run render   --topic local_economy --name tact_iter0 --rho 0.1
    python -m tact.run render   --topic local_economy --name tact_full  --rho 0
    python -m tact.run truncate --topic local_economy
    python -m tact.run score    --topic local_economy --condition tact_iter0
    python -m tact.run compare  --topic local_economy

Milestone B (the contract loop) adds:

    python -m tact.run probes   --topic local_economy           # API: one call per document
    python -m tact.run loop     --topic local_economy           # reader + judge, from tact_iter0
    python -m tact.run score    --topic local_economy --condition tact_loop

``render``, ``truncate`` and ``score`` need the reader's llama-server: the
budget is counted in its tokens, as wicer's compile counts it.  ``inspect``
uses it when it is up and otherwise estimates tokens from words.  ``score``
serves a wiki over every question and judges the answers, exactly as wicer's
``--eval-full`` does; ``compare`` then restricts to the sweep's held-out set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import Profile, add_profile_arg
from wicer_core.corpus import load_topic
from wicer_core.records import wiki_paths
from tact import get_profile


def _reader(profile: Profile):
    from wicer_core.reader import ReaderClient

    reader = ReaderClient(profile.reader)
    if not reader.health():
        raise SystemExit(
            f"no reader at {profile.reader.base_url}: the budget is counted in its "
            "tokens -- start it with benchmark/start_server.sh"
        )
    return reader


def main(argv: list[str] | None = None, *, profile: Profile | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {name: sub.add_parser(name) for name in
            ("extract", "inspect", "render", "truncate", "probes", "loop", "score", "compare")}
    for p in cmds.values():
        add_profile_arg(p)
        p.add_argument("--topic", action="append", required=True, help="topic slug (repeatable)")
    cmds["extract"].add_argument("--workers", type=int, default=8)
    cmds["extract"].add_argument(
        "--embed", action="store_true",
        help="also link entity names by MiniLM similarity (needs the rag extra)")
    cmds["inspect"].add_argument("--rho", type=float, default=0.10)
    cmds["inspect"].add_argument("--baseline-dir", default=None)
    cmds["render"].add_argument("--name", default="tact_iter0")
    cmds["render"].add_argument("--rho", type=float, default=0.10,
                                help="headroom held back for pins (default: %(default)s)")
    cmds["render"].add_argument("--style", choices=("typed", "verbatim"), default="typed",
                                help="typed rows, or the source sentences behind the selected claims")
    cmds["render"].add_argument("--select", choices=("roundrobin", "next", "density", "paragraph"),
                                default="roundrobin",
                                help="verbatim only: entity round-robin (+ the next sentence), "
                                     "or sentences / paragraphs by claims per token")
    cmds["truncate"].add_argument("--name", default="truncate")
    cmds["probes"].add_argument("--m", type=int, default=5, help="probes per document")
    cmds["probes"].add_argument("--workers", type=int, default=8)
    cmds["loop"].add_argument("--start", default="tact_iter0",
                              help="iteration-0 wiki to refine (default: %(default)s)")
    cmds["loop"].add_argument("--max-iter", type=int, default=None)
    cmds["loop"].add_argument("--closure", choices=("paragraph", "none"), default="paragraph",
                              help="what pinning a probe closes over (default: %(default)s)")
    cmds["score"].add_argument("--condition", required=True,
                               help="wiki name; also the condition its answers are filed under")
    cmds["score"].add_argument("--resume", action="store_true")
    cmds["compare"].add_argument("--baseline-dir", default=None)
    args = ap.parse_args(argv)
    profile = profile or get_profile(args.profile)

    if args.cmd == "extract":
        from tact.ledger import Extractor, build_ledger, embedding_similar

        extractor = Extractor(profile.compiler)
        similar = embedding_similar() if args.embed else None
        for topic in args.topic:
            led = build_ledger(profile, topic, extractor, workers=args.workers, similar=similar)
            m = led.meta
            print(f"{topic:<26} claims={m['n_claims']} rejected={m['rejection_rate']:.1%} "
                  f"multispan={m['n_multispan']} relational={m['n_relational']} "
                  f"contradictions={m['n_contradictions']} "
                  f"collisions={m['entity_collision_rate']:.1%}")
        print(f"\nextractor usage: {extractor.usage_summary()}")

    elif args.cmd == "inspect":
        from tact.checks import estimate_tokens, inspect_topic, summary
        from tact.compare import BASELINE_DIR
        from wicer_core.reader import ReaderClient

        reader = ReaderClient(profile.reader)
        if reader.health():
            count_tokens, source = reader.count_tokens, "reader"
        else:
            count_tokens, source = estimate_tokens, "estimated from words (no reader up)"
        for topic in args.topic:
            print(summary(inspect_topic(profile, topic, rho=args.rho, count_tokens=count_tokens,
                                        baseline_dir=args.baseline_dir or BASELINE_DIR,
                                        token_source=source)))

    elif args.cmd == "render":
        from tact.render import render_topic

        reader = _reader(profile)
        for topic in args.topic:
            r = render_topic(profile, topic, name=args.name, rho=args.rho,
                             count_tokens=reader.count_tokens, style=args.style,
                             select=args.select)
            m = r.meta
            mix = " ".join(f"{k}={v['n_realized']}/{v['n_claims']}" for k, v in m["routing"].items())
            print(f"{topic:<26} {args.name}: {m['actual_tokens']:,}/{m['target_tokens']:,} tokens "
                  f"(ratio {m['actual_ratio_tokens']:.1%}) coverage={m['coverage']:.1%} {mix}")

    elif args.cmd == "truncate":
        from tact.truncate import truncate_topic

        reader = _reader(profile)
        for topic in args.topic:
            _, m = truncate_topic(profile, topic, name=args.name, count_tokens=reader.count_tokens)
            print(f"{topic:<26} {args.name}: {m['actual_tokens']:,}/{m['budget_tokens']:,} tokens "
                  f"(ratio {m['actual_ratio_tokens']:.1%})")

    elif args.cmd == "probes":
        from tact.probes import ProbeWriter, build_probes

        writer = ProbeWriter(profile.compiler)
        for topic in args.topic:
            _, m = build_probes(profile, topic, writer, m=args.m, workers=args.workers)
            print(f"{topic:<26} probes={m['n_probes']} rejected={m['rejection_rate']:.1%} "
                  f"relational={m['n_relational']} quarters={m['position_quartiles']} "
                  f"kinds={m['kinds']}")
        print(f"\nprobe writer usage: {writer.usage_summary()}")

    elif args.cmd == "loop":
        from tact.ledger import Extractor
        from tact.loop import run_loop
        from wicer_core.judge import Judge

        reader = _reader(profile)
        judge, extractor = Judge(profile.judge), Extractor(profile.compiler)
        for topic in args.topic:
            res = run_loop(profile, topic, reader=reader, judge=judge, extractor=extractor,
                           start=args.start, max_iter=args.max_iter, closure=args.closure)
            for r in res.iterations:
                imp = f"{r['improvement']:+.1%}" if r["improvement"] is not None else "   --"
                print(f"  iter {r['iteration']}  probe_mean={r['probe_mean']:.2f} "
                      f"failures={r['n_failures']:>3} {r['diagnoses']} "
                      f"+{r['rows_added']}/-{r['rows_evicted']} rows "
                      f"refused={r.get('pins_refused', 0)} "
                      f"headroom={r['headroom_tokens']:,} imp={imp}")
            print(f"{topic:<26} final={res.final_wiki} stop={res.stop_reason} "
                  f"pinned={len(res.pinned)} claims in {len(res.pinned_docs)} documents")

    elif args.cmd == "score":
        from benchmark.evaluate import evaluate_condition
        from benchmark.run_benchmark import build_wiki_context, run_condition

        reader = _reader(profile)
        for topic in args.topic:
            md = wiki_paths(profile, topic, args.condition)[0]
            if not md.exists():
                raise SystemExit(f"no wiki {md}: render or truncate it first")
            run_condition(profile, topic, args.condition, build_wiki_context(md.read_text()),
                          load_topic(profile, topic).qa, reader=reader, resume=args.resume,
                          progress=sys.stderr.isatty())
            # The answers were just regenerated, so their old scores are stale
            # unless this is a resumed run; the judge's disk cache makes
            # re-judging an unchanged answer free.
            s = evaluate_condition(profile, topic, args.condition, resume=args.resume)
            print(f"{topic:<26} {args.condition}: scored {s['n_scored']} "
                  f"(resumed {s.get('n_resumed', 0)})")

    elif args.cmd == "compare":
        from tact.compare import (
            BASELINE_DIR, compare_topic, displacement, render_displacement, render_table,
        )

        for topic in args.topic:
            base = args.baseline_dir or BASELINE_DIR
            print(render_table(topic, compare_topic(profile, topic, base)))
            print(render_displacement(topic, displacement(profile, topic, base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
