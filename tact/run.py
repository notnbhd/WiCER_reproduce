#!/usr/bin/env python3
"""TACT milestone A: the iteration-0 arm and the truncation baseline.

    python -m tact.run extract  --topic local_economy           # API: two calls per document
    python -m tact.run render   --topic local_economy --name tact_iter0 --rho 0.1
    python -m tact.run render   --topic local_economy --name tact_full  --rho 0
    python -m tact.run truncate --topic local_economy
    python -m tact.run score    --topic local_economy --condition tact_iter0
    python -m tact.run compare  --topic local_economy

``render``, ``truncate`` and ``score`` need the reader's llama-server: the
budget is counted in its tokens, as wicer's compile counts it.  ``score``
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
            ("extract", "render", "truncate", "score", "compare")}
    for p in cmds.values():
        add_profile_arg(p)
        p.add_argument("--topic", action="append", required=True, help="topic slug (repeatable)")
    cmds["extract"].add_argument("--workers", type=int, default=8)
    cmds["extract"].add_argument(
        "--embed", action="store_true",
        help="also link entity names by MiniLM similarity (needs the rag extra)")
    cmds["render"].add_argument("--name", default="tact_iter0")
    cmds["render"].add_argument("--rho", type=float, default=0.10,
                                help="headroom held back for pins (default: %(default)s)")
    cmds["truncate"].add_argument("--name", default="truncate")
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

    elif args.cmd == "render":
        from tact.render import render_topic

        reader = _reader(profile)
        for topic in args.topic:
            r = render_topic(profile, topic, name=args.name, rho=args.rho,
                             count_tokens=reader.count_tokens)
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
            s = evaluate_condition(profile, topic, args.condition)
            print(f"{topic:<26} {args.condition}: scored {s['n_scored']} "
                  f"(resumed {s.get('n_resumed', 0)})")

    elif args.cmd == "compare":
        from tact.compare import BASELINE_DIR, compare_topic, render_table

        for topic in args.topic:
            print(render_table(topic, compare_topic(profile, topic,
                                                    args.baseline_dir or BASELINE_DIR)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
