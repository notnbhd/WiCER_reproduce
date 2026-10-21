#!/usr/bin/env python3
"""Random-pinning ablation control (paper Section 7.5, Table 5).

This is the control that separates "diagnosis works" from "pinning anything
works".  For each catastrophic failure it pins a random 50-100 word passage
from a random source document instead of the diagnosed facts.  Everything else
-- probe set, compression level, iteration count, convergence rule, judge --
is identical to WiCER.

Two details the paper is specific about:

* "Each topic uses an independent blind compilation" (Section 7.5), which is
  why blind baselines differ slightly between Tables 4 and 5.  We compile a
  separate ``random_blind_moderate`` rather than reusing WiCER's W_0.
* The pinned passage is length-matched to a diagnosis (50-100 words), so the
  comparison isolates *what* is pinned, not *how much*.

Expected result: random pinning buys about +0.16 over blind; WiCER buys +0.95.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile
from wicer_core.corpus import Corpus, Document, QAPair, load_topic
from wicer_core.judge import Judge
from wicer_core.reader import ReaderClient
from wicer.compile import Compiler, PreservedFact
from wicer.diagnose import Diagnoser
from wicer.loop import LoopResult, evaluate_wikis_full, full_set_wikis, run_loop

BLIND_WIKI = "random_blind_moderate"
BLIND_CONDITION = "random_blind"
ITER_PREFIX = "random_pin_iter"
CONDITION = "random_pin"


class RandomPinner:
    """Pins a random length-matched passage from a random source document."""

    def __init__(self, corpus: Corpus, min_words: int, max_words: int, seed: int):
        self.documents = corpus.documents
        self.min_words = min_words
        self.max_words = max_words
        self.rng = random.Random(seed)

    def pin(self, probe: QAPair, iteration: int) -> PreservedFact:
        doc: Document = self.rng.choice(self.documents)
        words = doc.text.split()
        n = self.rng.randint(self.min_words, self.max_words)
        if len(words) <= n:
            start = 0
        else:
            start = self.rng.randrange(0, len(words) - n)
        return PreservedFact(
            document_id=doc.document_id,
            question_id=probe.question_id,
            text=" ".join(words[start : start + n]),
            iteration=iteration,
        )

    def __call__(
        self, failures: Sequence[tuple[QAPair, Document, str]], iteration: int
    ) -> list[PreservedFact]:
        return [self.pin(probe, iteration) for probe, _doc, _generated in failures]


def run_random_pin(
    profile: Profile,
    topic: str,
    *,
    reader: ReaderClient | None = None,
    judge: Judge | None = None,
    compiler: Compiler | None = None,
    diagnoser: Diagnoser | None = None,
    seed: int | None = None,
) -> LoopResult:
    corpus = load_topic(profile, topic)
    cfg = profile.wicer
    # A topic-dependent seed keeps topics independent while staying reproducible.
    # Not builtin hash(): string hashing is salted per process, so it would
    # silently give a different control arm on every run.
    digest = int.from_bytes(hashlib.sha256(topic.encode()).digest()[:4], "big")
    pin_seed = cfg.random_pin_seed + digest % 100_000
    pinner = RandomPinner(corpus, cfg.diagnosis_min_words, cfg.diagnosis_max_words, pin_seed)

    return run_loop(
        profile,
        topic,
        reader=reader,
        judge=judge,
        compiler=compiler,
        diagnoser=diagnoser,
        seed=seed,
        pin_fn=pinner,
        blind_wiki=BLIND_WIKI,
        iter_prefix=ITER_PREFIX,
        ledger_name="random_pin",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    ap.add_argument("--seed", type=int, help="compiler seed")
    ap.add_argument(
        "--eval-full",
        action="store_true",
        help="run the arm's blind ('random_blind') and its best pinned iteration "
        "('random_pin') over the full question set",
    )
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    reader = ReaderClient(profile.reader)
    judge = Judge(profile.judge)
    compiler = Compiler(
        profile.compiler, price_in=profile.judge.price_in_per_mtok,
        price_out=profile.judge.price_out_per_mtok,
    )
    diagnoser = Diagnoser(
        profile.compiler.model, profile.wicer, backend=compiler.backend,
        price_in=profile.judge.price_in_per_mtok, price_out=profile.judge.price_out_per_mtok,
    )

    for topic in args.topic or list(TOPICS):
        res = run_random_pin(
            profile, topic, reader=reader, judge=judge, compiler=compiler,
            diagnoser=diagnoser, seed=args.seed,
        )
        print(f"\n{topic}")
        for r in res.iterations:
            print(
                f"  iter {r['iteration']}  {r['wiki_name']:<20} "
                f"probe_mean={r['probe_mean']:.2f}  score1={r['probe_score_1_pct']:.1f}%  "
                f"failures={r['n_failures']:>3}  pinned={r['n_pinned_facts']:>3}"
            )
        print(
            f"  best: {res.best_wiki_name} (probe_mean {res.best_probe_mean:.2f} "
            f"vs blind {res.blind_probe_mean:.2f}, delta "
            f"{res.best_probe_mean - res.blind_probe_mean:+.2f})  stop={res.stop_reason}"
        )

        if args.eval_full:
            evaluate_wikis_full(
                profile, topic, full_set_wikis(res, BLIND_CONDITION, CONDITION), reader=reader
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
