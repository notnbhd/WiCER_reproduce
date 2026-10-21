# Representative Probes and Compiler-Noise Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw each document's WiCER probe at random from all five of its questions, score each arm's blind and best pinned wiki on every question, and report each arm's effect against the gap between two blind compilations of the same prompt.

**Architecture:** Probe selection becomes a seeded uniform draw per document (`wicer/probe.py`). One shared helper picks an arm's best pinned iteration; both arms' `--eval-full` score their blind plus that wiki (`wicer/loop.py`, `wicer/random_pin.py`). The report finds which condition scored which wiki from each condition's `context.txt`, and gains an "Effects against compiler noise" section (`benchmark/report.py`).

**Tech Stack:** Python 3.12, pytest, the repo's offline harness (`tests/fake_llama_server.py`, fake judge/compiler/diagnoser backends), ruff.

**Spec:** `docs/superpowers/specs/2026-10-21-representative-probes-and-noise-floor-design.md`

## Global Constraints

- Probe pool: all five questions per document, drawn uniformly; unanswerable questions eligible.
- `WicerConfig.probe_seed` default `20261021`; each document's draw is seeded from `sha256(f"{seed}:{document_id}")`.
- One compilation per arm. No replicates, no replicate directories.
- `--eval-full` scores each arm's blind and best pinned iteration: WiCER `blind_moderate` -> `wiki_moderate`, best pinned `wicer_iterN` -> `wicer_iterN`; random `random_blind_moderate` -> `random_blind`, best pinned `random_pin_iterN` -> `random_pin`.
- Best pinned iteration = highest `probe_mean` among iterations with `n_pinned_facts` > 0; `None` when nothing was pinned.
- The report identifies the condition that scored a wiki by the wiki its `context.txt` served, never by naming conventions.
- Noise band = 2 x RMS of the blind gaps over the topics present; mean +/- SE over the topics where a Δ exists; SE prints `--` with one topic; a wiki never scored on a question set prints `--`.
- Held-out = answerable questions the loop never saw (probes and unanswerable questions excluded).
- Tests run offline only: the `profile` fixture sets `WICER_JUDGE_BACKEND=fake` and `WICER_COMPILER_BACKEND=fake`. Never call an API.
- Run tests with `.venv/bin/python -m pytest -q` from the repo root.
- Commit straight to `main` (solo repo). Stage files by path; never `git add -A`, because `results-arts/` must stay untracked. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Match the surrounding code: `from __future__ import annotations`, type hints, docstrings and comments that say why. No new dependencies.
- No new ruff findings in changed files, compared against `HEAD` (`.venv/bin/ruff check`).

---

## Before you start: commit the step-1 work

The working tree carries the uncommitted step-1 fixes (matched recovery denominator, held-out without probes, unanswerable split, judge parser, UTF-8 decoding, the repaired `results-rental` scores). Tasks 3 and 4 edit `benchmark/report.py` and `tests/test_benchmark.py`, which step 1 also changed, so those changes must be committed first or they will land inside Task 3's commit. **Ask the user before running this commit.**

- [ ] **Confirm the suite is green**

Run: `.venv/bin/python -m pytest -q`
Expected: `245 passed`

- [ ] **Commit step 1 (with the user's OK)**

```bash
git add benchmark/evaluate.py benchmark/report.py wicer_core/corpus.py wicer_core/judge.py \
  wicer_core/reader.py tests/fake_llama_server.py tests/test_benchmark.py tests/test_corpus.py \
  tests/test_judge.py tests/test_reader.py \
  results-rental/local_economy/fc_raw/scores.jsonl \
  results-rental/local_economy/wicer_iter1/scores.jsonl \
  results-rental/local_economy/wiki_moderate/scores.jsonl \
  results-rental/per_question.csv results-rental/per_topic.csv results-rental/report.md
git commit -m "$(cat <<'EOF'
Score WiCER on matched question sets; fix judge parsing and stream decoding

Probe-set recovery now divides by full context on the same probes; held-out
excludes the probes and scores unanswerable questions apart; a random_pin
row that served the unpinned blind is marked. The judge reads verdicts whose
reasoning has unescaped quotes and no longer caches failures; seven
mis-recorded scores were repaired from their stored text. The reader decodes
the SSE stream as UTF-8.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `wicer_core/config.py` | Run profiles | `WicerConfig.probe_seed` |
| `wicer/probe.py` | Probe selection | Seeded uniform draw per document |
| `wicer/loop.py` | WiCER loop | `best_pinned_iteration`, `full_set_wikis`; `evaluate_wikis_full` takes (wiki, condition) pairs; `--eval-full` scores blind + best pinned |
| `wicer/random_pin.py` | Random-pin control | `BLIND_CONDITION = "random_blind"`; `--eval-full` scores blind + best pinned |
| `wicer_core/records.py` | Result layout | `CONDITIONS` gains `random_blind`, `wicer_iter3` |
| `benchmark/report.py` | Report | Served-wiki lookup; held-out uses best pinned WiCER; noise section |
| `tests/test_wicer.py` | Loop/probe tests | Probe draw, best pinned, `--eval-full` |
| `tests/test_benchmark.py` | Report tests | Fixture writes wikis and contexts; held-out and noise tests |
| `README.md`, `DEVIATIONS.md` | Docs | Seed advice, probe selection, noise floor |

---

### Task 1: Draw each document's probe uniformly from its five questions

**Files:**
- Modify: `wicer_core/config.py:122-132` (`WicerConfig`)
- Modify: `wicer/probe.py:1-35` (docstring, imports, `split_probes`), `wicer/probe.py:45` (`build`)
- Modify: `wicer/loop.py:392` (`_dry_run`)
- Test: `tests/test_wicer.py:1-41` (imports, probe tests), `tests/test_wicer.py:176-194` (loop test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `split_probes(qa: list[QAPair], seed: int) -> tuple[list[QAPair], list[QAPair]]` (seed now required); `WicerConfig.probe_seed: int = 20261021`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_wicer.py`, change the imports at the top to:

```python
from __future__ import annotations

import math
from collections import Counter

import pytest

from wicer_core.corpus import QAPair, load_topic
```

(keep the remaining import lines unchanged). Replace the three existing probe tests (lines 22-41) with the seeded calls, and add the new tests after them:

```python
def test_exactly_one_probe_per_source_document(profile, topic):
    corpus = load_topic(profile, topic)
    probes, heldout = split_probes(corpus.qa, profile.wicer.probe_seed)
    doc_ids = {d.document_id for d in corpus.documents}
    assert len(probes) == len(doc_ids)
    assert {p.document_id for p in probes} == doc_ids
    assert len(probes) + len(heldout) == len(corpus.qa)


def test_probes_and_heldout_are_disjoint(profile, topic):
    probes, heldout = split_probes(load_topic(profile, topic).qa, profile.wicer.probe_seed)
    assert not ({p.question_id for p in probes} & {h.question_id for h in heldout})


def test_probe_selection_is_order_independent(profile, topic):
    qa = load_topic(profile, topic).qa
    a, _ = split_probes(qa, profile.wicer.probe_seed)
    b, _ = split_probes(list(reversed(qa)), profile.wicer.probe_seed)
    assert [q.question_id for q in a] == [q.question_id for q in b]


def _documents(n_docs: int) -> list[QAPair]:
    """``n_docs`` documents with five questions each; the fifth is unanswerable."""
    return [
        QAPair(
            question_id=f"d{d:04d}-q{i}",
            document_id=f"d{d:04d}",
            question=f"question {i}?",
            answer="The answer is not found in the document." if i == 5 else f"answer {i}",
        )
        for d in range(n_docs)
        for i in range(1, 6)
    ]


def test_each_probe_is_drawn_uniformly_from_its_documents_five_questions():
    """Always taking the first question never probed an unanswerable one and
    favoured questions about a document's main subject, which a compiler keeps."""
    probes, _ = split_probes(_documents(1000), seed=20261021)
    positions = Counter(p.question_id.rsplit("-q", 1)[1] for p in probes)
    assert sorted(positions) == ["1", "2", "3", "4", "5"]
    # 200 expected per position; 150-250 is about +/-4 standard deviations.
    assert all(150 <= n <= 250 for n in positions.values()), positions


def test_a_different_seed_draws_different_probes():
    qa = _documents(50)
    first, _ = split_probes(qa, seed=1)
    second, _ = split_probes(qa, seed=2)
    assert [p.question_id for p in first] != [p.question_id for p in second]


def test_build_draws_with_the_profiles_probe_seed(profile, topic):
    from dataclasses import replace

    from wicer.probe import build

    a, _ = build(profile, topic)
    other = replace(profile, wicer=replace(profile.wicer, probe_seed=profile.wicer.probe_seed + 1))
    b, _ = build(other, topic)
    assert [p.question_id for p in a] != [p.question_id for p in b]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_wicer.py -k "probe or documents or seed"`
Expected: FAIL — `TypeError: split_probes() takes 1 positional argument but 2 were given` (and a `TypeError` from `replace(..., probe_seed=...)` because `WicerConfig` has no such field).

- [ ] **Step 3: Implement the seeded draw**

In `wicer_core/config.py`, add the field at the end of `WicerConfig` (after `random_pin_seed`):

```python
    random_pin_seed: int = 20260508
    # Seeds which of a document's questions becomes its probe (wicer/probe.py).
    probe_seed: int = 20261021
```

Replace `wicer/probe.py` lines 1-35 (module docstring through `split_probes`) with:

```python
#!/usr/bin/env python3
"""Diagnostic probe selection.

Paper Algorithm 1, line 1: "Select one QA pair per source document".  With 80
documents per topic that gives 80 probes and leaves ~320 pairs as the held-out
evaluation set.

One probe per document, not eighty probes sampled at random, is the point:
coverage is uniform, so every document gets an equal chance to reveal that the
compiler dropped it.  Within a document the probe is drawn uniformly from its
questions, so the probe set looks like the full set -- RepLiQA's unanswerable
questions included, at their usual rate.  The two pilots took each document's
first question instead, which never picked an unanswerable one (they are
almost all a document's 4th or 5th) and favoured questions about a document's
main subject, which a compiler keeps.  The draw is seeded per document, so a
rerun probes the same questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve
from wicer_core.corpus import QAPair, load_topic


def split_probes(qa: list[QAPair], seed: int) -> tuple[list[QAPair], list[QAPair]]:
    """(probes, held-out).  One probe per document, drawn uniformly from its questions."""
    by_doc: dict[str, list[QAPair]] = {}
    for q in sorted(qa, key=lambda q: (q.document_id, q.question_id)):
        by_doc.setdefault(q.document_id, []).append(q)
    probes = [_draw(by_doc[d], seed, d) for d in sorted(by_doc)]
    probe_ids = {q.question_id for q in probes}
    heldout = [q for q in qa if q.question_id not in probe_ids]
    return probes, heldout


def _draw(questions: list[QAPair], seed: int, document_id: str) -> QAPair:
    """Seeded per document, so the pick depends on neither input order nor
    which other documents are present."""
    digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).choice(questions)
```

In `wicer/probe.py` `build()`, change the split call:

```python
    probes, heldout = split_probes(corpus.qa, profile.wicer.probe_seed)
```

In `wicer/loop.py` `_dry_run()` (line 392), change the split call:

```python
        probes, _ = probe_mod.split_probes(corpus.qa, profile.wicer.probe_seed)
```

- [ ] **Step 4: Run the probe tests to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_wicer.py -k "probe or documents or seed"`
Expected: PASS

- [ ] **Step 5: Run the whole loop test file and expect one known failure**

Run: `.venv/bin/python -m pytest -q tests/test_wicer.py`
Expected: exactly one FAIL, `test_loop_diagnoses_failures_pins_them_and_improves`, at `assert res.best_probe_mean > res.blind_probe_mean` (`3.6666666666666665 > 3.6666666666666665`).

This is the test, not the code. With the new probes the fake loop's iteration 1 scores below the blind: its three diagnosed probes all go from 1 to 5, but five of the other nine probes get worse, because the fake compiler shares one hard word budget and pinning displaces coverage sharply (`DEVIATIONS.md` section 9). With first-question probes the overall mean happened to rise. What the loop promises is targeted repair, so the test should assert that.

- [ ] **Step 6: Rewrite that test around targeted repair**

Replace `test_loop_diagnoses_failures_pins_them_and_improves` (lines 176-194) with:

```python
def _probe_scores(profile, topic, wiki):
    """Per-probe judge scores from the loop's own run against ``wiki``."""
    import json

    from wicer_core.config import resolve

    path = resolve(profile.results_dir) / topic / "probe_runs" / f"{wiki}.jsonl"
    return {r["question_id"]: r["score"] for r in map(json.loads, path.open())}


def test_loop_diagnoses_failures_and_pins_back_what_they_needed(profile, topic):
    """Every probe diagnosed at iteration 0 passes at iteration 1.

    That targeted repair is the mechanism.  The overall probe mean is not
    promised to rise: pinning displaces other coverage (paper Appendix I), and
    sharply so in the fake compiler's fixed word budget (DEVIATIONS.md 9).
    """
    reader = ReaderClient(profile.reader)
    judge = Judge(profile.judge, backend="fake")
    compiler = Compiler(profile.compiler, backend="fake")
    diagnoser = Diagnoser(profile.compiler.model, profile.wicer, backend="fake")

    res = run_loop(profile, topic, reader=reader, judge=judge,
                   compiler=compiler, diagnoser=diagnoser)

    assert res.iterations[0]["wiki_name"] == BLIND_WIKI
    assert res.iterations[0]["n_pinned_facts"] == 0
    assert res.iterations[0]["n_failures"] > 0, "fixture must produce failures to diagnose"
    assert len(res.iterations) > 1, "loop must refine at least once"

    # Pinned facts accumulate and are never discarded (paper Appendix I).
    pinned = [r["n_pinned_facts"] for r in res.iterations]
    assert pinned == sorted(pinned)

    diagnosed = {f["question_id"] for f in res.preserved_facts if f["iteration"] == 0}
    after = _probe_scores(profile, topic, res.iterations[1]["wiki_name"])
    assert diagnosed and all(after[q] > 1 for q in diagnosed), {q: after[q] for q in diagnosed}
    assert res.stop_reason in {"no_failures", "insufficient_improvement", "max_iterations"}
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `248 passed`

- [ ] **Step 8: Commit**

```bash
git add wicer_core/config.py wicer/probe.py wicer/loop.py tests/test_wicer.py
git commit -m "$(cat <<'EOF'
Draw each document's probe uniformly from its five questions

The first question was never one of RepLiQA's unanswerable questions and
tended to ask about a document's main subject, which a compiler keeps, so
first-question probes ran 0.4-0.5 above the full set. The draw is seeded per
document from WicerConfig.probe_seed.

The loop test now checks targeted repair: with representative probes the
fake compiler's fixed budget makes the overall probe mean fall even though
every diagnosed probe recovers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Score each arm's blind and best pinned wiki on every question

**Files:**
- Modify: `wicer/loop.py:24-26` (docstring), `wicer/loop.py:293-313` (`evaluate_wikis_full`, plus two new functions before it), `wicer/loop.py:322-326` and `373-375` (`main`)
- Modify: `wicer/random_pin.py:38-42` (constants, import), `wicer/random_pin.py:114-118` and `151-159` (`main`)
- Modify: `wicer_core/records.py:21-31` (`CONDITIONS`)
- Modify: `benchmark/report.py:29` (import), `benchmark/report.py:287-298` (`_best_pinned`)
- Test: `tests/test_wicer.py`, `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `split_probes(qa, seed)` (Task 1, via `run_loop`).
- Produces:
  - `wicer.loop.best_pinned_iteration(iterations: Sequence[dict[str, Any]]) -> dict[str, Any] | None`
  - `wicer.loop.full_set_wikis(result: LoopResult, blind_condition: str, pinned_condition: str | None = None) -> list[tuple[str, str]]` — (wiki, condition) pairs
  - `wicer.loop.evaluate_wikis_full(profile, topic, wikis: Sequence[tuple[str, str]], *, reader=None) -> list[dict[str, Any]]`
  - `wicer.random_pin.BLIND_CONDITION = "random_blind"`
  - `wicer_core.records.CONDITIONS` includes `"random_blind"` and `"wicer_iter3"`

- [ ] **Step 1: Write the failing tests**

In `tests/test_wicer.py`, change the loop import line to:

```python
from wicer.loop import (
    BLIND_WIKI, LoopResult, best_pinned_iteration, evaluate_wikis_full, full_set_wikis, run_loop,
)
```

Add these tests after `test_random_pin_control_uses_an_independent_blind_compilation`:

```python
def _iteration(i: int, wiki: str, probe_mean: float, n_pinned: int) -> dict:
    return {"iteration": i, "wiki_name": wiki, "probe_mean": probe_mean,
            "n_pinned_facts": n_pinned}


def _loop_result(iterations: list[dict], best: str) -> LoopResult:
    return LoopResult(
        topic="t", best_iteration=0, best_wiki_name=best, best_probe_mean=0.0,
        blind_probe_mean=0.0, stop_reason="insufficient_improvement", iterations=iterations,
    )


def test_best_pinned_iteration_never_returns_the_unpinned_blind():
    iters = [_iteration(0, "blind", 3.20, 0), _iteration(1, "first", 3.06, 13),
             _iteration(2, "second", 2.90, 17)]
    assert best_pinned_iteration(iters)["wiki_name"] == "first"
    assert best_pinned_iteration(iters[:1]) is None


def test_eval_full_scores_the_pinned_wiki_even_when_best_of_is_the_blind():
    """Both pilots' random arm peaked at its blind, so the condition labelled
    random_pin was a second blind and no random-pinned wiki was ever scored."""
    res = _loop_result(
        [_iteration(0, "random_blind_moderate", 3.20, 0),
         _iteration(1, "random_pin_iter1", 3.06, 13)],
        best="random_blind_moderate",
    )
    assert full_set_wikis(res, "random_blind", "random_pin") == [
        ("random_blind_moderate", "random_blind"),
        ("random_pin_iter1", "random_pin"),
    ]


def test_eval_full_names_a_wicer_pinned_wiki_after_itself():
    res = _loop_result(
        [_iteration(0, "blind_moderate", 3.16, 0), _iteration(1, "wicer_iter1", 3.66, 13),
         _iteration(2, "wicer_iter2", 3.30, 17)],
        best="wicer_iter1",
    )
    assert full_set_wikis(res, "wiki_moderate") == [
        ("blind_moderate", "wiki_moderate"), ("wicer_iter1", "wicer_iter1"),
    ]


def test_eval_full_scores_only_the_blind_when_nothing_was_pinned():
    res = _loop_result([_iteration(0, "blind_moderate", 4.9, 0)], best="blind_moderate")
    assert full_set_wikis(res, "wiki_moderate") == [("blind_moderate", "wiki_moderate")]


def test_random_arm_scores_its_blind_and_its_pinned_wiki_on_every_question(profile, topic):
    from benchmark.run_benchmark import build_wiki_context
    from wicer.random_pin import BLIND_CONDITION, CONDITION
    from wicer_core.records import answers_path, read_jsonl, run_dir, wiki_paths

    res = run_random_pin(
        profile, topic,
        reader=ReaderClient(profile.reader),
        judge=Judge(profile.judge, backend="fake"),
        compiler=Compiler(profile.compiler, backend="fake"),
        diagnoser=Diagnoser(profile.compiler.model, profile.wicer, backend="fake"),
    )
    pairs = full_set_wikis(res, BLIND_CONDITION, CONDITION)
    assert len(pairs) == 2, "fixture must pin something"
    evaluate_wikis_full(profile, topic, pairs, reader=ReaderClient(profile.reader))

    n_questions = len(load_topic(profile, topic).qa)
    for wiki, condition in pairs:
        served = (run_dir(profile, topic, condition) / "context.txt").read_text()
        assert served == build_wiki_context(wiki_paths(profile, topic, wiki)[0].read_text())
        assert len(read_jsonl(answers_path(profile, topic, condition))) == n_questions
```

In `tests/test_benchmark.py`, add after `test_per_question_csv_marks_probes_and_unanswerable_questions`:

```python
def test_report_collects_every_condition_the_full_set_evaluation_can_write(profile, topic):
    """--eval-full can score the random arm's blind and a third iteration's wiki."""
    from wicer_core.records import CONDITIONS

    for condition in ("random_blind", "wicer_iter3"):
        _write_condition(profile, topic, condition, _SCORES["wiki_moderate"])
    found = {r.condition for r in collect(profile, [topic], list(CONDITIONS))}
    assert {"random_blind", "wicer_iter3"} <= found
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_wicer.py tests/test_benchmark.py -k "pinned or eval_full or random_arm or every_condition"`
Expected: FAIL — `ImportError: cannot import name 'best_pinned_iteration' from 'wicer.loop'` for `tests/test_wicer.py`, and the `every_condition` test failing its subset assertion.

- [ ] **Step 3: Implement the helper and the new `--eval-full`**

Line numbers below are as they stand before this task's edits; each edit shows the code to find.

In `wicer/loop.py`, replace the last bullet of the module docstring (lines 24-26) with:

```python
* The loop scores probes.  Paper Table 4 reports on those same 80 probes, so we
  record probe means for comparability -- and, with --eval-full, score the
  blind and the best pinned wiki over the full question set, so the report can
  measure both on questions the loop never saw.
```

Replace `evaluate_wikis_full` (lines 293-313) with these three functions:

```python
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
```

In `wicer/loop.py` `main()`, change the `--eval-full` help (lines 322-326):

```python
    ap.add_argument(
        "--eval-full",
        action="store_true",
        help="after the loop, run the blind wiki and the best pinned iteration "
        "over the full question set",
    )
```

and the `--eval-full` block (lines 373-375):

```python
        if args.eval_full:
            evaluate_wikis_full(profile, topic, full_set_wikis(res, BLIND_CONDITION), reader=reader)
```

In `wicer/random_pin.py`, change the loop import and constants (lines 38-42):

```python
from wicer.loop import LoopResult, evaluate_wikis_full, full_set_wikis, run_loop

BLIND_WIKI = "random_blind_moderate"
BLIND_CONDITION = "random_blind"
ITER_PREFIX = "random_pin_iter"
CONDITION = "random_pin"
```

Change the `--eval-full` help (lines 114-118):

```python
    ap.add_argument(
        "--eval-full",
        action="store_true",
        help="run the arm's blind ('random_blind') and its best pinned iteration "
        "('random_pin') over the full question set",
    )
```

and replace the `--eval-full` block (lines 151-159, including its local imports) with:

```python
        if args.eval_full:
            evaluate_wikis_full(
                profile, topic, full_set_wikis(res, BLIND_CONDITION, CONDITION), reader=reader
            )
```

In `wicer_core/records.py`, replace `CONDITIONS` (lines 21-31):

```python
CONDITIONS = (
    "fc_raw",
    "fc_raw_q4kv",
    "rag_top5",
    "wiki_light",
    "wiki_moderate",
    "wiki_aggressive",
    "wicer_iter1",
    "wicer_iter2",
    # max_iterations=3 lets the loop compile a third pinned wiki, and
    # --eval-full scores it under its own name when it is the best pinned.
    "wicer_iter3",
    "random_blind",
    "random_pin",
)
```

In `benchmark/report.py`, change the loop import (line 29):

```python
from wicer.loop import BLIND_CONDITION, BLIND_WIKI, best_pinned_iteration
```

and make `_best_pinned` (lines 287-298) delegate, keeping its docstring:

```python
def _best_pinned(ledger: dict[str, Any]) -> float | None:
    """Best score among the iterations that actually pinned something.

    ``max`` over every iteration includes iteration 0, the unpinned blind, so an
    arm whose pinning only ever made things worse reports its own baseline and a
    delta of exactly zero -- reading as "pinning did nothing" when the truth is
    "pinning hurt, and selection discarded it".  Identifies pinned iterations by
    ``n_pinned_facts`` rather than by index, so it holds if the loop ever stops
    numbering from zero.  Returns ``None`` when no iteration pinned anything.
    """
    best = best_pinned_iteration(ledger["iterations"])
    return best["probe_mean"] if best else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_wicer.py tests/test_benchmark.py`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `254 passed`

- [ ] **Step 6: Commit**

```bash
git add wicer/loop.py wicer/random_pin.py wicer_core/records.py benchmark/report.py \
  tests/test_wicer.py tests/test_benchmark.py
git commit -m "$(cat <<'EOF'
Score each arm's blind and best pinned wiki on every question

--eval-full scored the arm's best-of iteration, which can be its unpinned
blind: in both pilots the random arm's "random_pin" was a second blind, so
no random-pinned wiki was ever scored on held-out questions. Both arms now
score their blind (wiki_moderate / the new random_blind) and their best
pinned iteration. One helper picks that iteration for both arms and the
report. CONDITIONS gains random_blind and wicer_iter3.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Held-out compares the best pinned WiCER wiki, found by what each condition served

**Files:**
- Modify: `benchmark/report.py:29` (import), `benchmark/report.py:255-270` (replace `_best_condition` and `_result`), the `blind = _by(results, "wiki_moderate")` line in the Table 4 block, and the held-out block in `render()` (currently lines 530-562)
- Test: `tests/test_benchmark.py` (synthetic fixture and held-out tests)

**Interfaces:**
- Consumes: `best_pinned_iteration` (Task 2).
- Produces (in `benchmark/report.py`, used by Task 4):
  - `_scored(results: Sequence[CondResult], topic: str, wiki: str | None) -> CondResult | None`
  - `_held_out(profile: Profile, results: Sequence[CondResult], topic: str) -> tuple[frozenset[str], frozenset[str]]` — (answerable, unanswerable) held-out question ids
- Produces (in `tests/test_benchmark.py`, used by Task 4): `_write_wiki(profile, topic, name, n_pinned) -> str`, `_arm_ledger(topic, iterations) -> dict`, `_write_probes(profile, topic, qids) -> None`

- [ ] **Step 1: Give the synthetic fixture real wikis and contexts, and write the failing test**

In `tests/test_benchmark.py`, replace `_synthetic_wicer_topic` with these helpers and the rewritten fixture:

```python
def _write_wiki(profile, topic, name, n_pinned):
    """A compiled wiki on disk; returns the context a condition serving it holds."""
    from wicer_core.records import wiki_paths

    md, meta = wiki_paths(profile, topic, name)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(f"# {name}\n\nCompiled body.")
    meta.write_text(json.dumps({"n_preserved_facts": n_pinned}))
    return build_wiki_context(md.read_text())


def _arm_ledger(topic, iterations):
    """A loop ledger from (wiki_name, probe_mean, n_pinned_facts) per iteration."""
    return {"topic": topic, "iterations": [
        {"iteration": i, "wiki_name": wiki, "probe_mean": m, "probe_score_1_pct": 10.0,
         "n_failures": 1, "n_pinned_facts": n_pinned, "pinned_words": 100 * n_pinned,
         "actual_ratio_words": 0.15}
        for i, (wiki, m, n_pinned) in enumerate(iterations)
    ]}


def _write_probes(profile, topic, qids):
    from wicer.probe import probe_paths

    path = probe_paths(profile, topic)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {"question_id": q, "document_id": f"doc-{q}", "question": f"{q}?",
         "answer": _GOLD[q], "long_answer": ""} for q in qids
    ]))


def _synthetic_wicer_topic(profile, topic):
    from wicer_core.records import ledger_path

    served = {
        "wiki_moderate": _write_wiki(profile, topic, "blind_moderate", 0),
        "wicer_iter1": _write_wiki(profile, topic, "wicer_iter1", 2),
    }
    for condition, scores in _SCORES.items():
        _write_condition(profile, topic, condition, scores, context=served.get(condition))
    _write_probes(profile, topic, ("p1", "p2"))
    # The loop's own probe runs: blind 3.0, then 4.0 once two facts were pinned.
    ledger_path(profile, topic, "wicer").write_text(json.dumps(
        _arm_ledger(topic, [("blind_moderate", 3.0, 0), ("wicer_iter1", 4.0, 2)])
    ))
```

In `test_held_out_excludes_the_probes_and_scores_unanswerable_questions_apart`, rename the column key in both places it appears:

```python
    assert ans["Best pinned WiCER"] == "1.50 / 50.0%"
```

```python
    assert (un["n"], un["Blind (mean / score-1)"], un["Best pinned WiCER"], un["FC raw"]) == (
        "2", "3.00 / 0.0%", "2.50 / 0.0%", "1.50")
```

Replace `test_held_out_scores_the_wiki_the_loop_chose_not_the_best_held_out_one` with:

```python
def test_held_out_scores_the_wiki_the_loop_chose_not_the_best_held_out_one(profile, topic):
    """Choosing among full-set conditions by their score would select on the test set."""
    _synthetic_wicer_topic(profile, topic)
    # A wiki that does far better on held-out questions; the loop, judging on
    # its probes, never chose it.
    _write_condition(profile, topic, "wicer_iter2", dict(zip(_GOLD, (1, 1, 5, 5, 5, 5))),
                     context=_write_wiki(profile, topic, "wicer_iter2", 3))
    text = render(profile, collect(profile, [topic], [*_SCORES, "wicer_iter2"]), [topic])
    rows = {r["Questions"]: r for r in _table(_section(text, "## WiCER held-out"))}
    assert rows["answerable"]["Best pinned WiCER"] == "1.50 / 50.0%"    # wicer_iter1, not 5.00
```

Add the new test after it:

```python
def test_held_out_compares_the_best_pinned_wiki_even_when_best_of_is_the_blind(profile, topic):
    """Best-of includes the unpinned blind; scoring it would print a WiCER effect
    of exactly zero on held-out questions, whatever pinning did."""
    from wicer_core.records import ledger_path

    _synthetic_wicer_topic(profile, topic)
    # The blind beat the pinned wiki on the probes, so best-of is the blind.
    ledger_path(profile, topic, "wicer").write_text(json.dumps(
        _arm_ledger(topic, [("blind_moderate", 4.5, 0), ("wicer_iter1", 4.0, 2)])
    ))
    text = render(profile, collect(profile, [topic], list(_SCORES)), [topic])
    row = {r["Questions"]: r for r in _table(_section(text, "## WiCER held-out"))}["answerable"]
    assert row["Best pinned WiCER"] == "1.50 / 50.0%"     # wicer_iter1, not the blind's 2.00
    assert row["Iter"] == "1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_benchmark.py -k held_out`
Expected: FAIL — `KeyError: 'Best pinned WiCER'` (the column is still "Best WiCER").

- [ ] **Step 3: Implement the served-wiki lookup and the held-out change**

In `benchmark/report.py`, change the loop import (line 29):

```python
from wicer.loop import best_pinned_iteration
```

Replace `_best_condition` and `_result` (lines 255-270) with:

```python
def _scored(results: Sequence[CondResult], topic: str, wiki: str | None) -> CondResult | None:
    """The condition that scored ``wiki`` on the full set, by what it served.

    Read from each condition's context rather than from naming conventions: the
    pilots' ``random_pin`` condition served the random arm's blind.
    """
    if wiki is None:
        return None
    return next(
        (r for r in results if r.topic == topic and r.served_wiki == wiki and r.scores.n),
        None,
    )


def _held_out(
    profile: Profile, results: Sequence[CondResult], topic: str
) -> tuple[frozenset[str], frozenset[str]]:
    """(answerable, unanswerable) questions the loop never saw on a topic."""
    rs = [r for r in results if r.topic == topic]
    held = frozenset().union(*(r.by_question for r in rs)) - _probe_ids(profile, topic)
    unans = frozenset().union(*(r.unanswerable for r in rs))
    return held - unans, held & unans
```

In the Table 4 block of `render()`, delete the now-unused line:

```python
    blind = _by(results, "wiki_moderate")
```

Replace the held-out block of `render()` — from `parts.append("\n## WiCER held-out evaluation (questions the loop never saw)\n")` through its `md_table(...)` call — with:

```python
    parts.append("\n## WiCER held-out evaluation (questions the loop never saw)\n")
    rows = []
    for topic in topics:
        led = ledgers.get(topic)
        if not led or not led.get("iterations"):
            continue
        pinned = best_pinned_iteration(led["iterations"])
        b = _scored(results, topic, led["iterations"][0]["wiki_name"])
        best = _scored(results, topic, pinned["wiki_name"] if pinned else None)
        if not b and not best:
            continue
        f = fc.get(topic)
        answerable, unanswerable = _held_out(profile, results, topic)
        for label, subset in [("answerable", answerable), ("unanswerable", unanswerable)]:
            sb, sw, sf = (_stats(r, subset.__contains__) if r else None for r in (b, best, f))
            n = (sb or sw).n
            if not n:
                continue
            rec = (
                recovery_rate(sb.mean, sw.mean, sf.mean if sf and sf.n else None)
                if label == "answerable" and sb and sw and sw.n else None
            )
            rows.append([
                topic, label, n, _mean_and_s1(sb), _mean_and_s1(sw),
                fmt(sf.mean) if sf and sf.n else "--",
                f"{rec:.0f}%" if rec is not None else "--",
                pinned["iteration"] if pinned else "--",
            ])
    parts.append(md_table(
        ["Topic", "Questions", "n", "Blind (mean / score-1)", "Best pinned WiCER", "FC raw",
         "Recovery", "Iter"],
        rows,
    ))
```

(The note appended right after the table stays as it is; Task 4 edits its last sentence.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_benchmark.py`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `255 passed`

- [ ] **Step 6: Commit**

```bash
git add benchmark/report.py tests/test_benchmark.py
git commit -m "$(cat <<'EOF'
Compare the best pinned WiCER wiki on held-out questions

Best-of includes the unpinned blind, so scoring it on held-out questions
would print a WiCER effect of exactly zero whatever pinning did. The report
now finds the condition that scored each wiki from what its context served,
not from naming conventions -- the pilots' random_pin served the random
arm's blind.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Report each arm's effect against the gap between two blind compilations

**Files:**
- Modify: `benchmark/report.py` — new helpers after `_held_out`; call site in `render()` between the ablation notes and `# -- Table 13: latency`; last sentence of the held-out note
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `_scored`, `_held_out` (Task 3), `best_pinned_iteration` (Task 2), test helpers `_write_wiki`, `_arm_ledger`, `_write_probes` (Task 3), `_write_condition`, `_section`, `_table`, `_GOLD` (step 1).
- Produces: `_noise_section(profile, results, topics, ledgers) -> list[str]` and the report section headed `## Effects against compiler noise` with columns `Topic`, `Blind gap, probes`, `WiCER Δ, probes`, `Random Δ, probes`, `Blind gap, held-out`, `WiCER Δ, held-out`, `Random Δ, held-out`, and summary rows `**Noise band (2 × RMS gap)**` and `**Mean ± SE**`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benchmark.py`, after the held-out tests:

```python
_NOISE = "## Effects against compiler noise"
_BAND, _MEAN = "**Noise band (2 × RMS gap)**", "**Mean ± SE**"


def _noise_rows(text):
    return {r["Topic"]: r for r in _table(_section(text, _NOISE))}


def test_noise_section_sets_probe_effects_beside_the_gap_between_two_blinds(profile, topic):
    """Ledgers alone carry the probe columns; every value here is worked by hand."""
    from wicer_core.records import ledger_path

    arms = {
        topic: ([("blind_moderate", 3.0, 0), ("wicer_iter1", 3.5, 2)],
                [("random_blind_moderate", 3.2, 0), ("random_pin_iter1", 3.1, 2)]),
        "other_topic": ([("blind_moderate", 2.0, 0), ("wicer_iter1", 2.3, 2)],
                        [("random_blind_moderate", 2.4, 0), ("random_pin_iter1", 2.6, 2)]),
    }
    for t, (wicer, rnd) in arms.items():
        path = ledger_path(profile, t, "wicer")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_arm_ledger(t, wicer)))
        ledger_path(profile, t, "random_pin").write_text(json.dumps(_arm_ledger(t, rnd)))

    rows = _noise_rows(render(profile, [], list(arms)))
    probes = ("Blind gap, probes", "WiCER Δ, probes", "Random Δ, probes")
    assert [rows[topic][c] for c in probes] == ["0.20", "+0.50", "-0.10"]
    assert [rows["other_topic"][c] for c in probes] == ["0.40", "+0.30", "+0.20"]
    assert rows[topic]["Blind gap, held-out"] == "--"       # nothing scored on the full set
    assert rows[_BAND]["Blind gap, probes"] == "±0.63"      # 2 * sqrt((0.2² + 0.4²) / 2)
    assert rows[_MEAN]["WiCER Δ, probes"] == "+0.40 ± 0.10"
    assert rows[_MEAN]["Random Δ, probes"] == "+0.05 ± 0.15"


def _noise_topic(profile, topic, random_conditions):
    """Both arms' ledgers and wikis, the WiCER arm's two conditions, and the
    given random-arm conditions as (condition, served wiki, n_pinned, scores).
    Scores run over p1 p2 h1 h2 u1 u2; only h1 and h2 are held-out answerable."""
    from wicer_core.records import ledger_path

    conditions = [
        ("wiki_moderate", "blind_moderate", 0, (4, 4, 3, 3, 1, 1)),
        ("wicer_iter1", "wicer_iter1", 2, (5, 5, 2, 3, 1, 1)),
        *random_conditions,
    ]
    _write_wiki(profile, topic, "random_pin_iter1", 2)    # compiled even if never scored
    for condition, wiki, n_pinned, scores in conditions:
        _write_condition(profile, topic, condition, dict(zip(_GOLD, scores)),
                         context=_write_wiki(profile, topic, wiki, n_pinned))
    _write_probes(profile, topic, ("p1", "p2"))
    ledger_path(profile, topic, "wicer").write_text(json.dumps(
        _arm_ledger(topic, [("blind_moderate", 3.0, 0), ("wicer_iter1", 3.5, 2)])
    ))
    ledger_path(profile, topic, "random_pin").write_text(json.dumps(
        _arm_ledger(topic, [("random_blind_moderate", 3.2, 0), ("random_pin_iter1", 3.1, 2)])
    ))
    return [c for c, *_ in conditions]


def test_noise_section_sets_held_out_effects_beside_the_gap_between_two_blinds(profile, topic):
    conditions = _noise_topic(profile, topic, [
        ("random_blind", "random_blind_moderate", 0, (4, 4, 4, 3, 1, 1)),
        ("random_pin", "random_pin_iter1", 2, (4, 4, 4, 4, 1, 1)),
    ])
    rows = _noise_rows(render(profile, collect(profile, [topic], conditions), [topic]))
    assert rows[topic]["Blind gap, held-out"] == "0.50"     # |3.0 - 3.5| over h1, h2
    assert rows[topic]["WiCER Δ, held-out"] == "-0.50"      # 2.5 - 3.0
    assert rows[topic]["Random Δ, held-out"] == "+0.50"     # 4.0 - 3.5
    assert rows[_BAND]["Blind gap, held-out"] == "±1.00"
    assert rows[_MEAN]["WiCER Δ, held-out"] == "-0.50 ± --"   # no SE from one topic


def test_noise_section_reads_the_pilots_random_pin_as_the_random_arms_blind(profile, topic):
    """The pilots scored the random arm's blind under the name random_pin and
    never scored a random-pinned wiki on the full set."""
    conditions = _noise_topic(profile, topic, [
        ("random_pin", "random_blind_moderate", 0, (4, 4, 4, 3, 1, 1)),
    ])
    rows = _noise_rows(render(profile, collect(profile, [topic], conditions), [topic]))
    assert rows[topic]["Blind gap, held-out"] == "0.50"
    assert rows[topic]["Random Δ, held-out"] == "--"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_benchmark.py -k noise_section`
Expected: FAIL — `IndexError: list index out of range` in `_section`, because the report has no "## Effects against compiler noise" heading yet.

- [ ] **Step 3: Implement the section**

In `benchmark/report.py`, add after `_held_out`:

```python
@dataclass
class _Arm:
    """One arm's blind and best pinned wiki, on the probes and held out."""

    blind_probe: float
    pinned_probe: float | None
    blind_held: float | None
    pinned_held: float | None


def _arm(
    results: Sequence[CondResult], topic: str, ledger: dict[str, Any], held: frozenset[str]
) -> _Arm:
    blind = ledger["iterations"][0]
    pinned = best_pinned_iteration(ledger["iterations"])

    def held_mean(wiki: str) -> float | None:
        r = _scored(results, topic, wiki)
        s = _stats(r, held.__contains__) if r else None
        return s.mean if s and s.n else None

    return _Arm(
        blind_probe=blind["probe_mean"],
        pinned_probe=pinned["probe_mean"] if pinned else None,
        blind_held=held_mean(blind["wiki_name"]),
        pinned_held=held_mean(pinned["wiki_name"]) if pinned else None,
    )


def _gap(a: float | None, b: float | None) -> float | None:
    return abs(a - b) if a is not None and b is not None else None


def _delta(x: float | None, base: float | None) -> float | None:
    return x - base if x is not None and base is not None else None


def _cell(v: float | None, signed: bool) -> str:
    if v is None:
        return "--"
    return f"{v:+.2f}" if signed else fmt(v)


def _mean_se(xs: Sequence[float]) -> str:
    """Mean over topics and its standard error; no SE from a single topic."""
    if not xs:
        return "--"
    m = mean(xs)
    if len(xs) < 2:
        return f"{m:+.2f} ± --"
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return f"{m:+.2f} ± {sd / len(xs) ** 0.5:.2f}"


def _noise_section(
    profile: Profile,
    results: Sequence[CondResult],
    topics: Sequence[str],
    ledgers: dict[str, dict[str, Any] | None],
) -> list[str]:
    """Each arm's effect beside the gap between two blind compilations.

    Every run compiles the blind prompt twice -- the WiCER arm's W_0 and the
    random arm's own blind.  With no effect, an arm's delta is distributed like
    the gap between them: both are the difference of two independent
    compilations.  So the gaps set the band the deltas are read against.
    """
    gap_columns = (0, 3)
    rows: list[list[str]] = []
    columns: list[list[float]] = [[] for _ in range(6)]
    for topic in topics:
        led, rled = ledgers.get(topic), load_ledger(profile, topic, "random_pin")
        if not (led and led.get("iterations") and rled and rled.get("iterations")):
            continue
        held, _ = _held_out(profile, results, topic)
        w, rnd = _arm(results, topic, led, held), _arm(results, topic, rled, held)
        cells = [
            _gap(w.blind_probe, rnd.blind_probe),
            _delta(w.pinned_probe, w.blind_probe),
            _delta(rnd.pinned_probe, rnd.blind_probe),
            _gap(w.blind_held, rnd.blind_held),
            _delta(w.pinned_held, w.blind_held),
            _delta(rnd.pinned_held, rnd.blind_held),
        ]
        for column, value in zip(columns, cells):
            if value is not None:
                column.append(value)
        rows.append(
            [topic] + [_cell(v, signed=i not in gap_columns) for i, v in enumerate(cells)]
        )
    if not rows:
        return []
    rows.append(["**Noise band (2 × RMS gap)**"] + [
        (f"±{2 * (sum(v * v for v in col) / len(col)) ** 0.5:.2f}" if col else "--")
        if i in gap_columns else ""
        for i, col in enumerate(columns)
    ])
    rows.append(["**Mean ± SE**"] + [
        "" if i in gap_columns else _mean_se(col) for i, col in enumerate(columns)
    ])
    return [
        "\n## Effects against compiler noise\n",
        md_table(
            ["Topic", "Blind gap, probes", "WiCER Δ, probes", "Random Δ, probes",
             "Blind gap, held-out", "WiCER Δ, held-out", "Random Δ, held-out"],
            rows,
        ),
        "\n> *Blind gap* is how far a topic's two blind compilations land apart; "
        "*Δ* is an arm's best pinned wiki minus its own blind, on the probes and on "
        "held-out answerable questions. With no effect, a Δ is distributed like a "
        "blind gap -- both are the difference of two independent compilations -- so "
        "a Δ inside the noise band cannot be told apart from noise. '--' marks a "
        "wiki never scored on that question set.\n",
    ]
```

In `render()`, insert the call between the last ablation note and the latency block:

```python
    # -- effects against compiler noise
    parts.extend(_noise_section(profile, results, topics, ledgers))

    # -- Table 13: latency
```

In the held-out note in `render()`, replace its last sentence

```python
        "on them without having kept anything. With one compilation per arm, a "
        "held-out difference smaller than the gap between two compilations of the "
        "same prompt cannot be told apart from compiler noise.\n"
```

with

```python
        "on them without having kept anything. Whether a held-out difference "
        "exceeds compiler noise is in 'Effects against compiler noise' below.\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_benchmark.py`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `258 passed`

- [ ] **Step 6: Commit**

```bash
git add benchmark/report.py tests/test_benchmark.py
git commit -m "$(cat <<'EOF'
Report each arm's effect against the gap between two blind compilations

Every run compiles the blind prompt twice -- the WiCER arm's W_0 and the
random arm's own blind -- and with no effect an arm's delta is distributed
like the gap between them. The new section prints both, on the probes and on
held-out answerable questions, with a noise band of 2 x RMS of the gaps and
the mean effect with its standard error over topics.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Document probe selection and the noise floor

**Files:**
- Modify: `README.md:257-258`
- Modify: `DEVIATIONS.md:30` (section 2), `DEVIATIONS.md:158-163` (section 10), section 13 (append a paragraph), end of file (new section 14)
- Modify: `wicer/loop.py` and `wicer/compile.py` (`--seed` help strings)

**Interfaces:**
- Consumes: the behavior from Tasks 1-4.
- Produces: nothing code depends on. Prose and help text only, so no test.

- [ ] **Step 1: Replace the README's seed advice**

Replace `README.md` lines 257-258

```markdown
For a stronger result than the paper's, run at least three compiler seeds:
`wicer/compile.py --seed N` and `wicer/loop.py --seed N`.
```

with

```markdown
Compiler noise is measured, not assumed away. Every run compiles the blind
prompt twice -- the WiCER arm's W_0 and the random arm's own blind -- and the
report's "Effects against compiler noise" section sets each arm's effect beside
the gap between them. `--seed` on `wicer/compile.py` and `wicer/loop.py`
reaches only the offline fake compiler: the API takes no seed, so on a real run
it neither reproduces a compilation nor varies one.
```

- [ ] **Step 2: Correct DEVIATIONS section 2**

Replace `DEVIATIONS.md` line 30

```markdown
2. drop rows whose answer is marked unanswerable (in practice: none);
```

with

```markdown
2. drop rows whose answer is one of the markers in `UNANSWERABLE_MARKERS` (in
   practice: none -- RepLiQA's unanswerable questions carry the gold "The
   answer is not found in the document.", which is not one of them, so they
   stay; see section 14);
```

- [ ] **Step 3: Rewrite DEVIATIONS section 10**

Replace section 10 (from `## 10. Single seed by default` up to, not including, `## 11.`) with:

```markdown
## 10. One compilation per arm, with the noise measured

The paper reports no confidence intervals, and its own blind baselines differ
between Tables 4 and 5 through compiler non-determinism. This repo also
compiles each arm once, but measures what that costs: every run already
compiles the blind prompt twice -- the WiCER arm's W_0 and the random arm's own
blind -- and the report's "Effects against compiler noise" section sets each
arm's effect beside the gap between them. In the two pilots that gap was
0.04 / 0.22 on the probes and 0.26 / 0.40 on held-out answerable questions.

Repeating compilations on one topic narrows that topic's estimate slowly; the
sweep-level mean narrows faster from compilations spread across topics, which
a 17-topic sweep does anyway. So the sweep runs one compilation per arm per
topic (design: `docs/superpowers/specs/2026-10-21-representative-probes-and-noise-floor-design.md`).
`--seed` reaches only the offline fake compiler; the API takes no seed.

```

- [ ] **Step 4: Append to DEVIATIONS section 13**

Add this paragraph at the end of section 13 (after its last paragraph, before section 14):

```markdown
On the full question set, `--eval-full` scores each arm's blind and its best
*pinned* iteration. The pilots scored the random arm's best-of instead, which
was its blind both times, so no random-pinned wiki was ever scored on held-out
questions; the report marks that `random_pin` row with †.
```

- [ ] **Step 5: Add DEVIATIONS section 14**

Append at the end of `DEVIATIONS.md`:

```markdown

## 14. Probe selection

Paper Algorithm 1 selects "one QA pair per source document" and never says
which. This repo draws each document's probe uniformly from its five
questions, seeded per document from `WicerConfig.probe_seed` and the document
id, so a rerun probes the same questions. Every question is eligible,
including RepLiQA's unanswerable ones (gold: "The answer is not found in the
document."): about one per document, almost always the 4th or 5th. The probe
set therefore mirrors the full 400, as the paper's appears to -- its probe
blind (2.17) roughly equals its full-set blind (2.25). The loop diagnoses a
failed unanswerable probe like any other, as the paper's would have.

The two pilots took each document's first question instead. That never picked
an unanswerable question and favoured questions about a document's main
subject, which a compiler keeps: their probe-set blind ran 0.4-0.5 above the
full set. Their probe-set numbers are not comparable to the paper's Table 4.
```

- [ ] **Step 6: Stop the `--seed` help from repeating the old advice**

In `wicer/loop.py` `main()`, replace

```python
    ap.add_argument("--seed", type=int, help="compiler seed (run >=3 seeds if budget allows)")
```

with

```python
    ap.add_argument(
        "--seed", type=int,
        help="compiler seed; reaches only the offline fake compiler -- the API takes none",
    )
```

In `wicer/compile.py` `main()`, replace

```python
    ap.add_argument("--seed", type=int, help="compiler seed; vary it to expose non-determinism")
```

with

```python
    ap.add_argument(
        "--seed", type=int,
        help="compiler seed; reaches only the offline fake compiler -- the API takes none",
    )
```

Run: `.venv/bin/python wicer/loop.py --help | grep -A1 -- --seed && .venv/bin/python -m pytest -q`
Expected: the new help text, then `258 passed`.

- [ ] **Step 7: Commit**

```bash
git add README.md DEVIATIONS.md wicer/loop.py wicer/compile.py
git commit -m "$(cat <<'EOF'
Document probe selection and the measured compiler noise

The README and DEVIATIONS advised running several compiler seeds, but the API
takes no seed; the noise is now measured from the two blind compilations each
run makes. New DEVIATIONS section 14 records the probe rule and why the
pilots' first-question probes were biased; section 2 no longer claims the
corpus holds no unanswerable questions.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Verify end to end and regenerate the pilot reports

**Files:**
- Regenerate: `results-rental/report.md`, `results-rental/per_topic.csv`, `results-rental/per_question.csv` (tracked); `results-arts/report.md` and CSVs (untracked, stays untracked)

**Interfaces:**
- Consumes: everything above.
- Produces: nothing new; verification evidence.

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `258 passed`

- [ ] **Step 2: No new lint findings**

Run:

```bash
BASE=$(git log --format=%H -1 --grep="Score WiCER on matched question sets")
for f in wicer_core/config.py wicer/probe.py wicer/loop.py wicer/compile.py wicer/random_pin.py \
         wicer_core/records.py benchmark/report.py tests/test_wicer.py tests/test_benchmark.py; do
  base=$(git show $BASE:$f | .venv/bin/ruff check --stdin-filename $f --output-format concise - 2>/dev/null | grep -c "^$f")
  now=$(.venv/bin/ruff check --output-format concise $f 2>/dev/null | grep -c "^$f")
  [ "$now" -gt "$base" ] && echo "NEW in $f: $base -> $now"
done; echo done
```

Expected: only `done`. `BASE` is the step-1 commit; if step 1 was committed under another message, set `BASE` to that commit's hash.

- [ ] **Step 3: Smoke run**

Run: `bash scripts/smoke.sh > /dev/null && grep -c "## Effects against compiler noise" results/test/report.md && grep -c "| mini_topic | random_blind |" results/test/report.md`
Expected: exit 0, then `1` and `1`.

- [ ] **Step 4: Regenerate both pilot reports**

Run:

```bash
for R in results-rental results-arts; do
  WICER_RESULTS_DIR=$R .venv/bin/python benchmark/report.py --profile paper > /dev/null
  grep -A7 "## Effects against compiler noise" $R/report.md | grep "^| local\|Noise band\|Mean"
done
```

Expected (computed independently from the pilots' raw files):

```
| local_economy | 0.04 | +0.50 | -0.14 | 0.26 | -0.05 | -- |
| **Noise band (2 × RMS gap)** | ±0.08 |  |  | ±0.53 |  |  |
| **Mean ± SE** |  | +0.50 ± -- | -0.14 ± -- |  | -0.05 ± -- | -- |
| local_arts_and_culture | 0.23 | +0.15 | -0.06 | 0.40 | -0.24 | -- |
| **Noise band (2 × RMS gap)** | ±0.45 |  |  | ±0.79 |  |  |
| **Mean ± SE** |  | +0.15 ± -- | -0.06 ± -- |  | -0.24 ± -- | -- |
```

The held-out section's "Best pinned WiCER" keeps step 1's values: `2.73 / 27.6%` (economy) and `2.68 / 25.6%` (arts).

- [ ] **Step 5: Commit the regenerated tracked report**

```bash
git add results-rental/report.md results-rental/per_topic.csv results-rental/per_question.csv
git commit -m "$(cat <<'EOF'
Regenerate the local_economy pilot report

Adds the effects-against-noise section: WiCER's +0.50 on the probes clears
the 0.04 gap between its two blind compilations; its -0.05 on held-out
questions sits inside the 0.26 gap. The pilot never scored a random-pinned
wiki on the full set.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```
