# Representative probes and a compiler-noise floor

**Date:** 2026-10-21 · **Status:** design approved in chat; written up for review

## Why

The two pilot runs (`results-rental/`, `results-arts/`) showed three problems
that rerunning cannot fix on its own:

1. **Every probe is a document's first question.** `wicer/probe.py` picks the
   lowest `question_id` per document. RepLiQA marks about one question in five
   unanswerable (gold: "The answer is not found in the document."), almost
   always a document's 4th or 5th, so no probe was ever unanswerable. First
   questions also tend to ask about a document's main subject, which a compiler
   keeps. Probe-set blind scores ran 0.4-0.5 above the full set. The paper's
   probe blind (2.17) roughly equals its full-set blind (2.25), so its probes
   were representative and ours were not.
2. **Nothing measures compiler noise.** Two compilations of the identical blind
   prompt differed by 0.04 / 0.22 on the probes and by 0.26 / 0.40 on held-out
   answerable questions -- as large as the effects under study. Every run
   already compiles that prompt twice, but the report never compares the two.
3. **No random-pinned wiki was ever scored on the full set.**
   `random_pin.py --eval-full` scores the arm's best-of iteration, which in both
   pilots was its unpinned blind.

## Decisions

- **Probe pool: all five questions per document, drawn uniformly.** This
  matches the paper's "select one QA pair per source document" and makes the
  probe set mirror the full 400. The loop will diagnose failed unanswerable
  probes too, as the paper's loop would have.
- **One compilation per arm; no replicates.** The noise floor comes from the
  two blind compilations every run already makes: the WiCER arm's W_0 and the
  random arm's own blind. At the measured noise, one topic's held-out effect is
  resolvable only to about +/-0.67 with one compilation and +/-0.39 with three,
  while the 17-topic mean at one compilation is resolvable to about +/-0.16.
  Spreading compilations across topics beats stacking them on one, and the
  sweep spreads them anyway. Replicates can be added later if a per-topic claim
  needs them.

## Design

### 1. Probe selection -- `wicer/probe.py`, `wicer_core/config.py`

- New `WicerConfig.probe_seed`, default `20261021`, recorded in every
  `run_meta.json` through the profile.
- `split_probes(qa, seed)` groups questions by document, orders each document's
  questions by `question_id`, and picks one uniformly with a generator seeded
  from `sha256(f"{seed}:{document_id}")`. The pick is reproducible, independent
  of input order and independent across documents. Unanswerable questions are
  eligible.
- `probes.json` and `heldout.json` keep their format. `scripts/run_e3_topic.sh`
  rebuilds them at the start of every run. `loop.py` run on its own reuses the
  `probes.json` it finds, so a results directory built under the old rule keeps
  first-question probes until `probe.py` runs again.

### 2. What `--eval-full` scores -- `wicer/loop.py`, `wicer/random_pin.py`, `wicer_core/records.py`

Both arms score two wikis on all 400 questions: their blind, and their **best
pinned** iteration, meaning the highest probe mean among iterations that
pinned something. If no iteration pinned anything, only the blind is scored.

| Arm | Blind wiki -> condition | Best pinned wiki -> condition |
|---|---|---|
| WiCER | `blind_moderate` -> `wiki_moderate` (unchanged) | `wicer_iterN` -> `wicer_iterN` (was the best-of) |
| Random pin | `random_blind_moderate` -> `random_blind` (new) | `random_pin_iterN` -> `random_pin` (was the best-of) |

The best-of and the best pinned iteration differ only when no pinned iteration
beat the blind on the probes. The probe-set tables keep the paper's best-of.
`records.CONDITIONS` gains `random_blind`. One helper picks the best pinned
iteration, shared by both arms and the report.

### 3. Report -- `benchmark/report.py`

- **Which condition scored which wiki** is read from each condition's
  `context.txt` (the served wiki the report already records), never from naming
  conventions. That keeps the pilots readable: their `random_pin` condition
  served the random arm's blind, so it counts as that arm's blind, and the
  arm's pinned wiki shows as never scored.
- **Held-out section:** the WiCER column becomes the best pinned WiCER wiki,
  the one `--eval-full` scores.
- **New section, "Effects against compiler noise"**, one row per topic:

  | Topic | Blind gap, probes | WiCER Δ, probes | Random Δ, probes | Blind gap, held-out | WiCER Δ, held-out | Random Δ, held-out |
  |---|---|---|---|---|---|---|

  - *Blind gap:* |WiCER arm's blind - random arm's blind|, two compilations of
    the identical prompt.
  - *Δ:* an arm's best pinned wiki minus its own blind.
  - *Held-out:* answerable questions the loop never saw, as in the held-out
    section.
  - A **noise band** row: 2 x RMS of the blind gaps over the topics present.
    With no effect, an arm's Δ is distributed like a blind gap -- both are the
    difference of two independent compilations -- so a Δ inside the band cannot
    be told apart from noise.
  - A **mean +/- SE** row: each Δ column's mean over the topics where that Δ
    exists, and its standard error; "--" for the SE with a single topic.
  - A cell is "--" when its wiki was never scored on that question set.

### 4. Docs

- README: replace the "run at least three compiler seeds" advice. `--seed`
  reaches only the offline fake compiler; the API takes no seed.
- DEVIATIONS: a new entry on probe selection (the rule, why the pilots'
  first-question probes were biased, the unanswerable share) and on the noise
  floor. Correct section 2's claim that no unanswerable rows exist, which the
  new entry contradicts.

## Testing

Test-first, on the offline harness (fake reader, judge and compiler):

- Probes: over a synthetic 500-document set each of the five positions is
  picked 15-25% of the time; unanswerable questions get picked; a different
  seed changes the pick; the existing determinism and order-independence tests
  keep passing.
- `--eval-full`: given a ledger whose best-of is the blind but which has a
  pinned iteration, each arm scores both the blind and the pinned wiki.
- Report: hand-computed blind gaps, Δs, noise band and SE on a synthetic topic;
  the held-out section uses the best pinned wiki; a pilot-style `random_pin`
  that served the blind is read as the random arm's blind.
- `scripts/smoke.sh` end to end.

## Out of scope

- Replicates.
- Running anything. Rerunning the pilots or the sweep needs a rented GPU. The
  change adds one 400-question evaluation per topic, so a topic should cost
  about $1.30 and 90 min, and the 17-topic sweep about $22 and 25 h.
- Rewriting the pilot write-up's numbers (step 3).
