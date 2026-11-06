# WiCER reproduction report

Profile: `paper`  ·  judge: `anthropic/claude-haiku-5.5`  ·  compiler: `anthropic/claude-haiku-5.5`

Reader: `models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`, n_ctx=98304, KV=q8_0/q8_0


## Per topic and condition

| Topic | Condition | n | Mean | %>=4 | %=1 | TTFT med (s) | tok/s | ctx tok |
|---|---|---|---|---|---|---|---|---|
| local_arts_and_culture | fc_raw | 400 | 3.58 | 61.0 | 10.8 | 0.183 | 35.1 | 93628 |
| local_arts_and_culture | wiki_light | 400 | 2.56 | 27.2 | 30.0 | 0.058 | 79.4 | 21125 |
| local_arts_and_culture | wiki_moderate | 400 | 2.77 | 34.2 | 25.8 | 0.053 | 83.5 | 18249 |
| local_arts_and_culture | wiki_aggressive | 400 | 2.62 | 32.5 | 31.2 | 0.056 | 80.2 | 20530 |
| local_arts_and_culture | wicer_iter1 | 400 | 2.62 | 29.5 | 31.5 | 0.053 | 82.1 | 19183 |
| local_arts_and_culture | random_pin † | 400 | 2.52 | 27.8 | 32.5 | 0.052 | 84.4 | 17607 |


> † Nothing pinned: `random_pin` on local_arts_and_culture served `random_blind_moderate`. `--eval-full` scores the arm's best iteration, and best-of includes iteration 0, the unpinned blind -- so the row is another blind compilation of the moderate prompt, not a pinning condition.


## Compilation: compression and quality (paper Table 3)

| Level | Target | Actual | Quality | Score-1 % | vs. FC raw | Quality (answerable) | vs. FC raw (answerable) |
|---|---|---|---|---|---|---|---|
| FC raw (80 docs) | -- | -- | 3.58 | 10.8% | -- | 4.05 | -- |
| Wiki-light | 75% | 15.8% | 2.56 | 30.0% | -1.02 | 2.71 | -1.34 |
| Wiki-moderate | 50% | 14.5% | 2.77 | 25.8% | -0.81 | 3.01 | -1.04 |
| Wiki-aggressive | 25% | 13.7% | 2.62 | 31.2% | -0.96 | 2.83 | -1.22 |


> The answerable columns leave out RepLiQA's unanswerable questions (gold: "The answer is not found in the document."), about a fifth of each topic. A compiled wiki makes the reader abstain more, which those questions reward, so they shrink the gap the other columns measure.


## WiCER across topics, probe set (paper Table 4)

| Topic | Blind (mean / score-1) | Best WiCER | FC raw (probes) | Recovery | Iter | Stop |
|---|---|---|---|---|---|---|
| local_arts_and_culture | 3.27 / 13.8% | 3.42 / 15.0% | 4.10 | 18% | 1 | insufficient_improvement |
| **Mean** | 3.27 | 3.42 / 15.0% | 4.10 | 18% | -- | -- |


> FC raw is scored on the same probes as Blind and Best WiCER, so recovery compares one question set throughout. The paper divides by FC over every question, which agrees with this only when the probes are as hard as the full set.


### Iteration trajectory

| Topic | Iter | Wiki | Probe mean | Score-1 % | Failures | Pinned facts | Pinned words | Kept verbatim | Actual ratio | Improvement |
|---|---|---|---|---|---|---|---|---|---|---|
| local_arts_and_culture | 0 | blind_moderate | 3.27 | 13.8 | 11 | 0 | 0 | -- | 14.5% | -- |
| local_arts_and_culture | 1 | wicer_iter1 | 3.42 | 15.0 | 12 | 11 | 831 | 0% | 13.0% | +4.6% |


## WiCER held-out evaluation (questions the loop never saw)

| Topic | Questions | n | Blind (mean / score-1) | Best pinned WiCER | FC raw | Recovery | Iter |
|---|---|---|---|---|---|---|---|
| local_arts_and_culture | answerable | 234 | 2.93 / 19.2% | 2.68 / 25.6% | 4.03 | -22% | 1 |
| local_arts_and_culture | unanswerable | 86 | 1.87 / 54.7% | 1.69 / 61.6% | 1.87 | -- | 1 |


> Every probe is left out, so these are questions the loop never saw. RepLiQA's unanswerable questions (gold: "The answer is not found in the document.") are scored apart and given no recovery: they reward a reader that abstains, which a smaller context encourages, so a wiki can beat FC raw on them without having kept anything. With one compilation per arm, a held-out difference smaller than the gap between two compilations of the same prompt cannot be told apart from compiler noise.


## Ablation: diagnosed vs. random pinning, probe set (paper Table 5)

| Topic | Blind (WiCER arm) | WiCER (best-of) | WiCER (pinned only) | Blind (abl. arm) | Random pin (best-of) | Random pin (pinned only) |
|---|---|---|---|---|---|---|
| local_arts_and_culture | 3.27 | 3.42 (+0.15) | 3.42 (+0.15) | 3.05 | 3.05 (+0.00) | 2.99 (-0.06) |
| **Mean delta vs. own blind** | -- | +0.15 | +0.15 | -- | +0.00 | -0.06 |


> This is the load-bearing comparison (paper Section 7.5): it is the only thing separating 'diagnosis works' from 'pinning anything works'. The paper measures +0.16 for random pinning against +0.95 for WiCER.


> Each arm compiles its own blind wiki -- paper Section 7.5, 'each topic uses an independent blind compilation; blind baselines differ slightly from Table 4 due to LLM compiler non-determinism' -- so every delta here is measured against the blind in its own column, never the other arm's. 'Best-of' follows the paper's 'best iteration per topic' and so includes iteration 0, the unpinned blind: an arm whose pinning only ever hurt therefore reports its own baseline and a delta of exactly +0.00. That is not a null effect, it is a selection artifact, and it is visible in the paper too -- seven of its seventeen topics print Random Pin identical to Blind. The pinned-only columns drop that selection and report what pinning actually did, for both arms alike -- an ablation that measured its two arms differently is what produced the incoherent row this replaced.


> One construction difference from the paper: its Table 5 compares the Table 4 WiCER score against the ablation arm's independent blind, mixing two compilations in one delta. Here each arm is measured against its own blind, so our WiCER delta and the paper's +0.95 do not share a denominator even when the method agrees.


## Latency by condition (paper Table 13)

| Condition | Context tokens | TTFT median (s) | tok/s |
|---|---|---|---|
| fc_raw | 93,628 | 0.183 | 35.1 |
| wiki_light | 21,125 | 0.058 | 79.4 |
| wiki_moderate | 18,249 | 0.053 | 83.5 |
| wiki_aggressive | 20,530 | 0.056 | 80.2 |
| wicer_iter1 | 19,183 | 0.053 | 82.1 |
| random_pin † | 17,607 | 0.052 | 84.4 |


> † Nothing pinned: `random_pin` on local_arts_and_culture served `random_blind_moderate`. `--eval-full` scores the arm's best iteration, and best-of includes iteration 0, the unpinned blind -- so the row is another blind compilation of the moderate prompt, not a pinning condition.


## Success criteria (spec.md Section 8)

| Criterion | Target | Measured | Delta | Verdict | Coverage |
|---|---|---|---|---|---|
| FC raw mean | 3.46 | 3.58 | +0.12 | PASS | n=1 topics |
| RAG mean | 3.63 | -- | -- | NOT RUN | -- |
| Wiki-moderate (blind) mean | 2.25 | 2.77 | +0.52 | MISS | n=1 topics |
| Wiki-moderate score-1 rate % | 57.10 | 25.75 | -31.35 | MISS | n=1 topics |
| Best WiCER mean | 3.18 | 3.42 | +0.24 | PASS | n=1 topics |
| Best WiCER score-1 rate % | 24.80 | 15.00 | -9.80 | PASS | n=1 topics |


> These targets describe the full 17-topic RepLiQA sweep. A partial or reduced-scale run will miss them by construction -- read the verdict column only when the coverage column says 17 topics.


> Directional agreement is the real bar (spec.md Section 8): the FC/RAG crossover exists, and targeted diagnosis beats random pinning by a wide margin. Exact numbers will move -- the compiler is stochastic and TTFT changes outright on different hardware.
