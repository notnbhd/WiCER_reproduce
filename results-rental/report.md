# WiCER reproduction report

Profile: `paper`  ·  judge: `anthropic/claude-haiku-5.5`  ·  compiler: `anthropic/claude-haiku-5.5`

Reader: `models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`, n_ctx=98304, KV=q8_0/q8_0


## Per topic and condition

| Topic | Condition | n | Mean | %>=4 | %=1 | TTFT med (s) | tok/s | ctx tok |
|---|---|---|---|---|---|---|---|---|
| local_economy | fc_raw | 400 | 3.63 | 62.0 | 11.0 | 0.175 | 31.2 | 91642 |
| local_economy | wiki_light | 400 | 2.87 | 39.0 | 24.0 | 0.059 | 69.9 | 23395 |
| local_economy | wiki_moderate | 400 | 2.79 | 35.0 | 26.2 | 0.055 | 81.7 | 15051 |
| local_economy | wiki_aggressive | 400 | 2.64 | 31.2 | 28.8 | 0.056 | 77.2 | 17748 |
| local_economy | wicer_iter1 | 400 | 2.87 | 38.5 | 23.8 | 0.061 | 74.1 | 19788 |
| local_economy | random_pin † | 400 | 2.94 | 39.5 | 21.0 | 0.052 | 77.2 | 17937 |


> † Nothing pinned: `random_pin` on local_economy served `random_blind_moderate`. `--eval-full` scores the arm's best iteration, and best-of includes iteration 0, the unpinned blind -- so the row is another blind compilation of the moderate prompt, not a pinning condition.


## Compilation: compression and quality (paper Table 3)

| Level | Target | Actual | Quality | Score-1 % | vs. FC raw | Quality (answerable) | vs. FC raw (answerable) |
|---|---|---|---|---|---|---|---|
| FC raw (80 docs) | -- | -- | 3.63 | 11.0% | -- | 4.01 | -- |
| Wiki-light | 75% | 19.0% | 2.87 | 24.0% | -0.76 | 3.02 | -0.99 |
| Wiki-moderate | 50% | 12.5% | 2.79 | 26.2% | -0.84 | 2.86 | -1.15 |
| Wiki-aggressive | 25% | 13.1% | 2.64 | 28.8% | -0.99 | 2.66 | -1.35 |


> The answerable columns leave out RepLiQA's unanswerable questions (gold: "The answer is not found in the document."), about a fifth of each topic. A compiled wiki makes the reader abstain more, which those questions reward, so they shrink the gap the other columns measure.


## WiCER across topics, probe set (paper Table 4)

| Topic | Blind (mean / score-1) | Best WiCER | FC raw (probes) | Recovery | Iter | Stop |
|---|---|---|---|---|---|---|
| local_economy | 3.16 / 16.2% | 3.66 / 5.0% | 4.00 | 60% | 1 | insufficient_improvement |
| **Mean** | 3.16 | 3.66 / 5.0% | 4.00 | 60% | -- | -- |


> FC raw is scored on the same probes as Blind and Best WiCER, so recovery compares one question set throughout. The paper divides by FC over every question, which agrees with this only when the probes are as hard as the full set.


### Iteration trajectory

| Topic | Iter | Wiki | Probe mean | Score-1 % | Failures | Pinned facts | Pinned words | Kept verbatim | Actual ratio | Improvement |
|---|---|---|---|---|---|---|---|---|---|---|
| local_economy | 0 | blind_moderate | 3.16 | 16.2 | 13 | 0 | 0 | -- | 12.5% | -- |
| local_economy | 1 | wicer_iter1 | 3.66 | 5.0 | 4 | 13 | 1157 | 0% | 15.0% | +15.8% |
| local_economy | 2 | wicer_iter2 | 3.30 | 20.0 | 16 | 17 | 1462 | 0% | 15.3% | -9.9% |


## WiCER held-out evaluation (questions the loop never saw)

| Topic | Questions | n | Blind (mean / score-1) | Best WiCER | FC raw | Recovery | Iter |
|---|---|---|---|---|---|---|---|
| local_economy | answerable | 239 | 2.77 / 25.5% | 2.73 / 27.6% | 4.01 | -4% | 1 |
| local_economy | unanswerable | 81 | 2.51 / 38.3% | 2.48 / 32.1% | 2.15 | -- | 1 |


> Every probe is left out, so these are questions the loop never saw. RepLiQA's unanswerable questions (gold: "The answer is not found in the document.") are scored apart and given no recovery: they reward a reader that abstains, which a smaller context encourages, so a wiki can beat FC raw on them without having kept anything. With one compilation per arm, a held-out difference smaller than the gap between two compilations of the same prompt cannot be told apart from compiler noise.


## Ablation: diagnosed vs. random pinning, probe set (paper Table 5)

| Topic | Blind (WiCER arm) | WiCER (best-of) | WiCER (pinned only) | Blind (abl. arm) | Random pin (best-of) | Random pin (pinned only) |
|---|---|---|---|---|---|---|
| local_economy | 3.16 | 3.66 (+0.50) | 3.66 (+0.50) | 3.20 | 3.20 (+0.00) | 3.06 (-0.14) |
| **Mean delta vs. own blind** | -- | +0.50 | +0.50 | -- | +0.00 | -0.14 |


> This is the load-bearing comparison (paper Section 7.5): it is the only thing separating 'diagnosis works' from 'pinning anything works'. The paper measures +0.16 for random pinning against +0.95 for WiCER.


> Each arm compiles its own blind wiki -- paper Section 7.5, 'each topic uses an independent blind compilation; blind baselines differ slightly from Table 4 due to LLM compiler non-determinism' -- so every delta here is measured against the blind in its own column, never the other arm's. 'Best-of' follows the paper's 'best iteration per topic' and so includes iteration 0, the unpinned blind: an arm whose pinning only ever hurt therefore reports its own baseline and a delta of exactly +0.00. That is not a null effect, it is a selection artifact, and it is visible in the paper too -- seven of its seventeen topics print Random Pin identical to Blind. The pinned-only columns drop that selection and report what pinning actually did, for both arms alike -- an ablation that measured its two arms differently is what produced the incoherent row this replaced.


> One construction difference from the paper: its Table 5 compares the Table 4 WiCER score against the ablation arm's independent blind, mixing two compilations in one delta. Here each arm is measured against its own blind, so our WiCER delta and the paper's +0.95 do not share a denominator even when the method agrees.


## Latency by condition (paper Table 13)

| Condition | Context tokens | TTFT median (s) | tok/s |
|---|---|---|---|
| fc_raw | 91,642 | 0.175 | 31.2 |
| wiki_light | 23,395 | 0.059 | 69.9 |
| wiki_moderate | 15,051 | 0.055 | 81.7 |
| wiki_aggressive | 17,748 | 0.056 | 77.2 |
| wicer_iter1 | 19,788 | 0.061 | 74.1 |
| random_pin † | 17,937 | 0.052 | 77.2 |


> † Nothing pinned: `random_pin` on local_economy served `random_blind_moderate`. `--eval-full` scores the arm's best iteration, and best-of includes iteration 0, the unpinned blind -- so the row is another blind compilation of the moderate prompt, not a pinning condition.


## Success criteria (spec.md Section 8)

| Criterion | Target | Measured | Delta | Verdict | Coverage |
|---|---|---|---|---|---|
| FC raw mean | 3.46 | 3.63 | +0.17 | PASS | n=1 topics |
| RAG mean | 3.63 | -- | -- | NOT RUN | -- |
| Wiki-moderate (blind) mean | 2.25 | 2.79 | +0.54 | MISS | n=1 topics |
| Wiki-moderate score-1 rate % | 57.10 | 26.25 | -30.85 | MISS | n=1 topics |
| Best WiCER mean | 3.18 | 3.66 | +0.48 | MISS | n=1 topics |
| Best WiCER score-1 rate % | 24.80 | 5.00 | -19.80 | MISS | n=1 topics |


> These targets describe the full 17-topic RepLiQA sweep. A partial or reduced-scale run will miss them by construction -- read the verdict column only when the coverage column says 17 topics.


> Directional agreement is the real bar (spec.md Section 8): the FC/RAG crossover exists, and targeted diagnosis beats random pinning by a wide margin. Exact numbers will move -- the compiler is stochastic and TTFT changes outright on different hardware.
