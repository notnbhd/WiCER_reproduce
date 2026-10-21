# Deviations from the paper

Everything here is a place where arXiv:2605.07068v1 under-specifies the
experiment and this repo had to make a choice, or where the choice was forced
by hardware. Read this before comparing any number to a published one.

## 1. Compilation and diagnosis prompts are reconstructed

**The largest source of divergence.** The paper publishes exactly one prompt:
the judge rubric (Appendix B), which is used here verbatim. The compilation
prompt (Section 6.1) and the diagnosis prompt (Section 7.2) are described in
prose and never given.

`wicer/prompts/*.md` reconstructs them from the stated constraints: knowledge
engineer framing, blindness to the evaluation questions, a stated word target,
verbatim preservation ordered before general coverage, and a 50-100 word
diagnosis budget. Compiled-wiki quality is sensitive to this wording, so
absolute wiki scores should be expected to differ from Table 3 and Table 4 even
on identical hardware.

## 2. The RepLiQA subsample is ours

The paper uses "17 topic domains with 80 documents and 400 QA pairs per topic"
(Section 3.1). The published dataset is larger -- split `repliqa_0` alone has
17,955 rows across the same 17 topics -- and the paper never says which 80.

Our rule (`wicer_core/corpus.py`), recorded in every manifest:

1. one split, `repliqa_0`;
2. drop rows whose answer is one of the markers in `UNANSWERABLE_MARKERS` (in
   practice: none -- RepLiQA's unanswerable questions carry the gold "The
   answer is not found in the document.", which is not one of them, so they
   stay; see section 14);
3. the **80 shortest documents by word count**, ties broken on `document_id`;
4. first 5 `question_id` values per document.

Step 3 was `first 80 document_id values lexicographically` until it was
measured against the reader's own tokeniser, and the earlier note here claimed
that landed on the paper's documents. It does not. The conversion that claim
rested on -- Company Policies 48,459 words against the paper's 54,987 tokens --
implies 1.135 tokens per word. Llama 3.1 tokenises this corpus at **1.29-1.32**,
measured through `/tokenize` on the model the paper itself specifies. The
paper's token counts cannot come from those 48,459 words.

What the measurements rule out, in order:

* **not a dataset version** -- the HuggingFace `sha` is `bc880adc...`, dated
  2025-06-09, and is what we build from;
* **not a different split** -- `repliqa_0` through `repliqa_4` agree on
  document length to within 2% (1,006-1,030 words/doc for Regional Cuisine);
* **not selection bias in the old rule** -- the lexicographic 80 average
  *exactly* the topic's population mean, 1.00x;
* **not prompt scaffolding** -- document headers and the preamble are 1.2%;
* **not a QA-count filter** -- every RepLiQA document carries exactly 5 QA
  pairs, so no "needs 5 questions" rule can skew the set.

What remains is a length-biased subsample, and `shortest` reproduces the paper
on three independent axes:

| topic | shortest-80 | paper | delta |
|---|---|---|---|
| Company Policies | 54,742 | 54,987 | -0.4% |
| Regional Folklore | 96,586 | 97,148 | -0.6% |
| Regional Cuisine | 96,546 | 97,251 | -0.7% |

It also reproduces the *count* of full-context exclusions (15 of 17 topics fit
a 96K window) and their *identity* -- Regional Cuisine and Regional Folklore,
precisely the two Section 5 footnote 2 names. Under the lexicographic rule,
sixteen topics overflowed and no recovery rate was computable for any of them.

This is a reconstruction, and it should be read as one: the paper states no
selection rule, and landing 0.4-0.7% *above* the shortest-possible floor means
it was not literally "sort by length and take 80" either -- more likely
documents packed against a token budget. `doc_selection: "lexicographic"`
(or `WICER_DOC_SELECTION=lexicographic`) restores the length-neutral rule, and
the manifest records which one built a corpus, because two corpora built by
different rules are not comparable.

## 3. Judge model is pinned; the paper's is not

The paper says "Claude Sonnet" without a version (spec.md Section 9 flags this
as a reproduction risk). We pin `claude-sonnet-5` in `wicer_core/config.py` and
additionally record the `model` the API echoes back on every scored row, so a
mid-sweep model change is visible after the fact rather than invisible.

Judge calls run with thinking disabled at `low` effort. The paper does not
describe its judge invocation; this keeps ~54K calls affordable.

## 4. Hardware: CUDA, not Apple Silicon

The paper runs on an Apple M4 Pro with Metal (Section 3.3). Appendix D projects
RTX 4090 behaviour and says two conclusions legitimately change on CUDA:

* **TTFT drops to ~0.2 s** warm (from 0.86 s), so absolute latency numbers and
  the FC/RAG latency ratio will differ.
* **Q4 KV has no penalty on CUDA** (fused Flash Attention handles quantised KV
  without dequantisation overhead), whereas on Metal at 80 documents Q8 beat Q4
  in 13 of 14 topics. The quantisation ablation is therefore expected *not* to
  reproduce.

Quality conclusions should hold. `q8_0` remains the headline config
(spec.md Section 2); run `--kv q4_0` only as the ablation arm.

## 5. WiCER results are reported twice

Paper Table 4 reports WiCER quality on "80 probes each" -- the same probes the
loop diagnoses against. `benchmark/report.py` reports that number, for
comparability, *and* a held-out number over the full question set.

They can disagree, and the difference is informative: a probe-set gain that
does not survive to held-out questions is the loop fitting its own probes,
because pinned facts consume budget that would otherwise buy general coverage
(the paper's own "random knowledge displacement", Appendix I). The paper does
not report a held-out figure.

## 6. Compression targets are measured, never enforced

Paper Section 6.2 reports that targets of 75/50/25% yield actual ratios of
35.4/12.2/8.2%. That over-compression is the finding, so nothing in
`wicer/compile.py` tries to make the compiler hit its target. Actual ratios are
measured in both words and reader tokens and reported.

If you change this, you have changed the experiment -- say so.

## 7. Full-context exclusions are measured, not hardcoded

Two topics exceed the 96K window at 80 documents and have no FC baseline
(Section 5, footnote 2). This repo does not carry an exclusion list: it
tokenises each corpus with the reader's own `/tokenize` and records
`skipped: context_overflow` when the corpus does not fit. Which topics drop out
therefore depends on the tokeniser and the configured window, as it should.

`context_limit()` takes the **smaller** of the configured and the
server-reported window, so a server launched with a larger context than the
profile asks for cannot silently turn a 96K run into a 128K one.

## 8. E1 (Policygenius) is not implemented

spec.md Section 0 marks it optional; the corpus is not redistributed by the
authors and would have to be scraped. Only E2 (scalability gap) and E3
(compilation gap + WiCER) are implemented. Table 1, Table 2, Table 9 and
Table 10 have no counterpart here.

## 9. The offline harness is not a model

`tests/fake_llama_server.py`, the `fake` judge backend and the `fake` compiler
backend exist to prove the pipeline runs end to end without a GPU or an API
key. They are stubs, and the numbers `scripts/smoke.sh` prints describe the
fixture, not the paper. Three specific ways they do not behave like the real
components:

* The fake compiler is a truncating extractor and **obeys its compression
  target**, so the smoke run does not reproduce the over-compression finding.
* The fake compiler shares one hard word budget across documents, so pinning
  displaces coverage far more sharply than an LLM compiler does.
* The fake judge scores by gold-token recall, with a strict rule that a missing
  gold figure is a 1. Without that rule it hands out partial credit for the
  right kind of answer with the wrong number, and the random-pinning control
  scores as well as diagnosis.

## 10. One compilation per arm

The paper reports no confidence intervals, and its own blind baselines differ
between Tables 4 and 5 through compiler non-determinism. This repo also
compiles each arm once. Every run still compiles the blind prompt twice -- the
WiCER arm's W_0 and the random arm's own blind -- so the ablation table's two
blind columns show how far two compilations of the same prompt land apart: in
the two pilots, 0.04 and 0.22 on the probes. A per-topic effect smaller than
that gap is not distinguishable from compiler noise. `--seed` reaches only the
offline fake compiler; the API takes no seed.

## 11. The OpenRouter backend is not the Anthropic path

`WICER_COMPILER_BACKEND=openrouter` and `WICER_JUDGE_BACKEND=openrouter` route
the compiler, the diagnoser and the judge through OpenRouter's
OpenAI-compatible endpoint instead of the Anthropic SDK. It exists so the
experiment can be run without an Anthropic API account. It is not equivalent,
in four ways that affect numbers rather than plumbing:

* **Prompt caching is equivalent, on `anthropic/*` slugs only.** OpenRouter
  passes `cache_control` through to Anthropic verbatim, including
  `ttl: "1h"`, and publishes the same write and read rates as the first-party
  API. The compile stage uses it; the judge and the diagnoser deliberately do
  not, because their prompts share no reusable prefix and a 1-hour write costs
  2x base input. On a non-`anthropic/*` slug the breakpoint is dropped
  entirely and there is no caching at all -- that costs roughly $29 of a
  17-topic sweep. Check `cache_read_tokens` in the first `run_meta.json`
  rather than assuming it worked.
* **Thinking and effort are approximated.** `thinking={"type": "disabled"}`
  becomes the absence of a `reasoning` block, and `output_config.effort`
  becomes `reasoning: {"effort": ...}`. Providers honour that hint to differing
  degrees; some ignore it.
* **Watch the Haiku prompt-size tier.** Claude Haiku 5.5 bills prompts over
  100,000 tokens at 5x its headline rate. A topic's source blob is ~130,000
  tokens, so every *compile* call lands in the expensive tier while judge and
  diagnosis prompts stay in the cheap one. Haiku is a far smaller saving on
  the compiler than its sticker price suggests.
* **Cost is measured, not estimated.** Requests set `usage: {"include": true}`
  and `run_meta.json` reports `cost_source: "openrouter"` with the figure
  OpenRouter billed. The Anthropic path still reports a price-table estimate
  tagged `cost_source: "estimate"`. Do not compare the two columns.
* **The model is a different model.** An Anthropic-native id such as
  `claude-sonnet-5` is rewritten to `anthropic/claude-sonnet-5`
  (`wicer_core/config.OPENROUTER_DEFAULT_MODEL`); `run_meta.json` records both
  the configured id and the resolved slug. Section 3 of this file already
  treats the judge model as a reproduction risk — changing it, or routing it
  through a different serving stack, is that risk. **Do not mix backends within
  a comparison**: the judge disk cache keys on the model name, so a switch
  re-scores rather than silently reusing another model's grades, but a report
  built from both is not a controlled comparison.

## 12. Two instruments the paper does not have

Neither of these changes the algorithm. Both exist because this reproduction
can be run on models the paper never used, and a model substitution can produce
a wrong answer that looks like a real one.

**Verbatim preservation rate.** WiCER's entire mechanism is pinning diagnosed
facts into the next compilation; the prompt calls them `MANDATORY PRESERVED
FACTS` and demands them verbatim. Nothing in the paper reports whether the
compiler complied. A compiler that silently ignores the instruction produces a
flat trajectory indistinguishable from "the method does not work", so
`CompileResult` now carries `n_preserved_found` and `preservation_rate`, the
loop records them per iteration, and `report.py` prints a **Kept verbatim**
column. Read it before reading the probe means next to it: a rate well below
1.0 invalidates that row. The rate is `None`, not `0.0`, for blind
compilation, which pins nothing.

**Judge agreement.** Paper Appendix J validates its judge against a human
rater (n=100, Pearson r 0.940, exact 75%, within-1 99%, MAE 0.26, bias +0.06;
per-condition r >= 0.89 and |dmean| <= 0.17). Repeating that needs a human.
`benchmark/judge_agreement.py` asks the question that governs a model swap
instead -- does the candidate judge track the one it replaces? -- and reports
the same statistics per condition against Appendix J's thresholds:

```bash
uv run python benchmark/judge_agreement.py --profile paper \
    --candidate-model anthropic/claude-haiku-5.5 --limit 100
```

It reads answers from finished runs, so only the candidate judge spends money,
and scores cache per model so the reference pass is never repeated. The
**bias** bound is the binding one: paper Section 5 reports the full-context /
RAG gap as 0.18 points (FC 3.47, RAG 3.64), so a condition-dependent bias of
0.17 can erase or invent E2 by itself. E3 and the ablation move 0.8-1.2 points
and tolerate far more. A group whose scores carry no variance is reported
`n/a` rather than `FAIL` -- an undefined correlation is a property of the
sample, not the judge.

## 13. Ablation deltas are measured against each arm's own blind

Paper Section 7.5 states that the random-pinning control uses "an independent
blind compilation" per topic, and that "blind baselines differ slightly from
Table 4 due to LLM compiler non-determinism". Its Table 5 therefore carries one
Blind column -- the ablation arm's -- and compares *Table 4's* WiCER score
against it. That mixes two compilations inside one delta.

This repo measures each arm against its own blind and prints both, so the row
is internally consistent. The cost is that our WiCER delta and the paper's
+0.95 do not share a denominator even when the method agrees; `report.py` says
so under the table. The random-pinning delta is unaffected, because the paper
already computes that one within the ablation arm.

**Best-of selection is reported alongside what pinning did.** Table 5's caption
is "best iteration per topic", and `max` over iterations includes iteration 0 --
the unpinned blind. An arm whose pinning only ever made things worse therefore
selects its own baseline and reports a delta of exactly +0.00, which reads as
"pinning did nothing" when the truth is "pinning hurt, and selection discarded
the evidence". This is visible in the paper: seven of its seventeen topics print
Random Pin identical to Blind, so its +0.16 for random pinning is biased upward
by the same mechanism.

The best-of columns stay, because dropping them would make our numbers
incomparable to the published ones. Beside each sits a **pinned only** column
over the iterations that actually pinned something (identified by
`n_pinned_facts`, not by index). Both arms get the column: the artifact is
symmetric, it can hide a diagnosed-pinning regression exactly as well as a
random one -- the paper's `local_education`, at 0% recovery, is such a topic --
and an ablation that measures its two arms by different rules is what produced
the incoherent `3.16 | 3.20 | +0.00` row this replaced.

On `local_economy` random pinning reports +0.00 best-of and -0.14 pinned-only.
That is not evidence that random pinning hurt: two blind compilations of the
same prompt landed 0.04 and 0.22 apart in the two pilots (section 10), so a
-0.14 is within compiler noise.

On the full question set, `--eval-full` scores each arm's blind and its best
*pinned* iteration. The pilots scored the random arm's best-of instead, which
was its blind both times, so no random-pinned wiki was ever scored on held-out
questions; the report marks that `random_pin` row with †.

## 14. Probe selection

Paper Algorithm 1 selects "one QA pair per source document" and never says
which. This repo draws each document's probe uniformly from its five
questions, seeded per document from `WicerConfig.probe_seed` and the document
id, so a rerun probes the same questions. RepLiQA's unanswerable questions
(gold: "The answer is not found in the document.", about one per document,
almost always the 4th or 5th) are eligible like any other, so the probe set
mirrors the full 400, as the paper's appears to: its probe blind (2.17)
roughly equals its full-set blind (2.25).

The two pilots took each document's first question instead. That never picked
an unanswerable question and favoured questions about a document's main
subject, which a compiler keeps: their probe-set blind ran 0.4-0.5 above the
full set, so their probe-set numbers are not comparable to the paper's Table 4.
