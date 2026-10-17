# WiCER reproduction

Reproduction of **WiCER: Wiki-memory Compile, Evaluate, Refine**
(arXiv:2605.07068v1) on RepLiQA.

Two experiments are implemented:

* **E2 — the scalability gap.** Full-context KV cache inference beats RAG on a
  small curated corpus, but loses to it at 80 raw documents per topic. The
  paper measures FC 3.46/3.47 against RAG 3.63.
* **E3 — the compilation gap and WiCER.** Blindly compiling those documents
  into a wiki collapses quality to 2.14–2.32. WiCER's compile → evaluate →
  diagnose → recompile loop recovers ~80% of the loss (to ~3.18/3.24) and cuts
  catastrophic failures from 55.1% to 24.8%. An ablation shows the gain comes
  from *targeted* diagnosis (+0.95), not from pinning text in general (+0.16).

E1 (Policygenius) is out of scope — see `DEVIATIONS.md`.

**Read `DEVIATIONS.md` before comparing any number here to a published one.**
The paper publishes only one of its prompts, never says which 80 documents it
sampled, and runs on hardware where two of its conclusions do not transfer.

A one-topic pilot has been run end to end: **`docs/pilot-2026-10-17.md`**. It
reproduces the paper's Table 4 recovery on `local_economy` (108% against 82%)
and finds that the gain does not survive on held-out questions (10%) -- the two
question sets disagree by a factor of ten on the same run. It also establishes
which 80 documents the paper used, which the first two attempts got wrong.

## Quick start (no GPU, no API key, no downloads)

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
uv run pytest -q          # 100 tests
scripts/smoke.sh          # whole pipeline against a fake reader and judge
```

`scripts/smoke.sh` boots `tests/fake_llama_server.py`, runs every condition
from spec.md Section 6 against a synthetic 12-document fixture, and writes
`results/test/report.md`. It proves the code works. It says nothing about the
paper's numbers — the stubs are stubs (`DEVIATIONS.md` §9).

## Layout

```
wicer_core/      corpus, reader client, judge, metrics, records, provenance
                 env.py (.env loading), openrouter.py (the OpenRouter backend)
                 agreement.py (Appendix J judge-agreement statistics)
benchmark/       full-context arm    config.py start_server.sh run_benchmark.py evaluate.py report.py
rag_benchmark/   RAG baseline        rag_config.py build_index.py run_rag_benchmark.py rag_evaluate.py rag_report.py
wicer/           the algorithm       compile.py probe.py diagnose.py loop.py random_pin.py prompts/
tests/           fake reader, fixture generator, test suite
```

The entry points match the paper's Appendix C. The shared `wicer_core` exists
so the judge, the chat template and the aggregation are written once — a second
judge implementation would silently break every cross-condition comparison.

## Profiles

Hardware lives in one place, `wicer_core/config.py`:

| Profile | Reader | Context | Documents | For |
|---|---|---|---|---|
| `paper` | Llama 3.1 8B Instruct Q5_K_M | 96K, q8_0 KV | 80/topic | the real run; needs ~12 GB VRAM |
| `smoke` | a small GGUF | 16K | 6/topic | a laptop; a different experiment, not a reproduction |
| `test` | `tests/fake_llama_server.py` | 8K | 12 (fixture) | CI and development |

Select with `--profile` or `WICER_PROFILE`. Individual knobs override via
`WICER_READER_HOST`, `WICER_READER_PORT`, `WICER_MODEL_PATH`, `WICER_N_CTX`,
`WICER_KV_TYPE`, `WICER_JUDGE_MODEL`, `WICER_COMPILER_MODEL`,
`WICER_RESULTS_DIR`.

The reader is reached over HTTP, so `paper` runs equally well against a rented
GPU: point `WICER_READER_HOST` at it.

## Running the real experiment

### 1. Reader

The RAG arm needs real embeddings, which the quick-start install leaves out:

```bash
uv pip install -e '.[rag,dev]'      # pulls sentence-transformers and torch
```

```bash
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf --local-dir models/

# llama.cpp with CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server

benchmark/start_server.sh --profile paper
```

24 GB is enough: ~5.7 GB of weights plus ~6.4 GB of q8_0 KV at 96K. The server
runs with `--parallel 1` deliberately — concurrent requests land in different
slots and destroy the prefix KV reuse the entire experiment measures.

### 2. Data

```bash
uv run python -m wicer_core.corpus --profile paper
```

17 topics × 80 documents × 400 QA = 6,800 questions. Each topic directory gets
a manifest with a content hash; `--check` asserts a rebuild reproduces it.

### 3. Credentials

The compiler, the diagnoser and the judge call an LLM. Two backends:

**Anthropic (default).** Export `ANTHROPIC_API_KEY`, or run `ant auth login` —
the SDK picks up the stored profile with no environment variable.

**OpenRouter.** Needs the `openai` client: `uv pip install -e '.[openrouter]'`.

```bash
OPENROUTER_API_KEY=sk-or-...
WICER_COMPILER_BACKEND=openrouter    # compiler *and* diagnoser
WICER_JUDGE_BACKEND=openrouter
```

Either backend reads credentials from a `.env` in the repo root — loaded
automatically, no `source` needed. The shell wins over the file, so an exported
variable is never silently overridden by a stale `.env`. `.env` is gitignored.

An Anthropic-native model id is rewritten to an OpenRouter slug
(`claude-sonnet-5` → `anthropic/claude-sonnet-5`); set `WICER_JUDGE_MODEL` /
`WICER_COMPILER_MODEL` to choose another. **The two backends are not
interchangeable** — approximated effort and a different served model. Read
`DEVIATIONS.md` §11 before comparing a number produced on one to a number
produced on the other. Pricing matches the first-party API line for line,
prompt caching included, so the estimate below holds for either.

### 4. A topic, end to end

```bash
TOPIC=company_policies

# E2: the two baselines
uv run python benchmark/run_benchmark.py     --profile paper --topic $TOPIC
uv run python rag_benchmark/build_index.py   --profile paper --topic $TOPIC
uv run python rag_benchmark/run_rag_benchmark.py --profile paper --topic $TOPIC

# E3: blind compilation at three levels, then the loop and its control
uv run python wicer/probe.py    --profile paper --topic $TOPIC
uv run python wicer/compile.py  --profile paper --topic $TOPIC
for L in light moderate aggressive; do
  uv run python benchmark/run_benchmark.py --profile paper --topic $TOPIC \
    --condition wiki_$L --context-file results/$TOPIC/wikis/blind_$L.md
done
uv run python wicer/loop.py       --profile paper --topic $TOPIC --eval-full
uv run python wicer/random_pin.py --profile paper --topic $TOPIC --eval-full

# judge and report
uv run python benchmark/evaluate.py --profile paper --topic $TOPIC
uv run python benchmark/report.py   --profile paper
```

Drop `--topic` to sweep all 17. **Pilot on three topics first.**

### 5. On a rented GPU

The reader speaks HTTP and 96K of q8_0 KV needs ~12.5 GB, so the `paper`
profile usually means a rented box. Three scripts cover it:

```bash
# on the box, once the repo is there
bash scripts/rental_setup.sh                 # deps, CUDA llama.cpp, model, corpus,
                                             # then verifies the profile actually fits
tmux new -s wicer
bash scripts/run_e3_topic.sh local_economy   # starts llama-server itself, then runs E3

# from your laptop, while that runs
scripts/pull_results.sh --watch [user@]host[:port]
```

`rental_setup.sh --verify` is worth running alone before anything expensive: it
reports VRAM in use, where ~18.5 GB instead of ~12.5 GB means flash attention
is off and fp16 KV is being allocated.

Everything a run produces lives in `results/` plus the judge cache in
`.cache/judge/`, both gitignored. `pull_results.sh` copies the remote `results/`
tree into `./results-rental` (`--with-cache` takes the judge cache too, which
makes a local re-judge free rather than ~$7). Pull *during* the run — a
reclaimed rental takes the only copy of the one artifact you cannot regenerate
without spending the API budget again.

`benchmark/report.py` reads nothing but the JSONL on disk, so the tables
regenerate anywhere, with no GPU, reader or API key:

```bash
WICER_RESULTS_DIR=results-rental uv run python benchmark/report.py --profile paper
```

### Cost and time

Every stage that spends money takes `--dry-run`:

```bash
uv run python wicer/loop.py    --profile paper --dry-run
uv run python wicer/compile.py --profile paper --dry-run
```

For the full 17-topic sweep, budget roughly **$250–350** of Claude API
(~54K judge calls dominate, plus ~100 compilations) and **40–50 GPU-hours**.
Three things keep that from being worse:

* judge results are cached content-addressed on disk, so reruns and the loop's
  repeated probe scoring are free after the first pass;
* source documents sit in a 1-hour-TTL cached system block, and the same
  collection is recompiled at three levels and once per iteration;
* probe answers are generated locally, at no API cost.

Resumability is uneven. `benchmark/run_benchmark.py` takes `--resume` and
skips questions already answered, which covers the long generation stages.
`wicer/loop.py` reuses an existing blind wiki (pass `--force-blind` to
recompile it) but its later iterations compile again; `wicer/compile.py`
always recompiles. Judge scores are cached content-addressed on disk, so
`benchmark/evaluate.py` and `benchmark/report.py` are free to re-run.

On the OpenRouter backend the cost figures are not estimates: every request
asks for usage accounting, and `run_meta.json` carries the billed total with
`cost_source: "openrouter"` alongside the backend and the resolved model slug.

## What the report tells you

`benchmark/report.py` writes `report.md`, `per_topic.csv` and
`per_question.csv`, shaped after paper Tables 3, 4, 5, 12 and 13, plus:

* an **iteration trajectory** per topic — probe mean, score-1 rate, failures,
  pinned facts and words, the share of pinned facts the compiler **kept
  verbatim**, measured compression ratio, improvement;
* **probe-set and held-out** WiCER numbers side by side (`DEVIATIONS.md` §5);
* a **success-criteria** table checking spec.md Section 8 target by target,
  with the delta and a coverage column. The verdicts are only meaningful once
  coverage says 17 topics.

Directional agreement is the bar: the FC/RAG crossover exists, and targeted
diagnosis beats random pinning by a wide margin. Exact numbers move — the
compiler is stochastic and TTFT changes outright on different hardware.

## Reproducibility

Every run writes `run_meta.json`: profile, model path and hash, the server's
own `/props`, the configured judge model *and* the model the API returned, all
seeds, platform and git revision. spec.md Section 9 names judge drift and
hardware as the two things that silently invalidate a comparison; both are on
disk after every run.

Corpus manifests carry a content hash. Probe selection, the RepLiQA subsample,
and the random-pinning control are all deterministic — the control seeds from a
SHA-256 of the topic name, not Python's per-process-salted `hash()`.

For a stronger result than the paper's, run at least three compiler seeds:
`wicer/compile.py --seed N` and `wicer/loop.py --seed N`.

Before trusting a cheaper judge model, measure it against the one it replaces
using the paper's own Appendix J statistics:

```bash
uv run python benchmark/judge_agreement.py --profile paper \
    --candidate-model anthropic/claude-haiku-5.5 --limit 100
```

The bar is per-condition Pearson r >= 0.89 and |bias| <= 0.17, and the bias
bound is the one that bites: the FC/RAG gap the sweep must detect is only 0.18
points. See `DEVIATIONS.md` §12.

## Licences

RepLiQA is CC-BY-4.0 (Montero et al., NeurIPS 2024 D&B). Llama 3.1 is under the
Llama 3.1 Community License; llama.cpp is MIT; Sentence-BERT is Apache 2.0.
