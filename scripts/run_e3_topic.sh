#!/usr/bin/env bash
# E3 on one topic: the compilation gap, WiCER, and the random-pinning ablation.
# Produces paper Tables 3, 4 and 5.  Skips the E2 arms (RAG) and the KV
# quantisation arm (fc_raw_q4kv) -- see DEVIATIONS.md.
#
#   bash scripts/run_e3_topic.sh local_economy
#   DRY_RUN=1 bash scripts/run_e3_topic.sh local_economy    # cost preview only
#
# Resumability is uneven, so re-running is NOT uniformly free:
#   benchmark/run_benchmark.py  --resume skips answered questions (the long stages)
#   wicer/loop.py               reuses an existing blind wiki; later iterations recompile
#   wicer/compile.py            always recompiles -- it will spend again
#   benchmark/evaluate.py       free to re-run: judge scores are cached on disk
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TOPIC="${1:?usage: run_e3_topic.sh <topic>}"
PROFILE="${WICER_PROFILE:-paper}"
PY="uv run python"

set -a; [[ -f .env ]] && source .env; set +a
: "${OPENROUTER_API_KEY:?paste your key into .env first}"

export WICER_COMPILER_BACKEND="${WICER_COMPILER_BACKEND:-openrouter}"
export WICER_JUDGE_BACKEND="${WICER_JUDGE_BACKEND:-openrouter}"
export WICER_COMPILER_MODEL="${WICER_COMPILER_MODEL:-anthropic/claude-sonnet-5.5}"
export WICER_JUDGE_MODEL="${WICER_JUDGE_MODEL:-anthropic/claude-sonnet-5.5}"

say() { printf '\n\033[1m== %s\033[0m  (%s)\n' "$*" "$(date +%H:%M:%S)"; }

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  $PY wicer/loop.py --profile "$PROFILE" --topic "$TOPIC" --dry-run
  $PY wicer/compile.py --profile "$PROFILE" --topic "$TOPIC" --dry-run
  exit 0
fi

say "reader health"
$PY - <<'PYEOF'
import os, sys
from wicer_core.config import get_profile
from wicer_core.reader import ReaderClient
p = get_profile(os.environ.get("WICER_PROFILE", "paper"))
r = ReaderClient(p.reader)
r.wait_for_health(timeout_s=120)
print("reader OK:", r.props().get("model_path", "?"))
PYEOF

say "probes"
$PY wicer/probe.py --profile "$PROFILE" --topic "$TOPIC"

say "FC raw (the recovery denominator -- slowest stage, ~400 questions at 96K)"
$PY benchmark/run_benchmark.py --profile "$PROFILE" --topic "$TOPIC" --resume

say "blind compilation at three levels"
$PY wicer/compile.py --profile "$PROFILE" --topic "$TOPIC"
for L in light moderate aggressive; do
  say "evaluating wiki_$L"
  $PY benchmark/run_benchmark.py --profile "$PROFILE" --topic "$TOPIC" \
    --condition "wiki_$L" --context-file "results/$TOPIC/wikis/blind_$L.md" --resume
done

say "WiCER loop"
$PY wicer/loop.py --profile "$PROFILE" --topic "$TOPIC" --eval-full

say "random-pinning control"
$PY wicer/random_pin.py --profile "$PROFILE" --topic "$TOPIC" --eval-full

say "judging"
$PY benchmark/evaluate.py --profile "$PROFILE" --topic "$TOPIC"

say "report"
$PY benchmark/report.py --profile "$PROFILE"

cat <<NEXT

Done. Check, in order:

  1. Kept verbatim column   -- results/*/report.md, iteration trajectory.
                               Well below 100% means the compiler ignored the
                               preservation instruction; the probe means beside
                               it are then not evidence about WiCER.
  2. Recovery               -- Table 4 section. Paper reports 82% for
                               $TOPIC (blind 2.16 -> WiCER 3.24, FC 3.48).
  3. measured_cost_usd      -- results/*/run_meta.json, cost_source=openrouter.
  4. cache_read_tokens      -- nonzero means the 1h prompt cache is working.

Copy results off the box before the rental ends:
  tar czf wicer-results.tgz results/ && echo "scp it down"
NEXT
