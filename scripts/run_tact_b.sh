#!/usr/bin/env bash
# TACT milestone B on one topic: the contract loop over self-generated probes,
# starting from milestone A's tact_iter0, then held-out scoring of its result.
#
#   bash scripts/run_tact_b.sh local_economy
#
# Needs the ledger (as for run_tact_a.sh).  Probes are generated if absent --
# API only, ~$0.5 a topic, cached -- so generating them on the laptop first
# (python -m tact.run probes) and copying result-tact/<topic>/self_probes/ up
# just saves doing it here.
#
# tact_iter0 is reused when milestone A left it: it is the loop's start, and
# its held-out scores are the "before" of the pinned/unpinned split, so it is
# never re-rendered under scores that describe an older wiki.  If it is absent
# it is rendered and scored first (~15 min).
#
# The loop: up to 3 iterations of 400 probes (~11 min each at ~45K tokens of
# context on a 3090), repair extractions on the API, then ~11 min of held-out
# answering for tact_loop.  Judge ~$0.04 per 400 answers.
#
# Then, from the laptop:
#   WICER_RESULTS_DIR=result-tact scripts/pull_results.sh [user@]host[:port]
#   uv run python -m tact.run compare --topic <topic>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TOPIC="${1:?usage: run_tact_b.sh <topic>}"
source scripts/tact_common.sh

# The loop's repair extractions (MISS-EXTRACT, MISS-RELATION) and the probe
# writer go through the compiler backend.
export WICER_COMPILER_BACKEND="${WICER_COMPILER_BACKEND:-openrouter}"
export WICER_COMPILER_MODEL="${WICER_COMPILER_MODEL:-anthropic/claude-haiku-5.5}"

TACT="$PY -m tact.run"
ARGS=(--profile "$PROFILE" --topic "$TOPIC")

say "probes"
$TACT probes "${ARGS[@]}"

read -r HAVE_WIKI HAVE_SCORES <<<"$($PY - "$PROFILE" "$TOPIC" <<'PYEOF'
import sys
from tact import get_profile
from wicer_core.records import scores_path, wiki_paths
p, t = get_profile(sys.argv[1]), sys.argv[2]
print(int(wiki_paths(p, t, "tact_iter0")[0].exists()), int(scores_path(p, t, "tact_iter0").exists()))
PYEOF
)"
[[ -n "$HAVE_WIKI" && -n "$HAVE_SCORES" ]] || die "could not check for tact_iter0 -- the probe above failed"
if [[ "$HAVE_WIKI" != "1" ]]; then
  say "rendering tact_iter0 (milestone A did not leave one)"
  $TACT render "${ARGS[@]}" --name tact_iter0 --rho 0.1
fi
if [[ "$HAVE_SCORES" != "1" ]]; then
  say "scoring tact_iter0 (the loop's held-out baseline)"
  $TACT score "${ARGS[@]}" --condition tact_iter0
fi

say "contract loop"
$TACT loop "${ARGS[@]}" --closure "${TACT_CLOSURE:-paragraph}"

say "answering and judging tact_loop"
$TACT score "${ARGS[@]}" --condition tact_loop "${RESUME_FLAG[@]}"

cat <<NEXT

Done. From the laptop:

  WICER_RESULTS_DIR=result-tact scripts/pull_results.sh [user@]host[:port]
  uv run python -m tact.run compare --topic $TOPIC

compare prints the held-out table and the pinned/unpinned split; the loop's
per-iteration record is result-tact/$TOPIC/tact_loop/loop.json.
The reader is still running (pid in results/server/llama-server-$PROFILE.pid).
NEXT
