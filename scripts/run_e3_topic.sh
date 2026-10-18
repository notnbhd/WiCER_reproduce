#!/usr/bin/env bash
# E3 on one topic: the compilation gap, WiCER, and the random-pinning ablation.
# Produces paper Tables 3, 4 and 5.  Skips the E2 arms (RAG) and the KV
# quantisation arm (fc_raw_q4kv) -- see DEVIATIONS.md.
#
#   bash scripts/run_e3_topic.sh local_economy
#   DRY_RUN=1 bash scripts/run_e3_topic.sh local_economy    # cost preview only
#
# Compiler and judge default to anthropic/claude-haiku-5.5, at the profile's
# effort=high with max_tokens raised to 64,000 so reasoning and the wiki are not
# competing for one budget.  A full 17-topic E3 sweep is roughly $13-17 on Haiku
# against $114 on Sonnet, which is what fits the key's credit -- but reasoning
# is billed, so effort=high lands above a low-effort estimate.  The one-topic
# pilot measures it before 17 topics commit the budget.  For Sonnet instead:
#   WICER_COMPILER_MODEL=anthropic/claude-sonnet-5.5 \
#   WICER_JUDGE_MODEL=anthropic/claude-sonnet-5.5 bash scripts/run_e3_topic.sh ...
# DRY_RUN prices every model at $3/$15 (wicer_core/config.py price_in/out, no
# override), so its preview overestimates Haiku by 30-40x.  Trust
# measured_cost_usd in run_meta.json instead.
#
# Starts llama-server itself if nothing is already answering on the profile's
# reader port, so rental_setup.sh -> this script is the whole chain. Set
# WICER_START_SERVER=0 to require a reader you launched yourself.
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
# uv installs to ~/.local/bin, and rental_setup.sh only puts that on PATH
# inside its own process -- a later tmux or ssh session starts without it and
# every `uv run` here dies as "command not found".  Same shape as the CUDA
# toolkit problem one step earlier in the chain: the tool is installed, the
# path is not inherited.
if ! command -v uv >/dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "FAILED: uv not on PATH; run scripts/rental_setup.sh first" >&2; exit 1; }

PY="uv run python"

set -a; [[ -f .env ]] && source .env; set +a
: "${OPENROUTER_API_KEY:?paste your key into .env first}"

export WICER_COMPILER_BACKEND="${WICER_COMPILER_BACKEND:-openrouter}"
export WICER_JUDGE_BACKEND="${WICER_JUDGE_BACKEND:-openrouter}"
export WICER_COMPILER_MODEL="${WICER_COMPILER_MODEL:-anthropic/claude-haiku-5.5}"
export WICER_JUDGE_MODEL="${WICER_JUDGE_MODEL:-anthropic/claude-haiku-5.5}"

# Reasoning bills as completion tokens and counts against max_tokens, so the
# compiler's budget is split between thinking and the wiki with no separate
# thinking allowance.  Haiku once spent an entire 4,000-token budget on
# reasoning and returned no wiki -- the fix is room, not less thinking: a
# paper-scale wiki runs ~16K tokens (paper 6.2) and Haiku's output ceiling is
# 128K, so 64,000 keeps reasoning and output from competing.  Effort stays at
# the profile's high.  Any other compiler keeps the profile's 32,000, because
# silently re-tuning Sonnet would change its output without saying so.
if [[ "$WICER_COMPILER_MODEL" == *haiku* ]]; then
  export WICER_COMPILER_MAX_TOKENS="${WICER_COMPILER_MAX_TOKENS:-64000}"
fi

say() { printf '\n\033[1m== %s\033[0m  (%s)\n' "$*" "$(date +%H:%M:%S)"; }
die() { printf '\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  $PY wicer/loop.py --profile "$PROFILE" --topic "$TOPIC" --dry-run
  $PY wicer/compile.py --profile "$PROFILE" --topic "$TOPIC" --dry-run
  exit 0
fi

say "reader"
read -r READER_HOST READER_PORT <<<"$($PY - "$PROFILE" <<'PYEOF'
import sys
from wicer_core.config import get_profile
r = get_profile(sys.argv[1]).reader
print(r.host, r.port)
PYEOF
)"

# `read` swallows the exit status of the substitution above, so a profile that
# fails to import leaves both of these empty and the dispatch below blames a
# "remote" reader host that is simply the empty string.  Fail on the real cause.
[[ -n "$READER_HOST" && -n "$READER_PORT" ]] ||
  die "could not read reader host/port from profile '$PROFILE' -- the probe above failed"

# Nothing else in the chain starts llama-server: rental_setup.sh boots one only
# to verify the profile fits, then kills it. Start it here -- but never touch a
# server that is already answering, because reloading 96K of context costs a
# minute and a re-run for --resume should not pay that again.  The health probe
# is time-bounded so a stale WICER_READER_HOST fails fast instead of hanging
# before any work has started.
if curl -fsS --connect-timeout 3 --max-time 5 "http://$READER_HOST:$READER_PORT/health" >/dev/null 2>&1; then
  echo "reader already up at $READER_HOST:$READER_PORT -- leaving it alone"
elif [[ "${WICER_START_SERVER:-1}" != "1" ]]; then
  die "no reader at $READER_HOST:$READER_PORT and WICER_START_SERVER=0"
elif [[ "$READER_HOST" == "127.0.0.1" || "$READER_HOST" == "localhost" || "$READER_HOST" == "0.0.0.0" ]]; then
  say "starting llama-server (loading 96K of context takes a minute)"
  benchmark/start_server.sh --profile "$PROFILE"
else
  die "reader host $READER_HOST is remote -- start llama-server on that box, or unset WICER_READER_HOST"
fi

# Confirm the server the client will actually talk to serves the right model.
$PY - "$PROFILE" <<'PYEOF'
import sys
from wicer_core.config import get_profile
from wicer_core.reader import ReaderClient
r = ReaderClient(get_profile(sys.argv[1]).reader)
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
                               Haiku reads ~0% here because it paraphrases
                               pinned facts rather than copying them, so on a
                               Haiku run this column cannot tell "dropped" from
                               "reworded".  Check whether the facts' distinctive
                               tokens (dates, proper nouns) reached the wiki
                               before concluding anything was lost.
  2. Recovery               -- Table 4 section. Paper reports 82% for
                               $TOPIC (blind 2.16 -> WiCER 3.24, FC 3.48).
  3. measured_cost_usd      -- results/*/run_meta.json, cost_source=openrouter.
  4. cache_read_tokens      -- nonzero means the 1h prompt cache is working.

Pull the results down from your laptop -- start this BEFORE the run finishes,
since a reclaimed rental takes the only copy with it:

  scripts/pull_results.sh --watch [user@]host[:port]

The reader is still running (pid in results/server/llama-server-$PROFILE.pid),
so a --resume re-run skips the context reload. Kill it when you are done.
NEXT
