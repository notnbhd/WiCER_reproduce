#!/usr/bin/env bash
# TACT milestone A on one topic: render the iteration-0 wikis and the
# truncation baseline from an already-extracted ledger, then answer and judge
# every question against each.
#
#   bash scripts/run_tact_a.sh local_economy
#
# Extraction is NOT run here.  It needs only the API, so it is done (and paid
# for) on the laptop; copy its ledger up with the repo:
#
#   result-tact/<topic>/ledger/        (ledger.jsonl, ledger_meta.json, extract/)
#
# The ledger records the corpus it was extracted from, and rendering refuses a
# different one -- so if this box built the corpus with another document rule,
# the run stops before the reader is loaded rather than after an hour.
#
# Arms (draft section 3.9): tact_iter0 fills (1 - 0.1) B, tact_full fills B,
# truncate cuts every document to its share of B.  ~11 min of generation per
# arm at ~45K tokens of context on a 3090, plus judging; ~$0.04 of judge per arm.
#
# Re-running regenerates and re-judges every arm (the judge's disk cache makes
# unchanged answers free).  RESUME=1 instead skips answered questions -- only
# safe when the wikis have not been re-rendered since.
#
# Then, from the laptop, which holds result-sweep/ for the comparison:
#   WICER_RESULTS_DIR=result-tact scripts/pull_results.sh [user@]host[:port]
#   uv run python -m tact.run compare --topic <topic>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TOPIC="${1:?usage: run_tact_a.sh <topic>}"
PROFILE="${WICER_PROFILE:-paper}"
# Same PATH repair as run_e3_topic.sh: rental_setup.sh installs uv into
# ~/.local/bin but a later ssh or tmux session does not inherit it.
if ! command -v uv >/dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "FAILED: uv not on PATH; run scripts/rental_setup.sh first" >&2; exit 1; }

PY="uv run python"

set -a; [[ -f .env ]] && source .env; set +a
: "${OPENROUTER_API_KEY:?paste your key into .env first}"

# tact.run sets no backend itself; without these the judge falls back to the
# anthropic SDK and its own key.
export WICER_JUDGE_BACKEND="${WICER_JUDGE_BACKEND:-openrouter}"
export WICER_JUDGE_MODEL="${WICER_JUDGE_MODEL:-anthropic/claude-haiku-5.5}"

RESUME_FLAG=()
[[ "${RESUME:-0}" == "1" ]] && RESUME_FLAG=(--resume)

say() { printf '\n\033[1m== %s\033[0m  (%s)\n' "$*" "$(date +%H:%M:%S)"; }
die() { printf '\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

say "ledger and corpus"
$PY - "$PROFILE" "$TOPIC" <<'PYEOF' || die "ledger/corpus check failed (above)"
import sys
from tact import get_profile
from tact.ledger import load_ledger
from tact.render import check_corpus
from wicer_core.corpus import build_topic, topic_dir

profile, topic = get_profile(sys.argv[1]), sys.argv[2]
ledger = load_ledger(profile, topic)
if ledger is None:
    sys.exit(f"no ledger under {profile.results_dir}/{topic}/ledger -- extract on the laptop "
             "(python -m tact.run extract) and copy that directory up")
corpus = build_topic(profile, topic)          # builds if absent, else loads
check_corpus(ledger, corpus)
print(f"ledger: {ledger.meta['n_claims']} claims, extracted with {ledger.meta.get('extracted_with')}")
print(f"corpus: {len(corpus.documents)} docs, sha {corpus.manifest['content_sha256'][:12]} -- matches")
PYEOF

say "reader"
read -r READER_HOST READER_PORT <<<"$($PY - "$PROFILE" <<'PYEOF'
import sys
from wicer_core.config import get_profile
r = get_profile(sys.argv[1]).reader
print(r.host, r.port)
PYEOF
)"
[[ -n "$READER_HOST" && -n "$READER_PORT" ]] ||
  die "could not read reader host/port from profile '$PROFILE' -- the probe above failed"

# Never touch a server that is already answering: reloading costs a minute.
if curl -fsS --connect-timeout 3 --max-time 5 "http://$READER_HOST:$READER_PORT/health" >/dev/null 2>&1; then
  echo "reader already up at $READER_HOST:$READER_PORT -- leaving it alone"
elif [[ "${WICER_START_SERVER:-1}" != "1" ]]; then
  die "no reader at $READER_HOST:$READER_PORT and WICER_START_SERVER=0"
elif [[ "$READER_HOST" == "127.0.0.1" || "$READER_HOST" == "localhost" || "$READER_HOST" == "0.0.0.0" ]]; then
  say "starting llama-server"
  benchmark/start_server.sh --profile "$PROFILE"
else
  die "reader host $READER_HOST is remote -- start llama-server on that box, or unset WICER_READER_HOST"
fi

$PY - "$PROFILE" <<'PYEOF'
import sys
from wicer_core.config import get_profile
from wicer_core.reader import ReaderClient
r = ReaderClient(get_profile(sys.argv[1]).reader)
r.wait_for_health(timeout_s=120)
print("reader OK:", r.props().get("model_path", "?"))
PYEOF

TACT="$PY -m tact.run"
ARGS=(--profile "$PROFILE" --topic "$TOPIC")

say "ledger checks, in reader tokens"
$TACT inspect "${ARGS[@]}"

say "rendering"
$TACT render "${ARGS[@]}" --name tact_iter0 --rho 0.1
$TACT render "${ARGS[@]}" --name tact_full --rho 0
$TACT truncate "${ARGS[@]}"

for ARM in tact_iter0 tact_full truncate; do
  say "answering and judging $ARM"
  $TACT score "${ARGS[@]}" --condition "$ARM" "${RESUME_FLAG[@]}"
done

cat <<NEXT

Done. From the laptop:

  WICER_RESULTS_DIR=result-tact scripts/pull_results.sh [user@]host[:port]
  uv run python -m tact.run compare --topic $TOPIC

The reader is still running (pid in results/server/llama-server-$PROFILE.pid).
Kill it when you are done.
NEXT
