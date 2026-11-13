#!/usr/bin/env bash
# Paged cache with whole-document fetch, at the paper's 80 documents a topic.
#
#   bash scripts/run_paged_doc.sh local_economy [more topics ...]
#   bash scripts/run_paged_doc.sh all
#
# Three arms a topic, in result-tact/<topic>/, all at 50% of the source tokens:
#   paged_none              plain verbatim, no stubs                  -- the baseline
#   paged_bare_reader_doc   a bare stub at each cut; the reader may reply FETCH <document>
#                           and gets that document's full text before the question
#   paged_bare_oracle_doc   the gold document restored whenever its gold passage was cut
#
# A topic with no ledger is extracted first (API: ~$1 and ~5 min).  The arms take
# ~18 min of an RTX 4090.  A topic that fails is tried once more with RESUME=1,
# which keeps generated answers and cached judgements, and the rest carry on; the
# failures are listed at the end.  Logs: result-tact/<topic>/box_logs/paged_doc.log.
#
# Results (2026-11-14, 17 topics): see `uv run python scripts/paged_report.py`.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"

[[ $# -gt 0 ]] || { echo "usage: run_paged_doc.sh <topic>... | all" >&2; exit 2; }
if [[ "$1" == all ]]; then
  read -r -a TOPICS <<<"$(uv run python -c 'from wicer_core.config import TOPICS; print(*TOPICS)')"
else
  TOPICS=("$@")
fi

one_topic() (
  set -euo pipefail
  TOPIC="$1"
  set -a; [[ -f .env ]] && source .env; set +a
  source scripts/api_backend.sh
  if [[ ! -f "result-tact/$TOPIC/ledger/ledger_meta.json" ]]; then
    # Eight workers, not more: the judge's threads share the organization's
    # concurrent-request limit with an extraction running alongside.
    uv run python -m tact.run extract --topic "$TOPIC" --workers 8
  fi
  source scripts/tact_common.sh
  TACT="$PY -m tact.run"
  A=(--profile "$PROFILE" --topic "$TOPIC")
  say rendering
  $TACT render "${A[@]}" --name paged_none --style verbatim --rho 0
  $TACT render "${A[@]}" --name paged_bare --style verbatim --rho 0 --stubs bare
  say "plain verbatim"
  $TACT score "${A[@]}" --condition paged_none "${RESUME_FLAG[@]}"
  say "document fetch"
  $TACT page "${A[@]}" --wiki paged_bare --mode reader --fetch doc "${RESUME_FLAG[@]}"
  say "document oracle"
  $TACT page "${A[@]}" --wiki paged_bare --mode oracle --fetch doc "${RESUME_FLAG[@]}"
)

FAILED=()
for T in "${TOPICS[@]}"; do
  mkdir -p "result-tact/$T/box_logs"
  LOG="result-tact/$T/box_logs/paged_doc.log"
  echo "== $T start ($(date +%H:%M:%S))"
  if RESUME=0 one_topic "$T" >"$LOG" 2>&1 || RESUME=1 one_topic "$T" >>"$LOG" 2>&1; then
    echo "== $T ok ($(date +%H:%M:%S))"
  else
    FAILED+=("$T"); echo "FAILED $T -- see $LOG"; tail -3 "$LOG"
  fi
done
echo "== done ($(date +%H:%M:%S)); failed: ${FAILED[*]:-none}"
echo "The reader is still running (pid in results/server/llama-server-*.pid)."
[[ ${#FAILED[@]} -eq 0 ]]
