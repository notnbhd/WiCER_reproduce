#!/usr/bin/env bash
# Paged cache on whole topics, larger than the reader's 96K window.
#
#   bash scripts/run_paged_full.sh local_economy cybersecurity_news small_medium_ent
#   BUDGET=55000 bash scripts/run_paged_full.sh local_economy      # a smaller context
#
# `--profile full` takes every document of a topic (137-233; 107K-298K tokens) into
# data/repliqa_full/ and writes to result-full/<topic>/.  Full context does not fit,
# so every arm is a context of BUDGET reader tokens (default 88,000, leaving room
# for a fetched document and the answer):
#   paged_none_b88               plain verbatim, no stubs                 -- the baseline
#   paged_bare_b88_reader_doc    a bare stub at each cut; the reader may FETCH <document>
#   paged_index_b88_reader_doc   no stubs at the cuts; each document that lost sentences
#                                ends with one line naming what was cut, from the ledger
#   paged_bare_b88_oracle_doc    the gold document restored whenever its gold passage was cut
#
# Per topic: build the corpus, seed the ledger's extraction cache from the
# 80-document ledger in result-tact/ (same documents, same text), extract the
# rest (API: ~$2 and ~20 min), then the arms (~1.6 h of an RTX 4090 for 1,000+
# questions).  A topic that fails is tried once more with RESUME=1.  Logs:
# result-full/<topic>/box_logs/paged_full.log.
#
# Results (2026-11-14): see `uv run python scripts/paged_report.py --full`.
# A 55K budget lost 0.58 to 88K on these topics: fill the window.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export WICER_PROFILE=full
BUDGET="${BUDGET:-88000}"
B="b$((BUDGET / 1000))"

[[ $# -gt 0 ]] || { echo "usage: run_paged_full.sh <topic>..." >&2; exit 2; }

one_topic() (
  set -euo pipefail
  TOPIC="$1"
  set -a; [[ -f .env ]] && source .env; set +a
  source scripts/api_backend.sh
  [[ -f "data/repliqa_full/$TOPIC/manifest.json" ]] ||
    uv run python -m wicer_core.corpus --profile full --topic "$TOPIC"
  if [[ ! -f "result-full/$TOPIC/ledger/ledger_meta.json" ]]; then
    mkdir -p "result-full/$TOPIC/ledger/extract"
    if [[ -d "result-tact/$TOPIC/ledger/extract" ]]; then
      cp -n "result-tact/$TOPIC/ledger/extract/"*.json "result-full/$TOPIC/ledger/extract/"
    fi
    uv run python -m tact.run extract --profile full --topic "$TOPIC" --workers 8
  fi
  source scripts/tact_common.sh
  TACT="$PY -m tact.run"
  A=(--profile full --topic "$TOPIC")
  R=(--style verbatim --rho 0 --budget-tokens "$BUDGET")
  say rendering
  $TACT render "${A[@]}" "${R[@]}" --name "paged_none_$B"
  $TACT render "${A[@]}" "${R[@]}" --name "paged_bare_$B" --stubs bare
  $TACT render "${A[@]}" "${R[@]}" --name "paged_index_$B" --stubs index
  say "plain verbatim"
  $TACT score "${A[@]}" --condition "paged_none_$B" "${RESUME_FLAG[@]}"
  say "document fetch, bare stubs"
  $TACT page "${A[@]}" --wiki "paged_bare_$B" --mode reader --fetch doc "${RESUME_FLAG[@]}"
  say "document fetch, cut index"
  $TACT page "${A[@]}" --wiki "paged_index_$B" --mode reader --fetch doc "${RESUME_FLAG[@]}"
  say "document oracle"
  $TACT page "${A[@]}" --wiki "paged_bare_$B" --mode oracle --fetch doc "${RESUME_FLAG[@]}"
)

FAILED=()
for T in "$@"; do
  mkdir -p "result-full/$T/box_logs"
  LOG="result-full/$T/box_logs/paged_full.log"
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
