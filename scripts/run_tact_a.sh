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
source scripts/tact_common.sh

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
