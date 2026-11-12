#!/usr/bin/env bash
# Paged cache on one topic: a verbatim context that marks its cuts, read with
# and without the option to fetch one back.
#
#   bash scripts/run_paged.sh local_economy
#
# Needs the topic's ledger, as run_tact_a.sh does (result-tact/<topic>/ledger/).
#
# Three contexts at the same token budget (rho 0), same selection rule:
#   paged_none   no stubs: the verbatim baseline, rendered by this code
#   paged_bare   a bare stub at each cut:            [cut 12.3]
#   paged_desc   a stub that names what was cut:     [cut 12.3: entity relation; ...]
#
# Seven runs, in the order that answers the cheapest question first:
#   paged_none, paged_bare, paged_desc      ordinary prompt -- what stubs cost unread
#   paged_bare_oracle, paged_desc_oracle    the gold passage's cuts restored -- the ceiling
#   paged_bare_reader, paged_desc_reader    the reader asks -- the method
#
# ~11 min of generation per run at ~45K tokens of context on a 3090 (a reader
# run adds a second pass for every question that fetched), ~$0.04 of judge each.
# STUBS="bare" runs one stub style; RESUME=1 skips answered questions.
#
# Then, from the laptop, which holds result-sweep/ for the comparison:
#   WICER_RESULTS_DIR=result-tact scripts/pull_results.sh [user@]host[:port]
#   uv run python -m tact.run compare --topic <topic>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TOPIC="${1:?usage: run_paged.sh <topic>}"
source scripts/tact_common.sh

TACT="$PY -m tact.run"
ARGS=(--profile "$PROFILE" --topic "$TOPIC")
declare -A STYLE=([bare]=bare [desc]=described)
read -r -a KINDS <<<"${STUBS:-bare desc}"

say "rendering"
$TACT render "${ARGS[@]}" --name paged_none --style verbatim --rho 0
for K in "${KINDS[@]}"; do
  $TACT render "${ARGS[@]}" --name "paged_$K" --style verbatim --rho 0 --stubs "${STYLE[$K]}"
done

for ARM in paged_none "${KINDS[@]/#/paged_}"; do
  say "ordinary prompt on $ARM"
  $TACT score "${ARGS[@]}" --condition "$ARM" "${RESUME_FLAG[@]}"
done

for MODE in oracle reader; do
  for K in "${KINDS[@]}"; do
    say "$MODE fetch on paged_$K"
    $TACT page "${ARGS[@]}" --wiki "paged_$K" --mode "$MODE" "${RESUME_FLAG[@]}"
  done
done

cat <<NEXT

Done. From the laptop:

  WICER_RESULTS_DIR=result-tact scripts/pull_results.sh [user@]host[:port]
  uv run python -m tact.run compare --topic $TOPIC

The reader is still running (pid in results/server/llama-server-$PROFILE.pid).
Kill it when you are done.
NEXT
