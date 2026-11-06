#!/usr/bin/env bash
# Confirmation on two unseen topics: extract (API, in parallel), then truncate / verb_r0 / next.
set -o pipefail
cd ~/WiCER
export PATH="$HOME/.local/bin:$PATH"
set -a; source .env; set +a
export WICER_RESULTS_DIR=result-tact WICER_COMPILER_BACKEND=openrouter WICER_COMPILER_MODEL=anthropic/claude-haiku-5.5
TOPICS="cybersecurity_news small_medium_ent"
for T in $TOPICS; do
  ( uv run python -m tact.run extract --topic $T > extract_$T.log 2>&1 && echo "EXTRACT DONE: $T" >> status.log || echo "EXTRACT FAILED: $T" >> status.log ) &
done
wait
for T in $TOPICS; do
  grep -q "EXTRACT DONE: $T" status.log || continue
  export TOPIC=$T
  ( source scripts/tact_common.sh
    $PY -m tact.run truncate --profile "$PROFILE" --topic $T
    $PY -m tact.run score --profile "$PROFILE" --topic $T --condition truncate ) > arm_${T}_truncate.log 2>&1 \
    && echo "ARM DONE: $T truncate" >> status.log || echo "ARM FAILED: $T truncate" >> status.log
  bash arm.sh next --style verbatim --select next --rho 0 > arm_${T}_next.log 2>&1 || echo "ARM FAILED: $T next" >> status.log
  bash arm.sh verb_r0 --style verbatim --rho 0 > arm_${T}_verb_r0.log 2>&1 || echo "ARM FAILED: $T verb_r0" >> status.log
done
echo "CONFIRM FINISHED" >> status.log
