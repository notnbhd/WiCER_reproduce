#!/usr/bin/env bash
# arm.sh NAME [render flags...]: render one TACT arm against the reader and score it.
set -euo pipefail
cd ~/WiCER
export PATH="$HOME/.local/bin:$PATH"
TOPIC=${TOPIC:-local_economy}
source scripts/tact_common.sh
NAME="$1"; shift
$PY -m tact.run render --profile "$PROFILE" --topic $TOPIC --name "$NAME" "$@"
$PY -m tact.run score --profile "$PROFILE" --topic $TOPIC --condition "$NAME"
echo "ARM DONE: $TOPIC $NAME" >> status.log
