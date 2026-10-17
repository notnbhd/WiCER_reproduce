#!/usr/bin/env bash
# Pull a run's results off the rented box -- repeatedly, while it is still running.
#
#   scripts/pull_results.sh root@ssh4.vast.ai:12345             # once
#   scripts/pull_results.sh --watch root@ssh4.vast.ai:12345     # every 5 min, Ctrl-C to stop
#   scripts/pull_results.sh --with-cache root@host:22           # also the judge score cache
#
# The results are the only thing on that box that cannot be recreated without
# spending the API budget again, and a rental can be reclaimed without warning.
# So pull DURING the run, not after it: run_e3_topic.sh only suggests a tarball
# on its last line, three hours in.
#
# The remote `results/` tree lands in ./results-rental, kept clear of the local
# results/test and results/haiku-pilot trees. Read it with no GPU, no reader and
# no API key -- benchmark/report.py is pure stdlib over the JSONL on disk:
#
#   WICER_RESULTS_DIR=results-rental uv run python benchmark/report.py \
#     --profile paper --topic local_economy
#
# Knobs: REMOTE_DIR (default /workspace/WiCER), DEST (results-rental),
# INTERVAL (300), SSH_OPTS (extra ssh flags, e.g. -i ~/.ssh/vast).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WATCH=0
WITH_CACHE=0
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)      WATCH=1; shift ;;
    --with-cache) WITH_CACHE=1; shift ;;
    --interval)   INTERVAL="$2"; shift 2 ;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
    *)            TARGET="$1"; shift ;;
  esac
done

: "${TARGET:?usage: pull_results.sh [--watch] [--with-cache] [user@]host[:port]}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/WiCER}"
DEST="${DEST:-results-rental}"
INTERVAL="${INTERVAL:-300}"

# [user@]host[:port] -- providers hand out a nonstandard port far more often than not.
PORT=22
HOSTSPEC="$TARGET"
if [[ "$TARGET" == *:* ]]; then
  PORT="${TARGET##*:}"
  HOSTSPEC="${TARGET%:*}"
fi

# Multiplex the connection so --watch authenticates once instead of every tick.
SSH_E="ssh -p $PORT -o ControlMaster=auto -o ControlPath=/tmp/wicer-pull-%C -o ControlPersist=10m ${SSH_OPTS:-}"

die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# rsync has to exist on BOTH ends. Provider CUDA images frequently ship without
# it, so fall back to streaming a tarball -- the whole tree is a few MB per
# topic, which makes resending it every tick perfectly affordable.
RSYNC_OK=""
probe_rsync() {
  [[ -n "$RSYNC_OK" ]] && return 0
  if command -v rsync >/dev/null && $SSH_E "$HOSTSPEC" 'command -v rsync >/dev/null' 2>/dev/null; then
    RSYNC_OK=1
  else
    RSYNC_OK=0
    echo "rsync unavailable on one end -- streaming tar instead"
  fi
}

pull_dir() {
  local remote="$1" local_dest="$2"
  mkdir -p "$local_dest"
  if [[ "$RSYNC_OK" == "1" ]]; then
    rsync -az --partial -e "$SSH_E" "$HOSTSPEC:$remote/" "$local_dest/"
  else
    # -C into the directory so the stream carries paths relative to it.
    $SSH_E "$HOSTSPEC" "cd '$remote' 2>/dev/null && tar czf - ." | tar xzf - -C "$local_dest"
  fi
}

pull() {
  probe_rsync
  pull_dir "$REMOTE_DIR/results" "$DEST"
  # Content-addressed by prompt hash, so merging the remote cache into the local
  # one is safe and makes a local re-judge free instead of ~$7. Optional extra:
  # early in a run nothing has been judged yet and the directory does not exist,
  # which must not take the results pull down with it.
  if [[ "$WITH_CACHE" == "1" ]]; then
    pull_dir "$REMOTE_DIR/.cache/judge" ".cache/judge" \
      || echo "judge cache not on the box yet -- skipped" >&2
  fi
  return 0
}

summarize() {
  local n sz
  n=$(find "$DEST" -type f 2>/dev/null | wc -l | tr -d ' ')
  sz=$(du -sh "$DEST" 2>/dev/null | cut -f1)
  printf '%s  %s files, %s in %s\n' "$(date +%H:%M:%S)" "$n" "${sz:-0}" "$DEST"
}

if [[ "$WATCH" == "1" ]]; then
  echo "pulling $HOSTSPEC:$REMOTE_DIR/results -> $DEST every ${INTERVAL}s (Ctrl-C to stop)"
  while :; do
    pull || echo "pull failed -- retrying in ${INTERVAL}s" >&2
    summarize
    sleep "$INTERVAL"
  done
fi

pull
summarize

if find "$DEST" -name 'scores.jsonl' -print -quit 2>/dev/null | grep -q .; then
  cat <<NEXT

Scored questions present. Regenerate the paper tables locally:

  WICER_RESULTS_DIR=$DEST uv run python benchmark/report.py --profile paper

Then check, in order: the Kept verbatim column, Recovery against the paper's
82% for local_economy, measured_cost_usd in */run_meta.json, and a nonzero
cache_read_tokens.
NEXT
else
  echo
  echo "No scores.jsonl yet -- the run has not reached benchmark/evaluate.py."
fi
