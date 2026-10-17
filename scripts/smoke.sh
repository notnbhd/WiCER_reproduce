#!/usr/bin/env bash
# End-to-end pipeline check with no GPU, no API key and no downloads.
#
# Boots the fake llama-server, then runs every condition from spec.md Section 6
# against the synthetic 6-document fixture: full-context, RAG, the three blind
# compression levels, the WiCER loop and the random-pinning control. Finishes
# by rendering the report.
#
# This is the acceptance gate for the code itself. It says nothing about the
# paper's numbers -- for those, see README.md 'Running the real experiment'.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

export WICER_PROFILE=test
export WICER_JUDGE_BACKEND=fake
export WICER_COMPILER_BACKEND=fake
export WICER_EMBED_BACKEND=fake
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TOPIC=mini_topic
PORT="$("$PY" -c "import sys;sys.path.insert(0,'.');from wicer_core.config import get_profile;print(get_profile('test').reader.port)")"

rm -rf results/test .cache/judge-test
mkdir -p results/test

"$PY" tests/fake_llama_server.py --port "$PORT" --n-ctx 8192 >results/test/fake-server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 100); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 0.1
done
echo "fake llama-server up on :$PORT (pid $SERVER_PID)"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "probe selection (one QA pair per source document)"
"$PY" wicer/probe.py --topic "$TOPIC"

step "full-context baseline"
"$PY" benchmark/run_benchmark.py --topic "$TOPIC" --condition fc_raw

step "RAG baseline"
"$PY" rag_benchmark/build_index.py --topic "$TOPIC"
"$PY" rag_benchmark/run_rag_benchmark.py --topic "$TOPIC"

step "blind compilation at three levels"
"$PY" wicer/compile.py --topic "$TOPIC"
for LEVEL in light moderate aggressive; do
  "$PY" benchmark/run_benchmark.py --topic "$TOPIC" --condition "wiki_$LEVEL" \
    --context-file "results/test/$TOPIC/wikis/blind_$LEVEL.md"
done

step "WiCER loop"
"$PY" wicer/loop.py --topic "$TOPIC" --eval-full

step "random-pinning ablation"
"$PY" wicer/random_pin.py --topic "$TOPIC" --eval-full

step "judging (fake judge)"
"$PY" benchmark/evaluate.py --topic "$TOPIC"

step "report"
"$PY" benchmark/report.py --topic "$TOPIC"

echo
echo "smoke run complete: results/test/report.md"
