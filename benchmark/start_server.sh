#!/usr/bin/env bash
# Launch llama-server for the reader under test.
#
# Paper Section 3.3: Llama 3.1 8B Instruct Q5_K_M, 96K context, q8_0 KV cache,
# Flash Attention, greedy decoding. Server flags come from the run profile so
# the benchmark and the server can never disagree about n_ctx or KV type.
#
#   benchmark/start_server.sh                      # profile from $WICER_PROFILE, default 'paper'
#   benchmark/start_server.sh --profile smoke
#   benchmark/start_server.sh --kv q4_0            # quantisation ablation arm
#   benchmark/start_server.sh --foreground
#
# spec.md Section 2: q8_0 is the headline config. Run q4_0 only as the ablation
# -- the paper found Q4 KV degrades quality at 80 documents (Q8 wins 13/14).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${WICER_PROFILE:-paper}"
KV_OVERRIDE=""
FOREGROUND=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --kv)      KV_OVERRIDE="$2"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

# Pull the server-relevant knobs out of the profile.
eval "$("$PY" - "$PROFILE" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from wicer_core.config import get_profile
r = get_profile(sys.argv[1]).reader
print(f'MODEL_PATH={r.model_path!r}')
print(f'HOST={r.host!r}')
print(f'PORT={r.port}')
print(f'N_CTX={r.n_ctx}')
print(f'CACHE_K={r.cache_type_k!r}')
print(f'CACHE_V={r.cache_type_v!r}')
print(f'NGL={r.n_gpu_layers}')
print(f'FLASH_ATTN={int(r.flash_attn)}')
PYEOF
)"

if [[ -n "$KV_OVERRIDE" ]]; then
  CACHE_K="$KV_OVERRIDE"
  CACHE_V="$KV_OVERRIDE"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  echo "Download a GGUF and point WICER_MODEL_PATH at it, e.g.:" >&2
  echo "  huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\" >&2
  echo "    Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf --local-dir models/" >&2
  exit 1
fi

command -v llama-server >/dev/null 2>&1 || {
  echo "llama-server not on PATH. Build llama.cpp with CUDA:" >&2
  echo "  cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server" >&2
  exit 1
}

ARGS=(
  --model "$MODEL_PATH"
  --ctx-size "$N_CTX"
  --cache-type-k "$CACHE_K"
  --cache-type-v "$CACHE_V"
  --n-gpu-layers "$NGL"
  --host "$HOST"
  --port "$PORT"
  # One slot: concurrent requests land in different slots and defeat the
  # prefix KV reuse the entire experiment is about.
  --parallel 1
  --cont-batching
  --metrics
)
[[ "$FLASH_ATTN" == "1" ]] && ARGS+=(--flash-attn on)
[[ ${#EXTRA[@]} -gt 0 ]] && ARGS+=("${EXTRA[@]}")

echo "profile=$PROFILE  model=$MODEL_PATH  n_ctx=$N_CTX  kv=$CACHE_K/$CACHE_V  ngl=$NGL"
echo "llama-server ${ARGS[*]}"

LOG_DIR="$REPO_ROOT/results/server"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/llama-server-$PROFILE.log"

if [[ "$FOREGROUND" == "1" ]]; then
  exec llama-server "${ARGS[@]}"
fi

llama-server "${ARGS[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$LOG_DIR/llama-server-$PROFILE.pid"
echo "started pid=$SERVER_PID  log=$LOG"

printf 'waiting for /health'
for _ in $(seq 1 600); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo " ok"
    exit 0
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "server died; see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
  printf '.'
  sleep 1
done
echo; echo "server did not become healthy; see $LOG" >&2
exit 1
