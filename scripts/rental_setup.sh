#!/usr/bin/env bash
# Prepare a rented CUDA box for a WiCER run, then verify it can actually hold
# the paper profile BEFORE anything expensive starts.
#
#   bash scripts/rental_setup.sh            # full setup
#   bash scripts/rental_setup.sh --verify   # just the VRAM/flash-attn check
#
# Assumes a CUDA base image with the driver present (nvidia-smi works) and
# this repo already copied to the box.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 3090 = Ampere = compute capability 8.6.  Set CUDA_ARCH for a different card:
#   Volta V100 70, Ampere A100 80, Ada L40S/4090 89, Hopper H100 90, Blackwell 120
# A mismatch is not subtle: the sm_86 SASS will not match the device and
# compute_86 PTX only JITs forward to newer cards, never back, so a V100 dies on
# the first kernel launch with "no kernel image is available for execution on
# the device".  CMake takes a list if you want one binary for both: "70;86".
CUDA_ARCH="${CUDA_ARCH:-86}"
MODEL="models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
N_CTX="${N_CTX:-98304}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

verify() {
  say "Verifying the paper profile fits (q8_0 KV at ${N_CTX})"
  command -v llama-server >/dev/null || die "llama-server not on PATH"
  [[ -f "$MODEL" ]] || die "model missing: $MODEL"

  llama-server --model "$MODEL" --ctx-size "$N_CTX" \
    --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 99 \
    --flash-attn on --parallel 1 --port 8080 --host 127.0.0.1 \
    > /tmp/llama-verify.log 2>&1 &
  local pid=$!

  for _ in $(seq 1 180); do
    curl -sf localhost:8080/health >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { cat /tmp/llama-verify.log; die "server died -- see log above"; }
    sleep 2
  done
  curl -sf localhost:8080/health >/dev/null || { cat /tmp/llama-verify.log; die "server never became healthy"; }

  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "VRAM in use: ${used} MiB (expect ~12500 with q8_0; ~18500 means fp16 -- flash attention is off)"
  grep -iE "flash|cache type|KV self" /tmp/llama-verify.log | head -5 || true

  # One real generation, to confirm it decodes rather than just loads.
  curl -sf localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":8}' \
    | head -c 200; echo

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo "verification passed"
}

if [[ "${1:-}" == "--verify" ]]; then verify; exit 0; fi

say "GPU"
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- not a GPU box, or the driver is missing"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "Build dependencies"
if command -v apt-get >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq cmake build-essential git curl libcurl4-openssl-dev tmux rsync >/dev/null
fi

# Docker ENV is not inherited by sshd sessions, so /usr/local/cuda/bin is off
# PATH even on a -devel image that does have the toolkit.  nvcc then looks
# missing and the CUDA build fails at configure time on a box that is in fact
# fine -- put the path back before cmake goes looking.  A no-op when nvcc is
# already reachable or when there is no CUDA directory to find.
if ! command -v nvcc >/dev/null; then
  for d in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then
      export PATH="$d:$PATH"
      echo "added $d to PATH (sshd does not inherit the image's ENV)"
      break
    fi
  done
fi

say "llama.cpp (CUDA arch sm_${CUDA_ARCH})"
if ! command -v llama-server >/dev/null; then
  command -v nvcc >/dev/null || die "nvcc not found: use a CUDA -devel image (nvidia/cuda:12.4.1-devel-ubuntu22.04); the runtime and PyTorch images ship no toolkit"
  [[ -d llama.cpp ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release
  cmake --build llama.cpp/build -j"$(nproc)" --target llama-server
  install -m755 llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
fi
llama-server --version 2>&1 | head -2 || true

say "Python environment"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12
uv pip install -e '.[openrouter,dev]' -q

say "Reader model (~5.4 GB)"
[[ -f "$MODEL" ]] || HF_HUB_DISABLE_XET=1 uv run hf download \
  bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf --local-dir models/

say "Corpus"
uv run python -m wicer_core.corpus --profile paper

say "Credentials"
[[ -f .env ]] || cp .env.example .env 2>/dev/null || true
if ! grep -q '^OPENROUTER_API_KEY=.\+' .env 2>/dev/null; then
  echo "WARNING: OPENROUTER_API_KEY is empty in .env -- paste it before running"
fi

verify

cat <<'NEXT'

Setup complete. Next, inside tmux so an SSH drop does not kill the run:

  tmux new -s wicer
  bash scripts/run_e3_topic.sh local_economy

It starts llama-server itself, so there is no separate step for that. From your
laptop, pull the results down while it runs:

  scripts/pull_results.sh --watch [user@]host[:port]

NEXT
