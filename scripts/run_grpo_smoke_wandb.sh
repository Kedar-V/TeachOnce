#!/usr/bin/env bash
# Driver-compatible 10-step GRPO smoke + W&B (Brev CUDA 12.7).
# Full NeMo-RL needs newer driver — see run_nemo_rl_smoke.sh / docs/brev.md.
set -euo pipefail

: "${WANDB_API_KEY:?export WANDB_API_KEY first (https://wandb.ai/authorize)}"

ROOT="${ROOT:-$HOME/teachonce}"
VENV="${GRPO_VENV:-$ROOT/grpo-venv}"
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-teachonce-nemo-rl}"
export WANDB_NAME="${WANDB_NAME:-grpo-smoke-10}"
export GRPO_MODEL="${GRPO_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export GRPO_STEPS="${GRPO_STEPS:-10}"

mkdir -p "$ROOT" "$HF_HOME" "$ROOT/logs"
# free GPU; do NOT pkill this script's own name (matches run_grpo_smoke_wandb.sh)
pkill -f 'vllm serve' 2>/dev/null || true
pkill -f '[p]ython -u .*grpo_smoke_wandb\\.py' 2>/dev/null || true

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.11
  # match Phase-0 CUDA 12.x driver
  uv pip install --python "$VENV/bin/python" \
    "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "$VENV/bin/python" \
    "transformers==4.57.6" "datasets" "accelerate" "wandb" "sentencepiece"
fi

SCRIPT="${GRPO_SCRIPT:-$ROOT/scripts/grpo_smoke_wandb.py}"
exec "$VENV/bin/python" -u "$SCRIPT" 2>&1 | tee "$ROOT/logs/grpo_smoke_wandb.log"
