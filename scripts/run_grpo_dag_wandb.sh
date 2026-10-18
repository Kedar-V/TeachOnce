#!/usr/bin/env bash
# GRPO on 25 NeMo-RL GT rows (video→DAG reward) + W&B. Driver-compatible HF stack.
set -euo pipefail

: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

ROOT="${ROOT:-$HOME/teachonce}"
VENV="${GRPO_VENV:-$ROOT/grpo-venv}"
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-teachonce-nemo-rl}"
export WANDB_NAME="${WANDB_NAME:-grpo-dag-fast50}"
export GRPO_MODEL="${GRPO_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export GRPO_STEPS="${GRPO_STEPS:-25}"
export GRPO_GROUP="${GRPO_GROUP:-4}"
export GRPO_BATCH="${GRPO_BATCH:-4}"
export GRPO_MAX_NEW="${GRPO_MAX_NEW:-256}"
export GRPO_MAX_SAMPLES="${GRPO_MAX_SAMPLES:-50}"
export GRPO_GT="${GRPO_GT:-$ROOT/data/generated/nemo_rl/gt.jsonl}"
export GRPO_CKPT="${GRPO_CKPT:-$ROOT/checkpoints/grpo_dag}"
export GRPO_LORA="${GRPO_LORA:-1}"

mkdir -p "$ROOT" "$HF_HOME" "$ROOT/logs" "$GRPO_CKPT"
pkill -f 'vllm serve' 2>/dev/null || true
pkill -f '[p]ython -u .*grpo_dag_wandb\\.py' 2>/dev/null || true

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.11
  uv pip install --python "$VENV/bin/python" \
    "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "$VENV/bin/python" \
    "transformers==4.57.6" "datasets" "accelerate" "wandb" "sentencepiece" "peft" "tqdm"
fi
# peft may be missing on older venv
"$VENV/bin/python" -c "import peft" 2>/dev/null || \
  uv pip install --python "$VENV/bin/python" peft

SCRIPT="${GRPO_SCRIPT:-$ROOT/scripts/grpo_dag_wandb.py}"
exec "$VENV/bin/python" -u "$SCRIPT" 2>&1 | tee "$ROOT/logs/grpo_dag_wandb.log"
