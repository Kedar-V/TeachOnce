#!/usr/bin/env bash
# GRPO on train-N GT (text+events proxy → dag_reward) + W&B.
# Holdout eval: set GRPO_HOLDOUT to the same 25 as baseline_vllm_dag.
set -euo pipefail

: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

ROOT="${ROOT:-$HOME/teachonce}"
VENV="${GRPO_VENV:-$ROOT/grpo-venv}"
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-teachonce-nemo-rl}"
export WANDB_NAME="${WANDB_NAME:-grpo-nemotron-train100}"
export GRPO_MODEL="${GRPO_MODEL:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16}"
export GRPO_STEPS="${GRPO_STEPS:-50}"
export GRPO_GROUP="${GRPO_GROUP:-4}"
export GRPO_BATCH="${GRPO_BATCH:-2}"
export GRPO_MAX_NEW="${GRPO_MAX_NEW:-768}"
export GRPO_MAX_SAMPLES="${GRPO_MAX_SAMPLES:-100}"
export GRPO_GT="${GRPO_GT:-$ROOT/data/generated/nemo_rl/gt_train_balanced100.jsonl}"
export GRPO_HOLDOUT="${GRPO_HOLDOUT:-$ROOT/data/generated/nemo_rl/gt_holdout_balanced24.jsonl}"
export GRPO_CKPT="${GRPO_CKPT:-$ROOT/checkpoints/grpo_dag_balanced100}"
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
"$VENV/bin/python" -c "import peft" 2>/dev/null || \
  uv pip install --python "$VENV/bin/python" peft

SCRIPT="${GRPO_SCRIPT:-$ROOT/scripts/grpo_dag_wandb.py}"
exec "$VENV/bin/python" -u "$SCRIPT" 2>&1 | tee "$ROOT/logs/grpo_dag_wandb.log"
