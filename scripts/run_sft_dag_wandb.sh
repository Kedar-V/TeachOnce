#!/usr/bin/env bash
# SFT LoRA on gold DAG JSON (schema learning) + W&B.
set -euo pipefail
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

ROOT="${ROOT:-$HOME/teachonce}"
VENV="${GRPO_VENV:-$ROOT/grpo-venv}"
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-teachonce-nemo-rl}"
export WANDB_NAME="${WANDB_NAME:-sft-dag-50}"
export SFT_MODEL="${SFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export SFT_STEPS="${SFT_EPOCHS:-5}"
export SFT_EPOCHS="${SFT_EPOCHS:-5}"
export SFT_BATCH="${SFT_BATCH:-8}"
export SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-1}"
export SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-50}"
export SFT_MAX_LEN="${SFT_MAX_LEN:-2048}"
export SFT_GT="${SFT_GT:-$ROOT/data/generated/nemo_rl/gt.jsonl}"
export SFT_CKPT="${SFT_CKPT:-$ROOT/checkpoints/sft_dag}"
export SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-10}"

mkdir -p "$ROOT/logs" "$SFT_CKPT" "$HF_HOME"
pkill -f '[p]ython -u .*grpo_dag_wandb' 2>/dev/null || true
pkill -f '[p]ython -u .*sft_dag_wandb' 2>/dev/null || true
pkill -f 'vllm serve' 2>/dev/null || true

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.11
  uv pip install --python "$VENV/bin/python" \
    "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "$VENV/bin/python" \
    "transformers==4.57.6" "datasets" "accelerate" "wandb" "sentencepiece" "peft" "tqdm"
fi
"$VENV/bin/python" -c "import peft" 2>/dev/null || \
  uv pip install --python "$VENV/bin/python" peft

exec "$VENV/bin/python" -u "$ROOT/scripts/sft_dag_wandb.py" 2>&1 | tee "$ROOT/logs/sft_dag_wandb.log"
