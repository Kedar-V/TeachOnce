#!/usr/bin/env bash
# Nemotron Nano 12B VL SFT on real videos → gold DAG (+ W&B).
set -euo pipefail
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

ROOT="${ROOT:-$HOME/teachonce}"
VENV="${NEMOTRON_VL_VENV:-$ROOT/nemotron-vl-venv}"
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-teachonce-nemo-rl}"
export WANDB_NAME="${WANDB_NAME:-sft-nemotron-vl-50}"
export SFT_MODEL="${SFT_MODEL:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16}"
export SFT_EPOCHS="${SFT_EPOCHS:-3}"
export SFT_BATCH="${SFT_BATCH:-1}"
export SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
export SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-50}"
export SFT_N_FRAMES="${SFT_N_FRAMES:-8}"
export SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-10}"
export SFT_GT="${SFT_GT:-$ROOT/data/generated/nemo_rl/gt.jsonl}"
export SFT_VIDEO_ROOT="${SFT_VIDEO_ROOT:-$ROOT/data/generated/demonstrations/videos}"
export SFT_CKPT="${SFT_CKPT:-$ROOT/checkpoints/sft_nemotron_vl}"
export ROOT

mkdir -p "$ROOT/logs" "$SFT_CKPT" "$HF_HOME"
pkill -f '[p]ython -u .*sft_dag_wandb' 2>/dev/null || true
pkill -f '[p]ython -u .*sft_nemotron_vl' 2>/dev/null || true
pkill -f '[p]ython -u .*grpo_dag' 2>/dev/null || true
pkill -f 'vllm serve' 2>/dev/null || true

need_deps=1
if [[ -x "$VENV/bin/python" ]]; then
  if "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available() and torch.version.cuda.startswith("12")' 2>/dev/null; then
    need_deps=0
  else
    rm -rf "$VENV"
  fi
fi
if [[ "$need_deps" -eq 1 ]]; then
  uv venv "$VENV" --python 3.11
  # Match driver: cu128 on H100 (CUDA≥12.8), else cu124 (teachonce-gpu 12.7)
  CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  major="${CUDA_VER%%.*}"; minor="${CUDA_VER#*.}"; minor="${minor%%.*}"
  if [[ "${major:-0}" -gt 12 ]] || { [[ "${major:-0}" -eq 12 ]] && [[ "${minor:-0}" -ge 8 ]]; }; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_PKGS=("torch==2.7.1" "torchvision==0.22.1")
  else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_PKGS=("torch==2.6.0" "torchvision==0.21.0")
  fi
  uv pip install --python "$VENV/bin/python" "${TORCH_PKGS[@]}" --index-url "$TORCH_INDEX"
  uv pip install --python "$VENV/bin/python" \
    "transformers>=4.53,<4.54" "accelerate" "peft" "wandb" "tqdm" "sentencepiece" \
    "timm" "open_clip_torch" "numpy" "pillow" "opencv-python-headless" "einops"
  # optional; skip if toolchain mismatches
  uv pip install --python "$VENV/bin/python" "causal-conv1d" "mamba-ssm==2.2.5" || true
fi

exec "$VENV/bin/python" -u "$ROOT/scripts/sft_nemotron_vl_wandb.py" 2>&1 | tee "$ROOT/logs/sft_nemotron_vl.log"
