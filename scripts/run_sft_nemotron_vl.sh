#!/usr/bin/env bash
# Nemotron Nano 12B VL SFT on real videos → gold DAG (+ W&B).
set -euo pipefail
ROOT="${ROOT:-$HOME/teachonce}"
if [[ -f "$ROOT/.env" ]]; then set -a; # shellcheck disable=SC1091
  source "$ROOT/.env"; set +a; fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
  echo "warn: WANDB_API_KEY unset → WANDB_MODE=$WANDB_MODE"
fi

# Prefer vllm-env: already has a working mamba-ssm build. Do NOT rebuild mamba.
VENV="${NEMOTRON_VL_VENV:-$ROOT/vllm-env}"
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
export SFT_CKPT_EVERY="${SFT_CKPT_EVERY:-10}"
export SFT_RESUME="${SFT_RESUME:-1}"
export SFT_GT="${SFT_GT:-$ROOT/data/generated/nemo_rl/gt.jsonl}"
export SFT_VIDEO_ROOT="${SFT_VIDEO_ROOT:-$ROOT/data/generated/demonstrations/videos}"
export SFT_CKPT="${SFT_CKPT:-$ROOT/checkpoints/sft_nemotron_vl}"
export SFT_PROGRESS="${SFT_PROGRESS:-$SFT_CKPT/progress.json}"
export ROOT

mkdir -p "$ROOT/logs" "$SFT_CKPT" "$HF_HOME"
echo "SFT venv → $VENV (reuse mamba; no rebuild)"
echo "SFT progress → $SFT_PROGRESS  ckpts → $SFT_CKPT (every ${SFT_CKPT_EVERY} steps)"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing $VENV — set NEMOTRON_VL_VENV to the env that already has mamba-ssm" >&2
  exit 1
fi
"$VENV/bin/python" -c "import mamba_ssm,torch; assert torch.cuda.is_available()" \
  || { echo "mamba/torch broken in $VENV" >&2; exit 1; }
# light SFT deps only — never force-build mamba here
uv pip install --python "$VENV/bin/python" \
  peft accelerate wandb tqdm pillow opencv-python-headless einops 2>/dev/null || true

exec "$VENV/bin/python" -u "$ROOT/scripts/sft_nemotron_vl_wandb.py" 2>&1 | tee "$ROOT/logs/sft_nemotron_vl.log"
