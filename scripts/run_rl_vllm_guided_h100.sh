#!/usr/bin/env bash
# Simple+fast guided RL on teachonce-rl (H100): vLLM guided rollouts → RFT LoRA.
# Kill free-generate GRPO first. Schema from baseline guided_schema.
set -euo pipefail

: "${WANDB_API_KEY:?export WANDB_API_KEY}"

ROOT="${ROOT:-$HOME/teachonce}"
PY="${RL_PY:-$ROOT/nemotron-vl-venv/bin/python}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export NVIDIA_VLLM_BASE_URL="${NVIDIA_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export LOCAL_VLLM_API_KEY="${LOCAL_VLLM_API_KEY:-teachonce-local-key}"
export RL_MODEL="${RL_MODEL:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16}"
export RL_GT="${RL_GT:-$ROOT/data/generated/nemo_rl/gt_train_balanced50.jsonl}"
export RL_HOLDOUT="${RL_HOLDOUT:-$ROOT/data/generated/nemo_rl/gt_holdout_balanced24.jsonl}"
export RL_CKPT="${RL_CKPT:-/ephemeral/teachonce/checkpoints/rl_vllm_guided_rft}"
export RL_GROUP="${RL_GROUP:-4}"
export RL_MAX_NEW="${RL_MAX_NEW:-512}"
export RL_ROUNDS="${RL_ROUNDS:-1}"
export RL_MAX_PROMPTS="${RL_MAX_PROMPTS:-50}"
export WANDB_PROJECT="${WANDB_PROJECT:-teachonce-nemo-rl}"
export WANDB_NAME="${WANDB_NAME:-rl-vllm-guided-rft-h100}"

mkdir -p "$ROOT/logs" "$RL_CKPT"
pkill -f grpo_dag_wandb.py || true
pkill -f rl_vllm_guided_rft.py || true

# start vLLM if down (reuse nvidia env if present)
if ! curl -sf -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" "$NVIDIA_VLLM_BASE_URL/models" >/dev/null 2>&1; then
  echo "starting vLLM..."
  VLLM_PY="${VLLM_PY:-$ROOT/nemotron-vl-venv/bin/python}"
  if [[ ! -x "$VLLM_PY" ]]; then VLLM_PY="$PY"; fi
  nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$RL_MODEL" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --api-key "$LOCAL_VLLM_API_KEY" \
    --served-model-name "$RL_MODEL" \
    >"$ROOT/logs/vllm_rl.log" 2>&1 &
  echo "vllm_pid=$!"
  for i in $(seq 1 90); do
    if curl -sf -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" "$NVIDIA_VLLM_BASE_URL/models" >/dev/null 2>&1; then
      echo "vLLM ready"
      break
    fi
    sleep 10
    echo "wait_vllm $i"
  done
fi

# rollout only while vLLM holds GPU
export RL_PHASE=rollout
"$PY" -u "$ROOT/scripts/rl_vllm_guided_rft.py" 2>&1 | tee "$ROOT/logs/rl_vllm_guided_rollout.log"

# free GPU for HF train
pkill -f 'vllm.entrypoints.openai.api_server' || true
pkill -f 'vllm serve' || true
sleep 5

export RL_PHASE=train
"$PY" -u "$ROOT/scripts/rl_vllm_guided_rft.py" 2>&1 | tee "$ROOT/logs/rl_vllm_guided_train.log"

echo "DONE ckpt=$RL_CKPT"
