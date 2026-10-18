#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/teachonce}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export VLLM_VIDEO_LOADER_BACKEND=opencv
# shellcheck disable=SC1091
source "$ROOT/vllm-nemotron-env/bin/activate"
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
nohup vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
  --host 127.0.0.1 --port 8000 --api-key teachonce-local-key \
  --trust-remote-code --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --allowed-local-media-path "$ROOT/data/generated/demonstrations/videos" \
  --media-io-kwargs '{"video":{"num_frames":8}}' \
  --video-pruning-rate 0.5 \
  --served-model-name nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
  >"$ROOT/vllm.log" 2>&1 &
echo "vllm_pid=$!"
for i in $(seq 1 150); do
  if curl -sf -H "Authorization: Bearer teachonce-local-key" http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "READY"
    break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "DIED"; tail -50 "$ROOT/vllm.log"; exit 1
  fi
  sleep 5
done
EVAL_GT="${EVAL_GT:-$ROOT/data/generated/nemo_rl/gt_train_balanced100.jsonl}"
EVAL_N="${EVAL_N:-50}"
SEED_ARGS=()
[[ -n "${EVAL_SEED:-}" ]] && SEED_ARGS=(--seed "$EVAL_SEED")
nohup "$ROOT/vllm-nemotron-env/bin/python" -u "$ROOT/scripts/baseline_vllm_dag.py" \
  --gt "$EVAL_GT" \
  --video-root "$ROOT/data/generated/demonstrations/videos" \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key teachonce-local-key \
  --model nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
  --max-samples "$EVAL_N" \
  "${SEED_ARGS[@]}" \
  --out "$ROOT/results/baseline_vllm_dag.json" \
  >"$ROOT/logs/baseline_eval.out" 2>&1 &
echo "eval_pid=$!"
