#!/usr/bin/env bash
# Serve merged SFT weights via vLLM + guided_json (Nemotron VL has no vLLM LoRA).
set -euo pipefail
ROOT="${ROOT:-$HOME/teachonce}"
export PATH="${HOME}/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export VLLM_VIDEO_LOADER_BACKEND=opencv

MERGED="${SFT_MERGED:-/data/merged-nemotron-sft}"
# name must contain Nemotron — baseline_vllm_dag.py guards on that
MODEL_NAME="${SFT_SERVED_NAME:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16-SFT}"
BASE_ID="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
VENV="$ROOT/vllm-nemotron-env"
KEY="${LOCAL_VLLM_API_KEY:-teachonce-local-key}"
PORT="${VLLM_PORT:-8000}"
VIDEO_ROOT="${VIDEO_ROOT:-$ROOT/data/generated/demonstrations/videos}"
LOG="$ROOT/logs/vllm_sft_merged.log"
PIDFILE="$ROOT/vllm.pid"
MAX_SAMPLES="${EVAL_N:-${SFT_EVAL_N:-50}}"
EVAL_GT="${EVAL_GT:-$ROOT/data/generated/nemo_rl/gt_train_balanced100.jsonl}"
# omit EVAL_SEED for a new random holdout-50 each run; set it to reproduce
SEED_ARGS=()
if [[ -n "${EVAL_SEED:-}" ]]; then
  SEED_ARGS=(--seed "$EVAL_SEED")
fi

[[ -d "$MERGED" ]] || { echo "missing merged model $MERGED"; exit 1; }
[[ -x "$VENV/bin/vllm" ]] || { echo "missing $VENV"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pkill -f 'vllm serve' 2>/dev/null || true
sleep 3

nohup vllm serve "$MERGED" \
  --host 127.0.0.1 --port "$PORT" \
  --api-key "$KEY" \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --allowed-local-media-path "$VIDEO_ROOT" \
  --media-io-kwargs '{"video":{"num_frames":8}}' \
  --video-pruning-rate 0.5 \
  --served-model-name "$MODEL_NAME" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "vLLM merged SFT starting pid=$(cat "$PIDFILE") model=$MERGED as $MODEL_NAME"

for i in $(seq 1 150); do
  if curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:${PORT}/v1/models" | grep -q "$MODEL_NAME\|Nemotron\|merged"; then
    echo "READY after ${i}x5s"
    curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:${PORT}/v1/models" | head -c 500
    echo
    break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "DIED"; tail -80 "$LOG"; exit 1
  fi
  sleep 5
done

"$VENV/bin/python" -u "$ROOT/scripts/baseline_vllm_dag.py" \
  --model "$MODEL_NAME" \
  --gt "$EVAL_GT" \
  --video-root "$VIDEO_ROOT" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-key "$KEY" \
  --max-samples "$MAX_SAMPLES" \
  "${SEED_ARGS[@]}" \
  --out "$ROOT/results/sft_lora_vllm_dag.json"

echo "=== BASELINE ($BASE_ID) ==="
"$VENV/bin/python" -c "import json;print(json.dumps(json.load(open('$ROOT/results/baseline_vllm_dag.json'))['summary'],indent=2))"
echo "=== SFT merged (guided_json) ==="
"$VENV/bin/python" -c "import json;print(json.dumps(json.load(open('$ROOT/results/sft_lora_vllm_dag.json'))['summary'],indent=2))"
