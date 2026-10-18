#!/usr/bin/env bash
# Serve Nemotron Nano 12B VL via vLLM (Red Hat AI Inference Server pattern:
# OpenAI-compatible + guided_json structured outputs). Run on Brev.
set -euo pipefail
ROOT="${ROOT:-$HOME/teachonce}"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
[[ -f "$HOME/teachonce/.env" ]] && source "$HOME/teachonce/.env"

: "${LOCAL_VLLM_API_KEY:?set LOCAL_VLLM_API_KEY}"
# Prefer caller-exported MODEL_ID; ignore Phase-0 Qwen default in .env
if [[ -z "${MODEL_ID:-}" || "$MODEL_ID" == *"Qwen"* ]]; then
  MODEL_ID="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
fi
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_VENV="${VLLM_VENV:-$HOME/teachonce/vllm-env}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
PIDFILE="${PIDFILE:-$HOME/teachonce/vllm.pid}"
LOGFILE="${LOGFILE:-$HOME/teachonce/vllm.log}"
VIDEO_ROOT="${VIDEO_ROOT:-$ROOT/data/generated/demonstrations/videos}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"

# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"

pkill -f 'vllm serve' 2>/dev/null || true
pkill -f '[p]ython -u .*sft_nemotron' 2>/dev/null || true
sleep 2

mkdir -p "$(dirname "$PIDFILE")"
# Same checkpoint we fine-tune; video inputs via video_url + local media path
nohup vllm serve "$MODEL_ID" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --api-key "$LOCAL_VLLM_API_KEY" \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"images":0,"videos":1}' \
  --allowed-local-media-path "$VIDEO_ROOT" \
  --guided-decoding-backend xgrammar \
  ${VIDEO_PRUNING:+--video-pruning-rate "$VIDEO_PRUNING"} \
  >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"
echo "vLLM Nemotron starting pid=$(cat "$PIDFILE") model=$MODEL_ID log=$LOGFILE"
echo "endpoint=http://${VLLM_HOST}:${VLLM_PORT}/v1"
