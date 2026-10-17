#!/usr/bin/env bash
# Start stock vLLM (OpenAI-compatible) on loopback. Run on the Brev VM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
[[ -f "$HOME/teachonce/.env" ]] && source "$HOME/teachonce/.env"

: "${LOCAL_VLLM_API_KEY:?set LOCAL_VLLM_API_KEY}"
: "${MODEL_ID:?set MODEL_ID}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_VENV="${VLLM_VENV:-$HOME/teachonce/vllm-env}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
PIDFILE="${PIDFILE:-$HOME/teachonce/vllm.pid}"
LOGFILE="${LOGFILE:-$HOME/teachonce/vllm.log}"

# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"

mkdir -p "$(dirname "$PIDFILE")"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "vLLM already running pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup vllm serve "$MODEL_ID" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --api-key "$LOCAL_VLLM_API_KEY" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"
echo "vLLM starting pid=$(cat "$PIDFILE") log=$LOGFILE"
echo "endpoint=http://${VLLM_HOST}:${VLLM_PORT}/v1"
