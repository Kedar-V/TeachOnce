#!/usr/bin/env bash
# Serve merged SFT Nemotron VL via vLLM (Red Hat AI Inference Server pattern:
# OpenAI-compatible + guided_json). For app / Mac tunnel on teachonce-gpu.
set -euo pipefail
ROOT="${ROOT:-$HOME/teachonce}"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
[[ -f "$HOME/teachonce/.env" ]] && source "$HOME/teachonce/.env"

export PATH="${HOME}/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export VLLM_VIDEO_LOADER_BACKEND=opencv

# Key from .env; SFT serve defaults win over Phase-0 Qwen .env knobs.
LOCAL_VLLM_API_KEY="${LOCAL_VLLM_API_KEY:-teachonce-local-key}"
MERGED="${SFT_MERGED:-/data/merged-nemotron-sft}"
MODEL_NAME="${SFT_SERVED_NAME:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16-SFT}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_VENV="${SFT_VLLM_VENV:-$ROOT/vllm-nemotron-env}"
VLLM_GPU_MEMORY_UTILIZATION="${SFT_GPU_MEM:-0.90}"
VLLM_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-32768}"
VIDEO_ROOT="${VIDEO_ROOT:-$ROOT/data/generated/demonstrations/videos}"
PIDFILE="${PIDFILE:-$ROOT/vllm.pid}"
LOGFILE="${LOGFILE:-$ROOT/logs/vllm_sft_merged.log}"

[[ -d "$MERGED" ]] || { echo "missing merged model $MERGED"; exit 1; }
[[ -x "$VLLM_VENV/bin/vllm" ]] || { echo "missing $VLLM_VENV"; exit 1; }
# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"

mkdir -p "$(dirname "$LOGFILE")" "$(dirname "$PIDFILE")"

# Stop prior serve — kill EngineCore + api server by PID (avoid `pkill -f`, kills brev SSH).
stop_vllm() {
  local p cmd
  for p in $(pgrep -f '[v]llm serve' || true); do
    kill -9 "$p" 2>/dev/null || true
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
    cmd="$(ps -p "$p" -o args= 2>/dev/null || true)"
    case "$cmd" in
      *vllm*|*VLLM*|*EngineCore*) kill -9 "$p" 2>/dev/null || true ;;
    esac
  done
  sleep 4
}
stop_vllm
# Port must be free before bind
if ss -ltn 2>/dev/null | grep -q ":${VLLM_PORT} "; then
  echo "port ${VLLM_PORT} still busy; force stop again"
  stop_vllm
fi

nohup vllm serve "$MERGED" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --api-key "$LOCAL_VLLM_API_KEY" \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"images":0,"videos":1}' \
  --allowed-local-media-path "$VIDEO_ROOT" \
  --media-io-kwargs '{"video":{"num_frames":8}}' \
  --video-pruning-rate "${VIDEO_PRUNING:-0.5}" \
  --guided-decoding-backend xgrammar \
  --served-model-name "$MODEL_NAME" \
  >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"

echo "vLLM SFT starting pid=$(cat "$PIDFILE") model=$MERGED as $MODEL_NAME"
echo "log=$LOGFILE"
echo "endpoint=http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo "app: NVIDIA_VLLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1 MODEL_ID=$MODEL_NAME"
echo "     LOCAL_VLLM_API_KEY=$LOCAL_VLLM_API_KEY"
echo "Mac: ./scripts/tunnel_vllm.sh"

READY=0
for i in $(seq 1 120); do
  body="$(curl -sf -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" \
    "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" || true)"
  # Must match SFT served name — do not accept leftover base Nemotron.
  if echo "$body" | grep -q "$MODEL_NAME"; then
    echo "READY after $((i * 5))s"
    echo "$body"
    READY=1
    break
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "DIED"; tail -80 "$LOGFILE"; exit 1
  fi
  sleep 5
done
[[ "$READY" -eq 1 ]] || { echo "TIMEOUT"; tail -80 "$LOGFILE"; exit 1; }
