#!/usr/bin/env bash
# Start NeMo Gym wired to local vLLM only (no third-party LLM). Run on Brev VM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
[[ -f "$HOME/teachonce/.env" ]] && source "$HOME/teachonce/.env"

: "${LOCAL_VLLM_API_KEY:?set LOCAL_VLLM_API_KEY}"
: "${MODEL_ID:?set MODEL_ID}"
NVIDIA_VLLM_BASE_URL="${NVIDIA_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
GYM_DIR="${GYM_DIR:-$HOME/teachonce}"
GYM_VENV="${GYM_VENV:-$GYM_DIR/gym-venv}"
RESOURCES_SERVER="${GYM_RESOURCES_SERVER:-mcqa}"
PIDFILE="${GYM_PIDFILE:-$HOME/teachonce/gym.pid}"
LOGFILE="${GYM_LOGFILE:-$HOME/teachonce/gym.log}"

cd "$GYM_DIR"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source "$GYM_VENV/bin/activate"

# NVIDIA-only: never point at api.openai.com
cat > "$GYM_DIR/env.yaml" <<EOF
policy_base_url: ${NVIDIA_VLLM_BASE_URL}
policy_api_key: ${LOCAL_VLLM_API_KEY}
policy_model_name: ${MODEL_ID}
EOF

mkdir -p "$(dirname "$PIDFILE")"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Gym already running pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup gym env start \
  --resources-server "$RESOURCES_SERVER" \
  --model-type vllm_model \
  --model-url "$NVIDIA_VLLM_BASE_URL" \
  --model "$MODEL_ID" \
  --model-api-key "$LOCAL_VLLM_API_KEY" \
  >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"
echo "Gym starting pid=$(cat "$PIDFILE") log=$LOGFILE env.yaml=$GYM_DIR/env.yaml"
