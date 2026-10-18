#!/usr/bin/env bash
# Mac → Brev: forward local 8000 to VM loopback vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"

: "${BREV_INSTANCE_NAME:?set BREV_INSTANCE_NAME in .env}"
# Remote vLLM is always :8000 on the VM. Local port defaults to 8101 so it
# doesn't collide with teachonce-web API (:8000) or mock model (:8100).
REMOTE_PORT="${VLLM_PORT:-8000}"
LOCAL_PORT="${VLLM_LOCAL_PORT:-8101}"

echo "Forwarding localhost:${LOCAL_PORT} → ${BREV_INSTANCE_NAME}:${REMOTE_PORT}"
echo "App/vLLM: NVIDIA_VLLM_BASE_URL=http://127.0.0.1:${LOCAL_PORT}/v1"
echo "Then: curl http://127.0.0.1:${LOCAL_PORT}/v1/models -H \"Authorization: Bearer \$LOCAL_VLLM_API_KEY\""
exec brev port-forward "$BREV_INSTANCE_NAME" -p "${LOCAL_PORT}:${REMOTE_PORT}"
