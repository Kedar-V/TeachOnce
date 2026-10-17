#!/usr/bin/env bash
# Mac → Brev: forward local 8000 to VM loopback vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"

: "${BREV_INSTANCE_NAME:?set BREV_INSTANCE_NAME in .env}"
PORT="${VLLM_PORT:-8000}"

echo "Forwarding localhost:${PORT} → ${BREV_INSTANCE_NAME}:${PORT}"
echo "Then: curl http://127.0.0.1:${PORT}/v1/models -H \"Authorization: Bearer \$LOCAL_VLLM_API_KEY\""
exec brev port-forward "$BREV_INSTANCE_NAME" -p "${PORT}:${PORT}"
