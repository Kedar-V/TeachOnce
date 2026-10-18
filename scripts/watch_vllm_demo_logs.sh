#!/usr/bin/env bash
# Isolated vLLM-only log for demo recording (no uvicorn noise).
# Usage:  ./scripts/watch_vllm_demo_logs.sh
set -euo pipefail
LOG="${VLLM_DEMO_LOG:-/tmp/flowstate-vllm-demo.log}"
touch "$LOG"
clear
printf '\033[92m=== Flowstate · live vLLM requests (isolated) ===\033[0m\n'
printf 'Log: %s\n\n' "$LOG"
# Reset for a clean take (optional: KEEP=1 to append)
if [[ "${KEEP:-}" != "1" ]]; then
  : > "$LOG"
  printf '(cleared — set KEEP=1 to preserve prior lines)\n\n'
fi
exec tail -n 0 -F "$LOG"
