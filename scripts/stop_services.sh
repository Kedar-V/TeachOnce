#!/usr/bin/env bash
# Stop vLLM + Gym started by start_*.sh. Run on Brev VM.
set -euo pipefail
HOME_TO="${HOME}/teachonce"
for name in vllm gym; do
  pidfile="$HOME_TO/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      # vLLM may leave children
      pkill -P "$pid" 2>/dev/null || true
      echo "stopped $name pid=$pid"
    else
      echo "$name pidfile stale"
    fi
    rm -f "$pidfile"
  else
    echo "no $name pidfile"
  fi
done
# belt-and-suspenders
pkill -f 'vllm serve' 2>/dev/null || true
pkill -f 'gym env start' 2>/dev/null || true
echo "done"
