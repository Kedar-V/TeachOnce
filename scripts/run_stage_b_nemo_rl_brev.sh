#!/usr/bin/env bash
# Start Stage-B → NeMo-RL incremental processor on Brev (watch mode).
set -euo pipefail
ROOT="${ROOT:-$HOME/teachonce}"
# Stage-B landings live in the repo tree on Brev (synced continuously).
VIDEOS="${STAGE_B_VIDEOS:-$ROOT/data/generated/demonstrations/videos}"
OUT="${NEMO_RL_OUT:-$ROOT/data/generated/nemo_rl}"
INTERVAL="${NEMO_RL_INTERVAL:-30}"
LOGDIR="${ROOT}/logs"
PIDFILE="${OUT}/processor.pid"
LOGFILE="${LOGDIR}/stage_b_nemo_rl.log"

mkdir -p "$OUT" "$LOGDIR" "$VIDEOS"

export STAGE_B_VIDEOS="$VIDEOS"
export NEMO_RL_OUT="$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE") progress=$OUT/progress.json"
  exit 0
fi

nohup python3 -u scripts/process_stage_b_nemo_rl.py --watch --interval "$INTERVAL" \
  --videos "$VIDEOS" --out "$OUT" \
  >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "videos=$VIDEOS"
echo "out=$OUT"
echo "log=$LOGFILE"
echo "progress=$OUT/progress.json"
sleep 2
tail -n 40 "$LOGFILE" || true
