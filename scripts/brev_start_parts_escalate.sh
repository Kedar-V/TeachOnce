#!/usr/bin/env bash
# Start parts + escalate Stage-B gens in parallel (same host, two processes).
set -euo pipefail
TO="${TEACHONCE_ROOT:-$HOME/teachonce}"
VID="$TO/data/generated/demonstrations/videos"
LOG="$TO/logs"
mkdir -p "$LOG" "$VID"

# stop anything already running for these jobs
touch "$VID/.pause" || true
pkill -f "gen_stage_b_parallel.py --procedure parts" 2>/dev/null || true
pkill -f "gen_stage_b_parallel.py --procedure escalate" 2>/dev/null || true
sleep 2
rm -f "$VID/.pause"

cd "$TO"
nohup gen-venv/bin/python -u scripts/gen_stage_b_parallel.py \
  --procedure parts --count 228 --workers 6 \
  >>"$LOG/stage_b_parts.log" 2>&1 &
echo $! >"$LOG/stage_b_parts.pid"
echo "parts pid=$(cat "$LOG/stage_b_parts.pid")"

nohup gen-venv/bin/python -u scripts/gen_stage_b_parallel.py \
  --procedure escalate --count 228 --workers 6 \
  >>"$LOG/stage_b_escalate.log" 2>&1 &
echo $! >"$LOG/stage_b_escalate.pid"
echo "escalate pid=$(cat "$LOG/stage_b_escalate.pid")"

sleep 4
echo "--- procs ---"
pgrep -af "gen_stage_b_parallel.py" || true
echo "--- parts ---"
tail -8 "$LOG/stage_b_parts.log" || true
echo "--- escalate ---"
tail -8 "$LOG/stage_b_escalate.log" || true
echo "pause: touch $VID/.pause"
echo "resume: rm $VID/.pause && bash $TO/scripts/brev_start_parts_escalate.sh"
