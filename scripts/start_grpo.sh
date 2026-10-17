#!/usr/bin/env bash
set -euo pipefail
chmod 600 "$HOME/teachonce/.wandb.env"
chmod +x "$HOME/teachonce/scripts/run_grpo_smoke_wandb.sh"
mkdir -p "$HOME/teachonce/logs"
# kill prior python train only (not this launcher)
pkill -f 'python -u .*/grpo_smoke_wandb.py' 2>/dev/null || true
set -a
# shellcheck disable=SC1091
source "$HOME/teachonce/.wandb.env"
set +a
nohup bash "$HOME/teachonce/scripts/run_grpo_smoke_wandb.sh" \
  >"$HOME/teachonce/logs/grpo_launch.out" 2>&1 &
echo "STARTED:$!"
sleep 8
wc -l "$HOME/teachonce/logs/grpo_launch.out" || true
tail -40 "$HOME/teachonce/logs/grpo_launch.out" || true
pgrep -af 'grpo_smoke_wandb.py' | head || echo "no python yet"
