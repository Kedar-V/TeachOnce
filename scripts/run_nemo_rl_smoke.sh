#!/usr/bin/env bash
# Full NeMo-RL GRPO (needs driver CUDA ≥12.8 / NGC). On teachonce-gpu (565/12.7)
# use scripts/run_grpo_smoke_wandb.sh instead.
set -euo pipefail
echo "Brev driver is CUDA 12.7; NeMo-RL v0.5+/main need cu128+. Use:" >&2
echo "  WANDB_API_KEY=... bash scripts/run_grpo_smoke_wandb.sh" >&2
exit 2
