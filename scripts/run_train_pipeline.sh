#!/usr/bin/env bash
# TeachOnce train pipeline with granular progress + checkpointing.
# Phases: path_gt → sft → (optional) post_eval
set -euo pipefail

ROOT="${ROOT:-$HOME/teachonce}"
PROGRESS="${TRAIN_PROGRESS:-$ROOT/checkpoints/train_pipeline_progress.json}"
GT="${SFT_GT:-$ROOT/data/generated/nemo_rl/gt.jsonl}"
CKPT="${SFT_CKPT:-$ROOT/checkpoints/sft_nemotron_vl}"
VENV="${NEMOTRON_VL_VENV:-$ROOT/vllm-env}"
PY="${VENV}/bin/python"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR" "$(dirname "$PROGRESS")" "$CKPT"

write_progress() {
  # usage: write_progress <phase> <status> [key=value ...]
  local phase="$1" status="$2"
  shift 2 || true
  EXTRA_ARGS=("$@")
  PHASE="$phase" STATUS="$status" PROG="$PROGRESS" ROOT="$ROOT" \
  EXTRA_KV="${EXTRA_ARGS[*]-}" "$PY" - <<'PY'
import json, os, time
from pathlib import Path
p = Path(os.environ["PROG"])
cur = {}
if p.exists():
    try:
        cur = json.loads(p.read_text())
    except Exception:
        cur = {}
cur.update({
    "phase": os.environ["PHASE"],
    "status": os.environ["STATUS"],
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
})
for tok in os.environ.get("EXTRA_KV", "").split():
    if "=" in tok:
        k, v = tok.split("=", 1)
        cur[k] = v
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(cur, indent=2) + "\n")
print(json.dumps(cur, indent=2))
PY
}

echo "== pipeline root=$ROOT =="
write_progress path_gt running gt="$GT"

# 1) Path-consistent GT (filmed tip only)
# Prefer system python for GT rewrite (no torch needed)
GT_PY="$PY"
command -v python3 >/dev/null && GT_PY=python3
"$GT_PY" -u "$ROOT/scripts/nemo_rl_gt.py" --rewrite-path-consistent "$GT" | tee "$LOG_DIR/path_gt.log"
write_progress path_gt done gt="$GT"

# 2) Free GPU from vLLM if present (SFT needs the card)
pkill -f '[v]llm serve' 2>/dev/null || true
sleep 2

# 3) SFT with step checkpoints + progress.json
if [[ -f "$ROOT/.env" ]]; then set -a; # shellcheck disable=SC1091
  source "$ROOT/.env"; set +a; fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "warn: WANDB_API_KEY unset → offline wandb; progress.json still tracks training"
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi
export SFT_GT="$GT"
export SFT_CKPT="$CKPT"
export SFT_PROGRESS="${SFT_PROGRESS:-$CKPT/progress.json}"
export SFT_CKPT_EVERY="${SFT_CKPT_EVERY:-10}"
export SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-10}"
export SFT_RESUME="${SFT_RESUME:-1}"
export SFT_EPOCHS="${SFT_EPOCHS:-3}"
export SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-25}"
write_progress sft running ckpt="$CKPT" sft_progress="$SFT_PROGRESS"

bash "$ROOT/scripts/run_sft_nemotron_vl.sh" 2>&1 | tee "$LOG_DIR/sft_pipeline.log"
write_progress sft done ckpt="$CKPT"

echo "== pipeline complete; progress at $PROGRESS =="
cat "$PROGRESS"
[[ -f "$SFT_PROGRESS" ]] && { echo "== sft progress =="; cat "$SFT_PROGRESS"; }
