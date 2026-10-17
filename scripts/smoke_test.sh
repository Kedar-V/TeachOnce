#!/usr/bin/env bash
# curl vLLM + optional Gym eval --limit 5. Works on VM or Mac (via tunnel).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
[[ -f "$HOME/teachonce/.env" ]] && source "$HOME/teachonce/.env"

: "${LOCAL_VLLM_API_KEY:?set LOCAL_VLLM_API_KEY}"
: "${MODEL_ID:?set MODEL_ID}"
BASE="${NVIDIA_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
RUN_GYM="${RUN_GYM:-0}"

echo "== GET $BASE/models =="
curl -sf "$BASE/models" -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" | head -c 2000
echo

echo "== POST $BASE/chat/completions =="
curl -sf "$BASE/chat/completions" \
  -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":32}"
echo

if [[ "$RUN_GYM" == "1" ]]; then
  GYM_DIR="${GYM_DIR:-$HOME/teachonce}"
  GYM_VENV="${GYM_VENV:-$GYM_DIR/gym-venv}"
  cd "$GYM_DIR"
  # shellcheck disable=SC1091
  source "$GYM_VENV/bin/activate"
  mkdir -p results
  # exact mcqa path (do not rglob — nested .venvs can match other example.jsonl)
  INPUT="${GYM_EVAL_INPUT:-}"
  if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
    INPUT="$(python -c "from importlib.resources import files; print(files('resources_servers')/'mcqa'/'data'/'example.jsonl')" 2>/dev/null || true)"
  fi
  if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
    INPUT="$GYM_VENV/lib/python3.13/site-packages/resources_servers/mcqa/data/example.jsonl"
  fi
  OUT="${GYM_EVAL_OUTPUT:-results/mcqa_rollouts.jsonl}"
  echo "== gym eval run --limit 5 input=$INPUT =="
  gym eval run --no-serve \
    --agent mcqa_simple_agent \
    --input "$INPUT" \
    --output "$OUT" \
    --limit 5 \
    --num-repeats 1
  echo "results: $GYM_DIR/$OUT"
  ls -la results/mcqa_rollouts* 2>/dev/null || true
fi

echo "smoke OK"
