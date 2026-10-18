#!/usr/bin/env bash
# Smoke: NeMo Gym ↔ local vLLM serving Nemotron VL (no third-party LLM).
# Prereq: Nemotron VL already up on :8000 (vllm-nemotron-env).
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
TO="${HOME}/teachonce"
cd "$TO"
# shellcheck disable=SC1091
source "$TO/.env" 2>/dev/null || true
source "$TO/gym-venv/bin/activate"

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16}"
KEY="${LOCAL_VLLM_API_KEY:-teachonce-local-key}"
BASE="${NVIDIA_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LIMIT="${GYM_SMOKE_LIMIT:-3}"

# NVIDIA-only Gym policy
cat > "$TO/env.yaml" <<EOF
policy_base_url: ${BASE}
policy_api_key: ${KEY}
policy_model_name: ${MODEL_ID}
EOF

echo "== models =="
curl -sf "$BASE/models" -H "Authorization: Bearer $KEY" | head -c 500
echo

# chat ping (text; VL accepts chat completions)
echo "== chat ping =="
curl -sf "$BASE/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":8}" 
echo

# stop stale gym
pkill -f 'gym env start' 2>/dev/null || true
sleep 2

echo "== gym env start (mcqa + vllm_model → Nemotron VL) =="
nohup env PATH="$HOME/.local/bin:$PATH" gym env start \
  --resources-server mcqa \
  --model-type vllm_model \
  --model-url "$BASE" \
  --model "$MODEL_ID" \
  --model-api-key "$KEY" \
  >"$TO/gym.log" 2>&1 &
echo $! >"$TO/gym.pid"
sleep 25
tail -30 "$TO/gym.log"

INPUT="$TO/gym-venv/lib/python3.13/site-packages/resources_servers/mcqa/data/example.jsonl"
mkdir -p "$TO/results"
OUT="$TO/results/nemotron_vl_mcqa_rollouts.jsonl"

echo "== gym eval run --limit $LIMIT =="
gym eval run --no-serve \
  --agent mcqa_simple_agent \
  --input "$INPUT" \
  --output "$OUT" \
  --limit "$LIMIT" \
  --num-repeats 1

echo "== metrics =="
ls -la "$TO/results"/nemotron_vl_mcqa* 2>/dev/null || ls -la "$TO/results"/mcqa* 2>/dev/null | tail
cat "$TO/results/nemotron_vl_mcqa_rollouts_aggregate_metrics.json" 2>/dev/null \
  || cat "$TO/results/"*"aggregate_metrics.json" 2>/dev/null | head -80
echo
echo "GYM_NEMOTRON_VL_SMOKE_OK"
