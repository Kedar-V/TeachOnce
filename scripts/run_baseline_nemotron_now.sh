#!/usr/bin/env bash
# Serve Nemotron VL on driver CUDA 12.7: vLLM 0.11.x + torch cu124 (not 0.31/cu130).
set -euo pipefail
ROOT="${ROOT:-$HOME/teachonce}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export VLLM_VIDEO_LOADER_BACKEND=opencv
LOG="$ROOT/logs/baseline_nemotron_now.log"
VENV="$ROOT/vllm-nemotron-env"
mkdir -p "$ROOT/logs" "$ROOT/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u) baseline start ==="
set -a
# shellcheck disable=SC1091
source "$ROOT/.env" 2>/dev/null || true
set +a
export MODEL_ID=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
export LOCAL_VLLM_API_KEY="${LOCAL_VLLM_API_KEY:-teachonce-local-key}"
VIDEO_ROOT="$ROOT/data/generated/demonstrations/videos"

need_install=1
if [[ -x "$VENV/bin/python" ]]; then
  if "$VENV/bin/python" - <<'PY' 2>/dev/null
import torch, vllm
from packaging.version import Version
assert torch.cuda.is_available()
assert str(torch.version.cuda).startswith("12")
assert Version(vllm.__version__.split("+")[0]) >= Version("0.11.1")
print("ok", torch.__version__, torch.version.cuda, vllm.__version__)
PY
  then need_install=0; fi
fi

if [[ "$need_install" -eq 1 ]]; then
  echo "Building venv with torch cu124 + vLLM 0.11.x"
  rm -rf "$VENV"
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" packaging
  # Force CUDA 12.4 wheels for driver 12.7
  uv pip install --python "$VENV/bin/python" \
    "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
  # Nemotron VL needs >=0.11.1; keep below 0.12 to stay on torch 2.6 if possible
  uv pip install --python "$VENV/bin/python" \
    "vllm==0.11.1" "opencv-python-headless" "requests" "tqdm" || \
  uv pip install --python "$VENV/bin/python" \
    --extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow \
    "vllm==0.11.1" opencv-python-headless requests tqdm
  # if vllm upgraded torch, force cu124 back
  "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    uv pip install --python "$VENV/bin/python" --reinstall \
      "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import torch,vllm; print('torch', torch.__version__, torch.version.cuda, 'cuda', torch.cuda.is_available()); print('vllm', vllm.__version__)"
python -c "import torch; assert torch.cuda.is_available(), 'need CUDA — wrong torch build'"

pkill -f "vllm serve" 2>/dev/null || true
pkill -f baseline_vllm_dag 2>/dev/null || true
sleep 2

nohup vllm serve "$MODEL_ID" \
  --host 127.0.0.1 --port 8000 \
  --api-key "$LOCAL_VLLM_API_KEY" \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --allowed-local-media-path "$VIDEO_ROOT" \
  --video-pruning-rate 0 \
  --served-model-name "$MODEL_ID" \
  >"$ROOT/vllm.log" 2>&1 &
echo "vLLM pid=$! model=$MODEL_ID"

for i in $(seq 1 180); do
  if curl -sf -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "vLLM ready after ${i}x5s"
    curl -s -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" http://127.0.0.1:8000/v1/models
    break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "vLLM died"; tail -80 "$ROOT/vllm.log"; exit 1
  fi
  sleep 5
done

python -u "$ROOT/scripts/baseline_vllm_dag.py" \
  --gt "$ROOT/data/generated/nemo_rl/gt.jsonl" \
  --video-root "$VIDEO_ROOT" \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key "$LOCAL_VLLM_API_KEY" \
  --model "$MODEL_ID" \
  --max-samples 25 \
  --out "$ROOT/results/baseline_vllm_dag.json"

python -c "import json; print(json.dumps(json.load(open('$ROOT/results/baseline_vllm_dag.json'))['summary'], indent=2))"
echo "=== $(date -u) DONE ==="
