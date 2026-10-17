#!/usr/bin/env bash
# Idempotent Phase 0 bootstrap. Run ON the Brev VM (brev shell / brev exec).
# Installs uv, vLLM venv, clones NeMo Gym, writes ~/teachonce/.env
set -euo pipefail

TO="$HOME/teachonce"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
LOCAL_VLLM_API_KEY="${LOCAL_VLLM_API_KEY:-teachonce-local-key}"
VLLM_PORT="${VLLM_PORT:-8000}"

mkdir -p "$TO"
cd "$TO"

# --- inspect dump ---
{
  echo "=== $(date -Is) Phase 0 inspect ==="
  nvidia-smi || true
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  nvcc --version 2>/dev/null || true
  cat /etc/os-release
  python3 --version
  df -h
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
} | tee "$TO/inspect.txt"

# --- uv ---
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version | tee -a "$TO/inspect.txt"

# --- vLLM venv (isolated) ---
if [[ ! -d "$TO/vllm-env" ]]; then
  uv venv "$TO/vllm-env"
fi
# shellcheck disable=SC1091
source "$TO/vllm-env/bin/activate"
# Driver reports CUDA 12.x — do not use cu129/auto (pulls libcudart.so.13)
# ponytail: pin cu124 until driver ≥ CUDA 13
uv pip install -U vllm hf_transfer --torch-backend=cu124
python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$TO/inspect.txt"
deactivate || true

# --- NeMo Gym (PyPI — Python 3.13.14 required) ---
# ponytail: PyPI over git clone; pin via nemo-gym== if installs drift
if [[ ! -d "$TO/gym-venv" ]]; then
  uv venv --python 3.13.14 "$TO/gym-venv"
fi
# shellcheck disable=SC1091
source "$TO/gym-venv/bin/activate"
uv pip install -U nemo-gym
GYM_VER="$(gym --version 2>&1 | head -5 | tr '\n' ' ')"
echo "Gym: $GYM_VER" | tee -a "$TO/inspect.txt"
gym --help >/dev/null
echo "gym CLI OK" | tee -a "$TO/inspect.txt"
deactivate || true

# --- env file ---
cat > "$TO/.env" <<EOF
LOCAL_VLLM_API_KEY=${LOCAL_VLLM_API_KEY}
NVIDIA_VLLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
MODEL_ID=${MODEL_ID}
VLLM_HOST=127.0.0.1
VLLM_PORT=${VLLM_PORT}
VLLM_GPU_MEMORY_UTILIZATION=0.6
VLLM_MAX_MODEL_LEN=4096
VLLM_VENV=${TO}/vllm-env
GYM_DIR=${TO}
GYM_VENV=${TO}/gym-venv
EOF

# NVIDIA-only Gym policy file (cwd for gym commands)
cat > "$TO/env.yaml" <<EOF
policy_base_url: http://127.0.0.1:${VLLM_PORT}/v1
policy_api_key: ${LOCAL_VLLM_API_KEY}
policy_model_name: ${MODEL_ID}
EOF

echo "bootstrap OK → $TO"
echo "next: start_vllm.sh ; start_gym.sh ; RUN_GYM=1 smoke_test.sh"
