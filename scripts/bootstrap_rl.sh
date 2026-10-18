#!/usr/bin/env bash
# TeachOnce RL box bootstrap (train-only). Learnings from teachonce-gpu:
# - pin torch to driver CUDA (cu124 on 12.x; cu128 only if driver ≥12.8)
# - transformers<5 with current stacks
# - Gym via PyPI + Python 3.13.14
# - no OpenAI; train owns the GPU (no long-lived vLLM here)
set -euo pipefail

TO="${ROOT:-$HOME/teachonce}"
export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "$TO" "$TO/logs" "$TO/checkpoints" "$TO/results" "$TO/configs"
cd "$TO"

{
  echo "=== $(date -Is) RL inspect ==="
  hostname
  nvidia-smi || true
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  cat /etc/os-release | head -5
  df -h / /data 2>/dev/null || df -h /
  python3 --version || true
} | tee "$TO/inspect_rl.txt"

# HF cache: prefer big disk
if [[ -d /data ]]; then
  export HF_HOME="${HF_HOME:-/data/hf-cache}"
  mkdir -p "$HF_HOME"
else
  export HF_HOME="${HF_HOME:-$TO/hf-cache}"
  mkdir -p "$HF_HOME"
fi

# --- uv ---
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# Driver CUDA major.minor from nvidia-smi header (e.g. 12.7 / 12.8)
CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
echo "detected CUDA_VER=$CUDA_VER" | tee -a "$TO/inspect_rl.txt"

torch_index="https://download.pytorch.org/whl/cu124"
torch_pkg="torch==2.6.0"
nemo_rl_ok=0
if [[ -n "$CUDA_VER" ]]; then
  # 12.8+ → official NeMo-RL path; cu128 wheels start ~torch 2.7
  major="${CUDA_VER%%.*}"
  minor="${CUDA_VER#*.}"; minor="${minor%%.*}"
  if [[ "$major" -gt 12 ]] || { [[ "$major" -eq 12 ]] && [[ "$minor" -ge 8 ]]; }; then
    torch_index="https://download.pytorch.org/whl/cu128"
    torch_pkg="torch==2.7.1"
    nemo_rl_ok=1
  fi
fi
echo "torch_index=$torch_index torch_pkg=$torch_pkg nemo_rl_ok=$nemo_rl_ok" | tee -a "$TO/inspect_rl.txt"

# --- GRPO/SFT train venv (always; works on 12.7 and 12.8) ---
VENV="$TO/grpo-venv"
if ! "$VENV/bin/python" -c "import torch" 2>/dev/null; then
  rm -rf "$VENV"
  uv venv "$VENV" --python 3.11
  uv pip install --python "$VENV/bin/python" \
    "$torch_pkg" --index-url "$torch_index"
  uv pip install --python "$VENV/bin/python" \
    "transformers==4.57.6" "datasets" "accelerate" "wandb" "sentencepiece" \
    "peft" "tqdm" "pillow" "opencv-python-headless"
fi
"$VENV/bin/python" -c "import torch; print('grpo-venv', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" \
  | tee -a "$TO/inspect_rl.txt"

# --- NeMo Gym (eval / reward servers; policy can point at serve box later) ---
GYM="$TO/gym-venv"
if [[ ! -x "$GYM/bin/python" ]]; then
  uv venv --python 3.13.14 "$GYM" || uv venv --python 3.13 "$GYM"
  uv pip install --python "$GYM/bin/python" -U "nemo-gym"
fi
# shellcheck disable=SC1091
source "$GYM/bin/activate"
gym --version | tee -a "$TO/inspect_rl.txt" || true
deactivate || true

# --- Official NeMo-RL (only if driver allows) ---
if [[ "$nemo_rl_ok" -eq 1 ]]; then
  if [[ ! -d /data/nemo-rl ]]; then
    sudo mkdir -p /data 2>/dev/null || mkdir -p "$TO/nemo-rl-src"
    NRL_DIR="/data/nemo-rl"
    [[ -d /data ]] || NRL_DIR="$TO/nemo-rl-src"
    git clone --depth 1 --branch v0.5.0 https://github.com/NVIDIA-NeMo/RL.git "$NRL_DIR" || true
  fi
  echo "NeMo-RL clone path ready (driver cu≥12.8)" | tee -a "$TO/inspect_rl.txt"
else
  echo "skip official NeMo-RL install (need CUDA≥12.8); use grpo-venv HF loop" | tee -a "$TO/inspect_rl.txt"
fi

# --- env ---
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16}"
cat > "$TO/.env" <<EOF
ROLE=rl
MODEL_ID=${MODEL_ID}
HF_HOME=${HF_HOME}
GRPO_VENV=${VENV}
GYM_VENV=${GYM}
GYM_DIR=${TO}
LOCAL_VLLM_API_KEY=teachonce-local-key
# Serve stays on teachonce-gpu; set if tunneling Gym rewards from there:
# NVIDIA_VLLM_BASE_URL=http://teachonce-gpu:8000/v1
NVIDIA_VLLM_BASE_URL=http://127.0.0.1:8000/v1
EOF

cat > "$TO/env.yaml" <<EOF
policy_base_url: http://127.0.0.1:8000/v1
policy_api_key: teachonce-local-key
policy_model_name: ${MODEL_ID}
EOF

chmod +x "$TO/scripts/"*.sh 2>/dev/null || true
echo "bootstrap_rl OK → $TO"
echo "next: source .wandb.env; bash scripts/run_grpo_dag_wandb.sh  # or run_sft_nemotron_vl.sh"
