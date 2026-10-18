#!/usr/bin/env bash
# Build mamba-ssm for A100 only into vllm-env, then start Nemotron vLLM + baseline.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
export MAMBA_FORCE_BUILD=TRUE
ROOT="${ROOT:-$HOME/teachonce}"
V="$ROOT/vllm-env/bin/python"
LOG="$ROOT/logs/fix_mamba_baseline.log"
mkdir -p "$ROOT/logs" /tmp/mamba_src
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u) fix_mamba_for_vllm start ==="
pkill -9 -f "selective_scan|ninja -v" 2>/dev/null || true
sleep 2

# pin torch to cu124 (do not let deps float to cu13)
uv pip install --python "$V" \
  "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
"$V" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.version.cuda)"

cd /tmp/mamba_src
[ -d causal-conv1d/.git ] || git clone --depth 1 --branch v1.5.0.post8 https://github.com/Dao-AILab/causal-conv1d.git
[ -d mamba-ssm/.git ] || git clone --depth 1 --branch v2.2.5 https://github.com/state-spaces/mamba.git mamba-ssm

# reset setup.py then force sm_80 after cc_flag = []
for repo in mamba-ssm causal-conv1d; do
  git -C /tmp/mamba_src/$repo checkout -- setup.py 2>/dev/null || true
done
python3 - <<'PY'
from pathlib import Path
for name in ("mamba-ssm/setup.py", "causal-conv1d/setup.py"):
    p = Path("/tmp/mamba_src") / name
    t = p.read_text()
    if "PATCHED_SM80" in t:
        print("already patched", p)
        continue
    needle = "cc_flag = []"
    if needle not in t:
        raise SystemExit(f"no cc_flag=[] in {p}")
    # after first cc_flag = [], replace later arch logic with sm_80 only
    i = t.index(needle)
    insert = (
        "cc_flag = []\n"
        "    # PATCHED_SM80: A100 only — skip multi-arch gencodes\n"
        '    cc_flag.extend(["-gencode", "arch=compute_80,code=sm_80"])\n'
        "    if False:  # PATCHED_SM80 disable stock arch detection\n"
        "        pass\n"
    )
    # wrap remainder of the if-not-SKIP block gently: just set flags and
    # comment out subsequent cc_flag.append("-gencode") pairs via rewrite
    t2 = t[:i] + insert + t[i + len(needle):]
    # kill remaining gencode appends
    import re
    t2 = re.sub(
        r'(\s*)cc_flag\.append\("-gencode"\)\s*\n\s*cc_flag\.append\("arch=compute_[^"]+"\)',
        r"\1pass  # PATCHED_SM80",
        t2,
    )
    t2 = "import os\nos.environ['TORCH_CUDA_ARCH_LIST']='8.0'  # PATCHED_SM80\n" + t2
    p.write_text(t2)
    print("patched", p)
PY

uv pip install --python "$V" --no-build-isolation /tmp/mamba_src/causal-conv1d
rm -rf /tmp/mamba_src/mamba-ssm/build
rm -f "$ROOT/vllm-env/lib/python3.12/site-packages/selective_scan_cuda"*.so
uv pip uninstall --python "$V" mamba-ssm || true
uv pip install --python "$V" --no-build-isolation --no-cache /tmp/mamba_src/mamba-ssm

cd /tmp
"$V" -c "import mamba_ssm,torch; print('mamba OK', mamba_ssm.__version__, torch.cuda.is_available())"

# env for serve
set -a
# shellcheck disable=SC1091
source "$ROOT/.env" 2>/dev/null || true
set +a
export MODEL_ID=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
export LOCAL_VLLM_API_KEY="${LOCAL_VLLM_API_KEY:-teachonce-local-key}"
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=8192
export HF_HOME=/data/hf-cache
chmod +x "$ROOT/scripts/start_vllm_nemotron.sh"
bash "$ROOT/scripts/start_vllm_nemotron.sh"

# wait for models
for i in $(seq 1 120); do
  if curl -sf -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "vLLM ready"
    break
  fi
  sleep 5
done

EVAL_GT="${EVAL_GT:-$ROOT/data/generated/nemo_rl/gt_train_balanced100.jsonl}"
EVAL_N="${EVAL_N:-50}"
SEED_ARGS=()
[[ -n "${EVAL_SEED:-}" ]] && SEED_ARGS=(--seed "$EVAL_SEED")
"$V" -u "$ROOT/scripts/baseline_vllm_dag.py" \
  --gt "$EVAL_GT" \
  --video-root "$ROOT/data/generated/demonstrations/videos" \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key "$LOCAL_VLLM_API_KEY" \
  --model "$MODEL_ID" \
  --max-samples "$EVAL_N" \
  "${SEED_ARGS[@]}" \
  --out "$ROOT/results/baseline_vllm_dag.json"

echo "=== $(date -u) DONE ==="
cat "$ROOT/results/baseline_vllm_dag.json" | "$V" -c "import sys,json; print(json.dumps(json.load(sys.stdin)['summary'], indent=2))"
