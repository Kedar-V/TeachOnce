#!/usr/bin/env bash
# Force-build mamba-ssm for nemotron-vl-venv (A100 sm_80) against installed torch.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

ROOT="${ROOT:-$HOME/teachonce}"
V="${NEMOTRON_VL_VENV:-$ROOT/nemotron-vl-venv}/bin/python"
LOG="$ROOT/logs/fix_mamba_sft.log"
mkdir -p "$ROOT/logs" /tmp/mamba_src
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u) fix_mamba_sft start ==="
"$V" -c "import torch; print('torch', torch.__version__, torch.version.cuda)"

uv pip uninstall --python "$V" mamba-ssm causal-conv1d 2>/dev/null || true

cd /tmp/mamba_src
[ -d causal-conv1d/.git ] || git clone --depth 1 --branch v1.5.0.post8 https://github.com/Dao-AILab/causal-conv1d.git
[ -d mamba-ssm/.git ] || git clone --depth 1 --branch v2.2.5 https://github.com/state-spaces/mamba.git mamba-ssm

python3 - <<'PY'
from pathlib import Path
import re
for name in ("mamba-ssm/setup.py", "causal-conv1d/setup.py"):
    p = Path("/tmp/mamba_src") / name
    t = p.read_text()
    if "PATCHED_SM80" in t:
        print("already patched", p)
        continue
    needle = "cc_flag = []"
    i = t.index(needle)
    insert = (
        "cc_flag = []\n"
        "    # PATCHED_SM80: A100 only\n"
        '    cc_flag.extend(["-gencode", "arch=compute_80,code=sm_80"])\n'
        "    if False:\n"
        "        pass\n"
    )
    t2 = t[:i] + insert + t[i + len(needle) :]
    t2 = re.sub(
        r'(\s*)cc_flag\.append\("-gencode"\)\s*\n\s*cc_flag\.append\("arch=compute_[^"]+"\)',
        r"\1pass  # PATCHED_SM80",
        t2,
    )
    t2 = "import os\nos.environ['TORCH_CUDA_ARCH_LIST']='8.0'  # PATCHED_SM80\n" + t2
    p.write_text(t2)
    print("patched", p)
PY

rm -rf /tmp/mamba_src/causal-conv1d/build /tmp/mamba_src/mamba-ssm/build
find "$(dirname "$V")/../lib" -name 'selective_scan_cuda*.so' -delete 2>/dev/null || true
find "$(dirname "$V")/../lib" -name 'causal_conv1d_cuda*.so' -delete 2>/dev/null || true

uv pip install --python "$V" --no-build-isolation --no-cache /tmp/mamba_src/causal-conv1d
uv pip install --python "$V" --no-build-isolation --no-cache /tmp/mamba_src/mamba-ssm
"$V" -c "import mamba_ssm,torch; print('mamba OK', mamba_ssm.__version__, torch.__version__)"
echo "=== $(date -u) fix_mamba_sft DONE ==="
