# TeachOnce

Hackathon demo: teach a warranty SOP once → executable DAG → NVIDIA Nemotron on Brev.

**AI policy:** NVIDIA / Brev / local vLLM only. No OpenAI, Anthropic, Gemini, or Cursor as inference backends.

## Phase 0 — Infra (Brev + vLLM + NeMo Gym)

Live VM facts: [`docs/brev.md`](docs/brev.md) · How-to: [`docs/implementation-stages.md`](docs/implementation-stages.md) · Tracker: [`docs/implementation.md`](docs/implementation.md)

### Mac once

```bash
brew install brevdev/homebrew-brev/brev
brev login --email <you@email>
cp .env.example .env   # set BREV_INSTANCE_NAME, LOCAL_VLLM_API_KEY, MODEL_ID
```

### Provision + bootstrap (GPU)

```bash
brev search gpu --gpu-name A100 --min-vram 40 --sort price
brev create teachonce-gpu -g A100 --min-vram 40 --stoppable --timeout 600
# copy repo or bootstrap script onto VM, then:
brev shell teachonce-gpu
# on VM:
bash scripts/bootstrap_brev.sh
bash scripts/start_vllm.sh
# wait for model load, then:
bash scripts/start_gym.sh
RUN_GYM=1 bash scripts/smoke_test.sh
```

### Mac ↔ vLLM tunnel

```bash
./scripts/tunnel_vllm.sh
# other terminal:
source .env && ./scripts/smoke_test.sh
```

### Notebook smoke (Mac)

```bash
brev port-forward teachonce-gpu -p 8000:8000   # leave running
jupyter notebook smoke-test/phase0_integration.ipynb
```

See [`smoke-test/README.md`](smoke-test/README.md).

### Stop

On VM: `./scripts/stop_services.sh` · On Mac: `brev stop teachonce-gpu` when idle (save credits).

| Script | Where | Does |
|--------|-------|------|
| `scripts/bootstrap_brev.sh` | VM | uv, vLLM venv, NeMo Gym, `.env` |
| `scripts/start_vllm.sh` | VM | serve smoke model on `127.0.0.1:8000` |
| `scripts/start_gym.sh` | VM | Gym → **local** vLLM (`vllm_model`) |
| `scripts/smoke_test.sh` | VM or Mac | curl models + chat; `RUN_GYM=1` for eval |
| `scripts/stop_services.sh` | VM | kill vLLM / Gym |
| `scripts/tunnel_vllm.sh` | Mac | `brev port-forward … 8000:8000` |

## Pre-SFT baselines (`dag_reward`)

Same GT: `data/generated/nemo_rl/gt.jsonl` (25 rows). Scorer: `schema_ok×0.4 + critical_ops×0.3 + taken_branch×0.3` via `scripts/nemo_rl_gt.py` / `scripts/score_dag_preds.py`.

| Run | Model | Input | mean reward | schema_ok | critical_ops | taken_branch | Artifact |
|-----|-------|-------|-------------|-----------|--------------|--------------|----------|
| Nemotron VL | `NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` @ vLLM | full `video.mp4` + `guided_json` (registry `oneOf`) | **0.649** | 1.0 | 0.19 | 0.64 | `results/baseline_vllm_dag.json` |
| Grok fair | Cursor Grok `grok-4.7-medium-fast` subagents | 8 frames/video + GT `prompt` + registry list only | **0.809** | 1.0 | 0.56 | 0.80 | `results/baseline_grok_fair_dag.json` |
| Grok unfair (invalid) | same Grok | frames + GT prompt + **extra branch-tip coaching** | 1.000 | 1.0 | 1.0 | 1.0 | `results/baseline_grok_dag.json` — do not cite |

**Fair protocol (Grok):** inputs stripped to `results/grok_fair_inputs.jsonl` (`id`, `prompt`, `frames_dir` only — no gold). Registry lock = `results/grok_fair_registry.json` (same op/tool pairs as Nemotron guided schema). Agents forbidden from opening `gt.jsonl`. No Active/Expired→op hints from the harness.

**Still asymmetric:** Grok sees frames; Nemotron sees video. Both share the GT `prompt` text (goal/family). Grok is a **reference ceiling**, not a TeachOnce inference backend (see AI policy above). Per-sample fair: Grok 17 / Nemotron 3 / ties 0.

Nemotron script: `scripts/baseline_vllm_dag.py` (Brev `teachonce-gpu`).
