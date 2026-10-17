# TeachOnce — Brev / vLLM / NeMo Gym (Phase 0)

**Status:** smoke green (vLLM + Gym↔vLLM)  
**Last updated:** 2026-10-17  

## Instance

| Field | Value |
|-------|--------|
| Instance name | `teachonce-gpu` |
| Instance ID | `r2hn7d18o` |
| GPU | 1× A100 80GB PCIe (`a100-80gb.1x`) |
| $/hr | **~$1.98** (stoppable) |
| Provider | Crusoe |
| Disk | 124GB root + 118GB `/data` (~108GB free at bootstrap) |
| Note | Swapped off GCP `a2-ultragpu-1g` (~$6.03/hr) |

## Versions

| Component | Version |
|-----------|---------|
| Driver | 565.57.01 |
| CUDA (driver) | 12.7 |
| OS | Ubuntu 22.04.5 LTS |
| System Python | 3.10.12 |
| uv | 0.13.1 |
| vLLM | **0.8.5.post1** (`--torch-backend=cu124`; auto/cu129 needs libcudart.so.13) |
| torch | 2.6.0+cu124 |
| transformers | **4.57.6** (5.x broke Qwen tokenizer with vLLM 0.8.5) |
| NeMo Gym | **v0.7.0** (PyPI; Python 3.13.14) |
| Smoke model | `Qwen/Qwen2.5-1.5B-Instruct` (~1.5B) |
| GPU mem after load | ~**49227 MiB** @ `gpu_memory_utilization=0.6` |

### Mac Gym (also preinstalled)

```bash
source ~/teachonce/gym-venv/bin/activate
gym --version   # NeMo Gym v0.7.0
cat ~/teachonce/env.yaml
```

## Endpoints

| What | Value |
|------|--------|
| vLLM on VM | `http://127.0.0.1:8000/v1` |
| API key | `teachonce-local-key` (`$LOCAL_VLLM_API_KEY`) |
| Gym `env.yaml` | `~/teachonce/env.yaml` → **local vLLM only** |

## Mac tunnel

```bash
brev port-forward teachonce-gpu -p 8000:8000
# or: ./scripts/tunnel_vllm.sh
source .env && ./scripts/smoke_test.sh
```

## Restart after reboot

```bash
brev shell teachonce-gpu
cd ~/teachonce && source .env
export PATH="$HOME/.local/bin:$PATH"   # Gym spawns need uv
./scripts/start_vllm.sh                 # wait for /v1/models
./scripts/start_gym.sh
RUN_GYM=1 ./scripts/smoke_test.sh
```

Stop: `./scripts/stop_services.sh` · idle credits: `brev stop teachonce-gpu`

## Gym smoke results (2026-10-17)

| Artifact | Path on VM |
|----------|------------|
| Rollouts | `~/teachonce/results/mcqa_rollouts.jsonl` |
| Metrics | `~/teachonce/results/mcqa_rollouts_aggregate_metrics.json` |
| Inspect | `~/teachonce/inspect.txt` |

Key metrics: `mean/reward` **0.4**, `pass@1/accuracy` **40%** (score irrelevant — E2E green).  
Wired with `--model-type vllm_model` → `http://127.0.0.1:8000/v1`.

## NVIDIA-only confirmation

- Gym `policy_base_url` = local vLLM (`http://127.0.0.1:8000/v1`)
- `--model-type vllm_model`
- **Zero** OpenAI / Anthropic / Gemini / OpenRouter keys used

## NeMo-RL / W&B smoke

| Item | Value |
|------|--------|
| Driver | 565.57.01 → **CUDA 12.7 max** |
| Official NeMo-RL | `main`/v0.5+ need torch **cu128+** → fails on this image |
| Cloned (reference) | `/data/nemo-rl` @ **v0.5.0** |
| Runnable smoke | `scripts/run_grpo_smoke_wandb.sh` (HF GRPO-style, cu124) |
| Live view | `smoke-test/phase0_integration.ipynb` §4 + W&B |

```bash
export WANDB_API_KEY=…   # https://wandb.ai/authorize
brev exec teachonce-gpu 'bash ~/teachonce/scripts/run_grpo_smoke_wandb.sh'
```

## Unresolved / notes

- Prefer `uv pip install vllm --torch-backend=cu124` on driver CUDA 12.x
- Pin `transformers<5` with this vLLM build
- Ensure `~/.local/bin` on `PATH` before `gym env start` (needs `uv`)
- Eval input must be **mcqa** `example.jsonl` under site-packages (not nested rglob)
- Full NeMo-RL: need newer Brev driver / NGC container, then `run_nemo_rl_smoke.sh`
