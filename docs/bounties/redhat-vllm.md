# Bounty: Best Use of Red Hat vLLM

Project: Flowstate (repo: TeachOnce)  
Required: vLLM as the serving layer for a core model; show it is central, not bolted on

## Where vLLM sits

vLLM is the inference engine for every generative call in Flowstate:

```
Expert screen video
        │
        ▼
 Red Hat vLLM  (Nemotron VL, guided_json)
        │
        ▼
   registry DAG JSON
        │
        ├─ Teach Studio (app)
        ├─ Holdout eval (baseline / SFT / RL)
        └─ NeMo Gym policy (--model-type vllm_model)
```

There is no third-party chat API in the inference path. App env points at the local OpenAI-compatible server:

- `NVIDIA_VLLM_BASE_URL=http://127.0.0.1:8000/v1` (VM) or Mac tunnel `./scripts/tunnel_vllm.sh`
- `MODEL_ID=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16-SFT`
- `LOCAL_VLLM_API_KEY=teachonce-local-key`

Live Studio path: `teachonce-web/mock-server/app/vllm_backend.py` → `POST /v1/chat/completions` with video + `guided_json`.

## Config and optimizations

| Knob | Choice | Why |
|------|--------|-----|
| Version | vLLM **0.8.5.post1** | Stable with this Nemotron VL + CUDA 12.7 driver |
| Torch backend | `--torch-backend=cu124` | Driver CUDA 12.7; cu129 wants libcudart.so.13 |
| transformers | pin `<5` (4.57.6) | 5.x broke Qwen tokenizer with this vLLM |
| Serve script | `scripts/start_vllm_sft.sh` | Merged SFT weights (no LoRA for this arch in vLLM) |
| Media | `--allowed-local-media-path`, `--limit-mm-per-prompt videos:1`, `num_frames:8`, video pruning | Real mp4 in, not frame dumps only |
| Structured out | `--guided-decoding-backend xgrammar` + request `guided_json` | Force registry `oneOf` DAG schema |
| Context | `--max-model-len` up to 32k on SFT serve | Room for video + schema |
| GPU mem | `gpu_memory_utilization` ~0.90 on SFT serve | Fill A100 for VL |
| Bind | `127.0.0.1:8000` | Loopback only; Mac uses `brev port-forward` |

Dual-box layout (see [`../brev.md`](../brev.md)):

| Box | Role |
|-----|------|
| `teachonce-gpu` (A100) | Always-on vLLM serve + Gym eval + app tunnel |
| `teachonce-rl` (H100) | Train; optional guided rollouts via vLLM then LoRA |

RL hot path (when NeMo-RL owns the GPU): vLLM as generation backend with `expose_http_server` + weight update. Diagram: [`../rl-vllm-gym-architecture.md`](../rl-vllm-gym-architecture.md).

## What we got out of it

1. App must work through vLLM: Flowstate Studio generate calls the same `/v1` surface as eval.
2. Schema-valid DAGs at decode: without `guided_json`, free generate fails `schema_ok` and RL advantages collapse (documented in the RL architecture note).
3. Reproducible holdout: baseline and SFT both scored through the same vLLM + guided protocol; SFT ops F1 0.716 vs baseline 0.275 on balanced-24 ([`nemo-gym.md`](nemo-gym.md)).
4. Gym on vLLM: `gym env start --model-type vllm_model --model-url http://127.0.0.1:8000/v1`.

## How to reproduce serve

```bash
# On Brev teachonce-gpu
bash scripts/start_vllm_sft.sh
bash scripts/start_gym.sh          # optional
RUN_GYM=1 bash scripts/smoke_test.sh

# On Mac
./scripts/tunnel_vllm.sh
cd teachonce-web && VLLM_BASE_URL=http://127.0.0.1:8101/v1 ./run.sh
```

Smoke details: [`../brev.md`](../brev.md) · Phase 0 scripts under `scripts/`.

## One-line submission blurb

Red Hat vLLM serves fine-tuned Nemotron VL for Flowstate end-to-end: video ingest, guided_json DAG decode, NeMo Gym policy, and holdout eval.
