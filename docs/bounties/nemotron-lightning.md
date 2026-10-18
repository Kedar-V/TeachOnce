# Bounty: Best Use of NVIDIA Nemotron Lightning

Project: Flowstate (repo: TeachOnce)  
Challenge track fit: Fine-tuned & Specialized (≤33B)

## What Nemotron does here

Flowstate turns one expert screen recording into an executable support workflow (DAG). The generative model that watches the video and emits the DAG is NVIDIA Nemotron Nano 12B v2 VL:

| Item | Value |
|------|--------|
| Base | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` (~12.6B) |
| Served (app) | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16-SFT` (merged LoRA; SFT + RFT best eval) |
| Job | screen `video.mp4` + goal/SOP text → registry-locked DAG JSON |
| Serve | Red Hat vLLM on Brev (`teachonce-gpu`), OpenAI-compatible `/v1` |

Nemotron is the compile step: Teach Studio upload → model job → DAG → human confirm assumptions → publish workflow.

## Why this Nemotron SKU

The task is video understanding + structured output, not open chat. Nano VL accepts the full demo mp4 under vLLM (`--allowed-local-media-path`, video frames, guided decoding). We specialize it so it emits tool-registry ops (`CREATE_REPAIR`, `RECORD_PAID_OPTIONS`, …) instead of free-form clicks.

Note on the bounty name: the hackathon bounty asks for Nemotron Lightning (text MoE). Our core path needs vision, so production is Nano VL. Params stay under the 33B track cap.

## How we use / optimize it

1. Decode constraint: vLLM `guided_json` with registry `oneOf` on ACTION nodes (`op` + `tool` locked) and `maxItems` on nodes.
2. Product SFT: LoRA on warranty / parts / escalate demos (`teachonce-rl` H100), then merge to disk (vLLM has no LoRA path for `NemotronH_Nano_VL_V2`).
3. Guided RFT on top of SFT (`scripts/rl_vllm_guided_rft.py`): same `guided_json` + `dag_reward` as eval.
4. Holdout protocol: balanced-50 videos scored with `dag_reward` under the same guided schema as the app.
5. App wiring: `teachonce-web/mock-server` → live vLLM when `VLLM_BASE_URL` is set (`vllm_backend.py`).

## Holdout (balanced-50)

| Metric | Baseline | SFT | RL-only | SFT + RFT |
|--------|----------|-----|---------|-----------|
| Ops F1 | 0.306 | 0.720 | 0.330 | 0.805 |
| Mean reward | 0.587 | 0.814 | 0.615 | 0.881 |

Full tables: [`nemo-gym.md`](nemo-gym.md).

## Where to look in the repo

| Path | Role |
|------|------|
| `teachonce-web/` | Flowstate UI + API; Studio calls the model |
| `scripts/start_vllm_sft.sh` | Serve merged SFT weights |
| `scripts/baseline_vllm_dag.py` | Base Nemotron VL eval |
| `scripts/run_sft_nemotron_vl.sh` / `scripts/sft_nemotron_vl_wandb.py` | SFT train |
| `scripts/rl_vllm_guided_rft.py` | Guided RFT |
| `docs/training-design.md` | Train stages |
| [`nemo-gym.md`](nemo-gym.md) | Baseline → SFT → RFT numbers |

## One-line submission blurb

Nemotron Nano VL watches the expert screen recording and emits a registry-valid support DAG; we SFT + RFT it for our ops, serve it on vLLM with `guided_json`, and run it as the Teach Studio compile model inside Flowstate.
