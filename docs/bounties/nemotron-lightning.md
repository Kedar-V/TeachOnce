# Bounty: Best Use of NVIDIA Nemotron Lightning

Project: Flowstate (repo: TeachOnce)  
Challenge track fit: Fine-tuned & Specialized (≤33B)

## What Nemotron does here

Flowstate turns one expert screen recording into an executable support workflow (DAG). The generative model that watches the video and emits the DAG is NVIDIA Nemotron Nano 12B v2 VL:

| Item | Value |
|------|--------|
| Base | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` (~12.6B) |
| Served (app) | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16-SFT` (merged LoRA) |
| Job | screen `video.mp4` + goal/SOP text → registry-locked DAG JSON |
| Serve | Red Hat vLLM on Brev (`teachonce-gpu`), OpenAI-compatible `/v1` |

Nemotron is not a side feature. It is the compile step: Teach Studio upload → model job → DAG → human confirm assumptions → publish workflow.

## Why this Nemotron SKU

The task is **video understanding + structured output**, not open chat. Nano VL is the Nemotron vision model that accepts the full demo mp4 under vLLM (`--allowed-local-media-path`, video frames, guided decoding). We specialize it with product SFT so it emits our tool-registry ops (`CREATE_REPAIR`, `RECORD_PAID_OPTIONS`, …) instead of free-form clicks.

Note on the bounty name: the hackathon bounty asks for Nemotron Lightning. Lightning is the text MoE SKU. Our core path needs vision, so the model in production is Nano VL. Total generative params stay under the 33B track cap.

## How we use / optimize it

1. Decode constraint: vLLM `guided_json` with registry `oneOf` on ACTION nodes (`op` + `tool` locked) and `maxItems` on nodes so the model cannot invent tools or loop ACTION nodes forever.
2. Product SFT: LoRA on Northstar warranty / parts / escalate demos (`teachonce-rl` H100), then merge to disk because vLLM has no LoRA path for `NemotronH_Nano_VL_V2`.
3. Same protocol for eval: holdout videos scored with `dag_reward` (`scripts/nemo_rl_gt.py`) under the same guided schema as the app.
4. App wiring: `teachonce-web/mock-server` talks to live vLLM when `VLLM_BASE_URL` is set (`vllm_backend.py`); mock corpus fallback when the tunnel is down.

## Where to look in the repo

| Path | Role |
|------|------|
| `teachonce-web/` | Flowstate UI + API; Studio calls the model |
| `scripts/start_vllm_sft.sh` | Serve merged SFT weights |
| `scripts/baseline_vllm_dag.py` | Base Nemotron VL eval |
| `scripts/run_sft_nemotron_vl.sh` / `scripts/sft_nemotron_vl_wandb.py` | SFT train |
| `docs/training-design.md` | Train stages |
| [`nemo-gym.md`](nemo-gym.md) | Baseline → SFT numbers |

## One-line submission blurb

Nemotron Nano VL watches the expert screen recording and emits a registry-valid support DAG; we SFT it for our ops, serve it on vLLM with `guided_json`, and run it as the Teach Studio compile model inside Flowstate.
