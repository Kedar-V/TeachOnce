# Phase 2 — Nemotron Fine-Tune + Serve

**Status:** `not_started`  
**Depends on:** Phase 1 dataset + Gym IR env  
**Tracker:** [`../implementation.md`](../implementation.md)

## Goal

Fine-tune **NVIDIA Nemotron Nano 12B v2 VL** (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`, 12.6B &lt; 33B):

1. **Stage A** — LoRA/PEFT on mapped public screen videos → IR/DAG  
2. **Stage B** — continue SFT on **50** warranty videos (recipe: [`../prompts/synthetic-finetune-videos.md`](../prompts/synthetic-finetune-videos.md)) → our tool-registry DAG  

Serve on Brev with Red Hat vLLM (`--trust-remote-code`, video flags); wire `nvidia_inference.py` → `POST /workflows/generate`.

## Bounty alignment

- **Best Use of NVIDIA Nemotron Lightning** — if Lightning is the served/compile model, document its role  
- **Best Use of Red Hat vLLM** — serving must be central, not bolted on  
- **NeMo Gym** — continue IR eval baseline vs post-train scores  

## Out of scope until Phase 1 exits

Do not start full LoRA until IR corpus is accepted.

## Checklist (high level)

- [ ] Load Nemotron on Brev; one base inference  
- [ ] Short LoRA smoke → merge (Mamba / `vllm_merge_lora` as needed)  
- [ ] Full TeachOnce LoRA on Phase 1 JSONL  
- [ ] Serve via Red Hat AI Inference Server / vLLM  
- [ ] `NVIDIA_VLLM_BASE_URL` + allowlists; no third-party LLMs  
- [ ] `/workflows/generate` returns validated IR/DAG from **Nemotron**  
- [ ] Gym: baseline vs post-train on holdout  
- [ ] Stop idle GPUs; reserve demo credits  

## Detail

Expand when Phase 1 completes. See also [`../plan.md`](../plan.md) §7 and [`../brev.md`](../brev.md).
