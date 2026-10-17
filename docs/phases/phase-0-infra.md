# Phase 0 — Brev + vLLM + NeMo Gym (Infra)

**Status:** `done` (2026-10-17)  
**Tracker:** [`../implementation.md`](../implementation.md) · **Live notes:** [`../brev.md`](../brev.md)  
**Handout:** [`../vLLM/brevsetup.pdf`](../vLLM/brevsetup.pdf)

## Goal

Prove Mac → Brev A100 → self-hosted OpenAI-compatible inference → NeMo Gym smoke, with **no** third-party LLM providers.

## What shipped

- Instance `teachonce-gpu` (Crusoe A100 80GB)
- vLLM serving a smoke model on `127.0.0.1:8000`
- NeMo Gym `vllm_model` → local vLLM; ≤5-sample MCQA eval
- Scripts: `start_vllm`, `start_gym`, `smoke_test`, `stop_services`, tunnel helpers
- NVIDIA-only `env.yaml`

## Important follow-on

Phase 0 smoke may have used a small non-Nemotron instruct model for speed. **Phase 2+ and bounty demos must use NVIDIA Nemotron** on Brev / Red Hat vLLM. Do not present Qwen (or similar) as the TeachOnce product model.

## Do not redo

Do not reinstall Brev/vLLM/Gym unless broken. Stop idle GPUs: `brev stop teachonce-gpu`.

## Exit criteria (met)

See checklists in [`../implementation.md`](../implementation.md) § Phase 0 and report in [`../brev.md`](../brev.md).
