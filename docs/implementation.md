# TeachOnce — Implementation Tracker

**Status:** Living document — update as work lands  
**Deadline:** 2026-10-18 11:00 AM local  
**Current phase:** `1` — Dataset + DAG runtime  
**Last updated:** 2026-10-17 (plans = video→DAG)  

| Doc | Role |
|-----|------|
| **This file** | What we’re doing, what’s done, what’s next — **maintain here** |
| [`phases/README.md`](phases/README.md) | **Phase plan index** |
| [`phases/phase-1-dataset.md`](phases/phase-1-dataset.md) | **Phase 1 deep plan (current)** |
| [`plan.md`](plan.md) | Product spec, NVIDIA-only policy, DoD |
| [`implementation-stages.md`](implementation-stages.md) | Phase 0 command how-to |
| [`hld.md`](hld.md) / [`lld.md`](lld.md) | Architecture and contracts |
| [`vLLM/brevsetup.pdf`](vLLM/brevsetup.pdf) | Brev + vLLM handout |
| [`brev.md`](brev.md) | Live VM facts |

---

## How to maintain this file

When you finish a task:

1. Flip the checkbox `[ ]` → `[x]`.  
2. Set phase status in the dashboard (`not_started` / `in_progress` / `blocked` / `done`).  
3. Add a row to **Working log** (date, what changed, blockers).  
4. Update **Current phase** and **Next up** at the top.  
5. Put secrets only in `.env` / Brev — never commit keys.  

Do not start a later phase until the current phase acceptance section is fully checked (unless the App lane is explicitly parallel — see Parallelism).

---

## Dashboard

| Phase | Name | Status | Owner | Notes |
|-------|------|--------|-------|-------|
| 0 | Infra: Brev + vLLM + NeMo Gym | `done` | | Crusoe A100; vLLM+Gym+Mac tunnel green |
| 1 | Workflow IR dataset + Gym env | `not_started` | | See [`phases/phase-1-dataset.md`](phases/phase-1-dataset.md) |
| 2 | Nemotron LoRA + serve | `not_started` | | Needs Phase 0 + train JSONL |
| 3 | FastAPI + TeachOnce UI | `not_started` | | Wire to real generate when Phase 2 ready |
| 4 | Demo polish (Flows A–C) | `not_started` | | |

**MVP lens:** Warranty only. Flows A–C only. **Primary product = expert video → DAG** (aligned trace compile; optional SOP). No Test Lab / Review Queue until A–C are solid.

**AI policy:** NVIDIA / Brev / local vLLM only. No OpenAI, Anthropic, Gemini, Cursor model APIs as backends. See [`plan.md`](plan.md).

---

## Next up

1. Execute Phase 1 per [`phases/phase-1-dataset.md`](phases/phase-1-dataset.md) (IR schema → expert video + aligned trace + warranty gold → corpus → Gym).  
2. Keep Brev stopped when idle: `brev stop teachonce-gpu`.  
3. Phase 2: Nemotron train/serve (not Phase 0 smoke Qwen) — [`phases/phase-2-nemotron.md`](phases/phase-2-nemotron.md).  

---

## Phase 0 — Brev + vLLM + NeMo Gym

**Goal:** Mac → SSH → Brev GPU serving a small instruct model via vLLM; NeMo Gym eval (≤5 samples) through that server.

**Out of scope:** datasets, DAG app, Nemotron fine-tune, frontend, custom Gym envs.

**Detail:** [`implementation-stages.md`](implementation-stages.md) § Phase 0.

### Checklist

#### 0.1 Provision
- [x] Brev account / credits
- [x] Instance selected (Crusoe A100 80GB ~$1.98/hr in `brev.md`)
- [x] `brev login` + `brev shell` / `brev exec` works from Mac

#### 0.2 Inspect
- [x] `nvidia-smi` healthy (A100 80GB, driver 565.57.01)
- [x] Ubuntu / Python / `uv` / disk recorded in `brev.md`
- [x] No conflicting GPU processes (or noted)

#### 0.3 vLLM
- [x] Isolated venv; vLLM 0.8.5.post1 + cu124
- [x] Smoke model `Qwen/Qwen2.5-1.5B-Instruct`
- [x] Server on `127.0.0.1:8000` with local API key
- [x] `GET /v1/models` OK
- [x] `POST /v1/chat/completions` OK
- [x] GPU memory shows model loaded (~49GB @ 0.6 util)

#### 0.4 NeMo Gym
- [x] Mac: `nemo-gym==0.7.0` in `~/teachonce/gym-venv`
- [x] Same on Brev VM
- [x] Gym venv isolated from vLLM
- [x] `env.yaml` NVIDIA-only (local vLLM)

#### 0.5 Gym ↔ vLLM
- [x] `--model-type vllm_model` → local vLLM
- [x] `env.yaml` does **not** use `api.openai.com`
- [x] Built-in eval ≤5 samples completes
- [x] Results + aggregate metrics on disk (`~/teachonce/results/`)

#### 0.6 Access + scripts
- [x] SSH tunnel from Mac documented
- [x] Mac tunnel proven (`brev port-forward` + local curl)
- [x] `scripts/start_vllm.sh`
- [x] `scripts/start_gym.sh`
- [x] `scripts/smoke_test.sh`
- [x] `scripts/stop_services.sh`
- [x] `.env.example` (no secrets committed)
- [x] README infra section
- [x] `scripts/bootstrap_brev.sh` + `scripts/tunnel_vllm.sh`

#### 0.7 Report
- [x] `docs/brev.md` complete (versions, model, VRAM, tunnel, restart, NVIDIA-only confirmation)

### Phase 0 exit criteria

All checkboxes above green → set dashboard Phase 0 = `done`.

**Captured facts (fill during Phase 0):**

| Field | Value |
|-------|--------|
| Instance name / GPU | `teachonce-gpu` / Crusoe A100 80GB |
| $/hr | ~$1.98 |
| Smoke model ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| vLLM port | `8000` |
| Gym commit | PyPI `nemo-gym==0.7.0` |
| Tunnel command | `brev port-forward teachonce-gpu -p 8000:8000` |

---

## Phase 1 — Workflow IR dataset + NeMo Gym

**Full plan:** [`phases/phase-1-dataset.md`](phases/phase-1-dataset.md)

**Depends on:** Phase 0 done. CPU-ok on laptop.

### Checklist (summary — detail in phase doc)
- [ ] Workflow IR v1 schema (Pydantic + JSON Schema)
- [ ] Warranty gold audited (unresolved missing-purchase flagged)
- [ ] Generator + validators + quarantine + manifest
- [ ] ~24–36 valid + ~8–12 labeled invalid across 4 families
- [ ] `docs/source-coverage.md` from WONDERBREAD/Mind2Web/WebLINX inspect
- [ ] NeMo Gym `teachonce_ir` baseline smoke (local vLLM only)
- [ ] Docs sync; Nemotron (not Qwen) for product path

**Exit:** Phase 1 acceptance in [`phases/phase-1-dataset.md`](phases/phase-1-dataset.md).

---

## Phase 2 — Nemotron Nano 12B VL LoRA + serve

**Depends on:** Phase 0 green + Phase 1 train JSONL.

### Checklist
- [ ] Load `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` on Brev; one **video** smoke inference
- [ ] Short NeMo AutoModel LoRA smoke; save adapter
- [ ] Merge LoRA for vLLM (`vllm_merge_lora` / Mamba constraint)
- [ ] Full TeachOnce LoRA (one practical run)
- [ ] Serve via Red Hat AI Inference Server when available; else vLLM
- [ ] `backend/app/services/nvidia_inference.py` + `NVIDIA_VLLM_BASE_URL` + allowlists
- [ ] `POST /workflows/generate` returns validated model DAG
- [ ] Stop idle GPUs; reserve demo credits

**Exit:** Live generate uses Brev-hosted Nemotron only.

---

## Phase 3 — TeachOnce UI + API glue

**Depends on:** Phase 1 APIs; Phase 2 generate (or temporary stub clearly labeled).

### Checklist
- [ ] React app (Teach Studio, Editor, Executions, Overview, Integrations)
- [ ] Generate → NVIDIA gateway → React Flow DAG
- [ ] Execute case → event timeline + artifacts
- [ ] Overview: seeded vs live metrics
- [ ] No frontend AI provider SDKs

**Exit:** Flows A–B work end-to-end.

---

## Phase 4 — Demo polish

### Checklist
- [ ] Flow C: same DAG, expired-warranty branch
- [ ] 3-minute demo script + README
- [ ] Repo audit: no third-party LLM keys/SDKs
- [ ] Brev stopped when idle; demo restart verified
- [ ] DoD in [`plan.md`](plan.md) §17 satisfied

**Exit:** Ready to present.

---

## Parallelism

| Lane | Phases | Notes |
|------|--------|-------|
| GPU | 0 → 2 | Brev credits; stop when idle |
| App | 1 → 3 | Laptop OK for dataset/runtime/UI shell |
| Sync | 2 + 3 | Teach Studio Generate needs live Nemotron |

---

## Blockers

| Date | Blocker | Owner | Resolution |
|------|---------|-------|------------|
| | | | |

---

## Working log

| Date | Change |
|------|--------|
| 2026-10-17 | Created `implementation.md` as maintained tracker; current phase = 0 |
| 2026-10-17 | Phase 0 local infra: brev CLI via brew; scripts; `.env.example`; `docs/brev.md` stub; README; awaiting `brev login` |
| 2026-10-17 | Mac NeMo Gym v0.7.0 (`~/teachonce/gym-venv`); NVIDIA-only `env.yaml`; bootstrap uses PyPI not git clone |
| 2026-10-17 | Phase 0 smoke: Crusoe A100; vLLM chat OK; Gym mcqa×5 via `vllm_model` (mean/reward 0.4); `brev.md` filled |
| 2026-10-17 | Phase 0 **done** — Mac `port-forward` models+chat OK; next = Phase 1 |
| 2026-10-17 | Added `docs/phases/*` phase plans; Phase 1 = Workflow IR dataset |
| 2026-10-17 | Product pivot: plans/HLD/LLD = **expert video → DAG** (aligned-trace compile) |
| 2026-10-17 | Model lock: Nano **12B v2 VL** (`…-VL-BF16`, 12.6B) for video input |

---

## Quick links

- Phase 0 commands: [`implementation-stages.md`](implementation-stages.md)  
- NVIDIA-only policy: [`plan.md`](plan.md)  
- Brev handout: [`vLLM/brevsetup.pdf`](vLLM/brevsetup.pdf)  
- NeMo Gym quickstart: https://docs.nvidia.com/nemo/gym/get-started/quickstart/  
- Gym ↔ vLLM: https://docs.nvidia.com/nemo/gym/main/model-server/vllm/  
