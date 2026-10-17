# TeachOnce — Stage-Wise Implementation How-To

**Status:** Detailed commands / acceptance criteria  
**Living tracker (checkboxes, log, next up):** [`implementation.md`](implementation.md) ← **maintain status there**  
**Source of truth for product:** [`plan.md`](plan.md) · Architecture: [`hld.md`](hld.md) · Contracts: [`lld.md`](lld.md)  
**Brev / vLLM how-to (hackathon handout):** [`vLLM/brevsetup.pdf`](vLLM/brevsetup.pdf)  
**AI policy:** NVIDIA-only — see [`plan.md`](plan.md) § Mandatory NVIDIA-only AI infrastructure policy  

This document is the deep how-to for each stage. Update progress in [`implementation.md`](implementation.md). **Do not start later stages until the current stage’s acceptance criteria pass.**

```text
Phase 0  Infra (Brev + vLLM + NeMo Gym)     ← NOW
Phase 1  Canonical dataset + DAG runtime
Phase 2  Nemotron Nano 9B LoRA + Red Hat serve
Phase 3  FastAPI app + TeachOnce UI
Phase 4  E2E demo polish (Flows A–C)
```

---

## Phase 0 — NVIDIA Brev: vLLM + NeMo Gym (infra only)

**Goal:** From a Mac, reach a healthy Brev GPU that serves a small instruction model via vLLM (OpenAI-compatible protocol on **our** host) and complete a built-in NeMo Gym smoke eval through that server.

**Out of scope for Phase 0:** datasets, DAG compiler, fine-tuning Nemotron, frontend, custom Gym environments, WONDERBREAD/Mind2Web, RL rewards.

### Target architecture

```text
Local Mac
    |
    | SSH tunnel (preferred) / brev shell
    v
NVIDIA Brev GPU (prefer 1× A100 80GB; see GPU note)
    |
    +-- vLLM inference server  (127.0.0.1:<port>)
    |      +-- small instruct model (smoke only)
    |      +-- OpenAI-compatible /v1/*
    |
    +-- NVIDIA NeMo Gym
           +-- model-type: vllm_model  → local vLLM only
           +-- built-in eval smoke (≤5 samples)
           +-- rollouts + metrics on disk
```

### GPU note (reconcile handout vs product lock)

| Source | Recommendation |
|--------|----------------|
| [`brevsetup.pdf`](vLLM/brevsetup.pdf) | Example mentions **L40S 48GB** |
| Product lock ([`plan.md`](plan.md)) | Prefer **1× A100 80GB** |

**Phase 0 rule:** Search Brev for **A100 80GB** first (record $/hr). If unavailable, use **A100 40GB** or **L40S 48GB** and document the choice in `docs/brev.md`. Do not block Phase 0 on A100 if another ≥40GB NVIDIA GPU is available for smoke.

### NVIDIA-only rule for Gym

NeMo Gym’s public quickstart often shows `policy_base_url: https://api.openai.com/v1`. **That is forbidden for TeachOnce.**

Phase 0 must configure Gym with:

- `policy_base_url` / `--model-url` → **local vLLM** (e.g. `http://127.0.0.1:8000/v1`)  
- `policy_model_name` / `--model` → **exact** id from `GET /v1/models`  
- `policy_api_key` → local dummy key matching vLLM config  
- `--model-type vllm_model` (see [NeMo Gym vLLM model server](https://docs.nvidia.com/nemo/gym/main/model-server/vllm/))  

No OpenAI / Anthropic / Gemini / OpenRouter keys. Cursor IDE is not an inference backend.

### Red Hat vLLM vs stock vLLM

| Track | When |
|-------|------|
| **Phase 0a** | Stock **vLLM** per [`brevsetup.pdf`](vLLM/brevsetup.pdf) + NeMo Gym docs — fastest path to green smoke |
| **Phase 0b** *(same phase if time)* | Swap / add **Red Hat AI Inference Server** (vLLM-based) if the Brev image or docs support it; re-run the same API + Gym smoke against that endpoint |

Product inference for Nemotron (Phase 2+) targets Red Hat AI Inference Server. Phase 0 proves the **pattern** (self-hosted OpenAI-compatible API + Gym) even if the first process is upstream vLLM.

---

### Stage 0.1 — Provision Brev (from `brevsetup.pdf`)

Handout steps (page 1):

1. Login / create account at [brev.nvidia.com](https://brev.nvidia.com)  
2. Select instance; add credits  
3. From Mac:

```bash
brew install brevdev/homebrew-brev/brev
brev login
brev shell <instance-name>   # handout example used a named instance
```

4. On VM: `sudo apt update` (upgrade only if needed; avoid disruptive changes)

**Deliverable:** Interactive shell on Brev GPU VM.

---

### Stage 0.2 — Inspect environment

On the Brev VM, record:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version 2>/dev/null || true
cat /etc/os-release
python3 --version
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
docker --version 2>/dev/null || true
df -h
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Identify CUDA / driver constraints **before** installing packages. Paste results into `docs/brev.md`.

**Acceptance:** GPU visible; enough free disk for one small model (~10–20 GB headroom).

---

### Stage 0.3 — Install and serve vLLM

Follow [`brevsetup.pdf`](vLLM/brevsetup.pdf) + isolate from Gym:

```bash
# Handout-aligned (adjust if PDF page 2 differs after inspect)
uv venv env
source env/bin/activate

# Prefer CUDA-matched install from handout:
CUDA_VERSION=$(python -c 'import torch; print(torch.version.cuda.replace(".", ""))' 2>/dev/null || true)
# If torch not present yet, use official vLLM install for the detected driver/CUDA from nvidia-smi
uv pip install -U vllm --torch-backend=auto
# Or nightly path from handout when needed:
# uv pip install -U vllm --torch-backend=cu${CUDA_VERSION} --extra-index-url "https://wheels.vllm.ai/nightly/cu${CUDA_VERSION}"
```

Also follow NeMo Gym’s companion guidance when using Gym’s Python 3.13.14+ stack: separate venv for vLLM vs Gym if dependency conflicts appear ([Gym vLLM docs](https://docs.nvidia.com/nemo/gym/main/model-server/vllm/)).

**Smoke model (Phase 0 only):** one small openly available instruct model (~1B–3B preferred; ≤4B OK). Examples to pick **one** after HF access works:

- Prefer smallest that loads on available VRAM with `gpu_memory_utilization≈0.5–0.7`  
- Do **not** download Nemotron 9B or multiple large models in Phase 0  

**Serve (security default for TeachOnce):**

```bash
# Bind loopback; Mac reaches via SSH tunnel (Phase 0.6)
vllm serve <MODEL_ID> \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$LOCAL_VLLM_API_KEY" \
  --gpu-memory-utilization 0.6 \
  --max-model-len 4096
```

If NeMo Gym docs require host `0.0.0.0` for same-node Gym, prefer still binding `127.0.0.1` when Gym runs on the same VM (`http://127.0.0.1:8000/v1`). Avoid public exposure.

**Verify:**

```bash
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $LOCAL_VLLM_API_KEY"
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $LOCAL_VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"ping"}],"max_tokens":32}'
nvidia-smi   # confirm GPU memory used by vLLM
```

**Acceptance:** `/v1/models` and `/v1/chat/completions` succeed; GPU shows model resident.

---

### Stage 0.4 — Install NVIDIA NeMo Gym

Official:

- Repo: https://github.com/NVIDIA-NeMo/Gym  
- Quickstart: https://docs.nvidia.com/nemo/gym/get-started/quickstart/  

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
git rev-parse HEAD   # record in docs/brev.md
# Follow current install: Python 3.13.14+ and uv per docs
# Verify:
gym --help
```

Inspect:

- Available model-server configs (`vllm_model` vs `openai_model`)  
- Built-in environments (e.g. `mcqa`)  

Do **not** build a custom Gym environment yet. Keep Gym venv isolated from the vLLM venv if needed.

**Acceptance:** `gym` CLI works; commit/hash recorded.

---

### Stage 0.5 — Connect Gym → vLLM + smoke eval

Use documented **vllm_model** integration ([docs](https://docs.nvidia.com/nemo/gym/main/model-server/vllm/)):

```bash
# Example shape — confirm flags against checked-out Gym version
gym env start \
  --resources-server mcqa \
  --model-type vllm_model \
  --model-url http://127.0.0.1:8000/v1 \
  --model <EXACT_MODEL_ID_FROM_VLLM> \
  --model-api-key "$LOCAL_VLLM_API_KEY"
```

`env.yaml` (if used) must **not** point at `api.openai.com`. Example shape:

```yaml
policy_base_url: http://127.0.0.1:8000/v1
policy_api_key: <same as LOCAL_VLLM_API_KEY>
policy_model_name: <exact id from /v1/models>
```

Eval (≤5 samples), per quickstart pattern adapted to `vllm_model`:

```bash
source .venv/bin/activate
gym eval run --no-serve \
  --agent mcqa_simple_agent \
  --input resources_servers/mcqa/data/example.jsonl \
  --output results/mcqa_rollouts.jsonl \
  --limit 5 \
  --num-repeats 1
```

If `mcqa` + agent names differ in the pinned Gym commit, use `gym list benchmarks` / `--help` and the smallest built-in example that works.

**Verify:** Gym starts → requests hit vLLM → responses scored → `results/` written → aggregate metrics inspectable. Score value is irrelevant.

**Acceptance:** End-to-end Gym↔vLLM smoke green.

---

### Stage 0.6 — Mac access + reproducibility artifacts

Prefer **SSH tunnel** from Mac (do not publicly expose vLLM):

```bash
# On Mac (ports illustrative — document actuals)
ssh -L 8000:127.0.0.1:8000 <brev-user@host-or-brev-proxy>
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $LOCAL_VLLM_API_KEY"
```

Create in repo (paths filled with **actual** VM locations after setup):

| Artifact | Purpose |
|----------|---------|
| `scripts/start_vllm.sh` | Activate venv, serve model on documented host/port |
| `scripts/start_gym.sh` | Start Gym with vllm_model → local vLLM |
| `scripts/smoke_test.sh` | curl models + chat + optional gym eval --limit 5 |
| `scripts/stop_services.sh` | Stop vLLM / Gym cleanly |
| `.env.example` | `NVIDIA_VLLM_BASE_URL`, `LOCAL_VLLM_API_KEY`, `MODEL_ID` — **no secrets committed** |
| `docs/brev.md` | Instance type, $/hr, versions, ports, tunnel, logs, restart after reboot |
| `README.md` (infra section) | Start / stop / smoke / tunnel |

**Acceptance:** Scripts work on a fresh shell; Mac can call tunneled `/v1/chat/completions`; credentials not in git.

---

### Stage 0.7 — Phase 0 final report (required)

Write into `docs/brev.md`:

1. Installed software versions (driver, CUDA, Python, uv, vLLM, Gym commit)  
2. Selected smoke model + parameter count  
3. GPU memory after load (`nvidia-smi`)  
4. Working API endpoint + SSH tunnel command  
5. NeMo Gym smoke-test result paths + key metrics JSON  
6. Exact restart commands  
7. Unresolved compatibility issues  
8. Confirmation: **zero** third-party LLM providers used  

### Phase 0 acceptance checklist

- [ ] GPU recognized and healthy  
- [ ] vLLM serves one small instruct model  
- [ ] Direct `/v1/models` + `/v1/chat/completions` succeed  
- [ ] NeMo Gym installed; `gym` CLI works  
- [ ] Gym communicates with **local** vLLM (`vllm_model`)  
- [ ] Built-in eval completes; results on disk  
- [ ] start / stop / smoke scripts work  
- [ ] SSH tunnel from Mac documented  
- [ ] NVIDIA-only policy held (no OpenAI Gym default)  

---

## Phase 1 — Canonical dataset + deterministic runtime

**Depends on:** Phase 0 green (optional parallel on laptop without GPU).

1. Deterministic generator → freeze `data/generated/` + manifest  
2. DAG schema + validator  
3. SQLite seed + mock tools + executor  
4. Prove active-warranty and expired-warranty paths on **ground-truth** DAG  
5. FastAPI stubs: cases, executions, admin reset  

**Exit:** Two cases execute without the model.

---

## Phase 2 — Nemotron Nano 9B LoRA + Red Hat serve

**Depends on:** Phase 0 (Brev habits, scripts) + Phase 1 training JSONL.

1. Milestone 0 model path on Brev: load `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, one inference  
2. Short NeMo AutoModel LoRA smoke → merge (Mamba / `vllm_merge_lora`)  
3. Full TeachOnce LoRA on canonical JSONL (single practical run)  
4. Serve merged checkpoint via **Red Hat AI Inference Server** when available; else vLLM with NVIDIA-only gateway  
5. Wire `backend/app/services/nvidia_inference.py` → `NVIDIA_VLLM_BASE_URL`  
6. `POST /workflows/generate` returns validated DAG from **Nemotron**  

**Exit:** Live generate path uses Brev-hosted Nemotron only.

---

## Phase 3 — TeachOnce application UI

**Depends on:** Phase 1 APIs; Phase 2 generate endpoint (or stub until ready).

1. Scaffold React app matching UI mock  
2. Teach Studio → generate → React Flow editor  
3. Executions timeline + artifacts  
4. Overview seeded vs live metrics  
5. Integrations page from tool registry  

**Exit:** Flows A–B clickable against real backend.

---

## Phase 4 — Demo polish

1. Flow C second branch on same DAG  
2. Thin Test Lab (optional)  
3. Demo script + README  
4. Budget: stop idle Brev GPUs; reserve credits for live demo  
5. Repo audit: no third-party LLM SDKs/keys  

**Exit:** Definition of Done in [`plan.md`](plan.md) §17.

---

## Parallelism (hackathon)

| Lane | Owner focus |
|------|-------------|
| GPU lane | Phase 0 → Phase 2 |
| App lane | Phase 1 → Phase 3 (CPU) |
| Sync point | Phase 2 generate + Phase 3 Teach Studio |

---

## Immediate next actions (start Phase 0)

1. Confirm Brev credits and pick GPU (A100 80GB → else A100 40 / L40S).  
2. Follow [`vLLM/brevsetup.pdf`](vLLM/brevsetup.pdf) Stages 0.1–0.3.  
3. Install NeMo Gym; connect with `vllm_model` to **local** vLLM only.  
4. Run ≤5-sample built-in eval; write `docs/brev.md` + scripts.  
5. Only then proceed to Phase 1 / Nemotron.  

---

## Document history

| Date | Change |
|------|--------|
| 2026-10-17 | Initial stage-wise plan; Phase 0 = Brev vLLM + NeMo Gym per brevsetup.pdf and Gym docs; NVIDIA-only Gym wiring |
