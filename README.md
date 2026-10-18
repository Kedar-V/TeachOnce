# Flowstate

Teach a support SOP once from an expert screen recording. Get an executable DAG. Run it on cases.

Hackathon stack: NVIDIA Nemotron Nano 12B v2 VL → Red Hat vLLM on Brev → NeMo Gym scoring → Flowstate web app. Inference is NVIDIA / Brev / local vLLM only (no OpenAI, Anthropic, Gemini, or Cursor as backends).

Challenge track: Fine-tuned & Specialized (~12.6B, under the 33B cap).

## Results (balanced holdout-24, same `guided_json`)

Base Nemotron VL vs product SFT vs one-round guided RFT. Scorer: `dag_reward` (`scripts/nemo_rl_gt.py`).

| Metric | Baseline | SFT | RL |
|--------|----------|-----|-----|
| mean reward | 0.594 | 0.821 | 0.608 |
| mean schema_ok | 0.958 | 0.958 | 1.000 |
| mean critical_ops | 0.202 | 0.792 | 0.192 |
| mean taken_branch | 0.500 | 0.667 | 0.500 |
| ops precision | 0.493 | 0.655 | 0.549 |
| ops recall | 0.202 | 0.792 | 0.192 |
| ops F1 | 0.275 | 0.716 | 0.269 |

Per-procedure mean reward (SFT wins all three):

| Procedure | Baseline | SFT | RL |
|-----------|----------|-----|-----|
| escalate | 0.625 | 0.688 | 0.618 |
| parts | 0.625 | 0.775 | 0.613 |
| warranty | 0.532 | 1.000 | 0.593 |

SFT is the improvement that stuck (~2.6× ops F1). Full write-up: [`docs/bounties/nemo-gym.md`](docs/bounties/nemo-gym.md).

## How to run

### 1. App (mock model, no GPU)

```bash
./teachonce-web/run.sh
# http://localhost:8000  ·  API docs /docs
```

Force mock even if a tunnel is up: `FORCE_MOCK=1 ./teachonce-web/run.sh`.

### 2. App against live SFT on Brev

```bash
cp .env.example .env   # BREV_INSTANCE_NAME, LOCAL_VLLM_API_KEY, MODEL_ID

# on teachonce-gpu:
bash scripts/bootstrap_brev.sh          # once
bash scripts/start_vllm_sft.sh          # merged SFT Nemotron VL on :8000

# on Mac:
./scripts/tunnel_vllm.sh                # localhost:8101 → VM:8000
cd teachonce-web && VLLM_BASE_URL=http://127.0.0.1:8101/v1 ./run.sh
```

Smoke the tunnel:

```bash
# with .env loaded
bash scripts/smoke_test.sh
# optional Gym MCQA: RUN_GYM=1 bash scripts/smoke_test.sh
```

### 3. NeMo Gym (on GPU box)

```bash
bash scripts/start_gym.sh
bash scripts/smoke_gym_nemotron_vl.sh
```

### 4. Holdout eval (on GPU box, vLLM up)

```bash
# base VL or SFT — whichever model vLLM is serving
python scripts/baseline_vllm_dag.py
# or compare runs:
bash scripts/eval_sft_vs_baseline.sh
```

More ops detail: [`docs/brev.md`](docs/brev.md) · train design: [`docs/training-design.md`](docs/training-design.md) · RL layout: [`docs/rl-vllm-gym-architecture.md`](docs/rl-vllm-gym-architecture.md).

## Demo

| Item | Path |
|------|------|
| 3–5 min video script | [`demo/VIDEO_SCRIPT.md`](demo/VIDEO_SCRIPT.md) · [`demo/FLOWSTATE_75S_SCRIPT.md`](demo/FLOWSTATE_75S_SCRIPT.md) |
| Example SOP | [`demo/SOP-CS-214_Laptop_Warranty.md`](demo/SOP-CS-214_Laptop_Warranty.md) |
| Web app | [`teachonce-web/`](teachonce-web/) |

## Bounty write-ups

| Bounty | Doc |
|--------|-----|
| Best Use of NVIDIA Nemotron Lightning | [`docs/bounties/nemotron-lightning.md`](docs/bounties/nemotron-lightning.md) |
| Best Model Improvement With NeMo Gym | [`docs/bounties/nemo-gym.md`](docs/bounties/nemo-gym.md) |
| Best Use of Red Hat vLLM | [`docs/bounties/redhat-vllm.md`](docs/bounties/redhat-vllm.md) |

## Repo map

| Path | What |
|------|------|
| `teachonce-web/` | Flowstate UI + API + model client (mock or live vLLM) |
| `scripts/` | vLLM/Gym bootstrap, SFT, eval, `dag_reward` |
| `configs/` | NeMo-RL + Gym sketch |
| `docs/bounties/` | Submission one-pagers |
| `demo/` | Video scripts + example SOP |

## Submission checklist

- [ ] GitHub repo link (this repo)
- [ ] Project description (this README + bounty docs)
- [ ] 3–5 minute demo video (record from `demo/` scripts)
- [ ] Point judges at the three bounty docs above
