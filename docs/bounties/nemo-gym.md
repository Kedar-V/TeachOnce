# Bounty: Best Model Improvement With NeMo Gym

Project: Flowstate (repo: TeachOnce)  
Required pieces: Gym environment · tools/tasks · scorer · baseline vs final on held-out tasks

## Task we improved

Fail mode: a VL model given a screen recording of a support SOP produces JSON that is schema-shaped but misses the critical registry ops and the filmed branch tip (e.g. Active warranty → `CREATE_REPAIR` vs Expired → `RECORD_PAID_OPTIONS`).

Success metric: `dag_reward` in `scripts/nemo_rl_gt.py`:

| Term | Weight | Meaning |
|------|--------|---------|
| `schema_ok` | 0.4 | Valid executable DAG |
| `critical_ops` | 0.3 | Family-defining ops present |
| `taken_branch` | 0.3 | Correct branch tip for the filmed path |

Also report ops precision / recall / F1 on critical ops (`scripts/baseline_ops_pr.py`).

## NeMo Gym setup

| Piece | Detail |
|-------|--------|
| Gym | NeMo Gym v0.7.0 |
| Policy URL | local vLLM only (`http://127.0.0.1:8000/v1`); no OpenAI/Anthropic |
| Config | `env.yaml` (`policy_base_url`, `policy_api_key`, `policy_model_name`) |
| Start | `scripts/start_gym.sh` → `gym env start --model-type vllm_model …` |
| Smoke | `scripts/smoke_gym_nemotron_vl.sh` → MCQA rollouts + aggregate metrics |
| RL bridge | `scripts/smoke_gym_rl_bridge.py` → Gym rewards → NeMo-RL-shaped batch |
| Architecture | [`../rl-vllm-gym-architecture.md`](../rl-vllm-gym-architecture.md) · sketch `configs/nemo_rl_gym_nemotron_vl.yaml` |

Verifier for product DAGs is `dag_reward` (same function Gym-adjacent scoring and holdout eval use). Schema is enforced at decode with vLLM `guided_json`; Gym / holdout scoring measures ops and branch quality.

## How to run scoring

On teachonce-gpu with vLLM serving the model under test: `scripts/baseline_vllm_dag.py` or `scripts/eval_sft_vs_baseline.sh`.

## Baseline vs final (balanced holdout-50, same guided_json)

Escalate 17 / parts 17 / warranty 16.

| Metric | Baseline | SFT | RL-only | SFT + RFT |
|--------|----------|-----|---------|-----------|
| Mean reward | 0.587 | 0.814 | 0.615 | 0.881 |
| Schema OK | 0.940 | 0.980 | 1.000 | 1.000 |
| Critical ops recall | 0.224 | 0.780 | 0.238 | 0.880 |
| Taken branch | 0.520 | 0.720 | 0.540 | 0.820 |
| Ops precision | 0.481 | 0.668 | 0.536 | 0.742 |
| Ops recall | 0.224 | 0.780 | 0.238 | 0.880 |
| Ops F1 | 0.306 | 0.720 | 0.330 | 0.805 |

Per-procedure mean reward:

| Procedure | Baseline | SFT | RL-only | SFT + RFT |
|-----------|----------|-----|---------|-----------|
| Escalate (17) | 0.610 | 0.760 | 0.630 | 0.850 |
| Parts (17) | 0.610 | 0.790 | 0.620 | 0.840 |
| Warranty (16) | 0.538 | 0.898 | 0.594 | 0.955 |
| Overall (50) | 0.587 | 0.814 | 0.615 | 0.881 |

## What improved the score

1. Path-consistent gold DAGs: GT ops match the filmed branch.
2. Product SFT LoRA: teach registry DAG format + MVP ops; merge for vLLM.
3. `guided_json` + `maxItems`: schema near ceiling at baseline; lifts concentrate on ops recall / F1.
4. RL-only on base ≈ baseline. SFT + RFT is best (ops F1 0.805, mean reward 0.881).

Ops F1 0.306 → 0.805 on balanced holdout-50 with SFT + RFT.

## One-line submission blurb

We wired NeMo Gym to local Nemotron-on-vLLM, scored video→DAG with `dag_reward`, and raised holdout ops F1 from 0.306 to 0.805 with product SFT + RFT under a fixed guided_json eval.
