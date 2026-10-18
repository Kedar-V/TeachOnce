# Bounty: Best Model Improvement With NeMo Gym

Project: Flowstate (repo: TeachOnce)  
Required pieces: Gym environment · tools/tasks · scorer · baseline vs final on held-out tasks

## Task we improved

Fail mode today: a small VL model given a screen recording of a support SOP produces JSON that is schema-shaped but misses the critical registry ops and the filmed branch tip (e.g. Active warranty → `CREATE_REPAIR` vs Expired → `RECORD_PAID_OPTIONS`).

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

## Baseline vs final (same holdout, same guided_json)

| Metric | Baseline (base VL) | SFT (primary) | RL guided RFT |
|--------|--------------------|---------------|---------------|
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

## What improved the score (ablations we can isolate)

1. Path-consistent gold DAGs: GT ops match the filmed branch (fixes label noise before train).
2. Product SFT LoRA: teach registry DAG format + MVP ops; merge for vLLM. W&B train 3: ~3 epochs, loss ~0.62 → ~0.48.
3. `guided_json` + `maxItems`: schema already near ceiling at baseline; lifts concentrate on ops recall / F1, not parse rate.
4. One-round guided RFT on base (`scripts/rl_vllm_guided_rft.py`) ≈ baseline on this holdout (rollout mean R≈0.995 → little advantage). SFT is the improvement that stuck.

SFT lifts ops F1 ~2.6× on the balanced holdout (0.275 → 0.716). Gains are strongest on warranty (train distribution); escalate/parts move less.

## One-line submission blurb

We wired NeMo Gym to local Nemotron-on-vLLM, scored video→DAG with `dag_reward`, and raised holdout ops F1 from 0.275 to 0.716 with product SFT under a fixed guided_json eval.
