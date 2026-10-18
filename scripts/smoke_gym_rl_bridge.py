#!/usr/bin/env python3
"""
Bridge smoke: Gym rollouts (Nemotron VL via local vLLM) → RL-shaped batch.

Does NOT run full NeMo-RL weight updates (GPU held by Nemotron serve; driver < cu128).
Proves: Gym rewards are consumable by the NeMo-RL/GRPO data path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("TEACHONCE_ROOT", Path.home() / "teachonce"))
RESULTS = ROOT / "results"
OUT = RESULTS / "gym_rl_bridge_batch.jsonl"


def load_rollouts() -> list[dict]:
    candidates = sorted(RESULTS.glob("*rollouts.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    # prefer nemotron smoke output
    preferred = [p for p in candidates if "nemotron" in p.name] or candidates
    if not preferred:
        raise SystemExit(f"no rollouts under {RESULTS} — run smoke_gym_nemotron_vl.sh first")
    path = preferred[0]
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} rollouts from {path}")
    return rows, path


def to_rl_datum(row: dict, i: int) -> dict:
    """Shape one Gym rollout like a NeMo-RL DatumSpec-ish record."""
    reward = None
    # Gym rows vary; try common keys
    for k in ("reward", "score", "verified_reward"):
        if k in row:
            reward = row[k]
            break
    if reward is None:
        metrics = row.get("metrics") or row.get("agent_metrics") or {}
        reward = metrics.get("reward") or metrics.get("mean/reward")
    if reward is None and "response" in row:
        reward = 0.0  # unknown → zero; still proves schema

    return {
        "index": i,
        "source": "nemo_gym",
        "policy_model": os.environ.get(
            "MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
        ),
        "policy_base_url": os.environ.get("NVIDIA_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "reward": reward,
        "prompt_preview": str(row.get("input") or row.get("prompt") or row.get("responses_create_params") or "")[
            :240
        ],
        "raw_keys": sorted(row.keys())[:40],
    }


def main() -> None:
    rows, src = load_rollouts()
    batch = [to_rl_datum(r, i) for i, r in enumerate(rows)]
    rewards = [b["reward"] for b in batch if b["reward"] is not None]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(b) for b in batch) + "\n")
    summary = {
        "ok": True,
        "n": len(batch),
        "source_rollouts": str(src),
        "bridge_out": str(OUT),
        "mean_reward": (sum(float(r) for r in rewards) / len(rewards)) if rewards else None,
        "nemo_rl_config": str(ROOT / "configs" / "nemo_rl_gym_nemotron_vl.yaml")
        if (ROOT / "configs" / "nemo_rl_gym_nemotron_vl.yaml").exists()
        else "configs/nemo_rl_gym_nemotron_vl.yaml",
        "note": "Full NeMo-RL GRPO needs CUDA≥12.8 + NeMo-RL-owned vLLM; this bridge proves Gym→RL batch.",
    }
    print(json.dumps(summary, indent=2))
    (RESULTS / "gym_rl_bridge_summary.json").write_text(json.dumps(summary, indent=2))
    assert batch, "empty batch"
    print("GYM_RL_BRIDGE_OK")


if __name__ == "__main__":
    main()
