#!/usr/bin/env python3
"""Simple+fast RL bridge: vLLM guided_json rollouts → reward-weighted SFT (RFT).

Schema enforced at decode (same guided_schema as baseline). RL signal = ops/branch.
Full NeMo-RL weight sync is the long-term path; this fits one H100 tonight.

Phases:
  1) rollout  — POST /v1/chat/completions + guided_json (G samples / prompt)
  2) train    — LoRA CE on completions, weight ∝ leave-one-out advantage (R>0 groups)

# ponytail: text+events prompts (no video) for speed; holdout eval stays video on A100.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import requests
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.baseline_vllm_dag import guided_schema  # noqa: E402
from scripts.nemo_rl_gt import dag_reward  # noqa: E402

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise SystemExit("need peft") from e

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

MODEL = os.environ.get("RL_MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
GT = Path(os.environ.get("RL_GT", str(ROOT / "data/generated/nemo_rl/gt_train_balanced50.jsonl")))
HOLDOUT = Path(
    os.environ.get("RL_HOLDOUT", str(ROOT / "data/generated/nemo_rl/gt_holdout_balanced24.jsonl"))
)
BASE_URL = os.environ.get("NVIDIA_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("LOCAL_VLLM_API_KEY", "teachonce-local-key")
GROUP = int(os.environ.get("RL_GROUP", "4"))
MAX_NEW = int(os.environ.get("RL_MAX_NEW", "1024"))
ROUNDS = int(os.environ.get("RL_ROUNDS", "3"))  # outer collect→train cycles
MAX_PROMPTS = int(os.environ.get("RL_MAX_PROMPTS", "50"))
LR = float(os.environ.get("RL_LR", "2e-5"))
SEED = int(os.environ.get("RL_SEED", "42"))
CKPT = Path(os.environ.get("RL_CKPT", str(ROOT / "checkpoints/rl_vllm_guided_rft")))
TEMP = float(os.environ.get("RL_TEMP", "0.7"))
PHASE = os.environ.get("RL_PHASE", "all")  # rollout | train | all


def load_rows(path: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if n > 0:
        rows = rows[:n]
    return rows


def build_prompt(gt: dict) -> str:
    events = gt.get("events") or []
    lines = []
    for e in events:
        if not isinstance(e, dict):
            continue
        op = e.get("op") or "?"
        lines.append(f"- t={e.get('t_start')}-{e.get('t_end')}s  {op}")
    timeline = "\n".join(lines) if lines else "(no events)"
    base = (gt.get("prompt") or "").rstrip()
    return (
        f"{base}\n\nVideo id: {gt.get('video_id')}\n"
        f"filmed_ops={gt.get('filmed_ops')}\n\n"
        f"## Filmed demonstration\n{timeline}\n\n"
        "Emit ONLY a single JSON object (the DAG)."
    )


def vllm_ready() -> bool:
    try:
        r = requests.get(
            f"{BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
        )
        return r.status_code < 400
    except Exception:
        return False


def generate_guided(prompt: str, *, n: int) -> list[str]:
    schema = guided_schema()
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    # one request with n completions when supported; else loop
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW,
        "temperature": TEMP,
        "n": n,
        "guided_json": schema,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    if r.status_code >= 400:
        # fallback: structured_outputs + single n=1 loop
        texts = []
        for _ in range(n):
            p2 = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_NEW,
                "temperature": TEMP,
                "guided_json": schema,
            }
            r2 = requests.post(url, headers=headers, json=p2, timeout=300)
            r2.raise_for_status()
            texts.append(r2.json()["choices"][0]["message"]["content"] or "")
        return texts
    data = r.json()
    return [(c.get("message") or {}).get("content") or "" for c in data["choices"]]


def score(text: str, gt: dict) -> dict:
    try:
        return dag_reward(text, gt)
    except Exception:
        return {"reward": 0.0, "schema_ok": 0.0, "critical_ops": 0.0, "taken_branch": 0.0}


def leave_one_out_adv(rewards: list[float]) -> list[float]:
    g = len(rewards)
    if g < 2:
        return [0.0] * g
    out = []
    for i, r in enumerate(rewards):
        base = (sum(rewards) - r) / (g - 1)
        out.append(r - base)
    mean = sum(out) / g
    var = sum((x - mean) ** 2 for x in out) / g
    if var < 1e-8:
        return out
    std = var**0.5
    return [(x - mean) / (std + 1e-8) for x in out]


def phase_rollout(gts: list[dict], out_path: Path) -> list[dict]:
    assert vllm_ready(), f"vLLM not ready at {BASE_URL}"
    rows = []
    for gt in tqdm(gts, desc="rollout"):
        prompt = build_prompt(gt)
        texts = generate_guided(prompt, n=GROUP)
        rewards, parts = [], []
        for t in texts:
            sc = score(t, gt)
            rewards.append(float(sc["reward"]))
            parts.append(sc)
        adv = leave_one_out_adv(rewards)
        for t, sc, a in zip(texts, parts, adv):
            rows.append(
                {
                    "video_id": gt.get("video_id"),
                    "id": gt.get("id"),
                    "prompt": prompt,
                    "completion": t,
                    "reward": sc["reward"],
                    "schema_ok": sc.get("schema_ok", 0),
                    "critical_ops": sc.get("critical_ops", 0),
                    "taken_branch": sc.get("taken_branch", 0),
                    "advantage": a,
                }
            )
        tqdm.write(
            f"id={gt.get('video_id')} R={[round(r,3) for r in rewards]} "
            f"schema={[p.get('schema_ok') for p in parts]}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n = max(len(rows), 1)
    summary = {
        "n": len(rows),
        "mean_reward": round(sum(r["reward"] for r in rows) / n, 4),
        "mean_schema_ok": round(sum(r["schema_ok"] for r in rows) / n, 4),
        "mean_critical_ops": round(sum(float(r["critical_ops"]) for r in rows) / n, 4),
        "mean_taken_branch": round(sum(float(r["taken_branch"]) for r in rows) / n, 4),
    }
    print("ROLLOUT", json.dumps(summary), flush=True)
    (out_path.parent / "rollout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb:
        wandb.log({f"rollout/{k}": v for k, v in summary.items()})
    return rows


def phase_train(rollouts: list[dict]) -> None:
    # keep positive-advantage or above-mean reward completions
    usable = [r for r in rollouts if r.get("advantage", 0) > 0 and r.get("completion")]
    if not usable:
        usable = [r for r in rollouts if float(r.get("reward") or 0) >= 0.4 and r.get("completion")]
    if not usable:
        print("train skip: no usable rollouts", flush=True)
        return

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # language_model path — VL.forward needs pixel_values
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    def lm(m):
        core = m.get_base_model() if hasattr(m, "get_base_model") else m
        if hasattr(core, "model") and hasattr(core.model, "language_model"):
            return core.model.language_model
        if hasattr(core, "language_model"):
            return core.language_model
        return core

    language = lm(model)
    random.shuffle(usable)
    losses = []
    model.train()
    for row in tqdm(usable, desc="rft"):
        text = row["prompt"] + "\n" + row["completion"]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        prompt_ids = tok(row["prompt"], return_tensors="pt", truncation=True, max_length=1536)
        pl = int(prompt_ids.input_ids.shape[1])
        labels = enc.input_ids.clone()
        labels[:, :pl] = -100
        w = max(float(row.get("advantage") or 0), 0.1)
        out = language(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
        loss = out.loss * w
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if wandb:
            wandb.log({"train/loss": losses[-1], "train/weight": w})

    CKPT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT / "lora")
    tok.save_pretrained(CKPT / "lora")
    meta = {"n_train": len(usable), "mean_loss": round(sum(losses) / max(len(losses), 1), 4)}
    (CKPT / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("TRAIN", json.dumps(meta), flush=True)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    gts = load_rows(GT, MAX_PROMPTS)
    if wandb and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "teachonce-nemo-rl"),
            name=os.environ.get("WANDB_NAME", "rl-vllm-guided-rft"),
            config={"gt": str(GT), "group": GROUP, "rounds": ROUNDS, "model": MODEL},
        )
    print(f"phase={PHASE} n_gt={len(gts)} group={GROUP} base={BASE_URL}", flush=True)

    rollout_path = CKPT / "rollouts.jsonl"
    for rd in range(ROUNDS if PHASE == "all" else 1):
        print(f"=== round {rd+1} ===", flush=True)
        if PHASE in ("all", "rollout"):
            rows = phase_rollout(gts, rollout_path)
        else:
            rows = [json.loads(l) for l in rollout_path.read_text().splitlines() if l.strip()]
        if PHASE in ("all", "train"):
            # free GPU from anything else before HF train — caller should stop vLLM
            phase_train(rows)

    if wandb:
        wandb.finish()
    print("DONE", CKPT)


if __name__ == "__main__":
    main()
