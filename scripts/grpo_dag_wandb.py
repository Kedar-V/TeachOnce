#!/usr/bin/env python3
"""GRPO on TeachOnce NeMo-RL GT (video→DAG) with W&B curves.

Uses data/generated/nemo_rl/gt.jsonl + scripts.nemo_rl_gt.dag_reward.
Same closed loop as grpo_smoke_wandb.py (rollout → LOO advantage → update),
but the task/reward are the real DAG ones.

# ponytail: text+events proxy for video (Brev CUDA 12.7 can't run full
# NeMo-RL VL GRPO). Swap prompt builder to multimodal when driver ≥575.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import wandb
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import dag_reward  # noqa: E402

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # optional; full FT if peft missing
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

MODEL = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GT_PATH = Path(os.environ.get("GRPO_GT", str(ROOT / "data/generated/nemo_rl/gt.jsonl")))
STEPS = int(os.environ.get("GRPO_STEPS", "50"))
GROUP = int(os.environ.get("GRPO_GROUP", "4"))
MAX_NEW = int(os.environ.get("GRPO_MAX_NEW", "1024"))
LR = float(os.environ.get("GRPO_LR", "2e-5"))
SEED = int(os.environ.get("GRPO_SEED", "42"))
KL_COEF = float(os.environ.get("GRPO_KL_COEF", "0.02"))
USE_LORA = os.environ.get("GRPO_LORA", "1") not in ("0", "false", "False")
CKPT_DIR = Path(os.environ.get("GRPO_CKPT", str(ROOT / "checkpoints/grpo_dag")))


def load_gt(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty GT: {path}")
    return rows


def build_prompt(gt: dict) -> str:
    """Text stand-in for video: GT prompt + filmed event timeline."""
    events = gt.get("events") or []
    lines = []
    for e in events:
        op = e.get("op") or "?"
        label = e.get("label") or ""
        obs = e.get("observed") or {}
        t0, t1 = e.get("t_start"), e.get("t_end")
        obs_s = ", ".join(f"{k}={v}" for k, v in obs.items()) if obs else ""
        bit = f"- t={t0}-{t1}s  {op}"
        if label:
            bit += f" ({label})"
        if obs_s:
            bit += f"  [{obs_s}]"
        lines.append(bit)
    timeline = "\n".join(lines) if lines else "(no aligned events; infer from goal)"
    return (
        f"{gt['prompt'].rstrip()}\n\n"
        f"Video id: {gt.get('video_id')}\n"
        f"Path type / taken branch (for reward only — still emit FULL DAG with both branches):\n"
        f"  filmed_ops={gt.get('filmed_ops')}\n\n"
        f"## Filmed demonstration (time-aligned)\n{timeline}\n\n"
        "Emit ONLY a single JSON object (the DAG). No markdown fences."
    )


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def score(pred_text: str, gt: dict) -> dict:
    return dag_reward(extract_json(pred_text), gt)


def attach_lora(model):
    if not USE_LORA or get_peft_model is None:
        return model
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def main() -> None:
    assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY"
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "need GPU"

    gts = load_gt(GT_PATH)
    project = os.environ.get("WANDB_PROJECT", "teachonce-nemo-rl")
    name = os.environ.get("WANDB_NAME", "grpo-dag-25")
    run = wandb.init(
        project=project,
        name=name,
        entity=os.environ.get("WANDB_ENTITY") or None,
        config={
            "model": MODEL,
            "gt_path": str(GT_PATH),
            "n_gt": len(gts),
            "steps": STEPS,
            "group": GROUP,
            "max_new": MAX_NEW,
            "lr": LR,
            "kl_coef": KL_COEF,
            "lora": USE_LORA,
            "algo": "grpo-loo-dag",
            "reward": "scripts.nemo_rl_gt.dag_reward",
            "stack": "hf-cu124-compat",
        },
    )
    print("WANDB_RUN_ID=", run.id)
    print("WANDB_URL=", run.url)
    print("GT=", GT_PATH, "n=", len(gts))

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model = attach_lora(model)
    model.train()

    # frozen ref = base (no LoRA) for KL
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    order = list(range(len(gts)))
    random.shuffle(order)

    pbar = tqdm(range(STEPS), desc="GRPO", unit="step")
    for step in pbar:
        t0 = time.time()
        gt = gts[order[step % len(order)]]
        prompt = build_prompt(gt)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        prompt_len = enc.input_ids.shape[1]

        gens, rewards, parts = [], [], []
        with torch.no_grad():
            for _ in range(GROUP):
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
                text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
                sc = score(text, gt)
                gens.append(out[0])
                rewards.append(float(sc["reward"]))
                parts.append(sc)

        r = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = []
        for i in range(GROUP):
            baseline = (r.sum() - r[i]) / max(GROUP - 1, 1)
            adv.append(r[i] - baseline)
        adv_t = torch.stack(adv)
        if float(adv_t.std()) < 1e-6:
            adv_t = r  # all equal → use raw reward signal
        else:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        losses, kls = [], []
        for i, seq in enumerate(gens):
            ids = seq.unsqueeze(0)
            labels = ids.clone()
            labels[:, :prompt_len] = -100
            logits = model(ids).logits[:, :-1]
            logp = F.log_softmax(logits, dim=-1)
            tgt = ids[:, 1:]
            mask = labels[:, 1:] != -100
            token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
            seq_logp = token_logp.sum() / mask.sum().clamp_min(1)

            with torch.no_grad():
                rlog = F.log_softmax(ref(ids).logits[:, :-1], dim=-1)
                ref_lp = rlog.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
                ref_seq = ref_lp.sum() / mask.sum().clamp_min(1)
            kl = (seq_logp.detach() - ref_seq).clamp_min(0)
            losses.append(-adv_t[i] * seq_logp + KL_COEF * kl)
            kls.append(float(kl))

        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        schema_m = sum(p.get("schema_ok", 0) for p in parts) / GROUP
        crit_m = sum(p.get("critical_ops", 0) for p in parts) / GROUP
        branch_m = sum(p.get("taken_branch", 0) for p in parts) / GROUP
        elapsed = time.time() - t0
        r_mean = float(r.mean())
        wandb.log(
            {
                "train/loss": float(loss.detach()),
                "train/reward_mean": r_mean,
                "train/reward_std": float(r.std(unbiased=False)),
                "train/reward_max": float(r.max()),
                "train/schema_ok": schema_m,
                "train/critical_ops": crit_m,
                "train/taken_branch": branch_m,
                "train/kl": sum(kls) / len(kls),
                "train/prompt_tokens": prompt_len,
                "data/gt_id": gt["id"],
                "data/family": gt.get("family"),
                "perf/step_seconds": elapsed,
            },
            step=step,
        )
        pbar.set_postfix(
            id=gt["id"][-16:],
            loss=f"{float(loss):.3f}",
            R=f"{r_mean:.2f}",
            schema=f"{schema_m:.2f}",
            s=f"{elapsed:.0f}",
        )

        if (step + 1) % 25 == 0 or step + 1 == STEPS:
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            out = CKPT_DIR / f"step_{step + 1}"
            model.save_pretrained(out)
            tok.save_pretrained(out)
            tqdm.write(f"ckpt={out}")

    wandb.finish()
    print("DONE", run.url)


if __name__ == "__main__":
    main()
