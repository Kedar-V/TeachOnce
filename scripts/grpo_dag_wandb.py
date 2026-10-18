#!/usr/bin/env python3
"""GRPO on TeachOnce NeMo-RL GT (video→DAG) with W&B curves.

# ponytail: text+events proxy for video; batched generate (B prompts × G samples).
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
except ImportError:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

MODEL = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GT_PATH = Path(os.environ.get("GRPO_GT", str(ROOT / "data/generated/nemo_rl/gt.jsonl")))
STEPS = int(os.environ.get("GRPO_STEPS", "25"))
GROUP = int(os.environ.get("GRPO_GROUP", "4"))  # samples per prompt (batched via num_return_sequences)
BATCH = int(os.environ.get("GRPO_BATCH", "4"))  # prompts per optimizer step
MAX_NEW = int(os.environ.get("GRPO_MAX_NEW", "256"))
MAX_SAMPLES = int(os.environ.get("GRPO_MAX_SAMPLES", "50"))
LR = float(os.environ.get("GRPO_LR", "2e-5"))
SEED = int(os.environ.get("GRPO_SEED", "42"))
KL_COEF = float(os.environ.get("GRPO_KL_COEF", "0.02"))
USE_LORA = os.environ.get("GRPO_LORA", "1") not in ("0", "false", "False")
CKPT_DIR = Path(os.environ.get("GRPO_CKPT", str(ROOT / "checkpoints/grpo_dag")))


def load_gt(path: Path, max_samples: int) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty GT: {path}")
    if max_samples > 0 and len(rows) > max_samples:
        rows = rows[:max_samples]
    return rows


def build_prompt(gt: dict) -> str:
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
        f"filmed_ops={gt.get('filmed_ops')}\n\n"
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


def leave_one_out_adv(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: (G,) → advantages."""
    g = rewards.numel()
    adv = torch.empty_like(rewards)
    for i in range(g):
        baseline = (rewards.sum() - rewards[i]) / max(g - 1, 1)
        adv[i] = rewards[i] - baseline
    if float(adv.std()) < 1e-6:
        return rewards
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def seq_logprob(model, ids: torch.Tensor, prompt_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean token logprob on completion + mask sum."""
    labels = ids.clone()
    labels[:, :prompt_len] = -100
    logits = model(ids).logits[:, :-1]
    logp = F.log_softmax(logits, dim=-1)
    tgt = ids[:, 1:]
    mask = labels[:, 1:] != -100
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
    return token_logp.sum() / mask.sum().clamp_min(1), mask


def main() -> None:
    assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY"
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "need GPU"

    gts = load_gt(GT_PATH, MAX_SAMPLES)
    project = os.environ.get("WANDB_PROJECT", "teachonce-nemo-rl")
    name = os.environ.get("WANDB_NAME", "grpo-dag-fast50")
    run = wandb.init(
        project=project,
        name=name,
        entity=os.environ.get("WANDB_ENTITY") or None,
        config={
            "model": MODEL,
            "gt_path": str(GT_PATH),
            "n_gt": len(gts),
            "max_samples": MAX_SAMPLES,
            "steps": STEPS,
            "group": GROUP,
            "batch": BATCH,
            "max_new": MAX_NEW,
            "lr": LR,
            "kl_coef": KL_COEF,
            "lora": USE_LORA,
            "algo": "grpo-loo-dag-batched",
            "reward": "scripts.nemo_rl_gt.dag_reward",
        },
    )
    print("WANDB_RUN_ID=", run.id)
    print("WANDB_URL=", run.url)
    print(f"GT={GT_PATH} n={len(gts)} batch={BATCH} group={GROUP} max_new={MAX_NEW} steps={STEPS}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # needed for batched generate

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model = attach_lora(model)
    model.train()

    ref = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    order = list(range(len(gts)))
    random.shuffle(order)
    cursor = 0

    pbar = tqdm(range(STEPS), desc="GRPO", unit="step", dynamic_ncols=True)
    for step in pbar:
        t0 = time.time()
        batch_idx = []
        for _ in range(BATCH):
            batch_idx.append(order[cursor % len(order)])
            cursor += 1
        batch_gts = [gts[i] for i in batch_idx]
        prompts = [build_prompt(g) for g in batch_gts]

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        # left-pad → prompt lens from attention mask
        prompt_lens = enc.attention_mask.sum(dim=1).tolist()
        pbar.set_postfix_str(
            f"phase=rollout B={BATCH}×G={GROUP} ids={[g['id'][-12:] for g in batch_gts]}"
        )

        t_gen = time.time()
        with torch.no_grad():
            # one batched decode: B prompts → B*GROUP sequences
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=GROUP,
                pad_token_id=tok.pad_token_id,
            )
        gen_s = time.time() - t_gen
        # out: (B*GROUP, seq)

        all_rewards, all_parts, all_losses, all_kls = [], [], [], []
        for bi, gt in enumerate(batch_gts):
            pl = int(prompt_lens[bi])
            chunk = out[bi * GROUP : (bi + 1) * GROUP]
            rewards, parts, gens = [], [], []
            for gi in range(GROUP):
                seq = chunk[gi]
                # strip left pad in prompt region for decode: use pl from original enc width
                # generated ids include left-padded prompt; completion starts after full enc width
                enc_width = enc.input_ids.shape[1]
                text = tok.decode(seq[enc_width:], skip_special_tokens=True)
                sc = score(text, gt)
                rewards.append(float(sc["reward"]))
                parts.append(sc)
                gens.append(seq)
                tqdm.write(
                    f"step={step} b={bi} gen={gi+1}/{GROUP} "
                    f"R={sc['reward']:.3f} schema={sc.get('schema_ok', 0)} "
                    f"id={gt['id'][-16:]}",
                    file=sys.stderr,
                )

            r = torch.tensor(rewards, device=device, dtype=torch.float32)
            adv = leave_one_out_adv(r)
            all_rewards.extend(rewards)
            all_parts.extend(parts)

            for i, seq in enumerate(gens):
                ids = seq.unsqueeze(0)
                # prompt_len for loss mask: left-padded so real tokens end at enc_width;
                # mask everything before last `pl` prompt tokens… simpler: mask first enc_width
                seq_lp, _ = seq_logprob(model, ids, enc_width)
                with torch.no_grad():
                    ref_lp, _ = seq_logprob(ref, ids, enc_width)
                kl = (seq_lp.detach() - ref_lp).clamp_min(0)
                all_losses.append(-adv[i] * seq_lp + KL_COEF * kl)
                all_kls.append(float(kl))

        loss = torch.stack(all_losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        r_t = torch.tensor(all_rewards, dtype=torch.float32)
        schema_m = sum(p.get("schema_ok", 0) for p in all_parts) / len(all_parts)
        crit_m = sum(p.get("critical_ops", 0) for p in all_parts) / len(all_parts)
        branch_m = sum(p.get("taken_branch", 0) for p in all_parts) / len(all_parts)
        elapsed = time.time() - t0
        r_mean = float(r_t.mean())
        # one commit per GRPO step so train/* actually lands on the chart
        wandb.log(
            {
                "train/loss": float(loss.detach()),
                "train/reward_mean": r_mean,
                "train/reward_std": float(r_t.std(unbiased=False)),
                "train/reward_max": float(r_t.max()),
                "train/schema_ok": schema_m,
                "train/critical_ops": crit_m,
                "train/taken_branch": branch_m,
                "train/kl": sum(all_kls) / max(len(all_kls), 1),
                "train/batch": BATCH,
                "train/group": GROUP,
                "perf/gen_seconds": gen_s,
                "perf/step_seconds": elapsed,
                "data/n_gt": len(gts),
            },
            step=step,
        )
        pbar.set_postfix(
            loss=f"{float(loss):.3f}",
            R=f"{r_mean:.2f}",
            schema=f"{schema_m:.2f}",
            gen_s=f"{gen_s:.0f}",
            s=f"{elapsed:.0f}",
        )

        if (step + 1) % 10 == 0 or step + 1 == STEPS:
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            out_dir = CKPT_DIR / f"step_{step + 1}"
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            tqdm.write(f"ckpt={out_dir}")

    wandb.finish()
    print("DONE", run.url)


if __name__ == "__main__":
    main()
