#!/usr/bin/env python3
"""10-step GRPO-style closed loop on Brev (CUDA 12.x driver).

NeMo-RL main/v0.5+ need torch cu128–cu130; this Brev image is driver 565
(CUDA 12.7 max). This smoke uses the same closed-loop shape
(rollout → reward → policy loss → update) with HF generate + verifiable
GSM8K reward, and logs every step to W&B for live notebook polling.

# ponytail: not full NeMo-RL; swap to `run_nemo_rl_smoke.sh` when driver ≥575.
"""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass

import torch
import wandb
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
STEPS = int(os.environ.get("GRPO_STEPS", "10"))
GROUP = int(os.environ.get("GRPO_GROUP", "4"))  # gens per prompt
MAX_NEW = int(os.environ.get("GRPO_MAX_NEW", "128"))
LR = float(os.environ.get("GRPO_LR", "5e-6"))
SEED = int(os.environ.get("GRPO_SEED", "42"))


def extract_answer(text: str) -> str:
    m = re.findall(r"####\s*(-?[0-9][0-9,\.]*)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def reward(pred: str, gold: str) -> float:
    # partial credit so LOO advantages aren't all-zero on a tiny model
    r = 0.0
    if "####" in pred:
        r += 0.25
    try:
        pa, ga = extract_answer(pred), extract_answer(str(gold))
        if pa and ga and float(pa) == float(ga):
            return 1.0
        if pa:
            r += 0.25  # produced a number
    except Exception:
        pass
    return r


@dataclass
class StepMetrics:
    step: int
    loss: float
    reward_mean: float
    reward_std: float
    kl: float
    elapsed_s: float


def main() -> None:
    assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY"
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "need GPU"

    project = os.environ.get("WANDB_PROJECT", "teachonce-nemo-rl")
    name = os.environ.get("WANDB_NAME", "grpo-smoke-10")
    run = wandb.init(
        project=project,
        name=name,
        entity=os.environ.get("WANDB_ENTITY") or None,
        config={
            "model": MODEL,
            "steps": STEPS,
            "group": GROUP,
            "max_new": MAX_NEW,
            "lr": LR,
            "algo": "grpo-loo-smoke",
            "stack": "hf-cu124-compat",
        },
    )
    print("WANDB_RUN_ID=", run.id)
    print("WANDB_URL=", run.url)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.train()
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    for step in range(STEPS):
        t0 = time.time()
        ex = ds[idxs[step % len(idxs)]]
        prompt = (
            "Solve the grade-school math problem. Put the final answer after ####.\n\n"
            f"Problem: {ex['question']}\nAnswer:"
        )
        gold = ex["answer"]

        enc = tok(prompt, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

        gens, rewards = [], []
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
                gens.append(out[0])
                rewards.append(reward(text, gold))

        r = torch.tensor(rewards, device=device, dtype=torch.float32)
        # leave-one-out baseline (GRPO-style); if all equal, use raw reward
        # so the policy still gets a non-zero learning signal
        adv = []
        for i in range(GROUP):
            baseline = (r.sum() - r[i]) / max(GROUP - 1, 1)
            adv.append(r[i] - baseline)
        adv_t = torch.stack(adv)
        if float(adv_t.std()) < 1e-6:
            adv_t = r
        elif float(adv_t.std()) > 1e-6:
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
            token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            token_logp = token_logp * mask
            seq_logp = token_logp.sum() / mask.sum().clamp_min(1)

            with torch.no_grad():
                rlog = F.log_softmax(ref(ids).logits[:, :-1], dim=-1)
                ref_lp = rlog.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
                ref_seq = ref_lp.sum() / mask.sum().clamp_min(1)
            kl = (seq_logp.detach() - ref_seq).clamp_min(0)
            losses.append(-adv_t[i] * seq_logp + 0.01 * kl)
            kls.append(float(kl))

        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        m = StepMetrics(
            step=step,
            loss=float(loss.detach()),
            reward_mean=float(r.mean()),
            reward_std=float(r.std(unbiased=False)),
            kl=sum(kls) / len(kls),
            elapsed_s=time.time() - t0,
        )
        wandb.log(
            {
                "train/loss": m.loss,
                "train/reward_mean": m.reward_mean,
                "train/reward_std": m.reward_std,
                "train/kl": m.kl,
                "perf/step_seconds": m.elapsed_s,
            },
            step=step,
        )
        print(
            f"step={m.step} loss={m.loss:.4f} reward={m.reward_mean:.3f} "
            f"kl={m.kl:.4f} {m.elapsed_s:.1f}s",
            flush=True,
        )

    wandb.finish()
    print("DONE", run.url)


if __name__ == "__main__":
    main()
