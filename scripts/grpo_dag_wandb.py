#!/usr/bin/env python3
"""GRPO on TeachOnce NeMo-RL GT (video→DAG) with W&B train/val curves + early stop.

# ponytail: text+events proxy; LM-only forward (VL needs pixel_values).
# Slow path was max_new×G decode (~2min/step). Defaults cut that ~3×.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
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

MODEL = os.environ.get("GRPO_MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
GT_PATH = Path(os.environ.get("GRPO_GT", str(ROOT / "data/generated/nemo_rl/gt_train_balanced50.jsonl")))
HOLDOUT_PATH = Path(
    os.environ.get("GRPO_HOLDOUT", str(ROOT / "data/generated/nemo_rl/gt_holdout_balanced24.jsonl"))
)
STEPS = int(os.environ.get("GRPO_STEPS", "40"))
GROUP = int(os.environ.get("GRPO_GROUP", "4"))
BATCH = int(os.environ.get("GRPO_BATCH", "1"))
MAX_NEW = int(os.environ.get("GRPO_MAX_NEW", "256"))  # was 512 — main speed lever
MAX_SAMPLES = int(os.environ.get("GRPO_MAX_SAMPLES", "50"))
LR = float(os.environ.get("GRPO_LR", "2e-5"))
SEED = int(os.environ.get("GRPO_SEED", "42"))
KL_COEF = float(os.environ.get("GRPO_KL_COEF", "0.02"))
USE_LORA = os.environ.get("GRPO_LORA", "1") not in ("0", "false", "False")
CKPT_DIR = Path(os.environ.get("GRPO_CKPT", str(ROOT / "checkpoints/grpo_dag_balanced50")))
VAL_EVERY = int(os.environ.get("GRPO_VAL_EVERY", "5"))
HOLDOUT_EVERY = int(os.environ.get("GRPO_HOLDOUT_EVERY", "20"))  # rare; expensive
EVAL_MAX_NEW = int(os.environ.get("GRPO_EVAL_MAX_NEW", "256"))
VAL_N = int(os.environ.get("GRPO_VAL_N", "9"))  # 3 per family
PATIENCE = int(os.environ.get("GRPO_PATIENCE", "3"))  # val evals without improve
MIN_DELTA = float(os.environ.get("GRPO_MIN_DELTA", "0.01"))


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


def split_train_val(rows: list[dict], val_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Balanced val slice; rest is train. Val never used for GRPO rollouts."""
    by_fam: dict[str, list] = defaultdict(list)
    for r in rows:
        by_fam[r.get("family") or "unknown"].append(r)
    rng = random.Random(seed)
    val, train = [], []
    fams = sorted(by_fam)
    if not fams:
        return rows, []
    base, rem = divmod(min(val_n, len(rows)), len(fams))
    for i, f in enumerate(fams):
        pool = list(by_fam[f])
        rng.shuffle(pool)
        q = min(len(pool), base + (1 if i < rem else 0))
        val.extend(pool[:q])
        train.extend(pool[q:])
    # leftovers if some fams short
    taken = {id(r) for r in val}
    for r in rows:
        if id(r) not in taken and r not in train:
            train.append(r)
    return train, val


def build_prompt(gt: dict) -> str:
    events = gt.get("events") or []
    lines = []
    for e in events:
        if not isinstance(e, dict):
            continue
        op = e.get("op") or "?"
        label = e.get("label") or ""
        obs = e.get("observed") or {}
        t0, t1 = e.get("t_start"), e.get("t_end")
        obs_s = ", ".join(f"{k}={v}" for k, v in obs.items()) if isinstance(obs, dict) else ""
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
    try:
        return dag_reward(extract_json(pred_text), gt)
    except Exception as e:
        return {
            "reward": 0.0,
            "schema_ok": 0.0,
            "critical_ops": 0.0,
            "taken_branch": 0.0,
            "error": f"score_fail:{type(e).__name__}",
        }


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


def _language_model(model):
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(core, "model") and hasattr(core.model, "language_model"):
        return core.model.language_model
    if hasattr(core, "language_model"):
        return core.language_model
    return core


def leave_one_out_adv(rewards: torch.Tensor) -> torch.Tensor:
    g = rewards.numel()
    adv = torch.empty_like(rewards)
    for i in range(g):
        baseline = (rewards.sum() - rewards[i]) / max(g - 1, 1)
        adv[i] = rewards[i] - baseline
    if float(adv.std()) < 1e-6:
        return rewards
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def seq_logprob_batch(model, ids: torch.Tensor, prompt_lens: list[int]) -> torch.Tensor:
    """ids (G,T) → mean completion logprob per row (G,)."""
    lm = _language_model(model)
    logits = lm(input_ids=ids).logits[:, :-1]
    logp = F.log_softmax(logits, dim=-1)
    tgt = ids[:, 1:]
    G, Tm = tgt.shape
    mask = torch.zeros(G, Tm, device=ids.device, dtype=torch.bool)
    for i, pl in enumerate(prompt_lens):
        # completion tokens are indices >= pl in full seq → in tgt that's >= pl-1? 
        # ids layout: [pad...][prompt][completion]; left-padded so real prompt ends at enc_width.
        # We pass enc_width as prompt_len for every row in a chunk (same prompt width).
        start = max(pl - 1, 0)
        mask[i, start:] = True
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
    return token_logp.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def eval_set(model, tok, device, rows: list[dict], *, prefix: str, step: int) -> dict:
    if not rows:
        return {}
    was = model.training
    model.eval()
    lm = _language_model(model)
    scores = []
    for gt in tqdm(rows, desc=prefix, leave=False):
        enc = tok(build_prompt(gt), return_tensors="pt", truncation=True, max_length=1536).to(device)
        out = lm.generate(
            **enc,
            max_new_tokens=EVAL_MAX_NEW,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
        scores.append(score(text, gt))
    if was:
        model.train()
    n = max(len(scores), 1)
    summary = {
        "n": len(scores),
        "reward_mean": round(sum(s["reward"] for s in scores) / n, 4),
        "schema_ok": round(sum(s["schema_ok"] for s in scores) / n, 4),
        "critical_ops": round(sum(s["critical_ops"] for s in scores) / n, 4),
        "taken_branch": round(sum(s["taken_branch"] for s in scores) / n, 4),
    }
    wandb.log({f"{prefix}/{k}": v for k, v in summary.items()}, step=step)
    print(prefix.upper(), json.dumps(summary), flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / f"{prefix.replace('/', '_')}_step{step}.json").write_text(
        json.dumps({"summary": summary, "rows": scores}, indent=2) + "\n"
    )
    return summary


def main() -> None:
    assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY"
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "need GPU"

    pool = load_gt(GT_PATH, MAX_SAMPLES)
    train_rows, val_rows = split_train_val(pool, VAL_N, SEED)
    holdout = load_gt(HOLDOUT_PATH, 0) if HOLDOUT_PATH.exists() else []

    project = os.environ.get("WANDB_PROJECT", "teachonce-nemo-rl")
    name = os.environ.get("WANDB_NAME", "grpo-nemotron-balanced50-h100")
    run = wandb.init(
        project=project,
        name=name,
        entity=os.environ.get("WANDB_ENTITY") or None,
        config={
            "model": MODEL,
            "gt_path": str(GT_PATH),
            "holdout_path": str(HOLDOUT_PATH),
            "n_pool": len(pool),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_holdout": len(holdout),
            "holdout_ids": [r.get("video_id") for r in holdout],
            "steps": STEPS,
            "group": GROUP,
            "batch": BATCH,
            "max_new": MAX_NEW,
            "eval_max_new": EVAL_MAX_NEW,
            "lr": LR,
            "kl_coef": KL_COEF,
            "lora": USE_LORA,
            "val_every": VAL_EVERY,
            "holdout_every": HOLDOUT_EVERY,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "algo": "grpo-loo-dag-batched",
        },
    )
    print("WANDB_RUN_ID=", run.id)
    print("WANDB_URL=", run.url)
    print(
        f"train={len(train_rows)} val={len(val_rows)} holdout={len(holdout)} "
        f"B={BATCH} G={GROUP} max_new={MAX_NEW} steps={STEPS} patience={PATIENCE}"
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model = attach_lora(model)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0
    best_val = -1.0
    bad = 0
    best_dir = CKPT_DIR / "best"

    # baseline curves at step 0
    if val_rows:
        s0 = eval_set(model, tok, device, val_rows, prefix="val", step=0)
        best_val = float(s0.get("reward_mean", 0.0))

    pbar = tqdm(range(STEPS), desc="GRPO", unit="step", dynamic_ncols=True)
    stopped_early = False
    for step in pbar:
        t0 = time.time()
        batch_idx = []
        for _ in range(BATCH):
            batch_idx.append(order[cursor % len(order)])
            cursor += 1
        batch_gts = [train_rows[i] for i in batch_idx]
        prompts = [build_prompt(g) for g in batch_gts]

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(device)
        enc_width = enc.input_ids.shape[1]
        pbar.set_postfix_str(f"rollout {[g.get('video_id','')[-10:] for g in batch_gts]}")

        t_gen = time.time()
        with torch.no_grad():
            lm = _language_model(model)
            out = lm.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=GROUP,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_s = time.time() - t_gen

        all_rewards, all_parts, all_losses, all_kls = [], [], [], []
        for bi, gt in enumerate(batch_gts):
            chunk = out[bi * GROUP : (bi + 1) * GROUP]
            rewards, parts = [], []
            for gi in range(GROUP):
                text = tok.decode(chunk[gi, enc_width:], skip_special_tokens=True)
                sc = score(text, gt)
                rewards.append(float(sc["reward"]))
                parts.append(sc)
            r = torch.tensor(rewards, device=device, dtype=torch.float32)
            adv = leave_one_out_adv(r)
            all_rewards.extend(rewards)
            all_parts.extend(parts)

            # one batched forward for G completions (policy + ref)
            ids = chunk  # (G, T)
            plens = [enc_width] * GROUP
            seq_lp = seq_logprob_batch(model, ids, plens)
            with torch.no_grad():
                if hasattr(model, "disable_adapter"):
                    with model.disable_adapter():
                        ref_lp = seq_logprob_batch(model, ids, plens)
                else:
                    ref_lp = seq_lp.detach()
            kl = (seq_lp.detach() - ref_lp).clamp_min(0)
            for i in range(GROUP):
                all_losses.append(-adv[i] * seq_lp[i] + KL_COEF * kl[i])
                all_kls.append(float(kl[i]))

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
                "perf/gen_seconds": gen_s,
                "perf/step_seconds": elapsed,
            },
            step=step + 1,
        )
        pbar.set_postfix(loss=f"{float(loss):.3f}", R=f"{r_mean:.2f}", schema=f"{schema_m:.2f}", gen_s=f"{gen_s:.0f}")

        # --- val curve + early stop ---
        if val_rows and ((step + 1) % VAL_EVERY == 0 or step + 1 == STEPS):
            vs = eval_set(model, tok, device, val_rows, prefix="val", step=step + 1)
            vr = float(vs.get("reward_mean", 0.0))
            if vr >= best_val + MIN_DELTA:
                best_val = vr
                bad = 0
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                tok.save_pretrained(best_dir)
                (best_dir / "best_meta.json").write_text(
                    json.dumps({"step": step + 1, "val_reward": best_val}, indent=2) + "\n"
                )
                tqdm.write(f"best val={best_val:.3f} @ step {step+1} → {best_dir}")
            else:
                bad += 1
                tqdm.write(f"val={vr:.3f} no improve ({bad}/{PATIENCE})")
                if bad >= PATIENCE:
                    tqdm.write(f"early stop at step {step+1} best_val={best_val:.3f}")
                    stopped_early = True
                    wandb.log({"early_stop/step": step + 1, "early_stop/best_val": best_val}, step=step + 1)
                    break

        if holdout and ((step + 1) % HOLDOUT_EVERY == 0):
            eval_set(model, tok, device, holdout, prefix="holdout", step=step + 1)

        if (step + 1) % 10 == 0:
            out_dir = CKPT_DIR / f"step_{step + 1}"
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)

    final_step = step + 1
    if holdout:
        eval_set(model, tok, device, holdout, prefix="holdout_final", step=final_step)

    wandb.summary["best_val_reward"] = best_val
    wandb.summary["stopped_early"] = stopped_early
    wandb.summary["final_step"] = final_step
    wandb.finish()
    print("DONE", run.url, "best_val", best_val, "early" if stopped_early else "full")


if __name__ == "__main__":
    main()
