#!/usr/bin/env python3
"""SFT LoRA: prompt(+events) → gold DAG JSON. Teaches schema before GRPO.

Batched training, W&B metrics, tqdm epoch/batch/eval bars.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import dag_reward  # noqa: E402

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise SystemExit("pip install peft") from e

MODEL = os.environ.get("SFT_MODEL", os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
GT_PATH = Path(os.environ.get("SFT_GT", os.environ.get("GRPO_GT", str(ROOT / "data/generated/nemo_rl/gt.jsonl"))))
MAX_SAMPLES = int(os.environ.get("SFT_MAX_SAMPLES", "50"))
EPOCHS = int(os.environ.get("SFT_EPOCHS", "5"))
BATCH = int(os.environ.get("SFT_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "1"))
LR = float(os.environ.get("SFT_LR", "2e-4"))
MAX_LEN = int(os.environ.get("SFT_MAX_LEN", "2048"))
SEED = int(os.environ.get("SFT_SEED", "42"))
EVAL_EVERY = int(os.environ.get("SFT_EVAL_EVERY", "10"))  # optimizer steps
CKPT_DIR = Path(os.environ.get("SFT_CKPT", str(ROOT / "checkpoints/sft_dag")))


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
    import re

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


class DagSFTDataset(Dataset):
    def __init__(self, rows: list[dict], tok, max_len: int):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        gt = self.rows[i]
        prompt = build_prompt(gt)
        # chat-ish: prompt then gold JSON only
        target = gt.get("gold_dag_json") or json.dumps(gt["gold_dag"], separators=(",", ":"))
        full = prompt + "\n" + target
        enc = self.tok(
            full,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        prompt_ids = self.tok(prompt + "\n", add_special_tokens=True, truncation=True, max_length=self.max_len)[
            "input_ids"
        ]
        input_ids = enc["input_ids"]
        labels = list(input_ids)
        pl = min(len(prompt_ids), len(labels))
        for j in range(pl):
            labels[j] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "gt": gt,
            "prompt": prompt,
        }


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "gts": [b["gt"] for b in batch],
        "prompts": [b["prompt"] for b in batch],
    }


def load_rows(path: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"empty {path}")
    if n > 0:
        rows = rows[:n]
    return rows


def attach_lora(model):
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


@torch.no_grad()
def eval_schema(model, tok, rows: list[dict], device, n: int = 8) -> dict:
    model.eval()
    sample = rows[:n]
    rewards, schemas, crits, branches = [], [], [], []
    bar = tqdm(sample, desc="  eval", leave=False, dynamic_ncols=True)
    for gt in bar:
        prompt = build_prompt(gt)
        enc = tok(prompt + "\n", return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        text = tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
        sc = dag_reward(extract_json(text), gt)
        rewards.append(sc["reward"])
        schemas.append(sc["schema_ok"])
        crits.append(sc["critical_ops"])
        branches.append(sc["taken_branch"])
        bar.set_postfix(R=f"{sc['reward']:.2f}", schema=sc["schema_ok"])
    model.train()
    return {
        "eval/reward_mean": sum(rewards) / len(rewards),
        "eval/schema_ok": sum(schemas) / len(schemas),
        "eval/critical_ops": sum(crits) / len(crits),
        "eval/taken_branch": sum(branches) / len(branches),
    }


def main() -> None:
    assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY"
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "need GPU"

    rows = load_rows(GT_PATH, MAX_SAMPLES)
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "teachonce-nemo-rl"),
        name=os.environ.get("WANDB_NAME", "sft-dag-50"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        config={
            "model": MODEL,
            "n_gt": len(rows),
            "epochs": EPOCHS,
            "batch": BATCH,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "max_len": MAX_LEN,
            "phase": "sft",
        },
    )
    print("WANDB_RUN_ID=", run.id)
    print("WANDB_URL=", run.url)
    print(f"SFT n={len(rows)} batch={BATCH} epochs={EPOCHS} model={MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model = attach_lora(model)
    model.train()
    # larger batch → more GPU use
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    ds = DagSFTDataset(rows, tok, MAX_LEN)
    loader = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    total_steps = max(1, (len(loader) * EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM)
    sched = get_linear_schedule_with_warmup(opt, max(1, total_steps // 10), total_steps)

    global_step = 0
    epoch_bar = tqdm(range(EPOCHS), desc="SFT epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        batch_bar = tqdm(loader, desc=f"  epoch {epoch+1}/{EPOCHS}", leave=False, dynamic_ncols=True)
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(batch_bar):
            t0 = time.time()
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            do_step = (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(loader)
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                wandb.log(
                    {
                        "train/loss": float(out.loss.detach()),
                        "train/lr": sched.get_last_lr()[0],
                        "train/epoch": epoch + (bi + 1) / len(loader),
                        "train/tokens": int(mask.sum()),
                        "perf/step_seconds": time.time() - t0,
                        "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                    },
                    step=global_step,
                )
                batch_bar.set_postfix(
                    loss=f"{float(out.loss):.3f}",
                    step=global_step,
                    tok=int(mask.sum()),
                )

                if global_step % EVAL_EVERY == 0 or (epoch == EPOCHS - 1 and bi + 1 == len(loader)):
                    metrics = eval_schema(model, tok, rows, device, n=min(8, len(rows)))
                    wandb.log(metrics, step=global_step)
                    tqdm.write(
                        f"eval step={global_step} schema={metrics['eval/schema_ok']:.2f} "
                        f"R={metrics['eval/reward_mean']:.3f}",
                        file=sys.stderr,
                    )

        epoch_bar.set_postfix(step=global_step)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = CKPT_DIR / f"epoch_{epoch+1}"
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        tqdm.write(f"ckpt={out_dir}")

    # final eval
    metrics = eval_schema(model, tok, rows, device, n=min(16, len(rows)))
    wandb.log({f"final/{k.split('/', 1)[-1]}": v for k, v in metrics.items()})
    print("FINAL", metrics)
    wandb.finish()
    print("DONE", run.url)


if __name__ == "__main__":
    main()
