#!/usr/bin/env python3
"""SFT LoRA on NVIDIA-Nemotron-Nano-12B-v2-VL with real video frames → gold DAG.

Uses GT video.mp4 (not text-only Qwen). Batch=1 + grad accum for A100 80GB.
W&B + tqdm. Eval logs schema_ok / reward.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import dag_reward  # noqa: E402

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise SystemExit("pip install peft") from e

try:
    import cv2
except ImportError as e:
    raise SystemExit("pip install opencv-python-headless") from e

MODEL = os.environ.get("SFT_MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
GT_PATH = Path(os.environ.get("SFT_GT", str(ROOT / "data/generated/nemo_rl/gt.jsonl")))
VIDEO_ROOT = Path(os.environ.get("SFT_VIDEO_ROOT", str(ROOT / "data/generated/demonstrations/videos")))
MAX_SAMPLES = int(os.environ.get("SFT_MAX_SAMPLES", "50"))
EPOCHS = int(os.environ.get("SFT_EPOCHS", "3"))
BATCH = int(os.environ.get("SFT_BATCH", "1"))  # video VL — keep 1 on 1×A100
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "8"))
LR = float(os.environ.get("SFT_LR", "1e-4"))
N_FRAMES = int(os.environ.get("SFT_N_FRAMES", "8"))
MAX_NEW_EVAL = int(os.environ.get("SFT_MAX_NEW_EVAL", "512"))
SEED = int(os.environ.get("SFT_SEED", "42"))
EVAL_EVERY = int(os.environ.get("SFT_EVAL_EVERY", "10"))
CKPT_DIR = Path(os.environ.get("SFT_CKPT", str(ROOT / "checkpoints/sft_nemotron_vl")))
VIDEO_FPS_META = float(os.environ.get("SFT_VIDEO_FPS", "1.0"))


def resolve_video(gt: dict) -> Path:
    raw = Path(gt.get("video_path") or "")
    if raw.exists():
        return raw
    vid = gt.get("video_id") or raw.parent.name
    alt = VIDEO_ROOT / vid / "video.mp4"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"video missing for {gt.get('id')}: tried {raw} and {alt}")


def sample_frames(video_path: Path, n: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(total - 1, 0), num=min(n, total)).astype(int)
    frames: list[Image.Image] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"need ≥2 frames from {video_path}, got {len(frames)}")
    # model card: min 8 frames preferred — pad by repeating last
    while len(frames) < min(8, n):
        frames.append(frames[-1])
    return frames[:n]


def user_text(gt: dict) -> str:
    return (
        f"{gt['prompt'].rstrip()}\n\n"
        f"Video id: {gt.get('video_id')}\n"
        "Watch the screen-recording video and emit ONLY valid DAG JSON "
        "(nodes START|ACTION|DECISION|END + edges). No markdown."
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


def load_rows(path: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"empty {path}")
    # keep only rows with resolvable video
    ok = []
    for r in rows:
        try:
            resolve_video(r)
            ok.append(r)
        except FileNotFoundError:
            continue
    if not ok:
        raise SystemExit("no GT rows with video on disk")
    if n > 0:
        ok = ok[:n]
    return ok


class VideoDagDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        gt = self.rows[i]
        path = resolve_video(gt)
        frames = sample_frames(path, N_FRAMES)
        target = gt.get("gold_dag_json") or json.dumps(gt["gold_dag"], separators=(",", ":"))
        return {"gt": gt, "frames": frames, "target": target, "video_path": str(path)}


def attach_lora(model):
    # language-side LoRA; vision frozen (fits 1×A100)
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


def build_train_tensors(tok, processor, device, frames, gt, target: str):
    messages = [
        {"role": "system", "content": "/no_think"},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": ""},
                {"type": "text", "text": "\n" + user_text(gt)},
            ],
        },
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    metadata = [{"fps": VIDEO_FPS_META, "frames_indices": list(range(len(frames)))}]
    try:
        proc = processor(
            text=[prompt],
            videos=[frames],
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
        )
    except TypeError:
        proc = processor(text=[prompt], videos=[frames], return_tensors="pt")

    prompt_ids = proc["input_ids"][0]
    resp = tok(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if tok.eos_token_id is not None:
        resp = torch.cat([resp, torch.tensor([tok.eos_token_id], dtype=resp.dtype)])

    input_ids = torch.cat([prompt_ids, resp]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[0]] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    # video pixels
    if "pixel_values_videos" in proc:
        out["pixel_values_videos"] = proc["pixel_values_videos"].to(device, dtype=torch.bfloat16)
    if "pixel_values" in proc:
        out["pixel_values"] = proc["pixel_values"].to(device, dtype=torch.bfloat16)
    for k in ("video_grid_thw", "image_grid_thw", "second_per_grid_ts"):
        if k in proc:
            out[k] = proc[k].to(device) if torch.is_tensor(proc[k]) else proc[k]
    return out, int(prompt_ids.shape[0])


@torch.no_grad()
def eval_schema(model, tok, processor, rows: list[dict], device, n: int = 4) -> dict:
    model.eval()
    sample = rows[:n]
    rewards, schemas = [], []
    bar = tqdm(sample, desc="  eval", leave=False)
    for gt in bar:
        frames = sample_frames(resolve_video(gt), N_FRAMES)
        messages = [
            {"role": "system", "content": "/no_think"},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": ""},
                    {"type": "text", "text": "\n" + user_text(gt)},
                ],
            },
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            proc = processor(text=[prompt], videos=[frames], return_tensors="pt")
        except Exception as e:
            tqdm.write(f"eval proc fail: {e}")
            continue
        proc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in proc.items()}
        if hasattr(model, "video_pruning_rate"):
            model.video_pruning_rate = 0.75
        gen_kwargs = {
            "input_ids": proc["input_ids"],
            "attention_mask": proc.get("attention_mask"),
            "max_new_tokens": MAX_NEW_EVAL,
            "do_sample": False,
            "eos_token_id": tok.eos_token_id,
        }
        if "pixel_values_videos" in proc:
            gen_kwargs["pixel_values_videos"] = proc["pixel_values_videos"]
        if "pixel_values" in proc:
            gen_kwargs["pixel_values"] = proc["pixel_values"]
        out = model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
        text = tok.decode(out[0, proc["input_ids"].shape[1] :], skip_special_tokens=True)
        sc = dag_reward(extract_json(text), gt)
        rewards.append(sc["reward"])
        schemas.append(sc["schema_ok"])
        bar.set_postfix(R=f"{sc['reward']:.2f}", schema=sc["schema_ok"])
    model.train()
    if not rewards:
        return {"eval/reward_mean": 0.0, "eval/schema_ok": 0.0}
    return {
        "eval/reward_mean": sum(rewards) / len(rewards),
        "eval/schema_ok": sum(schemas) / len(schemas),
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
        name=os.environ.get("WANDB_NAME", "sft-nemotron-vl-50"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        config={
            "model": MODEL,
            "n_gt": len(rows),
            "epochs": EPOCHS,
            "batch": BATCH,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "n_frames": N_FRAMES,
            "phase": "sft-nemotron-vl-video",
        },
    )
    print("WANDB_RUN_ID=", run.id)
    print("WANDB_URL=", run.url)
    print(f"Nemotron-VL SFT n={len(rows)} frames={N_FRAMES} batch={BATCH} accum={GRAD_ACCUM}")

    print("loading", MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = attach_lora(model)
    model.train()
    if hasattr(model, "video_pruning_rate"):
        model.video_pruning_rate = 0.5  # train a bit denser than infer default

    ds = VideoDagDataset(rows)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: b, num_workers=0)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    total_steps = max(1, (len(loader) * EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM)
    sched = get_linear_schedule_with_warmup(opt, max(1, total_steps // 10), total_steps)

    global_step = 0
    epoch_bar = tqdm(range(EPOCHS), desc="SFT-VL epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        batch_bar = tqdm(loader, desc=f"  epoch {epoch+1}/{EPOCHS}", leave=False, dynamic_ncols=True)
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(batch_bar):
            # batch is list of dicts (BATCH==1 typically)
            t0 = time.time()
            losses = []
            for item in batch:
                tensors, _ = build_train_tensors(
                    tok, processor, device, item["frames"], item["gt"], item["target"]
                )
                fwd = {k: v for k, v in tensors.items()}
                out = model(**fwd)
                losses.append(out.loss)
            loss = torch.stack(losses).mean() / GRAD_ACCUM
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
                        "train/loss": float(loss.detach() * GRAD_ACCUM),
                        "train/lr": sched.get_last_lr()[0],
                        "train/epoch": epoch + (bi + 1) / len(loader),
                        "perf/step_seconds": time.time() - t0,
                        "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                        "data/video": Path(batch[0]["video_path"]).parent.name,
                    },
                    step=global_step,
                )
                batch_bar.set_postfix(
                    loss=f"{float(loss.detach()*GRAD_ACCUM):.3f}",
                    step=global_step,
                    vid=Path(batch[0]["video_path"]).parent.name[-12:],
                    mem=f"{torch.cuda.memory_allocated()/1e9:.1f}G",
                )

                if global_step % EVAL_EVERY == 0:
                    metrics = eval_schema(model, tok, processor, rows, device, n=min(4, len(rows)))
                    wandb.log(metrics, step=global_step)
                    tqdm.write(
                        f"eval step={global_step} schema={metrics['eval/schema_ok']:.2f} "
                        f"R={metrics['eval/reward_mean']:.3f}",
                        file=sys.stderr,
                    )

        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = CKPT_DIR / f"epoch_{epoch+1}"
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        tqdm.write(f"ckpt={out_dir}")

    metrics = eval_schema(model, tok, processor, rows, device, n=min(8, len(rows)))
    wandb.log({f"final/{k.split('/', 1)[-1]}": v for k, v in metrics.items()})
    print("FINAL", metrics)
    wandb.finish()
    print("DONE", run.url)


if __name__ == "__main__":
    main()
