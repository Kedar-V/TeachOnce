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
from scripts.nemo_rl_gt import _ops_from_dag, dag_reward  # noqa: E402

try:
    from peft import LoraConfig, PeftModel, get_peft_model
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
CKPT_EVERY = int(os.environ.get("SFT_CKPT_EVERY", "10"))  # optimizer steps
RESUME = os.environ.get("SFT_RESUME", "1") != "0"
CKPT_DIR = Path(os.environ.get("SFT_CKPT", str(ROOT / "checkpoints/sft_nemotron_vl")))
PROGRESS_PATH = Path(os.environ.get("SFT_PROGRESS", str(CKPT_DIR / "progress.json")))
VIDEO_FPS_META = float(os.environ.get("SFT_VIDEO_FPS", "1.0"))


def write_progress(**fields: object) -> None:
    """Granular train status for external monitors (pipeline / loops)."""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cur: dict = {}
    if PROGRESS_PATH.exists():
        try:
            cur = json.loads(PROGRESS_PATH.read_text())
        except json.JSONDecodeError:
            cur = {}
    cur.update(fields)
    cur["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2) + "\n")
    tmp.replace(PROGRESS_PATH)


def save_checkpoint(model, tok, opt, sched, *, global_step: int, epoch: int, tag: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = CKPT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        out_dir / "trainer_state.pt",
    )
    latest = CKPT_DIR / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(out_dir.name)
    except OSError:
        (CKPT_DIR / "latest_path.txt").write_text(str(out_dir.resolve()) + "\n")
    write_progress(
        phase="sft",
        status="checkpointed",
        global_step=global_step,
        epoch=epoch,
        ckpt=str(out_dir),
        latest=str(out_dir),
    )
    return out_dir


def resolve_resume_dir() -> Path | None:
    if not RESUME:
        return None
    latest = CKPT_DIR / "latest"
    if latest.is_symlink() and latest.exists():
        return latest.resolve()
    path_file = CKPT_DIR / "latest_path.txt"
    if path_file.exists():
        p = Path(path_file.read_text().strip())
        if p.exists():
            return p
    # fall back to highest step_* or epoch_*
    cands = sorted(CKPT_DIR.glob("step_*")) + sorted(CKPT_DIR.glob("epoch_*"))
    return cands[-1] if cands else None


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
    # Nemotron processor expects video_metadata.fps (object), not a list/dict
    from types import SimpleNamespace

    metadata = SimpleNamespace(fps=VIDEO_FPS_META)
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

    pv = proc.get("pixel_values_videos", proc.get("pixel_values"))
    if pv is None:
        raise RuntimeError("processor returned no video/image pixels")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values_videos": pv.to(device, dtype=torch.bfloat16),
    }, int(prompt_ids.shape[0])


def _vl_core(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(m, "model") and hasattr(m.model, "extract_feature"):
        return m.model
    return m


def forward_video_loss(model, batch: dict):
    """Video train step: inject embeds like generate(); LM loss on language_model.

    Full VL `.forward()` requires image `pixel_values` and ignores video tokens.
    Vision is frozen; LoRA lives on language_model modules.
    """
    vl = _vl_core(model)
    with torch.no_grad():
        pv = batch["pixel_values_videos"].to(dtype=vl.vision_model.config.torch_dtype)
        video_vit = vl.extract_feature(pv)
    input_ids = batch["input_ids"]
    embeds = vl.language_model.get_input_embeddings()(input_ids)
    B, N, C = embeds.shape
    if B > 1:
        raise NotImplementedError("video SFT batch>1 unsupported")
    flat = embeds.reshape(B * N, C)
    ids_flat = input_ids.reshape(B * N)
    vid_tok = getattr(vl, "video_context_token_id", None)
    if vid_tok is None:
        raise RuntimeError("missing video_context_token_id — set after from_pretrained")
    mask = ids_flat == vid_tok
    n_tok = int(mask.sum())
    if n_tok == 0:
        raise RuntimeError("no video_context tokens in prompt")
    flat[mask] = video_vit.reshape(-1, C)[:n_tok].to(device=flat.device, dtype=flat.dtype)
    embeds = flat.reshape(B, N, C).detach()  # vision frozen; LM grads only
    embeds.requires_grad_(True)
    return vl.language_model(
        inputs_embeds=embeds,
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )



@torch.no_grad()
def eval_val_loss(model, tok, processor, rows: list[dict], device, n: int = 8) -> dict:
    """Teacher-forced val loss — reliable signal that training is working."""
    model.eval()
    val = [r for r in rows if r.get("split") == "val"] or rows
    sample = val[:n]
    losses = []
    for gt in tqdm(sample, desc="  val_loss", leave=False):
        try:
            frames = sample_frames(resolve_video(gt), N_FRAMES)
            target = gt.get("gold_dag_json") or json.dumps(gt["gold_dag"], separators=(",", ":"))
            tensors, _ = build_train_tensors(tok, processor, device, frames, gt, target)
            out = forward_video_loss(model, tensors)
            losses.append(float(out.loss.detach()))
        except Exception as e:
            tqdm.write(f"val_loss fail: {e}")
    model.train()
    if not losses:
        return {"eval/val_loss": float("nan"), "eval/val_n": 0}
    return {"eval/val_loss": sum(losses) / len(losses), "eval/val_n": len(losses)}


@torch.no_grad()
def eval_schema(model, tok, processor, rows: list[dict], device, n: int = 4) -> dict:
    model.eval()
    sample = rows[:n]
    rewards, schemas, precs, recs, branches = [], [], [], [], []
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
        # generate() accepts pixel_values_videos (video path)
        if "pixel_values_videos" in proc:
            gen_kwargs["pixel_values_videos"] = proc["pixel_values_videos"]
        elif "pixel_values" in proc:
            gen_kwargs["pixel_values"] = proc["pixel_values"]
        try:
            out = model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
        except Exception as e:
            # generate path still flaky (image_mask assert); train loop is the point
            tqdm.write(f"eval generate fail: {e}")
            continue
        text = tok.decode(out[0, proc["input_ids"].shape[1] :], skip_special_tokens=True)
        raw = extract_json(text)
        sc = dag_reward(raw, gt)
        rewards.append(sc["reward"])
        schemas.append(sc["schema_ok"])
        branches.append(sc["taken_branch"])
        try:
            pred = json.loads(raw) if isinstance(raw, str) else raw
            pred_ops = set(_ops_from_dag(pred)) if isinstance(pred, dict) else set()
        except json.JSONDecodeError:
            pred_ops = set()
        gold = set(gt.get("gold_ops") or [])
        inter = len(pred_ops & gold)
        precs.append(inter / len(pred_ops) if pred_ops else 0.0)
        recs.append(inter / len(gold) if gold else 1.0)
        bar.set_postfix(R=f"{sc['reward']:.2f}", schema=sc["schema_ok"], rec=f"{recs[-1]:.2f}")
    model.train()
    if not rewards:
        return {
            "eval/reward_mean": 0.0,
            "eval/schema_ok": 0.0,
            "eval/ops_precision": 0.0,
            "eval/ops_recall": 0.0,
            "eval/taken_branch": 0.0,
        }
    return {
        "eval/reward_mean": sum(rewards) / len(rewards),
        "eval/schema_ok": sum(schemas) / len(schemas),
        "eval/ops_precision": sum(precs) / len(precs),
        "eval/ops_recall": sum(recs) / len(recs),
        "eval/taken_branch": sum(branches) / len(branches),
    }


def main() -> None:
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        os.environ["WANDB_MODE"] = "offline"
        print("WANDB_API_KEY missing → WANDB_MODE=offline (local progress.json still written)")
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.forward calls get_rank(); don't init process group (breaks HF _tp_plan warmup)
    _orig_rank = torch.distributed.get_rank

    def _safe_rank(*_a, **_k):
        if not torch.distributed.is_initialized():
            return 0
        return _orig_rank()

    torch.distributed.get_rank = _safe_rank  # type: ignore[method-assign]
    assert device.type == "cuda", "need GPU"

    rows = load_rows(GT_PATH, MAX_SAMPLES)
    path_ok = sum(1 for r in rows if r.get("path_consistent"))
    write_progress(
        phase="sft",
        status="starting",
        n_gt=len(rows),
        path_consistent=path_ok,
        model=MODEL,
        epochs=EPOCHS,
        ckpt_every=CKPT_EVERY,
        eval_every=EVAL_EVERY,
    )

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "teachonce-nemo-rl"),
        name=os.environ.get("WANDB_NAME", "sft-nemotron-vl-50"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        resume="allow",
        config={
            "model": MODEL,
            "n_gt": len(rows),
            "path_consistent": path_ok,
            "epochs": EPOCHS,
            "batch": BATCH,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "n_frames": N_FRAMES,
            "ckpt_every": CKPT_EVERY,
            "phase": "sft-nemotron-vl-video",
        },
    )
    print("WANDB_RUN_ID=", run.id)
    print("WANDB_URL=", run.url)
    print(
        f"Nemotron-VL SFT n={len(rows)} path_consistent={path_ok} "
        f"frames={N_FRAMES} batch={BATCH} accum={GRAD_ACCUM} ckpt_every={CKPT_EVERY}"
    )

    print("loading", MODEL, flush=True)
    write_progress(status="loading_model")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    resume_dir = resolve_resume_dir()
    global_step = 0
    start_epoch = 0
    if resume_dir is not None:
        print(f"resuming LoRA from {resume_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
        state_path = resume_dir / "trainer_state.pt"
        trainer_state = torch.load(state_path, map_location="cpu") if state_path.exists() else {}
        global_step = int(trainer_state.get("global_step") or 0)
        start_epoch = int(trainer_state.get("epoch") or 0)
    else:
        model = attach_lora(model)
        trainer_state = {}

    model.train()
    vl = _vl_core(model)
    # sync multimodal token ids (generate path uses these; train injects the same)
    if getattr(vl, "video_context_token_id", None) is None:
        vl.video_context_token_id = getattr(model.config, "video_context_token_id", None) or tok.convert_tokens_to_ids(
            "<video>"
        )
    if getattr(vl, "img_context_token_id", None) is None:
        vl.img_context_token_id = getattr(model.config, "img_context_token_id", None)
    print("video_context_token_id=", vl.video_context_token_id, flush=True)
    if hasattr(model, "video_pruning_rate"):
        model.video_pruning_rate = 0.5  # train a bit denser than infer default
    if hasattr(vl, "video_pruning_rate"):
        vl.video_pruning_rate = 0.5

    train_rows = [r for r in rows if r.get("split") != "val"] or rows
    val_rows = [r for r in rows if r.get("split") == "val"] or rows[-max(1, len(rows)//10):]
    print(f"split train={len(train_rows)} val={len(val_rows)}", flush=True)
    ds = VideoDagDataset(train_rows)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: b, num_workers=0)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    total_steps = max(1, (len(loader) * EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM)
    sched = get_linear_schedule_with_warmup(opt, max(1, total_steps // 10), total_steps)
    if trainer_state.get("optimizer"):
        try:
            opt.load_state_dict(trainer_state["optimizer"])
            sched.load_state_dict(trainer_state["scheduler"])
        except Exception as e:
            tqdm.write(f"warn: could not restore opt/sched ({e}); continuing with fresh opt")

    write_progress(status="training", global_step=global_step, epoch=start_epoch, total_steps=total_steps)
    epoch_bar = tqdm(range(start_epoch, EPOCHS), desc="SFT-VL epochs", dynamic_ncols=True)
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
                out = forward_video_loss(model, tensors)
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
                loss_v = float(loss.detach() * GRAD_ACCUM)
                wandb.log(
                    {
                        "train/loss": loss_v,
                        "train/lr": sched.get_last_lr()[0],
                        "train/epoch": epoch + (bi + 1) / len(loader),
                        "perf/step_seconds": time.time() - t0,
                        "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                        "data/video": Path(batch[0]["video_path"]).parent.name,
                    },
                    step=global_step,
                )
                batch_bar.set_postfix(
                    loss=f"{loss_v:.3f}",
                    step=global_step,
                    vid=Path(batch[0]["video_path"]).parent.name[-12:],
                    mem=f"{torch.cuda.memory_allocated()/1e9:.1f}G",
                )
                write_progress(
                    status="training",
                    global_step=global_step,
                    epoch=epoch,
                    batch=bi + 1,
                    batches_per_epoch=len(loader),
                    last_loss=loss_v,
                    last_video=Path(batch[0]["video_path"]).parent.name,
                    pct=round(100.0 * global_step / total_steps, 2),
                )

                if global_step % EVAL_EVERY == 0:
                    metrics = eval_val_loss(model, tok, processor, val_rows, device, n=min(8, len(val_rows)))
                    try:
                        metrics.update(eval_schema(model, tok, processor, val_rows, device, n=min(2, len(val_rows))))
                    except Exception as e:
                        tqdm.write(f"schema eval skipped: {e}")
                    wandb.log(metrics, step=global_step)
                    write_progress(status="eval", last_eval=metrics, global_step=global_step)
                    tqdm.write(
                        f"eval step={global_step} val_loss={metrics.get('eval/val_loss')} "
                        f"schema={metrics.get('eval/schema_ok')} R={metrics.get('eval/reward_mean')}",
                        file=sys.stderr,
                    )
                    write_progress(status="training")

                if global_step % CKPT_EVERY == 0:
                    out_dir = save_checkpoint(
                        model, tok, opt, sched, global_step=global_step, epoch=epoch, tag=f"step_{global_step}"
                    )
                    tqdm.write(f"ckpt={out_dir}")

        out_dir = save_checkpoint(
            model, tok, opt, sched, global_step=global_step, epoch=epoch + 1, tag=f"epoch_{epoch+1}"
        )
        tqdm.write(f"epoch_ckpt={out_dir}")

    write_progress(status="final_eval")
    metrics = eval_val_loss(model, tok, processor, val_rows, device, n=min(10, len(val_rows)))
    try:
        metrics.update(eval_schema(model, tok, processor, val_rows, device, n=min(4, len(val_rows))))
    except Exception as e:
        tqdm.write(f"final schema eval skipped: {e}")
    wandb.log({f"final/{k.split('/', 1)[-1]}": v for k, v in metrics.items()})
    final_dir = save_checkpoint(
        model, tok, opt, sched, global_step=global_step, epoch=EPOCHS, tag="final"
    )
    write_progress(phase="sft", status="done", last_eval=metrics, ckpt=str(final_dir), global_step=global_step)
    print("FINAL", metrics)
    wandb.finish()
    print("DONE", run.url)


if __name__ == "__main__":
    main()
