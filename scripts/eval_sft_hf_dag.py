#!/usr/bin/env python3
"""Score SFT LoRA (HF) with dag_reward — same metrics as baseline_vllm_dag summary.

Note: baseline used vLLM guided_json (schema forced). This path is free generate
via the working video-embed inject used in SFT train.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import dag_reward  # noqa: E402
from scripts.sft_nemotron_vl_wandb import (  # noqa: E402
    N_FRAMES,
    VIDEO_FPS_META,
    _vl_core,
    extract_json,
    resolve_video,
    sample_frames,
    user_text,
)

try:
    from peft import PeftModel
except ImportError as e:
    raise SystemExit("need peft") from e


MODEL = os.environ.get("SFT_MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
ADAPTER = Path(os.environ.get("SFT_ADAPTER", str(ROOT / "checkpoints/sft_nemotron_vl_100/step_30")))
GT = Path(os.environ.get("SFT_GT", str(ROOT / "data/generated/nemo_rl/gt.jsonl")))
OUT = Path(os.environ.get("SFT_EVAL_OUT", str(ROOT / "results/sft_hf_dag.json")))
MAX_SAMPLES = int(os.environ.get("SFT_EVAL_N", "25"))
MAX_NEW = int(os.environ.get("SFT_MAX_NEW_EVAL", "1024"))
SHARD = int(os.environ.get("SFT_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("SFT_NUM_SHARDS", "1"))


def main() -> None:
    device = torch.device("cuda")
    _orig = torch.distributed.get_rank

    def _safe_rank(*_a, **_k):
        if not torch.distributed.is_initialized():
            return 0
        return _orig()

    torch.distributed.get_rank = _safe_rank  # type: ignore[method-assign]

    rows = [json.loads(l) for l in GT.read_text().splitlines() if l.strip()][:MAX_SAMPLES]
    if NUM_SHARDS > 1:
        rows = rows[SHARD::NUM_SHARDS]
        print(f"shard {SHARD}/{NUM_SHARDS} n={len(rows)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("loading", MODEL, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if ADAPTER.exists():
        print("loading adapter", ADAPTER, flush=True)
        model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False)
    model.eval()
    vl = _vl_core(model)
    if getattr(vl, "video_context_token_id", None) is None:
        vl.video_context_token_id = tok.convert_tokens_to_ids("<video>")
        if vl.video_context_token_id is None or vl.video_context_token_id < 0:
            # fallback used in train script
            for cand in ("<|video_pad|>", "<video>"):
                tid = tok.convert_tokens_to_ids(cand)
                if tid is not None and tid >= 0:
                    vl.video_context_token_id = tid
                    break
    print("video_context_token_id=", vl.video_context_token_id, flush=True)

    from types import SimpleNamespace

    per = []
    sums = {k: 0.0 for k in ("reward", "schema_ok", "critical_ops", "taken_branch", "latency_s")}
    n_ok = 0
    for gt in tqdm(rows, desc="sft-eval"):
        t0 = time.time()
        try:
            path = resolve_video(gt)
            frames = sample_frames(path, N_FRAMES)
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
            proc = processor(
                text=[prompt],
                videos=[frames],
                videos_kwargs={"video_metadata": SimpleNamespace(fps=VIDEO_FPS_META)},
                return_tensors="pt",
            )
            input_ids = proc["input_ids"].to(device)
            attn = proc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            pv = proc.get("pixel_values_videos", proc.get("pixel_values"))
            if pv is None:
                raise RuntimeError("no pixels")
            pv = pv.to(device=device, dtype=vl.vision_model.config.torch_dtype)
            with torch.no_grad():
                video_vit = vl.extract_feature(pv)
                embeds = vl.language_model.get_input_embeddings()(input_ids)
                B, N, C = embeds.shape
                flat = embeds.reshape(B * N, C)
                ids_flat = input_ids.reshape(B * N)
                mask = ids_flat == vl.video_context_token_id
                n_tok = int(mask.sum())
                if n_tok == 0:
                    raise RuntimeError("no video_context tokens in prompt")
                flat[mask] = video_vit.reshape(-1, C)[:n_tok].to(device=flat.device, dtype=flat.dtype)
                embeds = flat.reshape(B, N, C)
                out_ids = vl.language_model.generate(
                    inputs_embeds=embeds,
                    attention_mask=attn,
                    max_new_tokens=MAX_NEW,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            # generate with inputs_embeds returns only new tokens on some versions; handle both
            if out_ids.shape[1] > input_ids.shape[1]:
                gen = out_ids[0, input_ids.shape[1] :]
            else:
                gen = out_ids[0]
            text = tok.decode(gen, skip_special_tokens=True)
            text = extract_json(text)
            scores = dag_reward(text, gt)
            scores.update(
                {
                    "id": gt.get("id"),
                    "video": str(path),
                    "latency_s": round(time.time() - t0, 3),
                    "raw_preview": text[:240],
                }
            )
            n_ok += 1
        except Exception as e:
            scores = {
                "id": gt.get("id"),
                "reward": 0.0,
                "schema_ok": 0.0,
                "critical_ops": 0.0,
                "taken_branch": 0.0,
                "latency_s": round(time.time() - t0, 3),
                "error": str(e)[:400],
            }
        per.append(scores)
        for k in sums:
            sums[k] += float(scores.get(k) or 0)
        tqdm.write(
            f"{scores.get('id')} R={scores.get('reward')} schema={scores.get('schema_ok')} "
            f"crit={scores.get('critical_ops')} branch={scores.get('taken_branch')} err={scores.get('error','')[:80]}"
        )

    n = max(len(per), 1)
    summary = {
        "n": len(per),
        "n_success": n_ok,
        "model": MODEL,
        "adapter": str(ADAPTER) if ADAPTER.exists() else None,
        "input": "hf_video_frames",
        "guided_json": False,
        "mean_reward": round(sums["reward"] / n, 4),
        "mean_schema_ok": round(sums["schema_ok"] / n, 4),
        "mean_critical_ops": round(sums["critical_ops"] / n, 4),
        "mean_taken_branch": round(sums["taken_branch"] / n, 4),
        "mean_latency_s": round(sums["latency_s"] / n, 3),
        "baseline_ref": {
            "mean_reward": 0.649,
            "mean_schema_ok": 1.0,
            "mean_critical_ops": 0.19,
            "mean_taken_branch": 0.64,
            "guided_json": True,
            "note": "baseline used vLLM guided_json; this eval is unconstrained HF generate",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "per_sample": per}, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
