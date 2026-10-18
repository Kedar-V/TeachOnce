#!/usr/bin/env python3
"""Merge SFT LoRA into base Nemotron VL for vLLM (arch has no runtime LoRA).

Frees HF cache after load so a ~24GB save fits on tight disks.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import scan_cache_dir
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

MODEL = os.environ.get("SFT_MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
ADAPTER = Path(os.environ.get("SFT_ADAPTER", "checkpoints/sft_nemotron_vl_100/step_30"))
OUT = Path(os.environ.get("SFT_MERGED_OUT", "/data/merged-nemotron-sft"))
FREE_CACHE = os.environ.get("SFT_FREE_HF_CACHE", "1") == "1"


def _drop_hf_model(repo_id: str) -> None:
    try:
        info = scan_cache_dir()
    except Exception as e:
        print("cache scan skip", e, flush=True)
        return
    for repo in info.repos:
        if repo.repo_id != repo_id:
            continue
        print("deleting HF cache", repo.repo_path, flush=True)
        shutil.rmtree(repo.repo_path, ignore_errors=True)
        return
    print("no HF cache entry for", repo_id, flush=True)


def main() -> None:
    _orig = torch.distributed.get_rank

    def _safe_rank(*_a, **_k):
        if not torch.distributed.is_initialized():
            return 0
        return _orig()

    torch.distributed.get_rank = _safe_rank  # type: ignore[method-assign]

    OUT.mkdir(parents=True, exist_ok=True)
    print("loading tokenizer/processor", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    try:
        proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    except Exception as e:
        proc = None
        print("processor skip:", e, flush=True)

    print("loading base on cuda", MODEL, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print("loading adapter", ADAPTER, flush=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    print("merge_and_unload", flush=True)
    model = model.merge_and_unload()

    if FREE_CACHE:
        # weights are in VRAM; reclaim disk for save_pretrained
        _drop_hf_model(MODEL)
        print("disk after cache drop:", flush=True)
        os.system("df -h / | tail -1")

    print("saving", OUT, flush=True)
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(OUT)
    if proc is not None:
        proc.save_pretrained(OUT)
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
