#!/usr/bin/env python3
"""Apply LoRA deltas onto base safetensors shards — no model class / mamba needed."""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER = Path(os.environ["SFT_ADAPTER"])
BASE = Path(os.environ["SFT_BASE_SNAPSHOT"])
OUT = Path(os.environ.get("SFT_MERGED_OUT", "/data/merged-nemotron-sft"))
ALPHA = float(os.environ.get("SFT_LORA_ALPHA", "32"))
R = float(os.environ.get("SFT_LORA_R", "16"))
SCALE = ALPHA / R


def adapter_to_base(k: str) -> tuple[str, str] | None:
    # base_model.model.language_model....q_proj.lora_A.weight
    if not k.startswith("base_model.model."):
        return None
    core = k[len("base_model.model.") :]
    if core.endswith(".lora_A.weight"):
        return core[: -len(".lora_A.weight")] + ".weight", "A"
    if core.endswith(".lora_B.weight"):
        return core[: -len(".lora_B.weight")] + ".weight", "B"
    return None


def main() -> None:
    cfg = json.loads((ADAPTER / "adapter_config.json").read_text())
    scale = float(cfg.get("lora_alpha", ALPHA)) / float(cfg.get("r", R))
    print(f"scale={scale} alpha={cfg.get('lora_alpha')} r={cfg.get('r')}", flush=True)

    ad_path = ADAPTER / "adapter_model.safetensors"
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(ad_path, framework="pt") as f:
        for k in f.keys():
            mapped = adapter_to_base(k)
            if not mapped:
                continue
            base_k, which = mapped
            pairs[base_k][which] = f.get_tensor(k)
    print(f"lora modules={len(pairs)}", flush=True)

    idx = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = idx["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for bk in pairs:
        if bk not in weight_map:
            raise SystemExit(f"base missing {bk}")
        by_shard[weight_map[bk]].append(bk)

    if OUT.exists():
        shutil.rmtree(OUT)
    print(f"copy base → {OUT}", flush=True)
    shutil.copytree(BASE, OUT, symlinks=False)

    for shard, keys in by_shard.items():
        sp = OUT / shard
        print(f"patch {shard} n={len(keys)}", flush=True)
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for bk in keys:
            A, B = pairs[bk]["A"], pairs[bk]["B"]
            # peft: delta = B @ A * scale  (B: out x r, A: r x in)
            delta = (B.float() @ A.float()) * scale
            w = tensors[bk]
            tensors[bk] = (w.float() + delta.to(w.device)).to(w.dtype).contiguous()
        save_file(tensors, sp)
        print(f"  wrote {sp}", flush=True)

    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
