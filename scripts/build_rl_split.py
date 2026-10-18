#!/usr/bin/env python3
"""Freeze holdout = existing baseline/Grok gt.jsonl; sample train-N from the rest.

Usage:
  python3 scripts/build_rl_split.py \\
    --holdout data/generated/nemo_rl/gt.jsonl \\
    --videos data/generated/demonstrations/videos \\
    --train-n 100 --seed 42 \\
    --out-dir data/generated/nemo_rl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import from_sample_dir, save_gt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--train-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    hold_rows = [json.loads(l) for l in args.holdout.read_text().splitlines() if l.strip()]
    hold_ids = {r["video_id"] for r in hold_rows}
    if len(hold_ids) != len(hold_rows):
        raise SystemExit("duplicate video_id in holdout")

    cands = []
    for d in sorted(args.videos.iterdir()):
        if not d.is_dir() or d.name in hold_ids:
            continue
        if (d / "video.mp4").exists() and (d / "expected_dag.json").exists():
            cands.append(d)
    if len(cands) < args.train_n:
        raise SystemExit(f"need {args.train_n} train candidates, have {len(cands)}")

    rng = random.Random(args.seed)
    picked = sorted(rng.sample(cands, args.train_n), key=lambda p: p.name)

    train_rows = []
    for d in picked:
        r = from_sample_dir(d)
        r["split"] = "train"
        train_rows.append(r)

    # rewrite holdout split tag + ensure path-consistent already in file
    for r in hold_rows:
        r["split"] = "holdout"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hold_path = out / "gt_holdout_25.jsonl"
    train_path = out / f"gt_train_{args.train_n}.jsonl"
    save_gt(hold_rows, hold_path)
    save_gt(train_rows, train_path)
    (out / "rl_split_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "holdout_source": str(args.holdout.resolve()),
                "holdout_n": len(hold_rows),
                "holdout_ids": sorted(hold_ids),
                "train_n": len(train_rows),
                "train_ids": [r["video_id"] for r in train_rows],
                "train_candidates_available": len(cands),
                "note": "holdout = same 25 as baseline_vllm_dag / grok eval",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "holdout": str(hold_path),
                "holdout_n": len(hold_rows),
                "train": str(train_path),
                "train_n": len(train_rows),
                "excluded_from_train": sorted(hold_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
