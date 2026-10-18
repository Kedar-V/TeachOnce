#!/usr/bin/env python3
"""Balanced train JSONL excluding holdout ids. ~equal per family."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--train-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hold = {json.loads(l)["video_id"] for l in args.holdout.read_text().splitlines() if l.strip()}
    by_fam: dict[str, list] = defaultdict(list)
    for line in args.pool.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("video_id") in hold:
            continue
        by_fam[r.get("family") or "unknown"].append(r)

    fams = sorted(by_fam)
    if not fams:
        raise SystemExit("no families left after holdout exclude")
    # equal split; remainder to first families
    base, rem = divmod(args.train_n, len(fams))
    quotas = {f: base + (1 if i < rem else 0) for i, f in enumerate(fams)}
    rng = random.Random(args.seed)
    picked = []
    for f, q in quotas.items():
        pool = by_fam[f]
        if len(pool) < q:
            raise SystemExit(f"{f}: need {q}, have {len(pool)}")
        picked.extend(rng.sample(pool, q))
    picked.sort(key=lambda r: r.get("video_id") or "")
    for r in picked:
        r["split"] = "train"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in picked) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n": len(picked),
                "per_family": {f: quotas[f] for f in fams},
                "holdout_excluded": len(hold),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
