#!/usr/bin/env python3
"""
Parallel Stage-B warranty video gen (all CPU cores).

  python3 scripts/gen_stage_b_parallel.py
  python3 scripts/gen_stage_b_parallel.py --workers 12
  touch data/generated/demonstrations/videos/.pause   # stop after in-flight jobs
  # resume: re-run same command (skips checkpoint completed)

Checkpoint: data/generated/demonstrations/videos/.checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_one_synthetic_sample import generate_sample  # noqa: E402

VIDEOS = ROOT / "data" / "generated" / "demonstrations" / "videos"
CKPT = VIDEOS / ".checkpoint.json"
PAUSE = VIDEOS / ".pause"
TRAIN = ROOT / "data" / "generated" / "splits" / "mvp_train.jsonl"

NAMES = [
    "Ava Chen", "Sam Ortiz", "Riley Park", "Jordan Lee", "Casey Ng",
    "Morgan Diaz", "Quinn Brooks", "Jamie Cole", "Taylor Kim", "Drew Patel",
    "Alex Rivera", "Blake Soto", "Cameron Wu", "Dana Price", "Elliot Shaw",
    "Finley Ames", "Gray Benton", "Harper Cruz", "Indie Vale", "Jules Hart",
    "Kai Monroe", "Lane Foster",
]


def build_jobs(count: int) -> list[dict]:
    # ~64% active / ~36% expired (matches 14/22 recipe). Last 5% → val.
    n_val = max(2, count // 20)
    jobs = []
    for i in range(1, count + 1):
        # deterministic mix: every 11th + remainder block → expired (~36%)
        path_type = "expired" if (i % 11 in (0, 3, 7, 10)) else "active"
        vid = f"warranty_{i:04d}"
        n = NAMES[(i - 1) % len(NAMES)]
        first, last = n.split()
        split = "val" if i > count - n_val else "train"
        jobs.append(
            {
                "video_id": vid,
                "path_type": path_type,
                "split": split,
                "case": {
                    "customer_id": f"C-{1000 + i}",
                    "customer_name": n,
                    "email": f"{first.lower()}.{last.lower()}{i}@example.com",
                    "device_id": f"D-{2000 + i}",
                    "serial": f"SN-{9000 + i:04d}-X{i % 100:02d}",
                    "model": "Northstar Pro 15" if i % 2 else "Northstar Air 13",
                    "ticket_id": f"CS-{48000 + i}",
                    "ticket_subject": "Laptop fails to boot — HDD click",
                    "warranty_id": f"W-{9000 + i}",
                    "warranty_status": "ACTIVE" if path_type == "active" else "EXPIRED",
                    "warranty_end": "2027-03-14" if path_type == "active" else "2024-01-10",
                    "repair_id": f"SR-{88000 + i}",
                    "agent": NAMES[i % len(NAMES)],
                },
            }
        )
    return jobs


def load_ckpt() -> dict:
    if CKPT.exists():
        return json.loads(CKPT.read_text())
    return {"completed": {}, "failed": {}, "started_at": None}


def save_ckpt(ckpt: dict) -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    tmp = CKPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, indent=2) + "\n")
    tmp.replace(CKPT)


def _worker(spec: dict) -> dict:
    try:
        return generate_sample(spec)
    except Exception as e:
        return {"ok": False, "video_id": spec["video_id"], "error": str(e)}


def merge_train(ckpt: dict) -> None:
    rows = []
    for vid, info in sorted(ckpt.get("completed", {}).items()):
        row_path = VIDEOS / vid / "train_row.json"
        if row_path.exists():
            rows.append(json.loads(row_path.read_text()))
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    TRAIN.write_text("".join(json.dumps(r) + "\n" for r in rows))
    active = sum(1 for r in rows if r.get("path_type") == "active")
    expired = sum(1 for r in rows if r.get("path_type") == "expired")
    (VIDEOS / "manifest.json").write_text(
        json.dumps(
            {
                "version": "1",
                "family": "warranty_resolution",
                "stage": "B",
                "count": len(rows),
                "active": active,
                "expired": expired,
                "val": [r["input"]["video_id"] for r in rows if r.get("split") == "val"],
                "source_dataset": "teachonce_self_synthetic",
                "model_target": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
                "adapter": "scripts/tool_registry_adapter.py",
                "incomplete": False,
                "checkpoint": str(CKPT),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000, help="total samples to generate")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0, help="only first N jobs (smoke)")
    ap.add_argument("--force", action="store_true", help="redo completed ids")
    args = ap.parse_args()

    try:
        from tqdm import tqdm
    except ImportError:
        print("install tqdm: pip install tqdm", file=sys.stderr)
        raise

    jobs = build_jobs(args.count)
    if args.limit:
        jobs = jobs[: args.limit]

    ckpt = load_ckpt()
    if not ckpt.get("started_at"):
        ckpt["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.force:
        ckpt["completed"] = {}
        ckpt["failed"] = {}

    pending = [j for j in jobs if args.force or j["video_id"] not in ckpt["completed"]]
    print(
        f"workers={args.workers} total={len(jobs)} pending={len(pending)} "
        f"done={len(ckpt['completed'])} pause_file={PAUSE}"
    )
    if not pending:
        merge_train(ckpt)
        print("nothing to do; train merged")
        return

    if PAUSE.exists():
        print(f"remove {PAUSE} to run (pause flag set)")
        return

    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
        print("\nSIGINT — finishing in-flight, then exit (checkpoint saved)", flush=True)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, j): j["video_id"] for j in pending}
        pbar = tqdm(total=len(pending), desc="stage-b", unit="vid")
        try:
            for fut in as_completed(futures):
                vid = futures[fut]
                result = fut.result()
                if result.get("ok"):
                    ckpt["completed"][vid] = {
                        "video": result["video"],
                        "path_type": result["path_type"],
                        "size_mb": result["size_mb"],
                        "ts": time.time(),
                    }
                    ckpt["failed"].pop(vid, None)
                else:
                    ckpt["failed"][vid] = result.get("error", "unknown")
                save_ckpt(ckpt)
                pbar.update(1)
                pbar.set_postfix(
                    done=len(ckpt["completed"]),
                    fail=len(ckpt["failed"]),
                    last=vid,
                )
                if stop["flag"] or PAUSE.exists():
                    # cancel not-yet-started
                    for f in futures:
                        f.cancel()
                    reason = "pause" if PAUSE.exists() else "signal"
                    pbar.set_description(f"stopped ({reason})")
                    break
        finally:
            pbar.close()

    merge_train(ckpt)
    print(
        json.dumps(
            {
                "completed": len(ckpt["completed"]),
                "failed": ckpt["failed"],
                "train": str(TRAIN),
                "checkpoint": str(CKPT),
                "paused": PAUSE.exists(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
