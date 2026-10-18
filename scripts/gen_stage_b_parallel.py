#!/usr/bin/env python3
"""
Parallel Stage-B video gen (warranty / parts / escalate).

  python3 scripts/gen_stage_b_parallel.py --procedure parts --count 228
  python3 scripts/gen_stage_b_parallel.py --procedure escalate --count 228 --workers 12

Pause / resume (any time):
  touch data/generated/demonstrations/videos/.pause   # finish in-flight only, then exit
  rm    data/generated/demonstrations/videos/.pause   # clear flag
  # re-run the same command — skips checkpoint + complete dirs on disk

SIGINT/SIGTERM: same as pause (drain in-flight, save checkpoint, exit).

Checkpoint: data/generated/demonstrations/videos/.checkpoint_{proc}.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_one_synthetic_sample import generate_sample  # noqa: E402

VIDEOS = ROOT / "data" / "generated" / "demonstrations" / "videos"
PAUSE = VIDEOS / ".pause"
TRAIN = ROOT / "data" / "generated" / "splits" / "mvp_train.jsonl"
PREFIX = {"warranty": "warranty_", "parts": "parts_", "escalate": "escalate_"}

NAMES = [
    "Ava Chen", "Sam Ortiz", "Riley Park", "Jordan Lee", "Casey Ng",
    "Morgan Diaz", "Quinn Brooks", "Jamie Cole", "Taylor Kim", "Drew Patel",
    "Alex Rivera", "Blake Soto", "Cameron Wu", "Dana Price", "Elliot Shaw",
    "Finley Ames", "Gray Benton", "Harper Cruz", "Indie Vale", "Jules Hart",
    "Kai Monroe", "Lane Foster",
]


def ckpt_path(procedure: str) -> Path:
    if procedure == "warranty":
        # keep legacy filename for warranty resume
        return VIDEOS / ".checkpoint.json"
    return VIDEOS / f".checkpoint_{procedure}.json"


def build_jobs(count: int, procedure: str = "warranty") -> list[dict]:
    n_val = max(2, count // 20)
    jobs = []
    for i in range(1, count + 1):
        n = NAMES[(i - 1) % len(NAMES)]
        first, last = n.split()
        split = "val" if i > count - n_val else "train"
        base_case = {
            "customer_id": f"C-{1000 + i}",
            "customer_name": n,
            "email": f"{first.lower()}.{last.lower()}{i}@example.com",
            "device_id": f"D-{2000 + i}",
            "serial": f"SN-{9000 + i:04d}-X{i % 100:02d}",
            "model": "Northstar Pro 15" if i % 2 else "Northstar Air 13",
            "ticket_id": f"CS-{48000 + i}",
            "ticket_subject": "Laptop fails to boot — HDD click",
            "agent": NAMES[i % len(NAMES)],
        }
        if procedure == "parts":
            # ~56% in_stock / ~44% backorder (9/16 recipe)
            path_type = "backorder" if (i % 9 in (0, 3, 5, 8)) else "in_stock"
            vid = f"parts_{i:04d}"
            base_case.update(
                {
                    "part_sku": "SSD-1TB-NS" if path_type == "in_stock" else "BATT-14-NS",
                    "part_name": "1TB NVMe SSD" if path_type == "in_stock" else "Pro 14 battery pack",
                    "stock": "IN_STOCK" if path_type == "in_stock" else "OUT_OF_STOCK",
                    "shipment_id": f"SH-{22000 + i}",
                    "backorder_id": f"BO-{9000 + i}",
                }
            )
        elif procedure == "escalate":
            # ~58% p1 / ~42% standard (7/12 recipe)
            path_type = "standard" if (i % 12 in (0, 3, 6, 9, 11)) else "p1"
            vid = f"escalate_{i:04d}"
            base_case.update(
                {
                    "ticket_subject": "Production fleet — unit offline (P1)" if path_type == "p1" else "Keyboard ghosting under warranty",
                    "severity": "P1" if path_type == "p1" else "P2",
                    "oncall": "Sam Okonkwo",
                    "l2_queue": "Hardware-L2",
                }
            )
        else:
            path_type = "expired" if (i % 11 in (0, 3, 7, 10)) else "active"
            vid = f"warranty_{i:04d}"
            base_case.update(
                {
                    "warranty_id": f"W-{9000 + i}",
                    "warranty_status": "ACTIVE" if path_type == "active" else "EXPIRED",
                    "warranty_end": "2027-03-14" if path_type == "active" else "2024-01-10",
                    "repair_id": f"SR-{88000 + i}",
                }
            )
        jobs.append(
            {
                "video_id": vid,
                "procedure": procedure,
                "path_type": path_type,
                "split": split,
                "case": base_case,
            }
        )
    return jobs


def load_ckpt(procedure: str) -> dict:
    p = ckpt_path(procedure)
    if p.exists():
        return json.loads(p.read_text())
    return {"completed": {}, "failed": {}, "started_at": None, "procedure": procedure}


def save_ckpt(ckpt: dict, procedure: str) -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    p = ckpt_path(procedure)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, indent=2) + "\n")
    tmp.replace(p)


def sync_disk_completed(ckpt: dict, procedure: str) -> int:
    """Pick up finished samples even if checkpoint was lost mid-run."""
    prefix = PREFIX[procedure]
    added = 0
    if not VIDEOS.exists():
        return 0
    for d in VIDEOS.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        vid = d.name
        if vid in ckpt["completed"]:
            continue
        video = d / "video.mp4"
        row = d / "train_row.json"
        if video.exists() and video.stat().st_size > 0 and row.exists():
            meta = {}
            try:
                meta = json.loads(row.read_text())
            except Exception:
                pass
            ckpt["completed"][vid] = {
                "video": str(video),
                "procedure": procedure,
                "path_type": meta.get("path_type") or "unknown",
                "size_mb": round(video.stat().st_size / 1e6, 2),
                "ts": time.time(),
                "from_disk": True,
            }
            ckpt["failed"].pop(vid, None)
            added += 1
    return added


def _worker(spec: dict) -> dict:
    try:
        return generate_sample(spec)
    except Exception as e:
        return {"ok": False, "video_id": spec["video_id"], "error": str(e)}


def merge_train(ckpt: dict, procedure: str) -> None:
    # merge ALL procedure train_row.json files under videos/
    rows = []
    for d in sorted(VIDEOS.iterdir()):
        if not d.is_dir():
            continue
        row_path = d / "train_row.json"
        if row_path.exists():
            rows.append(json.loads(row_path.read_text()))
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    TRAIN.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_proc: dict[str, int] = {}
    for r in rows:
        p = r.get("procedure") or (
            "warranty" if str(r.get("id", "")).find("warranty") >= 0 else "unknown"
        )
        # prefer family → procedure map
        fam = r.get("family", "")
        if fam == "parts_fulfillment":
            p = "parts"
        elif fam == "severity_escalation":
            p = "escalate"
        elif fam == "warranty_resolution":
            p = "warranty"
        by_proc[p] = by_proc.get(p, 0) + 1
    (VIDEOS / "manifest.json").write_text(
        json.dumps(
            {
                "version": "2",
                "stage": "B",
                "count": len(rows),
                "by_procedure": by_proc,
                "val": [r["input"]["video_id"] for r in rows if r.get("split") == "val"],
                "source_dataset": "teachonce_self_synthetic",
                "model_target": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
                "adapter": "scripts/tool_registry_adapter.py",
                "incomplete": False,
                "checkpoint": str(ckpt_path(procedure)),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", choices=["warranty", "parts", "escalate"], default="warranty")
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

    jobs = build_jobs(args.count, args.procedure)
    if args.limit:
        jobs = jobs[: args.limit]

    ckpt = load_ckpt(args.procedure)
    if not ckpt.get("started_at"):
        ckpt["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.force:
        ckpt["completed"] = {}
        ckpt["failed"] = {}
    else:
        n_disk = sync_disk_completed(ckpt, args.procedure)
        if n_disk:
            save_ckpt(ckpt, args.procedure)
            print(f"synced {n_disk} finished samples from disk into checkpoint")

    pending = [j for j in jobs if args.force or j["video_id"] not in ckpt["completed"]]
    print(
        f"procedure={args.procedure} workers={args.workers} total={len(jobs)} "
        f"pending={len(pending)} done={len(ckpt['completed'])} pause_file={PAUSE}"
    )
    if not pending:
        merge_train(ckpt, args.procedure)
        print("nothing to do; train merged")
        return

    if PAUSE.exists():
        print(f"paused — remove {PAUSE} and re-run to resume")
        return

    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
        print("\nSIGINT/SIGTERM — draining in-flight only, then exit", flush=True)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Cap in-flight so pause/signal does not leave hundreds queued.
    max_inflight = max(args.workers * 2, args.workers)
    queue = list(pending)
    futures: dict = {}
    paused = False
    pbar = tqdm(total=len(pending), desc=f"stage-b:{args.procedure}", unit="vid")

    def _record(fut, vid: str) -> None:
        result = fut.result()
        if result.get("ok"):
            ckpt["completed"][vid] = {
                "video": result["video"],
                "procedure": result.get("procedure", args.procedure),
                "path_type": result["path_type"],
                "size_mb": result["size_mb"],
                "ts": time.time(),
            }
            ckpt["failed"].pop(vid, None)
        else:
            ckpt["failed"][vid] = result.get("error", "unknown")
        save_ckpt(ckpt, args.procedure)
        pbar.update(1)
        pbar.set_postfix(
            done=len(ckpt["completed"]),
            fail=len(ckpt["failed"]),
            last=vid,
            q=len(queue),
        )

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            while queue or futures:
                want_stop = stop["flag"] or PAUSE.exists()
                while queue and len(futures) < max_inflight and not want_stop:
                    j = queue.pop(0)
                    futures[ex.submit(_worker, j)] = j["video_id"]
                if not futures:
                    paused = True
                    break
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    vid = futures.pop(fut)
                    _record(fut, vid)
                if stop["flag"] or PAUSE.exists():
                    paused = True
                    # drain in-flight only; do not pull more from queue
                    while futures:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        for fut in done:
                            vid = futures.pop(fut)
                            _record(fut, vid)
                    break
    finally:
        if paused:
            reason = "pause" if PAUSE.exists() else "signal"
            pbar.set_description(f"stopped ({reason})")
        pbar.close()

    merge_train(ckpt, args.procedure)
    left = len(queue)
    print(
        json.dumps(
            {
                "procedure": args.procedure,
                "completed": len(ckpt["completed"]),
                "failed": ckpt["failed"],
                "remaining": left,
                "train": str(TRAIN),
                "checkpoint": str(ckpt_path(args.procedure)),
                "paused": PAUSE.exists() or paused,
                "resume": (
                    f"rm {PAUSE}; re-run same command"
                    if (PAUSE.exists() or left)
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
