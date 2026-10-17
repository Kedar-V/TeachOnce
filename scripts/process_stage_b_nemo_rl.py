#!/usr/bin/env python3
"""Incrementally convert Stage-B sample folders → NeMo-RL GT.

Skips already-processed video_ids (unless DAG fingerprint changed).
Writes progress.json for live tracking. Optional --watch poll loop.

On Brev:
  STAGE_B_VIDEOS=/data/teachonce/demonstrations/videos \\
  NEMO_RL_OUT=~/teachonce/data/generated/nemo_rl \\
  python3 scripts/process_stage_b_nemo_rl.py --watch --interval 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import dag_reward, from_sample_dir  # noqa: E402

DEFAULT_VIDEOS = ROOT / "data" / "generated" / "demonstrations" / "videos"
DEFAULT_OUT = ROOT / "data" / "generated" / "nemo_rl"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def discover_ready(videos_root: Path) -> list[Path]:
    """Folders with both video.mp4 and expected_dag.json."""
    ready = []
    if not videos_root.is_dir():
        return ready
    for d in sorted(videos_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "video.mp4").exists() and (d / "expected_dag.json").exists():
            ready.append(d)
    return ready


def _append_gt_line(gt_path: Path, record: dict[str, Any]) -> None:
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with gt_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _drop_gt_rows(gt_path: Path, *, video_ids: set[str], gt_ids: set[str]) -> None:
    """Remove rows for reprocessed samples (fingerprint changed)."""
    if not gt_path.exists() or (not video_ids and not gt_ids):
        return
    kept = []
    for line in gt_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") in gt_ids or row.get("video_id") in video_ids:
            continue
        kept.append(row)
    gt_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + ("\n" if kept else ""))


def process_once(
    videos_root: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "gt.jsonl"
    state_path = out_dir / "processed.json"
    progress_path = out_dir / "progress.json"

    state: dict[str, Any] = {} if force else _load_json(state_path, {})
    if force and gt_path.exists():
        gt_path.write_text("")
    ready = discover_ready(videos_root)

    pending: list[str] = []
    skipped: list[str] = []
    processed_now: list[str] = []
    failed: list[dict[str, str]] = []

    for sample_dir in ready:
        dag_path = sample_dir / "expected_dag.json"
        meta_path = sample_dir / "meta.json"
        video_id = sample_dir.name
        if meta_path.exists():
            try:
                video_id = json.loads(meta_path.read_text()).get("video_id") or video_id
            except json.JSONDecodeError:
                pass

        fingerprint = _sha_file(dag_path) + ":" + _sha_file(sample_dir / "video.mp4")
        prev = state.get(video_id)
        if not force and prev and prev.get("fingerprint") == fingerprint:
            skipped.append(video_id)
            continue

        pending.append(video_id)
        try:
            if prev:
                _drop_gt_rows(
                    gt_path,
                    video_ids={video_id},
                    gt_ids={prev.get("gt_id")} if prev.get("gt_id") else set(),
                )

            record = from_sample_dir(sample_dir)
            # rewrite video_path to absolute under videos_root (Brev-stable)
            record["video_path"] = str((sample_dir / "video.mp4").resolve())
            score = dag_reward(record["gold_dag"], record)
            if score["reward"] < 0.99:
                raise ValueError(f"gold self-score {score}")

            _append_gt_line(gt_path, record)
            state[video_id] = {
                "gt_id": record["id"],
                "fingerprint": fingerprint,
                "family": record.get("family"),
                "path_type": record.get("path_type"),
                "processed_at": _now(),
                "reward_self": score["reward"],
            }
            processed_now.append(video_id)
            print(f"[ok] {video_id} → {record['id']}", flush=True)
        except Exception as e:
            failed.append({"video_id": video_id, "error": str(e)})
            print(f"[fail] {video_id}: {e}", flush=True)

    # refresh manifest from gt.jsonl
    n_lines = 0
    ids = []
    if gt_path.exists():
        for line in gt_path.read_text().splitlines():
            if line.strip():
                n_lines += 1
                try:
                    ids.append(json.loads(line)["id"])
                except Exception:
                    pass
    _save_json(
        out_dir / "manifest.json",
        {
            "version": "1",
            "task": "video_to_dag",
            "count": n_lines,
            "path": str(gt_path.resolve()),
            "ids": ids,
            "reward": "scripts.nemo_rl_gt.dag_reward",
            "updated_at": _now(),
        },
    )
    _save_json(state_path, state)

    still_pending = [
        d.name
        for d in sorted(videos_root.iterdir())
        if d.is_dir()
        and d.name not in state
        and not ((d / "video.mp4").exists() and (d / "expected_dag.json").exists())
    ]

    progress = {
        "updated_at": _now(),
        "videos_root": str(videos_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "ready": len(ready),
        "processed_total": len(state),
        "processed_this_run": processed_now,
        "skipped_this_run": skipped,
        "failed_this_run": failed,
        "incomplete_folders": still_pending,  # dirs missing video or dag
        "gt_rows": n_lines,
        "pct_ready_done": round(100.0 * len(state) / len(ready), 1) if ready else 100.0,
    }
    _save_json(progress_path, progress)
    return progress


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental Stage-B → NeMo-RL GT")
    ap.add_argument(
        "--videos",
        type=Path,
        default=Path(os.environ.get("STAGE_B_VIDEOS", DEFAULT_VIDEOS)),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("NEMO_RL_OUT", DEFAULT_OUT)),
    )
    ap.add_argument("--watch", action="store_true", help="poll forever for new samples")
    ap.add_argument("--interval", type=int, default=30, help="watch poll seconds")
    ap.add_argument("--force", action="store_true", help="reprocess all")
    ap.add_argument("--once", action="store_true", help="single pass (default if no --watch)")
    args = ap.parse_args()

    def run():
        p = process_once(args.videos, args.out, force=args.force)
        print(
            json.dumps(
                {
                    "ready": p["ready"],
                    "processed_total": p["processed_total"],
                    "new": len(p["processed_this_run"]),
                    "skipped": len(p["skipped_this_run"]),
                    "failed": len(p["failed_this_run"]),
                    "incomplete": len(p["incomplete_folders"]),
                    "pct": p["pct_ready_done"],
                    "progress": str((args.out / "progress.json").resolve()),
                },
                indent=2,
            ),
            flush=True,
        )
        return p

    if args.watch:
        print(f"watching {args.videos} every {args.interval}s → {args.out}", flush=True)
        while True:
            run()
            time.sleep(args.interval)
    else:
        run()


if __name__ == "__main__":
    main()
