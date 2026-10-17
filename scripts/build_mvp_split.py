#!/usr/bin/env python3
"""Build MVP train/val JSONL from Stage B warranty videos only (no VideoCUA)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tool_registry_adapter import adapt_record  # noqa: E402

VIDEOS = ROOT / "data" / "generated" / "demonstrations" / "videos"
OUT = ROOT / "data" / "generated" / "splits"
QUAR = ROOT / "data" / "generated" / "quarantine"


def row_from_meta(meta_path: Path) -> dict | None:
    meta = json.loads(meta_path.read_text())
    vid = meta["video_id"]
    video = meta_path.parent / "video.mp4"
    steps = meta.get("events") or [{"op": o} for o in meta.get("ops", [])]
    adapted = adapt_record(
        meta.get("source_dataset", "teachonce_self_synthetic"),
        steps,
        workflow_id=f"WF-{vid}",
        name=f"{meta.get('family', 'workflow')} ({meta.get('path_type', 'unknown')})",
    )
    if not adapted["ok"]:
        QUAR.mkdir(parents=True, exist_ok=True)
        (QUAR / f"{vid}.json").write_text(json.dumps(adapted, indent=2))
        print(f"quarantine {vid}: {adapted['errors']}", file=sys.stderr)
        return None

    # prefer checked-in expected_dag if present and valid
    dag_path = meta_path.parent / "expected_dag.json"
    dag = adapted["dag"]
    if dag_path.exists():
        dag = json.loads(dag_path.read_text())

    return {
        "id": f"train-{vid}",
        "family": meta.get("family", "warranty_resolution"),
        "split": meta.get("split", "train"),
        "path_type": meta.get("path_type"),
        "procedure": meta.get("procedure"),
        "source_dataset": meta.get("source_dataset", "teachonce_self_synthetic"),
        "input": {
            "goal": meta.get("goal"),
            "video_id": vid,
            "video_path": str(video.resolve()) if video.exists() else None,
            "sop": meta.get("sop"),
            "policies": [],
            "demonstrations": [{"video_id": vid, "events": meta.get("events", [])}],
            "tools_ref": meta.get("tools_ref", "mvp_registry_v1"),
        },
        "target_dag": dag,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metas = sorted(
        list(VIDEOS.glob("warranty_*/meta.json"))
        + list(VIDEOS.glob("parts_*/meta.json"))
        + list(VIDEOS.glob("escalate_*/meta.json"))
    )
    rows = [r for m in metas if (r := row_from_meta(m))]

    # Prefer meta.split; else hold out rows marked val, or last 2 when ≥3
    train, val = [], []
    for r in rows:
        (val if r.get("split") == "val" else train).append(r)
    if not val and len(rows) >= 3:
        val, train = rows[-2:], rows[:-2]
        for r in train:
            r["split"] = "train"
        for r in val:
            r["split"] = "val"

    train_path = OUT / "mvp_train.jsonl"
    val_path = OUT / "mvp_val.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train) + ("\n" if train else ""))
    val_path.write_text("\n".join(json.dumps(r) for r in val) + ("\n" if val else ""))

    print(f"mvp_train={len(train)} → {train_path}")
    print(f"mvp_val={len(val)} → {val_path}")
    print("skip: VideoCUA/AgentNet (Stage A) — see docs/phases/mvp-train-subset.md")


if __name__ == "__main__":
    main()
