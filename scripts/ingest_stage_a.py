#!/usr/bin/env python3
"""
Stage A (continuous video only): OSTask .mkv + VideoCUA .mp4
→ keep source JSON with each sample
→ try LLD §3.3 train rows via tool_registry_adapter (quarantine if ops don't map)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tool_registry_adapter import MVP_OPS, adapt_record, validate_registry_dag  # noqa: E402

RAW = ROOT / "data" / "raw"
RECORDS = ROOT / "data" / "generated" / "stage_a_records"
OUT_SPLIT = ROOT / "data" / "generated" / "splits"
QUAR = ROOT / "data" / "generated" / "quarantine"
INDEX = OUT_SPLIT / "stage_a_index.jsonl"
TRAIN = OUT_SPLIT / "stage_a_train.jsonl"


def steps_from_text(text: str) -> list[dict]:
    chunks = [c.strip() for c in text.replace("?", ".").split(".") if c.strip()]
    return [{"label": c, "description": c} for c in (chunks or [text])]


def steps_from_videocua_log(action_log: list) -> list[dict]:
    """Prefer typed text; never use raw x/y as schema events."""
    steps = []
    for a in action_log:
        params = a.get("action_params") or {}
        text = params.get("text") or a.get("action_type") or ""
        steps.append(
            {
                "action": a.get("action_type"),
                "label": str(text),
                "description": str(text),
                "timestamp": a.get("timestamp"),
            }
        )
    return steps


def validate_row(row: dict) -> list[str]:
    errs = []
    if row.get("family") != "warranty_resolution":
        errs.append("family must be warranty_resolution")
    if row.get("input", {}).get("tools_ref") != "mvp_registry_v1":
        errs.append("tools_ref")
    for ev in (row.get("input", {}).get("demonstrations") or [{}])[0].get("events", []):
        if ev.get("op") not in MVP_OPS:
            errs.append(f"bad op {ev.get('op')}")
    errs.extend(validate_registry_dag(row.get("target_dag") or {}))
    return errs


def write_record_dir(video_id: str, *, video: Path, source: dict, goal: str, source_dataset: str, source_record_id: str) -> Path:
    d = RECORDS / video_id
    d.mkdir(parents=True, exist_ok=True)
    # copy source JSON next to a pointer for the video
    (d / "source.json").write_text(json.dumps(source, indent=2))
    meta = {
        "video_id": video_id,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "goal": goal,
        "video_path": str(video.resolve()) if video.exists() else None,
        "video_exists": video.exists(),
        "source_json": str((d / "source.json").resolve()),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    # soft-copy video into record dir for a self-contained sample folder
    if video.exists():
        dest = d / video.name
        if not dest.exists():
            shutil.copy2(video, dest)
        meta["video_local"] = str(dest.resolve())
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def try_train_row(
    *,
    video_id: str,
    goal: str,
    video: Path,
    steps: list,
    source_dataset: str,
    source_record_id: str,
) -> dict | None:
    adapted = adapt_record(
        source_dataset,
        steps,
        workflow_id=f"WF-{video_id}",
        name=goal[:80] or video_id,
        require_min_ops=2,
    )
    if not adapted["ok"]:
        QUAR.mkdir(parents=True, exist_ok=True)
        (QUAR / f"{video_id}.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "video_id": video_id,
                    "source_dataset": source_dataset,
                    "source_record_id": source_record_id,
                    "goal": goal,
                    "video_path": str(video) if video.exists() else None,
                    "errors": adapted["errors"],
                    "ops": adapted["ops"],
                },
                indent=2,
            )
        )
        print(f"quarantine {video_id}: {adapted['errors']}")
        return None

    events = []
    t = 0.0
    for op in adapted["ops"]:
        events.append(
            {
                "t_start": t,
                "t_end": t + 5.0,
                "op": op,
                "label": op.replace("_", " ").title(),
                "observed": {},
            }
        )
        t += 5.0

    row = {
        "id": f"train-{video_id}",
        "family": "warranty_resolution",
        "split": "train",
        "path_type": None,
        "source_dataset": source_dataset,
        "source_record_id": str(source_record_id),
        "input": {
            "goal": goal,
            "video_id": video_id,
            "video_path": str(video.resolve()) if video.exists() else None,
            "sop": None,
            "policies": [],
            "demonstrations": [{"video_id": video_id, "events": events}],
            "tools_ref": "mvp_registry_v1",
        },
        "target_dag": adapted["dag"],
    }
    verrs = validate_row(row)
    if verrs:
        QUAR.mkdir(parents=True, exist_ok=True)
        (QUAR / f"{video_id}.json").write_text(json.dumps({"ok": False, "errors": verrs, "row": row}, indent=2))
        print(f"quarantine {video_id} (schema): {verrs}")
        return None
    print(f"accept {video_id} ops={[e['op'] for e in events]}")
    return row


def collect_ostask() -> list[dict]:
    raw = RAW / "ostask-demo"
    out = []
    for jp in sorted(raw.glob("*.json")):
        if jp.name.startswith("DOWNLOAD"):
            continue
        task = jp.stem
        source = json.loads(jp.read_text())
        goal = str(source.get("instruction") or source.get("goal") or task)
        video = raw / f"{task}.mkv"
        video_id = f"ostask_{task}"
        write_record_dir(
            video_id,
            video=video,
            source=source,
            goal=goal,
            source_dataset="open-world-agents/OSTask-demo",
            source_record_id=task,
        )
        row = try_train_row(
            video_id=video_id,
            goal=goal,
            video=video,
            steps=steps_from_text(goal),
            source_dataset="open-world-agents/OSTask-demo",
            source_record_id=task,
        )
        out.append(
            {
                "video_id": video_id,
                "source_dataset": "open-world-agents/OSTask-demo",
                "goal": goal,
                "video_path": str(video.resolve()) if video.exists() else None,
                "source_json": str((RECORDS / video_id / "source.json").resolve()),
                "record_dir": str((RECORDS / video_id).resolve()),
                "train_ok": row is not None,
            }
        )
        if row:
            out[-1]["train_row"] = row
    return out


def collect_videocua() -> list[dict]:
    root = RAW / "videocua" / "Conky"
    out = []
    if not root.exists():
        return out
    for task_dir in sorted(root.glob("task_*")):
        log_path = task_dir / "action_log.json"
        video = task_dir / "video" / "video.mp4"
        if not log_path.exists():
            continue
        source = json.loads(log_path.read_text())
        goal = str(source.get("task_instruction") or task_dir.name)
        tid = str(source.get("task_id") or task_dir.name)
        video_id = f"videocua_{tid}"
        write_record_dir(
            video_id,
            video=video,
            source=source,
            goal=goal,
            source_dataset="ServiceNow/VideoCUA",
            source_record_id=tid,
        )
        steps = steps_from_text(goal) + steps_from_videocua_log(source.get("action_log") or [])
        row = try_train_row(
            video_id=video_id,
            goal=goal,
            video=video,
            steps=steps,
            source_dataset="ServiceNow/VideoCUA",
            source_record_id=tid,
        )
        out.append(
            {
                "video_id": video_id,
                "source_dataset": "ServiceNow/VideoCUA",
                "goal": goal,
                "video_path": str(video.resolve()) if video.exists() else None,
                "source_json": str((RECORDS / video_id / "source.json").resolve()),
                "record_dir": str((RECORDS / video_id).resolve()),
                "train_ok": row is not None,
            }
        )
        if row:
            out[-1]["train_row"] = row
    return out


def main() -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.mkdir(parents=True, exist_ok=True)
    items = collect_ostask() + collect_videocua()
    if not items:
        print("no video samples — run scripts/download_video_sources.py", file=sys.stderr)
        sys.exit(1)

    # index: every sample with video + JSON paths
    index_lines = []
    train_rows = []
    for it in items:
        index_lines.append({k: v for k, v in it.items() if k != "train_row"})
        if it.get("train_row"):
            train_rows.append(it["train_row"])

    INDEX.write_text("\n".join(json.dumps(x) for x in index_lines) + "\n")
    TRAIN.write_text("\n".join(json.dumps(r) for r in train_rows) + ("\n" if train_rows else ""))
    print(f"records={len(items)} → {RECORDS}")
    print(f"index={len(index_lines)} → {INDEX}")
    print(f"train_ok={len(train_rows)} quarantined={len(items)-len(train_rows)} → {TRAIN}")


if __name__ == "__main__":
    main()
