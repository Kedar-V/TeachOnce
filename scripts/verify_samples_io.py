#!/usr/bin/env python3
"""Verify raw samples → TeachOnce I/O (schema + registry adapter)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tool_registry_adapter import MVP_OPS, adapt_record, validate_registry_dag  # noqa: E402


def check_train_row(row: dict) -> list[str]:
    errs: list[str] = []
    for k in ("id", "family", "split", "input", "target_dag"):
        if k not in row:
            errs.append(f"missing {k}")
    if row.get("family") != "warranty_resolution":
        errs.append("family != warranty_resolution")
    inp = row.get("input") or {}
    if not inp.get("video_id"):
        errs.append("input.video_id missing")
    if inp.get("tools_ref") != "mvp_registry_v1":
        errs.append("tools_ref != mvp_registry_v1")
    demos = inp.get("demonstrations") or []
    if not demos:
        errs.append("demonstrations empty")
    else:
        for ev in demos[0].get("events") or []:
            if ev.get("op") not in MVP_OPS:
                errs.append(f"bad event op {ev.get('op')}")
    errs.extend(validate_registry_dag(row.get("target_dag") or {}))
    vp = inp.get("video_path")
    if vp and not Path(vp).exists():
        errs.append(f"video_path missing on disk: {vp}")
    return errs


def verify_stage_b() -> None:
    print("\n=== Stage B (TeachOnce gold) ===")
    split = ROOT / "data/generated/splits/mvp_train.jsonl"
    if not split.exists() or split.stat().st_size == 0:
        print("FAIL no mvp_train.jsonl")
        return
    for line in split.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        errs = check_train_row(row)
        vid = row["input"].get("video_id")
        n_ev = len(row["input"]["demonstrations"][0]["events"])
        n_nodes = len(row["target_dag"]["nodes"])
        status = "PASS" if not errs else "FAIL"
        print(f"{status} {row['id']} video={vid} events={n_ev} dag_nodes={n_nodes}")
        print(f"  INPUT  goal={row['input'].get('goal')!r}")
        print(f"         ops={[e['op'] for e in row['input']['demonstrations'][0]['events']]}")
        print(f"  OUTPUT dag.id={row['target_dag']['id']} actions="
              f"{[n['op'] for n in row['target_dag']['nodes'] if n.get('type')=='ACTION']}")
        for e in errs:
            print(f"  ! {e}")


def verify_videocua() -> None:
    print("\n=== VideoCUA raw (Conky) ===")
    base = ROOT / "data/raw/videocua/Conky"
    if not base.exists():
        print("FAIL no VideoCUA samples")
        return
    for task in sorted(base.glob("task_*")):
        log = task / "action_log.json"
        mp4 = next(task.rglob("video.mp4"), None)
        if not log.exists():
            print(f"FAIL {task.name}: no action_log.json")
            continue
        data = json.loads(log.read_text())
        instruction = data.get("task_instruction") or ""
        actions = data.get("action_log") or []
        steps = [{"label": a.get("action_type", ""), "description": json.dumps(a.get("action_params", {}))} for a in actions]
        steps.append({"label": instruction, "description": instruction})
        adapted = adapt_record("videocua", steps, workflow_id=f"WF-{task.name}", require_min_ops=2)
        mp4_ok = mp4 is not None and mp4.exists()
        print(f"{'RAW_OK' if mp4_ok and log.exists() else 'RAW_BAD'} {task.name}")
        print(f"  INPUT  instruction={instruction!r}")
        print(f"         video={'OK '+str(mp4.stat().st_size)+'B' if mp4_ok else 'MISSING'}")
        print(f"         raw_actions={[a.get('action_type') for a in actions]}")
        print(f"  OUTPUT teachonce_adapter ok={adapted['ok']} ops={adapted['ops']} errors={adapted['errors']}")


def verify_ostask() -> None:
    print("\n=== OSTask-demo raw ===")
    base = ROOT / "data/raw/ostask-demo"
    for jf in sorted(base.glob("*.json")):
        if jf.name == "DOWNLOAD_MANIFEST.json":
            continue
        task = jf.stem
        inst = json.loads(jf.read_text()).get("instruction", "")
        mkv = base / f"{task}.mkv"
        chunks = [c.strip() for c in inst.replace("?", ".").split(".") if c.strip()] or [inst]
        adapted = adapt_record("ostask", [{"label": c, "description": c} for c in chunks], require_min_ops=2)
        print(f"{'RAW_OK' if mkv.exists() else 'RAW_BAD'} {task}")
        print(f"  INPUT  instruction={inst[:100]!r}…")
        print(f"         video={'OK' if mkv.exists() else 'MISSING'} ({mkv.stat().st_size if mkv.exists() else 0} B)")
        print(f"  OUTPUT teachonce_adapter ok={adapted['ok']} ops={adapted['ops']} errors={adapted['errors']}")


def verify_agentnet() -> None:
    print("\n=== AgentNet samples ===")
    p = ROOT / "data/raw/agentnet/ubuntu_samples_5.jsonl"
    if not p.exists():
        print("FAIL missing ubuntu_samples_5.jsonl")
        return
    for i, line in enumerate(p.read_text().splitlines()):
        if not line.strip():
            continue
        s = json.loads(line)
        inst = s.get("instruction") or s.get("natural_language_task") or ""
        traj = s.get("traj") or []
        steps = []
        for step in traj:
            val = step.get("value") or {}
            steps.append({"label": val.get("action") or "", "description": val.get("thought") or ""})
        steps.append({"label": inst, "description": inst})
        adapted = adapt_record("agentnet", steps, require_min_ops=2)
        print(f"RAW_OK traj[{i}] steps={len(traj)} (screenshots only — no MP4 in this pull)")
        print(f"  INPUT  instruction={inst[:100]!r}")
        print(f"  OUTPUT teachonce_adapter ok={adapted['ok']} ops={adapted['ops']} errors={adapted['errors']}")


def verify_videogui() -> None:
    print("\n=== VideoGUI Mid-Plan samples ===")
    p = ROOT / "data/raw/videogui/mid_plan_samples_3.json"
    if not p.exists():
        print("FAIL missing mid_plan_samples_3.json")
        return
    rows = json.loads(p.read_text())
    for i, row in enumerate(rows):
        q = row.get("subtask_query") or row.get("action_sequence") or ""
        start = ROOT / "data/raw/videogui" / f"sample_{i:02d}" / "screenshot_start.png"
        end = ROOT / "data/raw/videogui" / f"sample_{i:02d}" / "screenshot_end.png"
        adapted = adapt_record(
            "videogui",
            [{"label": str(q), "description": str(row.get("action_sequence") or q)}],
            require_min_ops=2,
        )
        print(f"{'RAW_OK' if start.exists() else 'RAW_BAD'} sample_{i:02d} (screenshots — no MP4 on HF)")
        print(f"  INPUT  query={str(q)[:100]!r}")
        print(f"         screenshots start={start.exists()} end={end.exists()}")
        print(f"  OUTPUT teachonce_adapter ok={adapted['ok']} ops={adapted['ops']} errors={adapted['errors']}")


def main() -> None:
    verify_stage_b()
    verify_videocua()
    verify_ostask()
    verify_agentnet()
    verify_videogui()
    print("\n=== verdict ===")
    print("TeachOnce train I/O = schema-valid Stage B (and any adapter-ok Stage A).")
    print("Raw public videos/screenshots alone are NOT train outputs until adapter ok=true.")


if __name__ == "__main__":
    main()
