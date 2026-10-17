#!/usr/bin/env python3
"""Validate train JSONL against frozen schemas + registry DAG rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tool_registry_adapter import MVP_OPS, validate_registry_dag  # noqa: E402

FAMILY = "warranty_resolution"


def check_row(row: dict, i: int) -> list[str]:
    errs: list[str] = []
    for k in ("id", "family", "split", "input", "target_dag"):
        if k not in row:
            errs.append(f"row{i}: missing {k}")
    if row.get("family") != FAMILY:
        errs.append(f"row{i}: family != {FAMILY}")
    inp = row.get("input") or {}
    if inp.get("tools_ref") != "mvp_registry_v1":
        errs.append(f"row{i}: tools_ref")
    if not inp.get("video_id"):
        errs.append(f"row{i}: video_id required")
    demos = inp.get("demonstrations") or []
    if not demos:
        errs.append(f"row{i}: demonstrations required")
    for ev in demos[0].get("events", []) if demos else []:
        if ev.get("op") not in MVP_OPS:
            errs.append(f"row{i}: bad event op {ev.get('op')}")
    dag = row.get("target_dag") or {}
    for k in ("id", "name", "version", "nodes", "edges"):
        if k not in dag:
            errs.append(f"row{i}: dag missing {k}")
    errs.extend(f"row{i}: {e}" for e in validate_registry_dag(dag))
    return errs


def main(paths: list[str]) -> None:
    if not paths:
        paths = [
            str(ROOT / "data/generated/splits/mvp_train.jsonl"),
            str(ROOT / "data/generated/splits/stage_a_train.jsonl"),
        ]
    all_errs: list[str] = []
    n = 0
    for path in paths:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            print(f"skip missing/empty {p}")
            continue
        for i, line in enumerate(p.read_text().splitlines()):
            if not line.strip():
                continue
            n += 1
            all_errs.extend(check_row(json.loads(line), i))
    if all_errs:
        print("FAIL")
        print("\n".join(all_errs[:50]))
        sys.exit(1)
    print(f"OK rows={n} files={paths}")


if __name__ == "__main__":
    main(sys.argv[1:])
