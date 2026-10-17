#!/usr/bin/env python3
"""NeMo-RL ground-truth layer on top of video→DAG SFT rows.

Does not change the generate path. Builds reward-ready GT records from
(video_path, target_dag [, goal/events]) and writes JSONL for Gym / GRPO.

Usage:
  python3 scripts/nemo_rl_gt.py \\
    --from-split data/generated/splits/mvp_train.jsonl \\
    --out data/generated/nemo_rl/gt.jsonl

  python3 scripts/nemo_rl_gt.py \\
    --video path/to/video.mp4 --dag path/to/expected_dag.json \\
    --goal "..." --out data/generated/nemo_rl/one.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tool_registry_adapter import MVP_OPS, validate_registry_dag  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "generated" / "nemo_rl" / "gt.jsonl"

PROMPT_TMPL = """Compile an executable workflow DAG JSON from the expert screen-recording video.

Goal: {goal}
Family: {family}
Tools ref: {tools_ref}

Return ONLY valid DAG JSON with nodes (START|ACTION|DECISION|END) and edges.
ACTION.op must be registry ops only. Do not invent tools.
"""


def _ops_from_dag(dag: dict[str, Any]) -> list[str]:
    ops: list[str] = []
    for n in dag.get("nodes") or []:
        if n.get("type") == "ACTION" and n.get("op"):
            op = n["op"]
            if op not in ops:
                ops.append(op)
    return ops


def _critical_ops(dag: dict[str, Any]) -> set[str]:
    """Ops that define family topology (decision + both branch tips if present)."""
    ops = set(_ops_from_dag(dag))
    # keep decision-defining tips even if only one filmed
    tips = {
        "CREATE_REPAIR",
        "RECORD_PAID_OPTIONS",
        "CREATE_PARTS_SHIPMENT",
        "CREATE_BACKORDER",
        "PAGE_ONCALL",
        "ASSIGN_L2_QUEUE",
        "CHECK_WARRANTY",
        "CHECK_PARTS_STOCK",
        "ASSESS_SEVERITY",
    }
    return ops & (set(MVP_OPS) | tips)


def _taken_branch(path_type: str | None, ops: list[str]) -> str | None:
    if path_type in ("active", "expired", "in_stock", "backorder", "p1", "standard"):
        return path_type
    # infer from ops if path_type missing
    for tip, name in (
        ("CREATE_REPAIR", "active"),
        ("RECORD_PAID_OPTIONS", "expired"),
        ("CREATE_PARTS_SHIPMENT", "in_stock"),
        ("CREATE_BACKORDER", "backorder"),
        ("PAGE_ONCALL", "p1"),
        ("ASSIGN_L2_QUEUE", "standard"),
    ):
        if tip in ops:
            return name
    return None


def build_gt(
    *,
    video_path: str | Path,
    dag: dict[str, Any],
    goal: str = "Compile workflow DAG from expert video",
    family: str = "warranty_resolution",
    path_type: str | None = None,
    video_id: str | None = None,
    tools_ref: str = "mvp_registry_v1",
    events: list[dict[str, Any]] | None = None,
    row_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one NeMo-RL GT record from video + gold DAG."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    errs = validate_registry_dag(dag)
    if errs:
        raise ValueError(f"invalid gold DAG: {errs}")

    vid = video_id or video_path.parent.name or video_path.stem
    ops = _ops_from_dag(dag)
    filmed = [e.get("op") for e in (events or []) if e.get("op")]
    branch = _taken_branch(path_type, filmed or ops)

    record = {
        "id": row_id or f"nemo-rl-{vid}",
        "task": "video_to_dag",
        "family": family,
        "path_type": path_type,
        "split": (extra or {}).get("split", "train"),
        # --- model input ---
        "prompt": PROMPT_TMPL.format(goal=goal, family=family, tools_ref=tools_ref),
        "video_path": str(video_path.resolve()),
        "video_id": vid,
        "tools_ref": tools_ref,
        # --- ground truth for reward / SFT target reuse ---
        "gold_dag": dag,
        "gold_dag_json": json.dumps(dag, separators=(",", ":"), sort_keys=True),
        "gold_ops": ops,
        "filmed_ops": filmed,
        "taken_branch": branch,
        "critical_ops": sorted(_critical_ops(dag)),
        "events": events or [],
        # --- reward contract (Gym / GRPO read this; generate path ignores it) ---
        "reward_spec": {
            "type": "dag_match",
            "weights": {
                "schema_ok": 0.4,
                "critical_ops": 0.3,
                "taken_branch": 0.3,
            },
            "schema": "validate_registry_dag",
        },
    }
    if extra:
        for k, v in extra.items():
            if k not in record:
                record[k] = v
    return record


def dag_reward(pred: dict[str, Any] | str, gt: dict[str, Any]) -> dict[str, Any]:
    """Score a predicted DAG against a GT record. Layer on top of generate."""
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return {"reward": 0.0, "schema_ok": 0.0, "critical_ops": 0.0, "taken_branch": 0.0, "error": "bad_json"}

    w = (gt.get("reward_spec") or {}).get("weights") or {
        "schema_ok": 0.4,
        "critical_ops": 0.3,
        "taken_branch": 0.3,
    }
    errs = validate_registry_dag(pred) if isinstance(pred, dict) else ["not_dict"]
    schema_ok = 1.0 if not errs else 0.0

    gold_crit = set(gt.get("critical_ops") or [])
    pred_ops = set(_ops_from_dag(pred)) if isinstance(pred, dict) else set()
    if gold_crit:
        critical = len(gold_crit & pred_ops) / len(gold_crit)
    else:
        critical = 1.0 if pred_ops else 0.0

    branch = gt.get("taken_branch")
    branch_score = 0.0
    if not branch:
        branch_score = 1.0  # no branch to check
    else:
        tip = {
            "active": "CREATE_REPAIR",
            "expired": "RECORD_PAID_OPTIONS",
            "in_stock": "CREATE_PARTS_SHIPMENT",
            "backorder": "CREATE_BACKORDER",
            "p1": "PAGE_ONCALL",
            "standard": "ASSIGN_L2_QUEUE",
        }.get(branch)
        branch_score = 1.0 if tip and tip in pred_ops else 0.0

    reward = (
        w.get("schema_ok", 0.4) * schema_ok
        + w.get("critical_ops", 0.3) * critical
        + w.get("taken_branch", 0.3) * branch_score
    )
    return {
        "reward": round(reward, 4),
        "schema_ok": schema_ok,
        "critical_ops": round(critical, 4),
        "taken_branch": branch_score,
        "errors": errs,
    }


def from_train_row(row: dict[str, Any]) -> dict[str, Any]:
    inp = row.get("input") or {}
    video = inp.get("video_path")
    if not video:
        raise ValueError(f"{row.get('id')}: missing input.video_path")
    events = []
    for d in inp.get("demonstrations") or []:
        events.extend(d.get("events") or [])
    return build_gt(
        video_path=video,
        dag=row["target_dag"],
        goal=inp.get("goal") or "Compile workflow DAG from expert video",
        family=row.get("family") or "warranty_resolution",
        path_type=row.get("path_type"),
        video_id=inp.get("video_id"),
        tools_ref=inp.get("tools_ref") or "mvp_registry_v1",
        events=events,
        row_id=f"nemo-rl-{row.get('id', Path(video).stem)}",
        extra={"split": row.get("split", "train"), "source_id": row.get("id"), "procedure": row.get("procedure")},
    )


def from_sample_dir(sample_dir: Path) -> dict[str, Any]:
    """Load warranty_XX/ / parts_XX/ folder with video.mp4 + expected_dag.json (+ meta.json)."""
    sample_dir = Path(sample_dir)
    video = sample_dir / "video.mp4"
    dag_path = sample_dir / "expected_dag.json"
    meta_path = sample_dir / "meta.json"
    if not video.exists() or not dag_path.exists():
        raise FileNotFoundError(f"need video.mp4 + expected_dag.json in {sample_dir}")
    dag = json.loads(dag_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return build_gt(
        video_path=video,
        dag=dag,
        goal=meta.get("goal") or "Compile workflow DAG from expert video",
        family=meta.get("family") or "warranty_resolution",
        path_type=meta.get("path_type"),
        video_id=meta.get("video_id") or sample_dir.name,
        tools_ref=meta.get("tools_ref") or "mvp_registry_v1",
        events=meta.get("events") or [],
        row_id=f"nemo-rl-{meta.get('video_id') or sample_dir.name}",
        extra={"split": meta.get("split", "train"), "procedure": meta.get("procedure")},
    )


def save_gt(records: list[dict[str, Any]], out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""))
    # sidecar index for Gym loaders
    index = {
        "version": "1",
        "task": "video_to_dag",
        "count": len(records),
        "path": str(out.resolve()),
        "ids": [r["id"] for r in records],
        "reward": "scripts.nemo_rl_gt.dag_reward",
    }
    (out.parent / "manifest.json").write_text(json.dumps(index, indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NeMo-RL GT from video + DAG")
    ap.add_argument("--from-split", type=Path, help="SFT JSONL (mvp_train.jsonl)")
    ap.add_argument("--from-dir", type=Path, nargs="*", help="sample folder(s) with video.mp4 + expected_dag.json")
    ap.add_argument("--video", type=Path, help="single video path")
    ap.add_argument("--dag", type=Path, help="single expected_dag.json")
    ap.add_argument("--goal", type=str, default="Compile workflow DAG from expert video")
    ap.add_argument("--family", type=str, default="warranty_resolution")
    ap.add_argument("--path-type", type=str, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    records: list[dict[str, Any]] = []

    if args.from_split:
        for line in args.from_split.read_text().splitlines():
            if not line.strip():
                continue
            records.append(from_train_row(json.loads(line)))

    if args.from_dir:
        for d in args.from_dir:
            records.append(from_sample_dir(d))

    if args.video and args.dag:
        dag = json.loads(args.dag.read_text())
        records.append(
            build_gt(
                video_path=args.video,
                dag=dag,
                goal=args.goal,
                family=args.family,
                path_type=args.path_type,
            )
        )

    if not records:
        ap.error("provide --from-split and/or --from-dir and/or --video+--dag")

    path = save_gt(records, args.out)
    # sanity: gold scores 1.0 against itself
    bad = []
    for r in records:
        sc = dag_reward(r["gold_dag"], r)
        if sc["reward"] < 0.99:
            bad.append((r["id"], sc))
    print(json.dumps({"wrote": str(path), "count": len(records), "self_score_fail": bad or None}, indent=2))


def _selfcheck() -> None:
    dag = {
        "id": "WF-t",
        "name": "t",
        "version": "1.0.0",
        "nodes": [
            {"id": "start", "type": "START"},
            {
                "id": "a0",
                "type": "ACTION",
                "label": "Check Warranty",
                "tool": "warranty",
                "op": "CHECK_WARRANTY",
                "inputs": {"device_id": {"ref": "$case.device_id"}},
            },
            {
                "id": "decide_warranty",
                "type": "DECISION",
                "label": "Warranty status",
                "branches": [{"name": "Active"}, {"name": "Expired"}],
            },
            {
                "id": "branch_active",
                "type": "ACTION",
                "label": "Create Repair",
                "tool": "repair",
                "op": "CREATE_REPAIR",
                "inputs": {"device_id": {"ref": "$case.device_id"}, "ticket_id": {"ref": "$case.ticket_id"}},
            },
            {"id": "end_active", "type": "END", "outcome": "SUCCEEDED"},
            {
                "id": "branch_expired",
                "type": "ACTION",
                "label": "Record Paid Options",
                "tool": "repair",
                "op": "RECORD_PAID_OPTIONS",
                "inputs": {"device_id": {"ref": "$case.device_id"}, "ticket_id": {"ref": "$case.ticket_id"}},
            },
            {"id": "end_expired", "type": "END", "outcome": "SUCCEEDED"},
        ],
        "edges": [
            {"from": "start", "to": "a0", "when": None},
            {"from": "a0", "to": "decide_warranty", "when": None},
            {"from": "decide_warranty", "to": "branch_active", "when": "Active"},
            {"from": "branch_active", "to": "end_active", "when": None},
            {"from": "decide_warranty", "to": "branch_expired", "when": "Expired"},
            {"from": "branch_expired", "to": "end_expired", "when": None},
        ],
    }
    # minimal fake video: skip FileNotFound by patching — use real sample if present
    sample = ROOT / "data/generated/demonstrations/videos/parts_01/video.mp4"
    if sample.exists():
        gt = build_gt(video_path=sample, dag=dag, path_type="active", family="warranty_resolution")
        assert dag_reward(dag, gt)["reward"] >= 0.99
        empty = dag_reward({"nodes": [], "edges": []}, gt)
        assert empty["reward"] < 0.5
        print("selfcheck ok", gt["id"], "reward_self", dag_reward(dag, gt)["reward"])
    else:
        print("selfcheck skip (no sample video)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    else:
        main()
