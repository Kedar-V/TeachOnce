#!/usr/bin/env python3
"""Score predicted DAGs (JSONL: {id, pred|dag|raw}) against nemo_rl GT via dag_reward."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import dag_reward  # noqa: E402
from scripts.tool_registry_adapter import MVP_OPS  # noqa: E402


def _fill_tools(pred: dict) -> dict:
    for n in pred.get("nodes") or []:
        if n.get("type") == "ACTION" and n.get("op") in MVP_OPS:
            n["tool"] = MVP_OPS[n["op"]]["service"]
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(ROOT / "data/generated/nemo_rl/gt.jsonl"))
    ap.add_argument("--preds", required=True, help="JSONL with id + pred/dag/raw")
    ap.add_argument("--out", default=str(ROOT / "results/baseline_grok_dag.json"))
    ap.add_argument("--model", default="grok-subagent")
    args = ap.parse_args()

    gts = {r["id"]: r for r in (json.loads(l) for l in Path(args.gt).read_text().splitlines() if l.strip())}
    preds = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]

    per = []
    sums = {"reward": 0.0, "schema_ok": 0.0, "critical_ops": 0.0, "taken_branch": 0.0}
    n_ok = 0
    for row in preds:
        gid = row.get("id")
        gt = gts.get(gid)
        if not gt:
            per.append({"id": gid, "error": "unknown_id", "reward": 0.0, "schema_ok": 0.0, "critical_ops": 0.0, "taken_branch": 0.0})
            continue
        raw = row.get("pred") or row.get("dag") or row.get("raw")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                scores = dag_reward(raw, gt)
                scores["id"] = gid
                per.append(scores)
                for k in sums:
                    sums[k] += float(scores.get(k) or 0)
                continue
        if isinstance(raw, dict):
            raw = _fill_tools(raw)
            scores = dag_reward(raw, gt)
            n_ok += 1
        else:
            scores = {"reward": 0.0, "schema_ok": 0.0, "critical_ops": 0.0, "taken_branch": 0.0, "error": "bad_pred"}
        scores["id"] = gid
        per.append(scores)
        for k in sums:
            sums[k] += float(scores.get(k) or 0)

    n = max(len(per), 1)
    summary = {
        "n": len(per),
        "n_success": n_ok,
        "model": args.model,
        "input": "frames_from_video",
        "mean_reward": round(sums["reward"] / n, 4),
        "mean_schema_ok": round(sums["schema_ok"] / n, 4),
        "mean_critical_ops": round(sums["critical_ops"] / n, 4),
        "mean_taken_branch": round(sums["taken_branch"] / n, 4),
    }
    # pad missing GT ids as zeros so n matches full set when partial
    missing = [i for i in gts if i not in {p.get("id") for p in per}]
    for mid in missing:
        per.append({"id": mid, "reward": 0.0, "schema_ok": 0.0, "critical_ops": 0.0, "taken_branch": 0.0, "error": "missing_pred"})
        for k in sums:
            sums[k] += 0.0
    if missing:
        n = len(per)
        summary = {
            "n": n,
            "n_success": n_ok,
            "model": args.model,
            "input": "frames_from_video",
            "mean_reward": round(sums["reward"] / n, 4),
            "mean_schema_ok": round(sums["schema_ok"] / n, 4),
            "mean_critical_ops": round(sums["critical_ops"] / n, 4),
            "mean_taken_branch": round(sums["taken_branch"] / n, 4),
            "n_missing": len(missing),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_sample": per}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
