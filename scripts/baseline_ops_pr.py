#!/usr/bin/env python3
"""Op-set precision/recall for the Nemotron video→DAG baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_vllm_dag import call_vllm, guided_schema, resolve_video  # noqa: E402
from scripts.nemo_rl_gt import _ops_from_dag  # noqa: E402

BRANCH_TIP = {
    "active": "CREATE_REPAIR",
    "expired": "RECORD_PAID_OPTIONS",
    "in_stock": "CREATE_PARTS_SHIPMENT",
    "backorder": "CREATE_BACKORDER",
    "p1": "PAGE_ONCALL",
    "standard": "ASSIGN_L2_QUEUE",
}


def mean(xs: list[float]) -> float:
    return round(sum(xs) / max(len(xs), 1), 4)


def main() -> None:
    gt_path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/generated/nemo_rl/gt.jsonl")
    video_root = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/generated/demonstrations/videos")
    rows = [json.loads(l) for l in gt_path.read_text().splitlines() if l.strip()][:25]
    schema = guided_schema()
    prec_g, rec_g, prec_c, rec_c = [], [], [], []
    branch_ok = 0
    for gt in rows:
        path = resolve_video(gt, video_root)
        text, _ = call_vllm(
            base_url="http://127.0.0.1:8000/v1",
            api_key="teachonce-local-key",
            model="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
            prompt=gt.get("prompt") or "Emit executable DAG JSON.",
            video_path=path,
            schema=schema,
            max_tokens=1024,
            timeout=300,
        )
        pred = json.loads(text)
        pred_ops = set(_ops_from_dag(pred))
        gold = set(gt.get("gold_ops") or [])
        crit = set(gt.get("critical_ops") or [])
        inter_g = len(pred_ops & gold)
        inter_c = len(pred_ops & crit)
        p = inter_g / len(pred_ops) if pred_ops else 0.0
        r = inter_g / len(gold) if gold else 1.0
        pc = inter_c / len(pred_ops) if pred_ops else 0.0
        rc = inter_c / len(crit) if crit else 1.0
        prec_g.append(p)
        rec_g.append(r)
        prec_c.append(pc)
        rec_c.append(rc)
        tip = BRANCH_TIP.get(gt.get("taken_branch"))
        if tip and tip in pred_ops:
            branch_ok += 1
        print(f"{gt['id']}: ops P={p:.2f} R={r:.2f} critR={rc:.2f}", flush=True)

    mp, mr = mean(prec_g), mean(rec_g)
    out = {
        "n": len(rows),
        "ops_precision": mp,
        "ops_recall": mr,
        "ops_f1": round(2 * mp * mr / (mp + mr + 1e-12), 4),
        "critical_ops_precision_vs_pred": mean(prec_c),
        "critical_ops_recall": mean(rec_c),
        "taken_branch_acc": round(branch_ok / max(len(rows), 1), 4),
    }
    print(json.dumps(out, indent=2))
    Path(ROOT / "results/baseline_ops_pr.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
