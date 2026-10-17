#!/usr/bin/env python3
"""Record Stage-B samples by driving northstar-console (real UI) via Playwright."""

from __future__ import annotations

import http.server
import json
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tool_registry_adapter import adapt_record, validate_registry_dag  # noqa: E402

CONSOLE = ROOT / "northstar-console"
OUT_ROOT = ROOT / "data" / "generated" / "demonstrations" / "videos"
PORT = 8765
# Target ~35s total: hold scales with step count (5–7 ops).
HOLD_MS = 4200
INTRO_MS = 1500
OUTRO_MS = 2500
W, H, FPS = 1280, 720, 24

# 5 keepers spanning 3 families + main path forks
SAMPLES = [
    {
        "video_id": "warranty_01",
        "procedure": "warranty",
        "path_type": "active",
        "family": "warranty_resolution",
        "split": "train",
        "goal": "Resolve laptop hardware warranty and create repair if covered",
    },
    {
        "video_id": "warranty_15",
        "procedure": "warranty",
        "path_type": "expired",
        "family": "warranty_resolution",
        "split": "train",
        "goal": "Handle expired warranty — record paid service options",
    },
    {
        "video_id": "parts_01",
        "procedure": "parts",
        "path_type": "in_stock",
        "family": "parts_fulfillment",
        "split": "train",
        "goal": "Fulfill spare part for failed device when in stock",
    },
    {
        "video_id": "parts_10",
        "procedure": "parts",
        "path_type": "backorder",
        "family": "parts_fulfillment",
        "split": "train",
        "goal": "Backorder spare part when inventory is out of stock",
    },
    {
        "video_id": "escalate_01",
        "procedure": "escalate",
        "path_type": "p1",
        "family": "severity_escalation",
        "split": "train",
        "goal": "Triage P1 ticket and page on-call",
    },
]


def port_up() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
        return True
    except Exception:
        return False


def start_server() -> socketserver.TCPServer | None:
    if port_up():
        return None  # reuse existing northstar-console server

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(CONSOLE), **k)

        def log_message(self, *_a):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", PORT), Handler)
    httpd.allow_reuse_address = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def to_mp4(webm: Path, mp4: Path) -> float:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(webm),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-r", str(FPS),
            "-b:v", "2500k", "-minrate", "2000k", "-maxrate", "2500k",
            "-bufsize", "2500k", "-x264-params", "nal-hrd=cbr",
            "-an", str(mp4),
        ],
        check=True,
    )
    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(mp4),
        ],
        text=True,
    ).strip()
    return float(probe)


def case_fields(case: dict) -> dict:
    keys = (
        "customer_id", "device_id", "ticket_id", "warranty_id", "warranty_status",
        "repair_id", "part_sku", "shipment_id", "backorder_id", "severity",
    )
    return {k: case[k] for k in keys if case.get(k) is not None}


def record_sample(browser, sample: dict) -> dict:
    from playwright.sync_api import TimeoutError as PwTimeout

    vid = sample["video_id"]
    proc, path = sample["procedure"], sample["path_type"]
    url = f"http://127.0.0.1:{PORT}/?procedure={proc}&path={path}&hud=0"
    out_dir = OUT_ROOT / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    for p in raw_dir.glob("*"):
        p.unlink()

    context = browser.new_context(
        viewport={"width": W, "height": H},
        record_video_dir=str(raw_dir),
        record_video_size={"width": W, "height": H},
    )
    page = context.new_page()
    page.goto(url, wait_until="networkidle")
    page.wait_for_function("() => window.Northstar && window.Northstar.playbook")
    page.wait_for_timeout(INTRO_MS)

    playbook = page.evaluate("() => Northstar.playbook()")
    ops = [s["op"] for s in playbook["steps"]]
    obs = page.evaluate("() => Northstar.observe()")
    case = obs.get("case") or {}

    events = []
    t0 = time.time()
    for op in ops:
        step_start = time.time() - t0
        page.evaluate("(op) => Northstar.act('playbook_step', { op })", op)
        page.wait_for_timeout(HOLD_MS)
        step_end = time.time() - t0
        events.append(
            {
                "t_start": round(step_start, 2),
                "t_end": round(step_end, 2),
                "op": op,
                "label": op.replace("_", " ").title(),
                "observed": {},
            }
        )

    page.wait_for_timeout(OUTRO_MS)
    try:
        page.wait_for_function("() => Northstar.observe().flags?.done === true", timeout=5000)
    except PwTimeout:
        pass

    # enrich case from observe at end
    obs = page.evaluate("() => Northstar.observe()")
    case = obs.get("case") or case
    flags = obs.get("flags") or {}

    page.close()
    context.close()  # flushes video

    webms = list(raw_dir.glob("*.webm"))
    if not webms:
        raise RuntimeError(f"{vid}: no webm recorded")
    webm = webms[0]
    mp4 = out_dir / "video.mp4"
    duration = to_mp4(webm, mp4)
    webm.unlink()
    raw_dir.rmdir()

    # fill observed IDs lightly from case
    for ev in events:
        op = ev["op"]
        obs_map = {
            "LOOKUP_CUSTOMER": {"customer_id": case.get("customer_id")},
            "GET_TICKET": {"ticket_id": case.get("ticket_id")},
            "GET_DEVICE": {"device_id": case.get("device_id")},
            "CHECK_WARRANTY": {
                "warranty_id": case.get("warranty_id"),
                "status": case.get("warranty_status"),
            },
            "GET_DIAGNOSTICS": {"result": case.get("diag_result")},
            "CREATE_REPAIR": {"repair_id": case.get("repair_id")},
            "RECORD_PAID_OPTIONS": {"warranty_status": "EXPIRED"},
            "CHECK_PARTS_STOCK": {"part_sku": case.get("part_sku"), "stock": case.get("stock")},
            "CREATE_PARTS_SHIPMENT": {"shipment_id": case.get("shipment_id")},
            "CREATE_BACKORDER": {"backorder_id": case.get("backorder_id")},
            "ASSESS_SEVERITY": {"severity": case.get("severity")},
            "PAGE_ONCALL": {"oncall": case.get("oncall")},
            "ASSIGN_L2_QUEUE": {"l2_queue": case.get("l2_queue")},
            "UPDATE_TICKET": {"ticket_id": case.get("ticket_id")},
        }
        ev["observed"] = {k: v for k, v in (obs_map.get(op) or {}).items() if v is not None}

    meta = {
        "video_id": vid,
        "source_dataset": "teachonce_self_synthetic",
        "generation_method": "agent_driven_northstar_console",
        "procedure": proc,
        "family": sample["family"],
        "split": sample["split"],
        "path_type": path,
        "duration_sec": round(duration, 2),
        "fps": FPS,
        "resolution": f"{W}x{H}",
        "goal": sample["goal"],
        "sop": None,
        "case": case_fields(case) if case else {"customer_id": playbook.get("caseId")},
        "ops": ops,
        "events": events,
        "tools_ref": "mvp_registry_v1",
        "console_url": f"?procedure={proc}&path={path}&hud=0",
        "prompt_variant": "light",
        "notes": f"Recorded from northstar-console; done={flags.get('done')}",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    adapted = adapt_record(
        "teachonce_self_synthetic",
        [{"op": o} for o in ops],
        workflow_id=f"WF-{vid}",
        name=f"{sample['family']} ({path})",
    )
    assert adapted["ok"], (vid, adapted["errors"])
    assert not validate_registry_dag(adapted["dag"])
    (out_dir / "expected_dag.json").write_text(json.dumps(adapted["dag"], indent=2) + "\n")

    size_mb = mp4.stat().st_size / (1024 * 1024)
    return {"video_id": vid, "duration_sec": meta["duration_sec"], "size_mb": round(size_mb, 2), "ops": ops, "family": sample["family"]}


def write_manifest(results: list[dict]) -> None:
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "version": "2",
                "stage": "B",
                "count": len(results),
                "incomplete": True,
                "procedures": {
                    "warranty": {"family": "warranty_resolution", "ids": [r["video_id"] for r in results if r["video_id"].startswith("warranty_")]},
                    "parts": {"family": "parts_fulfillment", "ids": [r["video_id"] for r in results if r["video_id"].startswith("parts_")]},
                    "escalate": {"family": "severity_escalation", "ids": [r["video_id"] for r in results if r["video_id"].startswith("escalate_")]},
                },
                "val": [],
                "demo": ["warranty_01", "parts_01", "escalate_01"],
                "source_dataset": "teachonce_self_synthetic",
                "model_target": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
                "console": "northstar-console",
                "adapter": "scripts/tool_registry_adapter.py",
                "one_shot": True,
                "notes": "5-sample subset spanning warranty/parts/escalate path forks",
            },
            indent=2,
        )
        + "\n"
    )


def rebuild_split(results: list[dict]) -> None:
    split_dir = ROOT / "data" / "generated" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        meta = json.loads((OUT_ROOT / r["video_id"] / "meta.json").read_text())
        dag = json.loads((OUT_ROOT / r["video_id"] / "expected_dag.json").read_text())
        rows.append(
            {
                "id": f"train-{meta['video_id']}",
                "family": meta["family"],
                "split": meta["split"],
                "path_type": meta["path_type"],
                "procedure": meta["procedure"],
                "source_dataset": meta["source_dataset"],
                "input": {
                    "goal": meta["goal"],
                    "video_id": meta["video_id"],
                    "video_path": str((OUT_ROOT / meta["video_id"] / "video.mp4").resolve()),
                    "sop": meta.get("sop"),
                    "policies": [],
                    "demonstrations": [{"video_id": meta["video_id"], "events": meta["events"]}],
                    "tools_ref": meta.get("tools_ref", "mvp_registry_v1"),
                },
                "target_dag": dag,
            }
        )
    path = split_dir / "mvp_train.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    (split_dir / "mvp_val.jsonl").write_text("")
    print("wrote", path, "rows", len(rows))


def main() -> None:
    from playwright.sync_api import sync_playwright

    httpd = start_server()
    time.sleep(0.3)
    results = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for sample in SAMPLES:
                print("recording", sample["video_id"], sample["procedure"], sample["path_type"], flush=True)
                results.append(record_sample(browser, sample))
            browser.close()
    finally:
        if httpd is not None:
            httpd.shutdown()

    write_manifest(results)
    rebuild_split(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
