#!/usr/bin/env python3
"""
ONE-SHOT Stage A pull — OSTask-demo only (~66 MB, 5 tasks, typically <10 min).

Do NOT pull full VideoCUA (26 GB / 55h). Re-run is idempotent (skips existing files).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ostask-demo"
REPO = "https://huggingface.co/datasets/open-world-agents/OSTask-demo/resolve/main"
TASKS = ("chromium_01", "excel_01", "general_01", "vscode_01", "workflow_01")
EXTS = (".json", ".mcap", ".mkv")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"get  {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f" ok  {dest.stat().st_size/1e6:.1f} MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for task in TASKS:
        for ext in EXTS:
            name = f"{task}{ext}"
            dest = OUT / name
            download(f"{REPO}/{name}", dest)
            total += dest.stat().st_size
    manifest = {
        "source": "open-world-agents/OSTask-demo",
        "one_shot": True,
        "tasks": list(TASKS),
        "bytes": total,
        "mb": round(total / 1e6, 2),
        "note": "Stage A smoke corpus. Map via ingest_stage_a.py → LLD train schema only.",
    }
    (OUT / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"done → {OUT}")


if __name__ == "__main__":
    main()
