#!/usr/bin/env python3
"""
Stage A media with continuous video only:
  1. OSTask-demo  — 5× .mkv  (~66 MB)
  2. VideoCUA     — few tasks from one small app zip (Conky, ~7 MB) → .mp4

Skip AgentNet / VideoGUI (screenshots / Drive-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_stage_a_once import main as pull_ostask  # noqa: E402
from download_three_sources import download_videocua  # noqa: E402


def main() -> None:
    print("=== 1/2 OSTask-demo (mkv) ===")
    pull_ostask()
    print("\n=== 2/2 VideoCUA Conky (mp4, ≤3 tasks) ===")
    vc = download_videocua(3)
    manifest = {
        "policy": "continuous_video_only",
        "included": ["open-world-agents/OSTask-demo", "ServiceNow/VideoCUA (Conky subset)"],
        "excluded": ["xlangai/AgentNet (PNG)", "VideoGUI (Drive / screenshots on HF)"],
        "videocua": vc,
    }
    path = ROOT / "data" / "raw" / "VIDEO_SOURCES_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"manifest → {path}")


if __name__ == "__main__":
    main()
