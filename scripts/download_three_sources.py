#!/usr/bin/env python3
"""
Pull a *few* samples from the three Stage-A sources named in phase-1-dataset.md.

Reality check (media type):
  1. VideoCUA     — continuous MP4 ✅ (download one small app zip, keep ≤N tasks)
  2. AgentNet     — public HF release is screenshot PNG sequences + JSONL ❌ not MP4
                  (raw video exists in AgentNetTool captures, not on the HF dump)
  3. VideoGUI     — continuous recordings live on Google Drive; HF cards are
                  screenshot/plan parquet ❌ not MP4 on HF

This script downloads what HF can give quickly without 26GB VideoCUA / 50GB+ AgentNet images.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAX_TASKS = 3


def get(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"get  {url.split('/')[-1]} → {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f" ok  {dest.stat().st_size/1e6:.1f} MB")
    return dest


def download_videocua(n: int = MAX_TASKS) -> dict:
    """Smallest CRM-ish-ish zip with real MP4s: Conky (~7MB)."""
    out = RAW / "videocua" / "Conky"
    zpath = RAW / "videocua" / "Conky.zip"
    url = "https://huggingface.co/datasets/ServiceNow/VideoCUA/resolve/main/raw_data/Conky.zip"
    get(url, zpath)
    extract = RAW / "videocua" / "_extract_Conky"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract)
    # find task dirs with video.mp4
    videos = sorted(extract.rglob("video.mp4"))
    kept = []
    out.mkdir(parents=True, exist_ok=True)
    for i, vp in enumerate(videos[:n]):
        task_dir = vp.parent.parent if vp.parent.name == "video" else vp.parent
        dest = out / f"task_{i:02d}_{task_dir.name}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(task_dir, dest)
        kept.append(
            {
                "task_id": task_dir.name,
                "video": str(dest / "video" / "video.mp4") if (dest / "video" / "video.mp4").exists() else str(dest / "video.mp4"),
                "action_log": str(next(dest.rglob("action_log.json"), "")),
            }
        )
    return {
        "dataset": "ServiceNow/VideoCUA",
        "media": "continuous_mp4",
        "zip": "Conky.zip",
        "kept": kept,
        "available_tasks_in_zip": len(videos),
    }


def download_agentnet(n: int = 5) -> dict:
    """Metadata + first N trajectory stubs from ubuntu JSONL (no PNG pack — multi-GB)."""
    out = RAW / "agentnet"
    out.mkdir(parents=True, exist_ok=True)
    meta_url = "https://huggingface.co/datasets/xlangai/AgentNet/resolve/main/meta_data_merged.jsonl"
    get(meta_url, out / "meta_data_merged.jsonl")

    # Stream first N lines of ubuntu jsonl via HTTP range-unfriendly → read until N lines
    # Full file is 282MB; instead only save N sample rows by early-stop download to temp... 
    # urllib can't early-stop easily mid-jsonl without reading. Use a small request with 
    # Accept and read line-by-line until N.
    samples_path = out / f"ubuntu_samples_{n}.jsonl"
    if not samples_path.exists():
        url = "https://huggingface.co/datasets/xlangai/AgentNet/resolve/main/agentnet_ubuntu_5k.jsonl"
        print(f"stream first {n} traj from agentnet_ubuntu_5k.jsonl (stops early)…")
        req = urllib.request.Request(url)
        rows = []
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                rows.append(line)
                if len(rows) >= n:
                    break
        samples_path.write_text("\n".join(rows) + "\n")
        print(f" ok  wrote {n} trajectories → {samples_path}")
    else:
        print(f"skip {samples_path}")

    samples = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]
    summary = []
    for s in samples:
        traj = s.get("traj") or []
        summary.append(
            {
                "task_id": s.get("task_id"),
                "instruction": (s.get("instruction") or s.get("natural_language_task") or "")[:120],
                "n_steps": len(traj),
                "first_image": (traj[0].get("image") if traj else None),
                "media_note": "PNG frames referenced; full image zips are multi-GB — not downloaded",
            }
        )
    return {
        "dataset": "xlangai/AgentNet",
        "media": "screenshot_png_sequences_not_mp4",
        "samples": summary,
        "images_downloaded": False,
        "reason": "ubuntu_images/*.z## are ~5GB each; skip for few-sample pull",
    }


def download_videogui(n: int = 3) -> dict:
    """
    HF VideoGUI-* are screenshot/plan tables. Continuous screen recordings are on Google Drive.
    Pull Mid-Plan parquet only if user insists — 180MB. Prefer: save Drive pointer + try
    datasets streaming for n rows (may still fetch shard).
    """
    out = RAW / "videogui"
    out.mkdir(parents=True, exist_ok=True)
    note = {
        "dataset": "VideoGUI (showlab)",
        "media": "continuous_video_on_google_drive; HF=screenshots_plans",
        "google_drive": "See https://github.com/showlab/videogui (Google-Drive link in README)",
        "hf_action": "VideoGUI/VideoGUI-Action (~495MB parquet — not fetching full for few samples)",
        "hf_mid_plan": "VideoGUI/VideoGUI-Mid-Plan (~180MB)",
    }
    # Lightweight: write README only unless --full-mid-plan
    (out / "MEDIA_NOTE.json").write_text(json.dumps(note, indent=2))

    # Try streaming n rows without writing 180MB if datasets is available
    try:
        from datasets import load_dataset

        print(f"stream VideoGUI-Mid-Plan take({n})…")
        ds = load_dataset("VideoGUI/VideoGUI-Mid-Plan", split="mid_plan", streaming=True)
        rows = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            # drop heavy image bytes from JSON summary; save images separately
            rec = {k: v for k, v in row.items() if k not in ("screenshot_start", "screenshot_end")}
            rows.append(rec)
            for key in ("screenshot_start", "screenshot_end"):
                img = row.get(key)
                if img is None:
                    continue
                img_dir = out / f"sample_{i:02d}"
                img_dir.mkdir(exist_ok=True)
                path = img_dir / f"{key}.png"
                if hasattr(img, "save"):
                    img.save(path)
                elif isinstance(img, dict) and "bytes" in img:
                    path.write_bytes(img["bytes"])
        (out / f"mid_plan_samples_{n}.json").write_text(json.dumps(rows, indent=2, default=str))
        note["streamed_mid_plan_rows"] = n
        note["saved"] = str(out)
        print(f" ok  streamed {n} Mid-Plan rows (screenshots, not MP4)")
    except Exception as e:
        note["stream_error"] = str(e)
        print(f"warn VideoGUI stream failed: {e}")
        print("     Continuous videos: open GitHub README Google Drive manually.")

    (out / "MEDIA_NOTE.json").write_text(json.dumps(note, indent=2))
    return note


def main() -> None:
    report = {
        "videocua": download_videocua(MAX_TASKS),
        "agentnet": download_agentnet(5),
        "videogui": download_videogui(3),
    }
    path = RAW / "THREE_SOURCES_MANIFEST.json"
    path.write_text(json.dumps(report, indent=2))
    print("\n=== summary ===")
    print(json.dumps(
        {
            "videocua_mp4_tasks": len(report["videocua"].get("kept", [])),
            "agentnet_traj_samples": len(report["agentnet"].get("samples", [])),
            "agentnet_has_mp4": False,
            "videogui_has_mp4_on_hf": False,
            "manifest": str(path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
