#!/usr/bin/env python3
"""Convert .mkv → .mp4 via ffmpeg (H.264 + AAC for QuickTime)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def convert(src: Path, dest: Path | None = None) -> Path:
    dest = dest or src.with_suffix(".mp4")
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest}")
        return dest
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"ok {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main(paths: list[str]) -> None:
    if not paths:
        paths = [str(p) for p in Path("data/raw/ostask-demo").glob("*.mkv")]
    for p in paths:
        convert(Path(p))


if __name__ == "__main__":
    main(sys.argv[1:])
