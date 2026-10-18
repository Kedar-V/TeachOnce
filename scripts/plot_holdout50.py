#!/usr/bin/env python3
"""Regen docs/figures/holdout50-ops-f1.png from holdout-50 ops F1 numbers."""
from pathlib import Path

import matplotlib.pyplot as plt

METHODS = ["Baseline", "SFT", "RL-only", "SFT + RFT"]
OPS_F1 = [0.306, 0.720, 0.330, 0.805]
# mute context; accent winner (blue, not red/green)
COLORS = ["#9AA0A6", "#9AA0A6", "#9AA0A6", "#1F6FEB"]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "holdout50-ops-f1-v2.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=160)
    ypos = list(reversed(range(len(METHODS))))
    bars = ax.barh(
        ypos,
        list(reversed(OPS_F1)),
        color=list(reversed(COLORS)),
        height=0.62,
        edgecolor="none",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(list(reversed(METHODS)), fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Ops F1  (higher is better)", fontsize=10, color="#444")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, reversed(OPS_F1)):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=11,
            color="#1F6FEB" if val >= 0.8 else "#222",
            fontweight="bold" if val >= 0.8 else "regular",
        )

    # Title + subtitle above axes (avoid ax.set_title + overlapping text)
    fig.suptitle(
        "SFT + RFT lifts ops F1 from 0.306 to 0.805",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#111",
    )
    fig.text(
        0.01,
        0.915,
        "Balanced holdout-50 · Nemotron Nano 12B VL · same guided_json protocol",
        fontsize=9.5,
        color="#666",
        ha="left",
        va="top",
    )
    fig.text(
        0.01,
        0.02,
        "Source: dag_reward holdout eval (escalate 17 / parts 17 / warranty 16). "
        "RL-only ≈ baseline; SFT carries most of the gain.",
        fontsize=8.5,
        color="#777",
        ha="left",
        va="bottom",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.82, bottom=0.22)
    fig.savefig(OUT, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
