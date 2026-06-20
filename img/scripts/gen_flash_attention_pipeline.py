#!/usr/bin/env python3
"""Generate the Flash Attention 4 pipeline diagram."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "load": "#f6c7c8",
    "score": "#c7dbf4",
    "softmax": "#ffe3a6",
    "value": "#d8c3ef",
    "corr": "#9fd8d6",
    "store": "#c7ead2",
    "label": "#f8fafc",
}


def block(ax, x, y, w, h, text, color, fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, label=None, color="#4b5563", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        alpha=0.78,
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label, fontsize=7, color=color, ha="center", zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    ax.set_xlim(0, 13.3)
    ax.set_ylim(-0.25, 6.65)
    ax.axis("off")

    ax.text(6.65, 6.45, "Flash Attention 4 Pipeline Structure", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.65,
        6.18,
        "representative issue order; the MMA warp interleaves value MMA for current V with score MMA for next K",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    arrow(ax, 1.75, 5.95, 12.75, 5.95, color="#9ca3af")
    ax.text(7.25, 6.02, "time", ha="center", va="bottom", fontsize=8, color="#6b7280", style="italic", zorder=2)

    rows = [
        ("WG3 warp 1", "TMA load", 5.0),
        ("WG3 warp 0", "MMA issue", 4.0),
        ("WG0", "softmax Q stage 0", 3.0),
        ("WG1", "softmax Q stage 1", 2.0),
        ("WG2", "correction / epilogue", 1.0),
        ("WG3 warp 2", "TMA store", 0.0),
    ]
    for name, role, y in rows:
        block(ax, 0.15, y + 0.12, 1.35, 0.62, f"{name}\n{role}", COLORS["label"], fs=8)
        ax.plot([1.75, 13.0], [y + 0.43, y + 0.43], color="#e5e7eb", lw=1, zorder=0)

    # TMA load order from the source: Q0, K_last, Q1, V_last, then K/V stream.
    for x, text in [
        (2.0, "load Q0"),
        (3.1, "load K[n-1]"),
        (4.2, "load Q1"),
        (5.3, "load V[n-1]"),
        (6.7, "load K[n-2]"),
        (7.8, "load V[n-2]"),
        (9.2, "load K[n-3]"),
        (10.3, "load V[n-3]"),
    ]:
        block(ax, x, 5.12, 0.88, 0.62, text, COLORS["load"], fs=8)
    ax.text(11.45, 5.43, "...", fontsize=13, color="#6b7280")

    # MMA issue order: bootstrap scores, then interleave PV for current V with QK for next K.
    mma_blocks = [
        (4.0, "score\nQ0*K[n-1]", COLORS["score"]),
        (5.1, "score\nQ1*K[n-1]", COLORS["score"]),
        (6.35, "value\nP0*V[n-1]", COLORS["value"]),
        (7.45, "score\nQ0*K[n-2]", COLORS["score"]),
        (8.55, "value\nP1*V[n-1]", COLORS["value"]),
        (9.65, "score\nQ1*K[n-2]", COLORS["score"]),
        (10.75, "value\nP0*V[n-2]", COLORS["value"]),
    ]
    for x, text, color in mma_blocks:
        block(ax, x, 4.12, 0.98, 0.66, text, color, fs=8)
    ax.text(11.95, 4.43, "... after last K/V", fontsize=8, color="#6b7280")

    # Softmax and correction events. Keep one dependency loop per Q stage readable.
    block(ax, 4.75, 3.12, 1.05, 0.66, "softmax S0\nwrite P0", COLORS["softmax"], fs=8)
    block(ax, 5.85, 2.12, 1.05, 0.66, "softmax S1\nwrite P1", COLORS["softmax"], fs=8)
    block(ax, 6.05, 1.12, 1.02, 0.66, "release /\nrescale O0", COLORS["corr"], fs=8)
    block(ax, 8.25, 1.12, 1.02, 0.66, "release /\nrescale O1", COLORS["corr"], fs=8)
    block(ax, 8.35, 3.12, 1.05, 0.66, "softmax S0\nwrite P0", COLORS["softmax"], fs=8)
    block(ax, 10.25, 2.12, 1.05, 0.66, "softmax S1\nwrite P1", COLORS["softmax"], fs=8)
    block(ax, 11.25, 1.12, 1.08, 0.66, "normalize\nO0/O1", COLORS["corr"], fs=8)
    block(ax, 12.0, 0.12, 0.9, 0.62, "store O", COLORS["store"], fs=8)

    # Keep this as a timeline. Barrier-level dependencies are shown in
    # flash_attention_barrier_flow_v2.png; drawing them again here makes
    # the pipeline view harder to read.

    # Legend.
    legend = [
        ("TMA load", COLORS["load"]),
        ("score MMA", COLORS["score"]),
        ("softmax", COLORS["softmax"]),
        ("value MMA", COLORS["value"]),
        ("correction/epilogue", COLORS["corr"]),
        ("TMA store", COLORS["store"]),
    ]
    lx = 2.0
    for name, color in legend:
        block(ax, lx, -0.12, 0.22, 0.16, "", color, fs=1)
        ax.text(lx + 0.3, -0.04, name, fontsize=8, va="center", color="#4b5563")
        lx += 1.55

    fig.savefig("../flash_attention_pipeline_v2.png", dpi=160, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
