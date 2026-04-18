"""Regenerate img/memory_dataflow.png.

Illustrates the canonical data flow of a Blackwell GEMM-like kernel:

    GMEM --TMA load--> SMEM --tcgen05 MMA--> TMEM --tcgen05.ld--> RF
         --thread write--> SMEM --TMA store--> GMEM

Usage:
    python img/scripts/gen_memory_dataflow.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


BOXES = [
    ("Global\nMemory\n(GMEM)", "gmem"),
    ("Shared\nMemory\n(SMEM)", "smem"),
    ("Tensor\nMemory\n(TMEM)", "tmem"),
    ("Register\nFile\n(RF)", "rf"),
    ("Shared\nMemory\n(SMEM)", "smem"),
    ("Global\nMemory\n(GMEM)", "gmem"),
]

EDGE_LABELS = [
    "TMA load",
    "tcgen05 MMA",
    "tcgen05.ld",
    "thread write",
    "TMA store",
]

# Two colour families: GMEM/SMEM are "staging" (purple), TMEM/RF are
# Tensor-Core-resident (amber), matching the original hand-drawn diagram.
COLOURS = {
    "gmem": ("#E7E3F5", "#8470B0"),  # (fill, edge)
    "smem": ("#E7E3F5", "#8470B0"),
    "tmem": ("#FBE9D2", "#C98A3E"),
    "rf":   ("#FBE9D2", "#C98A3E"),
}

BOX_W = 1.6
BOX_H = 1.3
GAP = 1.6
Y = 0.0


def draw_box(ax, cx, cy, w, h, text, fill, edge):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5,
        facecolor=fill,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11)


def draw_arrow(ax, x_from, x_to, y, label):
    arrow = FancyArrowPatch(
        (x_from, y),
        (x_to, y),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.5,
        color="#333333",
    )
    ax.add_patch(arrow)
    ax.text(
        (x_from + x_to) / 2,
        y + 0.25,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
    )


def main():
    n_boxes = len(BOXES)
    total_w = n_boxes * BOX_W + (n_boxes - 1) * GAP
    fig, ax = plt.subplots(figsize=(total_w * 0.95, 1.9), dpi=200)

    centres = []
    x = BOX_W / 2
    for text, kind in BOXES:
        centres.append(x)
        fill, edge = COLOURS[kind]
        draw_box(ax, x, Y, BOX_W, BOX_H, text, fill, edge)
        x += BOX_W + GAP

    for i, label in enumerate(EDGE_LABELS):
        x_from = centres[i] + BOX_W / 2
        x_to = centres[i + 1] - BOX_W / 2
        draw_arrow(ax, x_from, x_to, Y, label)

    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.2)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "memory_dataflow.png"
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
