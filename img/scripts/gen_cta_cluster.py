"""Regenerate img/cta_cluster.png.

A 2-CTA cluster doing one cooperative MMA: each CTA owns half of A and half of B
in its own SMEM, the pair reads each other's B tile across the cluster (DSMEM),
and together they produce a 256x256 output tile instead of 128x128 per CTA.

Larger fonts than the original so the box labels are readable when embedded.

Usage:
    python img/scripts/gen_cta_cluster.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Font sizes (bumped up for legibility).
FS_TITLE = 24
FS_CTA = 20
FS_BOX = 18
FS_BOX_SUB = 16
FS_LABEL = 18
FS_OUT = 17
FS_CAPTION = 18


def box(ax, x, y, w, h, fill, edge, lw=2.0, rounding=0.10, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=lw,
            facecolor=fill,
            edgecolor=edge,
            zorder=z,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        7.5,
        9.55,
        "Step 8: 2-CTA Cluster - Cooperative MMA via Cross-CTA SMEM Read",
        ha="center",
        va="center",
        fontsize=FS_TITLE,
        fontweight="bold",
    )

    # --- Outer CTA boxes ---
    box(ax, 1.4, 2.3, 5.3, 6.0, "#EAF2FE", "#4A90E2", lw=2.5, rounding=0.18)
    box(ax, 8.3, 2.3, 5.3, 6.0, "#EAF8EE", "#34A853", lw=2.5, rounding=0.18)
    ax.text(
        4.05,
        7.85,
        "CTA 0 (SM-0)",
        ha="center",
        va="center",
        fontsize=FS_CTA,
        fontweight="bold",
        color="#2E6FC9",
    )
    ax.text(
        10.95,
        7.85,
        "CTA 1 (SM-1)",
        ha="center",
        va="center",
        fontsize=FS_CTA,
        fontweight="bold",
        color="#2E8B49",
    )

    # --- Sub-boxes: Asmem (own) + Bsmem in each CTA ---
    def subbox(cx, top_y, fill, name, sub):
        x = cx - 2.15
        box(ax, x, top_y, 4.3, 1.5, fill, "#222", lw=2.0, z=2)
        ax.text(
            cx,
            top_y + 0.92,
            name,
            ha="center",
            va="center",
            fontsize=FS_BOX,
            fontweight="bold",
            zorder=3,
        )
        ax.text(
            cx,
            top_y + 0.48,
            sub,
            ha="center",
            va="center",
            fontsize=FS_BOX_SUB,
            zorder=3,
        )

    # CTA 0
    subbox(4.05, 5.95, "#FBD0D0", "Asmem (own)", "A rows 0-127")
    subbox(4.05, 3.85, "#CFE0FB", "Bsmem", "B cols 0-127")
    # CTA 1
    subbox(10.95, 5.95, "#CDEFD3", "Asmem (own)", "A rows 128-255")
    subbox(10.95, 3.85, "#C7D8C0", "Bsmem", "B cols 128-255")

    # --- cross-CTA read (double arrow between the two Bsmem boxes) ---
    # Arrows span the full gap between the two Bsmem boxes (heads near each box);
    # the label box sits on top, so each arrowhead stays visible on its side.
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#E0533B")
    ax.add_patch(FancyArrowPatch((6.25, 4.88), (8.75, 4.88), **arrow_kw, zorder=2))
    ax.add_patch(FancyArrowPatch((8.75, 4.42), (6.25, 4.42), **arrow_kw, zorder=2))
    box(ax, 6.55, 4.20, 1.9, 0.92, "#FFFFFF", "#E0533B", lw=2.2, rounding=0.10, z=4)
    ax.text(
        7.5,
        4.84,
        "cross-CTA",
        ha="center",
        va="center",
        fontsize=FS_LABEL,
        fontweight="bold",
        color="#E0533B",
        zorder=5,
    )
    ax.text(
        7.5,
        4.46,
        "read",
        ha="center",
        va="center",
        fontsize=FS_LABEL,
        fontweight="bold",
        color="#E0533B",
        zorder=5,
    )

    # --- Output tiles ---
    def outbox(cx, label):
        x = cx - 2.15
        box(ax, x, 0.85, 4.3, 1.25, "#E9D9F6", "#222", lw=2.0, z=2)
        ax.text(
            cx,
            1.47,
            label,
            ha="center",
            va="center",
            fontsize=FS_OUT,
            fontweight="bold",
            zorder=3,
        )

    outbox(4.05, "D[0:128, 0:256]")
    outbox(10.95, "D[128:256, 0:256]")
    down_kw = dict(arrowstyle="-|>", mutation_scale=20, lw=2.0, color="#555")
    ax.add_patch(FancyArrowPatch((4.05, 3.80), (4.05, 2.15), **down_kw, zorder=1))
    ax.add_patch(FancyArrowPatch((10.95, 3.80), (10.95, 2.15), **down_kw, zorder=1))

    # --- Bottom caption ---
    box(ax, 3.3, 0.02, 8.4, 0.66, "#FFF8E1", "#F5A623", lw=2.2, rounding=0.12, z=2)
    ax.text(
        7.5,
        0.35,
        "Cluster output: 256 x 256  (was 128 x 128 per CTA)",
        ha="center",
        va="center",
        fontsize=FS_CAPTION,
        fontweight="bold",
        zorder=3,
    )

    out = os.path.join(os.path.dirname(__file__), "..", "cta_cluster.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
