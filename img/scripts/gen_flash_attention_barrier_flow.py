#!/usr/bin/env python3
"""Generate Flash Attention 4 barrier diagrams."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "tma": "#f6c7c8",
    "mma": "#c7dbf4",
    "softmax": "#ffe3a6",
    "wg2": "#cfead0",
    "store": "#c7ead2",
    "bar": "#eef6ff",
    "merge": "#eee7fb",
}


def box(ax, x, y, w, h, text, color, fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.15,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold")


def arrow(ax, x1, y1, x2, y2, color="#4b5563", rad=0.0, lw=1.25):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def label(ax, x, y, text, fs=8.5, color="#374151"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d1d5db"),
    )


def gen_main_handoff():
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(6.5, 6.62, "Flash Attention 4: MMA Input Gates", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.5,
        6.26,
        "inputs that must be ready before each MMA may fire",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # ---- Score MMA gate (top): Q and K must be in SMEM. ----
    ax.text(0.5, 5.95, "Score MMA gate", fontsize=11.5, weight="bold", color="#1f2937")
    box(ax, 0.6, 5.12, 1.95, 0.62, "Q tile\nin SMEM", COLORS["tma"], fs=8.8)
    box(ax, 0.6, 4.30, 1.95, 0.62, "K tile\nin SMEM", COLORS["tma"], fs=8.8)
    box(ax, 6.45, 4.55, 2.3, 0.95, "score MMA\nQ,K -> S", COLORS["mma"], fs=10.5)
    arrow(ax, 2.55, 5.43, 6.45, 5.18, rad=-0.04)
    arrow(ax, 2.55, 4.61, 6.45, 4.88, rad=0.04)
    label(ax, 4.45, 5.50, "q_load.full", fs=8.3)
    label(ax, 4.45, 4.54, "kv_load.full", fs=8.3)
    ax.text(7.6, 4.34, "fires when all inputs ready", ha="center", fontsize=7.8,
            color="#6b7280", style="italic")

    # ---- Value MMA gate (bottom): V in SMEM, P in TMEM (split), O slot safe. ----
    ax.text(0.5, 3.35, "Value MMA gate", fontsize=11.5, weight="bold", color="#1f2937")
    box(ax, 0.6, 2.55, 1.95, 0.62, "V tile\nin SMEM", COLORS["tma"], fs=8.8)
    box(ax, 0.6, 1.55, 3.15, 0.7, "P cols 0:96 in TMEM\n+ O-slot safe (WG2)", COLORS["softmax"], fs=8.2)
    box(ax, 0.6, 0.55, 3.15, 0.62, "P cols 96:128\nin TMEM", COLORS["softmax"], fs=8.4)
    box(ax, 6.45, 1.35, 2.3, 0.95, "value MMA\nP,V -> O", COLORS["mma"], fs=10.5)
    arrow(ax, 2.55, 2.86, 6.45, 2.05, rad=-0.05)
    arrow(ax, 3.75, 1.90, 6.45, 1.83, rad=0.0)
    arrow(ax, 3.75, 0.86, 6.45, 1.52, rad=0.06)
    label(ax, 4.95, 2.42, "kv_load.full", fs=8.3)
    label(ax, 4.95, 1.93, "p_o_rescale.full", fs=8.3)
    label(ax, 4.95, 1.08, "p_ready_2.full", fs=8.3)
    ax.text(7.6, 1.14, "two-part MMA: starts on cols 0:96, then 96:128", ha="center", fontsize=7.8,
            color="#6b7280", style="italic")

    # ---- Legend (right gap). ----
    lx = 9.55
    ax.text(lx, 5.95, "Legend", fontsize=11.5, weight="bold", color="#1f2937")

    def swatch(y, color, text):
        box(ax, lx, y, 0.42, 0.34, "", color)
        ax.text(lx + 0.6, y + 0.17, text, ha="left", va="center", fontsize=8.7, color="#374151")

    swatch(5.40, COLORS["tma"], "SMEM tile (TMA-loaded)")
    swatch(4.86, COLORS["softmax"], "TMEM tile (softmax output)")
    swatch(4.32, COLORS["mma"], "MMA operation")
    label(ax, lx + 0.5, 3.66, "barrier", fs=8.0)
    ax.text(lx + 1.05, 3.66, "gate that must signal\nbefore the MMA may fire",
            ha="left", va="center", fontsize=8.7, color="#374151")
    ax.text(lx, 2.55, "kv_load.full gates both the K and V loads,\nso it appears in both gates.",
            ha="left", va="center", fontsize=8.5, color="#6b7280", style="italic")

    fig.savefig("../flash_attention_main_handoff.png", dpi=170, bbox_inches="tight", facecolor="white")


def gen_softmax_correction():
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    ax.text(6.0, 5.0, "Softmax / WG2 Scale Slot Handshake", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.0,
        4.66,
        "softmax_corr.full and softmax_corr.empty protect one SMEM slot, not the P/O compute path",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # Main full/empty lifecycle.
    box(ax, 0.55, 3.35, 1.75, 0.72, "slot empty\nsoftmax may write", COLORS["bar"], fs=8.8)
    box(ax, 2.8, 3.35, 1.95, 0.72, "softmax writes\nacc_scale / row_sum", COLORS["softmax"], fs=8.8)
    box(ax, 5.25, 3.35, 1.7, 0.72, "bar_softmax\n_corr_full", COLORS["bar"], fs=8.4)
    box(ax, 7.45, 3.35, 1.9, 0.72, "WG2 reads\nthat SMEM slot", COLORS["wg2"], fs=8.8)
    box(ax, 9.85, 3.35, 1.7, 0.72, "bar_softmax\n_corr_empty", COLORS["bar"], fs=8.4)

    arrow(ax, 2.3, 3.71, 2.8, 3.71)
    arrow(ax, 4.75, 3.71, 5.25, 3.71)
    arrow(ax, 6.95, 3.71, 7.45, 3.71)
    arrow(ax, 9.35, 3.71, 9.85, 3.71)
    arrow(ax, 10.7, 3.35, 1.42, 3.35, color="#7c3aed", rad=-0.18, lw=1.35)
    label(ax, 5.95, 2.7, "empty goes back to softmax:\nthis slot can be overwritten next time", fs=8.4)

    ax.text(0.7, 4.25, "producer", fontsize=9, weight="bold", color="#92400e")
    ax.text(7.95, 4.25, "consumer", fontsize=9, weight="bold", color="#166534")
    ax.text(0.6, 1.95, "What the full/empty pair proves", fontsize=11, weight="bold")
    ax.text(
        0.6,
        1.58,
        "full: WG2 may read the scale or final row_sum from SMEM\n"
        "empty: softmax may reuse that same SMEM slot\n"
        "scope: one slot per Q stage, arrived by 128 warpgroup threads",
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    ax.text(7.05, 1.95, "What it does not prove", fontsize=11, weight="bold")
    ax.text(
        7.05,
        1.58,
        "not: P has been written to TMEM\n"
        "not: O has been rescaled\n"
        "not: value MMA may start\n"
        "those are covered by p_o_rescale.full and p_ready_2.full",
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    fig.savefig("../flash_attention_softmax_correction.png", dpi=170, bbox_inches="tight", facecolor="white")


def main():
    gen_main_handoff()
    gen_softmax_correction()


if __name__ == "__main__":
    main()
