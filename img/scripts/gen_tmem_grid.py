"""Generate the simple TMEM address-space grid for chapter_tmem ('Special Memory: TMEM').

The chapter's "A 2D Address Space" section promises a plain grid: 128 TLane rows by
up to 512 TCol columns, scoped to the CTA, with an accumulator written
S[(128, N) : (1@TLane, 1@TCol)]. This draws exactly that — no attention-kernel
S/P/O concepts (those live in the flash-attention figure, tmem_layout_v3.png).

Run from img/scripts/:  python gen_tmem_grid.py   ->  ../tmem_grid.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NCOL = 512   # TCol: up to 512 32-bit columns
NROW = 128   # TLane: 128 rows
ACC_N = 256  # example accumulator width (columns)


def main():
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=180)

    # The full address space.
    ax.add_patch(Rectangle((0, 0), NCOL, NROW, facecolor="#eef3fb",
                           edgecolor="#222222", linewidth=1.6, zorder=1))

    # Light gridlines: columns every 64, rows every 32 (the alloc granularity).
    for c in range(0, NCOL + 1, 64):
        ax.plot([c, c], [0, NROW], color="#c3d2e8", linewidth=0.7, zorder=2)
    for r in range(0, NROW + 1, 32):
        ax.plot([0, NCOL], [r, r], color="#c3d2e8", linewidth=0.7, zorder=2)

    # Example accumulator occupying the first ACC_N columns of all 128 rows.
    ax.add_patch(Rectangle((0, 0), ACC_N, NROW, facecolor="#cfe8d4",
                           edgecolor="#2f8f4e", linewidth=1.6, zorder=3))
    ax.text(ACC_N / 2, NROW / 2,
            "an accumulator\nS[(128, 256) : (1@TLane, 1@TCol)]",
            ha="center", va="center", fontsize=11.5, weight="bold",
            color="#1f5e36", zorder=5)

    # One highlighted element, to show the (row, column) addressing.
    cx, cy, cw = 352, 44, 10
    ax.add_patch(Rectangle((cx, cy), cw, 4, facecolor="#ffd166",
                           edgecolor="#b8860b", linewidth=1.2, zorder=4))
    ax.annotate("one element at\nrow TLane = l, column TCol = c",
                xy=(cx + cw, cy + 2), xytext=(cx + 30, cy + 46),
                fontsize=9.5, color="#7a5a00", ha="left", va="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1.3))

    # Row 0 at the top.
    ax.set_xlim(-10, NCOL + 12)
    ax.set_ylim(0, NROW)
    ax.invert_yaxis()
    ax.axis("off")

    # Column axis (TCol) along the bottom.
    for c in [0, 128, 256, 384, 512]:
        ax.plot([c, c], [NROW, NROW + 3], color="#222222", linewidth=1.0,
                clip_on=False, zorder=4)
        ax.text(c, NROW + 7, str(c), ha="center", va="top", fontsize=9,
                color="#333333")
    ax.text(NCOL / 2, NROW + 20,
            "TCol — up to 512 32-bit columns (allocated in units of 32)",
            ha="center", va="top", fontsize=10.5, color="#333333")

    # Row axis (TLane) on the left.
    for r in [0, 64, 127]:
        ax.text(-6, r, str(r), ha="right", va="center", fontsize=9,
                color="#333333")
    ax.text(-34, NROW / 2, "TLane — 128 rows", ha="center", va="center",
            fontsize=10.5, color="#333333", rotation=90)

    ax.set_title("TMEM is a 2D address space: 128 TLane rows × up to 512 "
                 "TCol columns, per CTA", fontsize=13, weight="bold", pad=14)

    fig.tight_layout(pad=0.5)
    fig.savefig("../tmem_grid.png", bbox_inches="tight", facecolor="white")
    print("Saved tmem_grid.png")


if __name__ == "__main__":
    main()
