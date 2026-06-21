"""How a wgmma/tcgen05 SMEM matrix descriptor lays an operand A(M x K) out in
shared memory. Params (ptx_*_encode_matrix_descriptor): start_address, ldo
(leading byte offset), sdo (stride byte offset), swizzle (layout_type).

The operand is a 2-D grid of swizzle ATOMS. The swizzle format sets the atom
shape (8 x 128 B for SWIZZLE_128B; 64/32/16 B otherwise) and the XOR pattern
inside it. Each atom is ONE CONTIGUOUS 8 x 128 B (1 KB) block: its 8 rows are
back-to-back in memory. ldo/sdo are the strides BETWEEN atoms: ldo along the
MAJOR dim, sdo along the OTHER. (K-major shown; MN-major swaps them.) Output SVG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path; OUT = str(Path(__file__).resolve().parent.parent)  # the repo img/ dir
TXT = "#1f2937"
C_A = "#3b82f6"
C_B = "#60a5fa"

fig, ax = plt.subplots(figsize=(9.8, 5.8))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 98, "SMEM matrix descriptor → how A(M×K) sits in shared memory",
        ha="center", va="top", fontsize=12.5, fontweight="bold", color=TXT)

NR, NC = 2, 3
X0, X1, YB, YT = 30, 88, 24, 72
bw = (X1 - X0) / NC
bh = (YT - YB) / NR

for br in range(NR):
    for bc in range(NC):
        x = X0 + bc * bw
        y = YT - (br + 1) * bh
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor=(C_A if (br + bc) % 2 == 0 else C_B),
                               edgecolor="white", linewidth=2.0, alpha=0.9))
        for k in range(1, 8):  # 8 rows, drawn back-to-back (contiguous)
            ax.plot([x, x + bw], [y + k * bh / 8, y + k * bh / 8], color="white", lw=0.4, alpha=0.5)
        # first atom: label shifted right to leave room for the contiguity ticks
        lx = x + bw * 0.62 if (br == 0 and bc == 0) else x + bw / 2
        ax.text(lx, y + bh / 2, "atom\n8 × 128 B", ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
    ax.text(X0 - 2, YT - (br + 0.5) * bh, ["rows 0–7", "rows 8–15"][br], ha="right", va="center",
            fontsize=8.5, fontweight="bold", color=TXT)

# contiguity detail inside the first (top-left) atom: 8 rows are consecutive 128 B
fx, fy = X0, YT - bh
ax.add_patch(Rectangle((fx, fy), bw, bh, facecolor="none", edgecolor="#111827", linewidth=2.4))
ax.annotate("", xy=(fx + 3.0, fy + 2.5), xytext=(fx + 3.0, fy + bh - 2.5),
            arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.5))
ax.text(fx + 4.2, fy + bh - 3.5, "byte 0", fontsize=7, color="#111827", va="center")
ax.text(fx + 4.2, fy + 3.5, "+896 B", fontsize=7, color="#111827", va="center")
ax.text(fx + bw / 2, fy + bh + 1.8, "contiguous 1 KB", ha="center", fontsize=7,
        color="#111827", fontweight="bold")

# axes
ax.text((X0 + X1) / 2, YB - 8.5, "K  (bytes) →", ha="center", fontsize=9.5, color=TXT)
ax.annotate("", xy=(X1 + 1.5, YT), xytext=(X1 + 1.5, YB), arrowprops=dict(arrowstyle="-|>", color=TXT, lw=1.3))
ax.text(X1 + 4, (YB + YT) / 2, "M ↓", ha="left", va="center", fontsize=9.5, color=TXT)

# start_address marker
ax.plot(X0, YT, marker="o", color="#111827", markersize=6)
ax.text(X0 - 2.5, YT + 6, "start_address (addr ≫ 4)", ha="left", va="bottom", fontsize=9, color="#111827")

# ldo: stride between atoms along the MAJOR dim (K here)
ax.annotate("", xy=(X0 + 2.5 * bw, YT - 3.2), xytext=(X0 + 1.5 * bw, YT - 3.2),
            arrowprops=dict(arrowstyle="<|-|>", color="#111827", lw=1.6))
ax.text(X0 + 2 * bw, YT - 7.5, "ldo  (major dim, K here)", ha="center", fontsize=8.5,
        fontweight="bold", color="#111827")

# sdo: stride between atoms along the OTHER dim (M here)
ax.annotate("", xy=(X0 - 9.5, YT - 1.5 * bh), xytext=(X0 - 9.5, YT - 0.5 * bh),
            arrowprops=dict(arrowstyle="<|-|>", color="#111827", lw=1.6))
ax.text(X0 - 12, YT - bh, "sdo\n(other\ndim, M)", ha="right", va="center", fontsize=8,
        fontweight="bold", color="#111827")

# swizzle-format note
ax.text((X0 + X1) / 2, YB - 4.0, "swizzle format sets the atom shape (8 × 128 B here; 64 / 32 / 16 B "
        "otherwise) and the XOR pattern inside it", ha="center", fontsize=8, color="#7c3aed")

ax.text(50, 7, "Each atom is one contiguous 8 × 128 B block — its 8 rows are back-to-back in memory; "
        "ldo / sdo jump between atoms.", ha="center", fontsize=8.2, color=TXT, style="italic")
ax.text(50, 3, "ldo = stride along the major dim, sdo = along the other.  K-major shown "
        "(major = K); an MN-major operand swaps ldo and sdo.", ha="center", fontsize=8.2,
        color=TXT, style="italic")

fig.savefig(f"{OUT}/smem_descriptor.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote smem_descriptor.svg")
