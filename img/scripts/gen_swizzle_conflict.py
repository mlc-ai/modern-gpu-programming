"""Bank-conflict schematic for the swizzle worked example: middle = an 8x8
row-major SMEM tile coloured by bank group (= column); left = the coalesced row
write (8 distinct banks, conflict-free); right = the ldmatrix column read (all
one bank, 8-way conflict). Outputs SVG into ../ (the img/ dir).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path; OUT = str(Path(__file__).resolve().parent.parent)  # the repo img/ dir
# 8 distinct colors = 8 bank groups
COLS = ["#ef4444", "#f59e0b", "#eab308", "#22c55e", "#14b8a6", "#3b82f6", "#8b5cf6", "#ec4899"]
TXT = "#1f2937"
GREEN = "#15803d"
RED = "#b91c1c"

fig, ax = plt.subplots(figsize=(9.4, 4.7))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 97, "Row-major SMEM: a row write hits 8 banks, a column read hits 1",
        ha="center", va="top", fontsize=11, fontweight="bold", color=TXT)

# ---- middle: SMEM 8x8 (color = bank group = column) ----
SX0, SY_TOP, SW, SH = 43, 80, 16, 60
cw, ch = SW / 8.0, SH / 8.0
WR_ROW, RD_COL = 2, 5
for r in range(8):
    for c in range(8):
        x = SX0 + c * cw
        y = SY_TOP - (r + 1) * ch
        ax.add_patch(Rectangle((x, y), cw, ch, facecolor=COLS[c], edgecolor="white",
                               linewidth=0.6, alpha=0.85))
ax.text(SX0 + SW / 2, SY_TOP + 2.5, "SMEM", ha="center", fontsize=8.5, fontweight="bold", color=TXT)
ax.text(SX0 + SW / 2, SY_TOP - SH - 3.5, "8×8 row-major\ncolour = bank group (= column)",
        ha="center", va="top", fontsize=7, color=TXT)
# highlight the written row and the read column
ax.add_patch(Rectangle((SX0, SY_TOP - (WR_ROW + 1) * ch), SW, ch, facecolor="none",
                       edgecolor="#111827", linewidth=2.2))
ax.add_patch(Rectangle((SX0 + RD_COL * cw, SY_TOP - SH), cw, SH, facecolor="none",
                       edgecolor="#111827", linewidth=2.2))

# ---- left: WRITE (one row, 8 distinct banks) ----
ax.text(15, 86, "WRITE — one row", ha="center", fontsize=8.8, fontweight="bold", color=TXT)
ax.text(15, 81.5, "(coalesced GMEM→SMEM)", ha="center", fontsize=6.8, color=TXT)
wy, wh = 50, 9
for c in range(8):
    x = 1.5 + c * 3.5
    ax.add_patch(Rectangle((x, wy), 3.5, wh, facecolor=COLS[c], edgecolor="white", linewidth=0.8))
    ax.text(x + 1.75, wy + wh + 2.2, f"T{c}", ha="center", fontsize=6, fontweight="bold", color=TXT)
    ax.text(x + 1.75, wy + wh / 2, f"b{c}", ha="center", va="center", fontsize=6,
            fontweight="bold", color="white")
ax.text(15, wy - 5, "banks {0,1,…,7}: all distinct", ha="center", fontsize=7, color=TXT)
ax.text(15, wy - 9.5, "✓ conflict-free", ha="center", fontsize=8.5, fontweight="bold", color=GREEN)
ax.annotate("", xy=(SX0 - 1, SY_TOP - (WR_ROW + 0.5) * ch), xytext=(30, wy + wh / 2),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.8))

# ---- right: READ (one column, all same bank) ----
ax.text(85, 86, "READ — one column", ha="center", fontsize=8.8, fontweight="bold", color=TXT)
ax.text(85, 81.5, "(ldmatrix)", ha="center", fontsize=6.8, color=TXT)
rx, rw = 82, 9
for r in range(8):
    y = SY_TOP - (r + 1) * ch
    ax.add_patch(Rectangle((rx, y), rw, ch, facecolor=COLS[RD_COL], edgecolor="white", linewidth=0.8))
    ax.text(rx + rw + 1.5, y + ch / 2, f"T{r}", ha="left", va="center", fontsize=6,
            fontweight="bold", color=TXT)
    ax.text(rx + rw / 2, y + ch / 2, f"b{RD_COL}", ha="center", va="center", fontsize=6,
            fontweight="bold", color="white")
ax.text(86, 15.5, "all bank 5: same", ha="center", fontsize=7, color=TXT)
ax.text(86, 11, "✗ 8-way conflict", ha="center", fontsize=8.5, fontweight="bold", color=RED)
ax.annotate("", xy=(rx - 1, SY_TOP - SH / 2), xytext=(SX0 + SW + 1, SY_TOP - SH / 2),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.8))

ax.text(50, 4.5, "A row spans all 8 bank groups (distinct); a column is one bank group (×8). "
        "Swizzle stores column c at c⊕r, making both distinct.",
        ha="center", fontsize=6.8, color=TXT, style="italic")

fig.savefig(f"{OUT}/swizzle_conflict.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote swizzle_conflict.svg")
