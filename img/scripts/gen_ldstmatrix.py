"""Schematic of ldmatrix/stmatrix data movement: an 8x8 b16 SMEM tile <-> the
warp register fragment. Ground truth: nymph-rust values/ldstmatrix.rs
(element_coord: lane l holds row l/4, cols 2*(l%4) and +1; row r address comes
from lane r). Outputs SVG into ../ (the img/ dir).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

from pathlib import Path; OUT = str(Path(__file__).resolve().parent.parent)  # the repo img/ dir
ROWC = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#0ea5e9", "#ec4899", "#65a30d"]
TXT = "#1f2937"

fig, ax = plt.subplots(figsize=(9.2, 4.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 97, "ldmatrix / stmatrix:  8×8 b16 SMEM tile  ↔  warp register fragment",
        ha="center", va="top", fontsize=11, fontweight="bold", color=TXT)

# --- geometry ---
TOP, BOT = 82, 14
H = (TOP - BOT) / 8.0          # row height
SX0, SW = 9, 32                # SMEM tile x-origin, width
FX0, FW = 59, 32              # fragment x-origin, width
cw_s = SW / 8.0
cw_f = FW / 8.0


def row_y(r):
    return TOP - (r + 1) * H


# headers
ax.text(SX0 + SW / 2, TOP + 3.5, "SMEM: 8×8 fp16 (row-major)", ha="center", fontsize=8.5,
        fontweight="bold", color=TXT)
ax.text(FX0 + FW / 2, TOP + 3.5, "Registers across 32 lanes", ha="center", fontsize=8.5,
        fontweight="bold", color=TXT)

# SMEM tile: 8 rows x 8 fp16, colored by row; address supplied by lane T{r}
for r in range(8):
    y = row_y(r)
    for c in range(8):
        ax.add_patch(Rectangle((SX0 + c * cw_s, y), cw_s, H, facecolor=ROWC[r],
                               edgecolor="white", linewidth=0.8, alpha=0.5))
    ax.text(SX0 - 1.5, y + H / 2, f"T{r}", ha="right", va="center", fontsize=7.5,
            fontweight="bold", color=ROWC[r])
ax.text(SX0 + SW / 2, BOT - 3.2, "row r address ← lane T{r}", ha="center", fontsize=7, color=TXT)

# Fragment: same 8x8, colored by row; each column-pair = one lane's b32 register
for r in range(8):
    y = row_y(r)
    for j in range(4):  # 4 lanes per row, each owns cols 2j, 2j+1
        lane = 4 * r + j
        x = FX0 + 2 * j * cw_f
        ax.add_patch(Rectangle((x, y), 2 * cw_f, H, facecolor=ROWC[r],
                               edgecolor="white", linewidth=1.6, alpha=0.9))
        ax.text(x + cw_f, y + H / 2, f"L{lane}", ha="center", va="center", fontsize=6.6,
                fontweight="bold", color="white")
ax.text(FX0 + FW / 2, BOT - 3.2, "lane l → row l/4, cols 2·(l%4), +1  (1 b32 = 2 fp16)",
        ha="center", fontsize=7, color=TXT)

# arrows between
ax.annotate("", xy=(FX0 - 2, 56), xytext=(SX0 + SW + 2, 56),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 60, "ldmatrix  (SMEM → reg)", ha="center", fontsize=7.8, color="#475569")
ax.annotate("", xy=(SX0 + SW + 2, 44), xytext=(FX0 - 2, 44),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 40, "stmatrix  (reg → SMEM)", ha="center", fontsize=7.8, color="#475569")

ax.text(50, 6, ".x1 loads one 8×8 (addresses from T0–T7); .x2 / .x4 load 2 / 4 matrices "
        "(T0–T15 / T0–T31).  .trans reads each 8×8 column-major.",
        ha="center", fontsize=6.8, color=TXT, style="italic")

fig.savefig(f"{OUT}/ldstmatrix.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote ldstmatrix.svg")
