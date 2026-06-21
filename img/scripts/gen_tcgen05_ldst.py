"""tcgen05.ld / st data movement: TMEM accumulator <-> the warp register fragment.
Ground truth: nymph-rust tcgen05 datapath atoms (CUTLASS SM100 TMEM-copy traits) —
ld/st distribute TMEM into registers in the m8n8 fragment (lane l -> row l/4, two
columns = one b32), the same fragment ldmatrix builds (Ampere) and wgmma outputs
(Hopper). Outputs SVG into ../ (the img/ dir).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path; OUT = str(Path(__file__).resolve().parent.parent)  # the repo img/ dir
ROWC = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#0ea5e9", "#ec4899", "#65a30d"]
TXT = "#1f2937"

fig, ax = plt.subplots(figsize=(9.2, 4.7))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 97, "tcgen05.ld / st:  TMEM accumulator  ↔  m8n8 register fragment",
        ha="center", va="top", fontsize=11, fontweight="bold", color=TXT)

TOP, BOT = 82, 16
H = (TOP - BOT) / 8.0
SX0, SW = 9, 32          # TMEM tile
FX0, FW = 59, 32         # fragment
cw_s = SW / 8.0
cw_f = FW / 8.0


def row_y(r):
    return TOP - (r + 1) * H


ax.text(SX0 + SW / 2, TOP + 3.5, "TMEM accumulator", ha="center", fontsize=8.5, fontweight="bold", color=TXT)
ax.text(FX0 + FW / 2, TOP + 3.5, "registers (m8n8 fragment)", ha="center", fontsize=8.5, fontweight="bold", color=TXT)

# TMEM tile: 8 TLanes x 8 TCols, colored by row (TLane)
for r in range(8):
    y = row_y(r)
    for c in range(8):
        ax.add_patch(Rectangle((SX0 + c * cw_s, y), cw_s, H, facecolor=ROWC[r],
                               edgecolor="white", linewidth=0.8, alpha=0.5))
    ax.text(SX0 - 1.5, y + H / 2, f"TLane {r}", ha="right", va="center", fontsize=6.8,
            fontweight="bold", color=ROWC[r])
ax.text(SX0 + SW / 2, BOT - 3.2, "row m → TLane m  (TCol → N)", ha="center", fontsize=7, color=TXT)

# fragment: each column-pair = one lane's b32 register; lane l -> row l/4, cols 2(l%4),+1
for r in range(8):
    y = row_y(r)
    for j in range(4):
        lane = 4 * r + j
        x = FX0 + 2 * j * cw_f
        ax.add_patch(Rectangle((x, y), 2 * cw_f, H, facecolor=ROWC[r], edgecolor="white",
                               linewidth=1.6, alpha=0.9))
        ax.text(x + cw_f, y + H / 2, f"L{lane}", ha="center", va="center", fontsize=6.6,
                fontweight="bold", color="white")
ax.text(FX0 + FW / 2, BOT - 3.2, "lane l → row l/4, cols 2·(l%4),+1  (1 b32 = 2 elems)",
        ha="center", fontsize=7, color=TXT)

# arrows
ax.annotate("", xy=(FX0 - 2, 56), xytext=(SX0 + SW + 2, 56),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 60, "tcgen05.ld  (TMEM → reg)", ha="center", fontsize=7.8, color="#475569")
ax.annotate("", xy=(SX0 + SW + 2, 44), xytext=(FX0 - 2, 44),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 40, "tcgen05.st  (reg → TMEM)", ha="center", fontsize=7.8, color="#475569")

ax.text(50, 6.5, "Warpgroup-cooperative and async (gated by tcgen05.wait). The fragment is the same "
        "m8n8 layout ldmatrix builds (Ampere) and wgmma outputs (Hopper).",
        ha="center", fontsize=7.2, color=TXT, style="italic")

fig.savefig(f"{OUT}/tcgen05_ldst.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote tcgen05_ldst.svg")
