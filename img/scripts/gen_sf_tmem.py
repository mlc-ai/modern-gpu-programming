"""Block-scaled MMA scale factors (SFA, SFB) in TMEM. Ground truth: nymph-rust +
tvm sf_tmem_layout (backend/cuda/.../gemm_async/tcgen05.py): rows must be a
multiple of 32; M = rows // 32; epc = 4 (four 8-bit SFs per 32-bit TMEM column);
the atom is one 32-row chunk with R[4 : 32@TLane] (a warpx4 broadcast).

Two distinct mappings, drawn as two panels:
  (1) Packing  — 128 M-rows occupy only 32 TMEM lanes (TLane = m % 32; the m // 32
      group runs along TCol).
  (2) Replication — those 32 stored lanes are broadcast (warpx4, R[4 : 32@TLane])
      to all 128 lanes of the reading warpgroup: lane l reads TLane (l mod 32).
Outputs SVG (and a PNG for inspection).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from pathlib import Path; OUT = str(Path(__file__).resolve().parent.parent)  # the repo img/ dir
TXT = "#1f2937"
PURPLE = ["#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"]

fig, ax = plt.subplots(figsize=(11.6, 5.7))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 98.5, "Scale factors in TMEM — packed into 32 lanes, then warpx4-broadcast to 128",
        ha="center", va="top", fontsize=12.8, fontweight="bold", color=TXT)

# ----------------------------------------------------------------------------
# Panel 1 — packing: 128 M-rows -> 32 TMEM lanes
# ----------------------------------------------------------------------------
ax.text(22, 90, "① Packed: 128 M-rows → 32 TMEM lanes", ha="center", fontsize=9.6,
        fontweight="bold", color=TXT)
ax.text(22, 86.3, "TLane = m % 32      m // 32 → TCol", ha="center", fontsize=7.6,
        color=TXT, style="italic")

X0, CW, YT, RH = 12, 7.4, 79, 8.0
rows = [0, 1, 2, None, 31]
for gi in range(4):                       # m // 32 group -> TCol
    cx = X0 + gi * CW
    ax.text(cx + CW / 2, YT + 1.3, str(gi), ha="center", va="bottom", fontsize=7,
            fontweight="bold", color=TXT)
    for ri, lane in enumerate(rows):
        y = YT - (ri + 1) * RH
        if lane is None:
            ax.text(cx + CW / 2, y + RH / 2, "⋮", ha="center", va="center", fontsize=11, color=TXT)
            continue
        ax.add_patch(Rectangle((cx, y), CW, RH, facecolor=PURPLE[gi], edgecolor="white",
                               linewidth=1.3, alpha=0.92))
        ax.text(cx + CW / 2, y + RH / 2, f"r{gi * 32 + lane}", ha="center", va="center",
                color="white", fontsize=6.4, fontweight="bold")
for ri, lane in enumerate(rows):          # left lane labels
    y = YT - (ri + 1) * RH
    lab = "⋮" if lane is None else f"TLane {lane}"
    ax.text(X0 - 1.3, y + RH / 2, lab, ha="right", va="center", fontsize=7, fontweight="bold", color=TXT)
ax.text(X0 + 2 * CW, YT + 4.6, "TCol →  (m // 32 group, then K)", ha="center", fontsize=7.3, color=TXT)
ax.text(22, 16, "Only 32 lanes hold all 128 M-rows.", ha="center", fontsize=7.6, color=TXT, style="italic")

# ----------------------------------------------------------------------------
# Bridge arrow — the warpx4 broadcast
# ----------------------------------------------------------------------------
ax.annotate("", xy=(50.5, 52), xytext=(43.5, 52),
            arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.4))
ax.text(47, 56, "warpx4\nbroadcast", ha="center", va="bottom", fontsize=7.6,
        fontweight="bold", color="#7c3aed")

# ----------------------------------------------------------------------------
# Panel 2 — replication: 32 stored lanes -> 128 reading lanes (4 copies)
# ----------------------------------------------------------------------------
ax.text(76, 90, "② Replicated to all 128 warpgroup lanes", ha="center", fontsize=9.6,
        fontweight="bold", color=TXT)
ax.text(76, 86.3, "R[4 : 32@TLane]  —  4 copies at lane stride 32", ha="center", fontsize=7.6,
        color=TXT, style="italic")

# source: the 32 stored lanes
SX, SY, SW, SH = 52.5, 38, 11, 28
ax.add_patch(Rectangle((SX, SY), SW, SH, facecolor="#ede9fe", edgecolor="#7c3aed", linewidth=1.8))
ax.text(SX + SW / 2, SY + SH / 2, "TLane\n0–31\n\n(stored\nonce)", ha="center", va="center",
        fontsize=7.4, fontweight="bold", color="#5b21b6")

# 4 destination quadrants of the reading warpgroup
DX, DW, DH = 78, 19, 11
ranges = ["lanes 0–31", "lanes 32–63", "lanes 64–95", "lanes 96–127"]
dys = [66, 52, 38, 24]
for i, (rg, dy) in enumerate(zip(ranges, dys)):
    ax.add_patch(Rectangle((DX, dy), DW, DH, facecolor=PURPLE[i], edgecolor="white",
                           linewidth=1.4, alpha=0.92))
    ax.text(DX + DW / 2, dy + DH * 0.66, rg, ha="center", va="center", color="white",
            fontsize=7.6, fontweight="bold")
    ax.text(DX + DW / 2, dy + DH * 0.30, "≡ TLane 0–31", ha="center", va="center",
            color="white", fontsize=6.6)
    ax.add_patch(FancyArrowPatch((SX + SW, SY + SH / 2), (DX, dy + DH / 2),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color="#8b5cf6", lw=1.3, shrinkA=0, shrinkB=0))

ax.text(76, 16, "lane l reads TLane (l mod 32) — no extra storage.", ha="center",
        fontsize=7.6, color=TXT, style="italic")

# ----------------------------------------------------------------------------
ax.text(50, 7.5, "Loaded SMEM→TMEM via `tcgen05.cp`; the block-scaled `tcgen05.mma` reads the "
        "scale factors from all 128 warpgroup lanes.", ha="center", fontsize=8.0, color=TXT, style="italic")

fig.savefig(f"{OUT}/sf_tmem.svg", facecolor="white", bbox_inches="tight")
fig.savefig("/tmp/sf_tmem_preview.png", dpi=130, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote sf_tmem.svg")
