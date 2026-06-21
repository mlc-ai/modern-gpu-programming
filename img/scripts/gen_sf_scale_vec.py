"""How a scale-factor u32 packs K-block scales for each scale_vec mode, mirroring
the PTX tcgen05 MMA scale-factor-A 1x / 2x / 4x layouts (self-drawn).

Each TMEM word (one u32 per (M-row, column)) has 4 bytes [0:7][8:15][16:23][24:31].
  1X (fp8 / mxfp8, SF_VEC=32): one scale, broadcast to all 4 bytes.
  2X (mxfp4,        SF_VEC=32): two scales, each duplicated.
  4X (nvfp4,        SF_VEC=16): four scales = four K-blocks.
Outputs SVG into ../ (the img/ dir).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path; OUT = str(Path(__file__).resolve().parent.parent)  # the repo img/ dir
TXT = "#1f2937"
SFC = {"SF0": "#ef4444", "SF1": "#3b82f6", "SF2": "#10b981", "SF3": "#f59e0b"}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 97, "scale_vec modes: how one TMEM u32 packs K-block scales", ha="center", va="top",
        fontsize=12, fontweight="bold", color=TXT)

SX, SW = 30, 44          # strip x-origin, width
bw = SW / 4
BYTES = ["[0:7]", "[8:15]", "[16:23]", "[24:31]"]

# byte-position header
for b in range(4):
    ax.text(SX + (b + 0.5) * bw, 86, BYTES[b], ha="center", va="center", fontsize=7.5, color=TXT)
ax.text(SX + SW / 2, 90, "bytes of the 32-bit word", ha="center", fontsize=8, color=TXT)

ROWS = [
    ("1X", "fp8 / mxfp8", ["SF0", "SF0", "SF0", "SF0"], "one scale, broadcast ×4   (SF_VEC = 32)"),
    ("2X", "mxfp4", ["SF0", "SF1", "SF0", "SF1"], "two scales, each ×2   (SF_VEC = 32)"),
    ("4X", "nvfp4", ["SF0", "SF1", "SF2", "SF3"], "four scales = four K-blocks   (SF_VEC = 16)"),
]
ys = [66, 46, 26]
sh = 13
for (mode, fmt, bytes_, cap), y in zip(ROWS, ys):
    ax.text(SX - 3, y + sh / 2, f"{mode}", ha="right", va="center", fontsize=11, fontweight="bold", color=TXT)
    ax.text(SX - 3, y - 2.5, fmt, ha="right", va="center", fontsize=7.5, color=TXT)
    for b, lab in enumerate(bytes_):
        ax.add_patch(Rectangle((SX + b * bw, y), bw, sh, facecolor=SFC[lab], edgecolor="white",
                               linewidth=1.6, alpha=0.92))
        ax.text(SX + (b + 0.5) * bw, y + sh / 2, lab, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.text(SX + SW + 3, y + sh / 2, cap, ha="left", va="center", fontsize=8, color=TXT)

ax.text(50, 9, "SFk = the scale for K-block k of one M-row.  sf_per_mma (the “×N”) = mma_k / SF_VEC: "
        "1 (fp8), 2 (mxfp4), 4 (nvfp4).", ha="center", fontsize=7.6, color=TXT, style="italic")
ax.text(50, 4, "Lane/column placement (TLane = m%32, m//32 → column) is the same for all three modes; "
        "only the byte packing differs.", ha="center", fontsize=7.6, color=TXT, style="italic")

fig.savefig(f"{OUT}/sf_scale_vec.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote sf_scale_vec.svg")
