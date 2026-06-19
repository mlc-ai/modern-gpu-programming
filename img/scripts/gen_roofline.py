"""Generate the roofline figure for chapter_performance ('What Makes a Kernel Fast').

Order-of-magnitude NVIDIA B200 ceilings (matching chapter_background's convention;
exact values depend on SKU and clock):
  - dense FP16/BF16 tensor-core peak ~ 2 PFLOP/s (2000 TFLOP/s)
  - HBM3e bandwidth                  ~ 8 TB/s (8000 GB/s)
Ridge point = 2000e12 / 8e12 = ~250 FLOP/byte.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PEAK_TFLOPS = 2000.0      # ~2 PFLOP/s dense fp16 tensor core (order of magnitude)
BW_TB_S = 8.0             # HBM3e, TB/s  (==> attainable = 8 * AI  TFLOP/s)
RIDGE = PEAK_TFLOPS / BW_TB_S   # ~281 FLOP/byte

ai = np.logspace(-1, 4.3, 500)            # arithmetic intensity, FLOP/byte
roof = np.minimum(PEAK_TFLOPS, BW_TB_S * ai)

fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
ax.plot(ai, roof, color='#222', lw=2.2, zorder=3)
ax.axhline(PEAK_TFLOPS, color='#888', ls='--', lw=1, alpha=0.6)
ax.axvline(RIDGE, color='#888', ls=':', lw=1, alpha=0.6)
ax.text(RIDGE * 1.1, 3.5, f'ridge ≈ {RIDGE:.0f} FLOP/byte', color='#555', fontsize=8.5, rotation=90, va='bottom')
ax.text(0.12, PEAK_TFLOPS * 1.07, f'compute roof ≈ {PEAK_TFLOPS/1000:.0f} PFLOP/s (fp16)', color='#555', fontsize=8.5)
ax.text(0.13, 8 * 0.13 * 1.15, f'memory roof: {BW_TB_S:.0f} TB/s', color='#555', fontsize=8.5, rotation=34)

# Example workloads: (label, arithmetic intensity, achieved TFLOP/s, color, offset, ha)
# The naive-GEMM label goes to the *right* of its point so it clears the vertical
# ridge label near x ~ 250; the SOTA label stays to the left.
pts = [
    ('Elementwise / RMSNorm\n(memory-bound)', 0.4, 8 * 0.4 * 0.7, '#ff6b6b', (14, 10), 'left'),
    ('GEMM 4096³ — naive\n(leaves the SM idle)', 1365, 2.9, '#ffa502', (12, -2), 'left'),
    ('GEMM 4096³ — SOTA\n(~⅔ of peak)', 1365, 1320, '#2ed573', (-8, 10), 'right'),
]
for label, x, y, c, xytext, ha in pts:
    ax.scatter([x], [y], s=70, color=c, zorder=5, edgecolor='white', linewidth=0.8)
    ax.annotate(label, (x, y), textcoords='offset points',
                xytext=xytext, ha=ha, fontsize=8.5, color='#333')
# the optimization gap arrow for GEMM
ax.annotate('', xy=(1365, 1320), xytext=(1365, 2.9),
            arrowprops=dict(arrowstyle='->', color='#2ed573', lw=1.6, alpha=0.8))
ax.text(1365 * 0.62, 70, 'optimization\nclimbs here', color='#2ed573', fontsize=8.5, ha='right')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.1, 2e4); ax.set_ylim(2, 4000)
ax.set_xlabel('Arithmetic intensity (FLOP / byte)')
ax.set_ylabel('Attainable performance (TFLOP/s)')
ax.set_title('Roofline (approx. B200) — where workloads live and what optimization buys')
ax.grid(which='both', ls='--', alpha=0.25)
plt.savefig('../roofline.png', dpi=150, bbox_inches='tight')
print('Saved roofline.png')
plt.close()
