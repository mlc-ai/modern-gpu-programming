"""Generate Warp Specialization Timeline diagram (chapter_gemm_advanced, Step 7)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# --- Top: Before (Sequential) ---
ax1.set_xlim(-0.5, 12.5)
ax1.set_ylim(0, 1.2)
ax1.axis('off')
ax1.set_title('Before (Step 4): Sequential - 50% hardware idle', fontsize=13, fontweight='bold', pad=5)

for i in range(5):
    x = i * 2.4
    ax1.add_patch(mpatches.FancyBboxPatch((x, 0.3), 1.0, 0.6, boxstyle='round,pad=0.05', fc='#ff6b6b', ec='black', lw=1.5))
    ax1.text(x + 0.5, 0.6, 'Load\nk=%d' % i, ha='center', va='center', fontsize=8, fontweight='bold')
    ax1.add_patch(mpatches.FancyBboxPatch((x + 1.1, 0.3), 1.0, 0.6, boxstyle='round,pad=0.05', fc='#4a90d9', ec='black', lw=1.5))
    ax1.text(x + 1.6, 0.6, 'MMA\nk=%d' % i, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
ax1.text(12.3, 0.6, '...', fontsize=14, ha='left', va='center')

# --- Bottom: After (Step 7, PIPE_DEPTH=2) ---
ax2.set_xlim(-2.8, 14.5)
ax2.set_ylim(-1.2, 3.2)
ax2.axis('off')
ax2.set_title('After (Step 7, PIPE_DEPTH=2): TMA at most 1 stage ahead of MMA', fontsize=13, fontweight='bold', pad=5)

tma_y = 2.2
mma_y = 1.2
wb_y = 0.2

# Fixed-size lane labels
label_w = 1.6
label_h = 0.55
label_x = -2.5
for label, y, color in [('TMA\n(WG1 warp 3)', tma_y, '#ff6b6b'), ('MMA\n(WG1 warp 0)', mma_y, '#4a90d9'), ('Writeback\n(WG0)', wb_y, '#27ae60')]:
    ax2.add_patch(mpatches.FancyBboxPatch((label_x, y - label_h/2), label_w, label_h,
                  boxstyle='round,pad=0.05', fc=color, ec='black', lw=1.5, alpha=0.3))
    ax2.text(label_x + label_w/2, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

sp = 1.85
bw = 1.5

shown_k = [0, 1, 2, None, -2, -1]
k_labels_tma = ['Load k=0', 'Load k=1', 'Load k=2', '...', 'Load k=N-2', 'Load k=N-1']
k_labels_mma = ['MMA k=0', 'MMA k=1', 'MMA k=2', '...', 'MMA k=N-2', 'MMA k=N-1']
buf_labels = ['buf 0', 'buf 1', 'buf 0', '', 'buf 0', 'buf 1']

last_mma_x = 0
for idx, (k, tl, ml, bl) in enumerate(zip(shown_k, k_labels_tma, k_labels_mma, buf_labels)):
    x = idx * sp
    if k is None:
        ax2.text(x + 0.75, tma_y, '...', fontsize=16, ha='center', va='center', color='#888')
        ax2.text(x + 0.75 + sp, mma_y, '...', fontsize=16, ha='center', va='center', color='#888')
    else:
        ax2.add_patch(mpatches.FancyBboxPatch((x, tma_y - 0.22), bw, 0.44, boxstyle='round,pad=0.05', fc='#ff6b6b', ec='black', lw=1.5))
        ax2.text(x + bw/2, tma_y, tl, ha='center', va='center', fontsize=8, fontweight='bold')
        if bl:
            ax2.text(x + bw/2, tma_y + 0.35, bl, ha='center', fontsize=7, color='#888', style='italic')
        mx = x + sp
        ax2.add_patch(mpatches.FancyBboxPatch((mx, mma_y - 0.22), bw, 0.44, boxstyle='round,pad=0.05', fc='#4a90d9', ec='black', lw=1.5))
        ax2.text(mx + bw/2, mma_y, ml, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        last_mma_x = mx

# Writeback box
wb_x = last_mma_x + bw + 0.3
ax2.add_patch(mpatches.FancyBboxPatch((wb_x, wb_y - 0.22), bw, 0.44, boxstyle='round,pad=0.05', fc='#27ae60', ec='black', lw=1.5))
ax2.text(wb_x + bw/2, wb_y, 'Writeback\nTMEM->GMEM', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Barrier: tma2mma
ax2.annotate('', xy=(sp + bw/2, mma_y + 0.22), xytext=(bw/2, tma_y - 0.22),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.8, linestyle='dashed'))
ax2.text(sp * 0.5 + bw/2 + 0.1, (tma_y + mma_y) / 2, 'tma2mma', ha='center', va='center', fontsize=8,
         color='#e74c3c', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='#e74c3c', lw=0.8, alpha=0.9))

# Barrier: mma2tma
ax2.annotate('', xy=(2*sp + bw/2, tma_y - 0.22), xytext=(sp + bw/2 + 0.3, mma_y + 0.22),
             arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.8, linestyle='dashed'))
ax2.text(sp * 1.5 + bw/2 + 0.3, (tma_y + mma_y) / 2, 'mma2tma', ha='center', va='center', fontsize=8,
         color='#e67e22', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='#e67e22', lw=0.8, alpha=0.9))

# Legend
legend_elements = [
    mpatches.Patch(fc='#ff6b6b', ec='black', label='TMA Load'),
    mpatches.Patch(fc='#4a90d9', ec='black', label='MMA Compute'),
    mpatches.Patch(fc='#27ae60', ec='black', label='Writeback'),
    plt.Line2D([0], [0], color='#e74c3c', linestyle='dashed', label='tma2mma'),
    plt.Line2D([0], [0], color='#e67e22', linestyle='dashed', label='mma2tma'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9,
           bbox_to_anchor=(1.0, -0.15), borderpad=0.4, handlelength=1.5)

plt.savefig('../warp_specialization_timeline.png', dpi=150, bbox_inches='tight')
print('Saved warp_specialization_timeline.png')
plt.close()
