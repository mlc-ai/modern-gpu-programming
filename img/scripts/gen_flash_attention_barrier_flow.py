"""Generate Flash Attention 4 Barrier Flow diagram (chapter_flash_attention)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(18, 22))
ax.axis('off')

wg3_x = 3.0
wg01_x = 9.0
wg2_x = 15.0

for label, x, color in [('WG3\n(TMA + MMA)', wg3_x, '#e8f4fd'), ('WG0 / WG1\n(Softmax)', wg01_x, '#fff3cd'), ('WG2\n(Correction)', wg2_x, '#d5f5e3')]:
    ax.add_patch(mpatches.FancyBboxPatch((x - 1.2, 0), 2.4, 0.8, boxstyle='round,pad=0.05',
                 fc=color, ec='black', lw=2))
    ax.text(x, 0.4, label, ha='center', va='center', fontsize=11, fontweight='bold')

for x_div in [6.0, 12.0]:
    ax.plot([x_div, x_div], [1.0, 21], color='#dddddd', lw=1, linestyle='--')

def task(x, y, text, color, w=2.2, h=0.6):
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle='round,pad=0.05', fc=color, ec='black', lw=1.5))
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

def barrier(src, dst, label, color='#e74c3c', label_pos=None, curve=0):
    style = 'arc3,rad=%.2f' % curve if curve else 'arc3,rad=0'
    ax.annotate('', xy=dst, xytext=src,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                connectionstyle=style))
    if label_pos:
        lx, ly = label_pos
    else:
        lx = (src[0] + dst[0]) / 2
        ly = (src[1] + dst[1]) / 2 - 0.3
    ax.text(lx, ly, label, ha='center', va='center', fontsize=7,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=color, lw=0.7, alpha=0.9))

# WG3 tasks
task(wg3_x, 2.0, 'TMA: Load Q', '#ffcccc')
task(wg3_x, 3.5, 'TMA: Load K', '#ffcccc')
task(wg3_x, 5.0, 'TMA: Load V', '#ffcccc')
task(wg3_x, 7.5, 'Score MMA\nQ @ K^T', '#cce0ff', h=0.8)
task(wg3_x, 14.0, 'Value MMA\nP @ V', '#cce0ff', h=0.8)
task(wg3_x, 19.5, 'TMA: Store O', '#ffcccc')

# WG0/WG1 tasks
task(wg01_x, 10.0, 'Read S from TMEM\nCompute softmax\nWrite P to TMEM', '#ffe0b2', w=2.6, h=1.2)
task(wg01_x, 12.5, 'Write acc_scale\n+ row_sum', '#ffe0b2', h=0.7)

# WG2 tasks
task(wg2_x, 12.5, 'Read acc_scale\nRescale O in TMEM', '#c8e6c9', h=0.7)
task(wg2_x, 17.5, 'Normalize O / sum\nWrite O to SMEM', '#c8e6c9', h=0.7)

# Barriers
barrier((wg3_x - 1.1, 2.3), (wg3_x - 1.1, 7.1),
        'bar_load_q_full', '#4a90d9', label_pos=(wg3_x - 2.0, 4.5), curve=-0.12)
barrier((wg3_x + 1.1, 5.3), (wg3_x + 1.1, 7.1),
        'bar_load_kv_full', '#4a90d9', label_pos=(wg3_x + 2.0, 6.2), curve=0.12)
barrier((wg3_x + 1.1, 7.5), (wg01_x - 1.3, 9.4),
        'bar_s_full', '#e74c3c', label_pos=(6.0, 8.0))
barrier((wg01_x - 1.3, 9.4), (wg3_x + 1.1, 3.8),
        'bar_load_kv_empty', '#16a085', label_pos=(6.0, 6.0), curve=-0.25)
barrier((wg01_x - 1.3, 10.7), (wg3_x + 1.1, 13.6),
        'bar_p_full_o_rescaled\n(Softmax)', '#8e44ad', label_pos=(5.5, 12.5))
barrier((wg2_x - 1.1, 12.9), (wg3_x + 1.1, 14.0),
        'bar_p_full_o_rescaled\n(WG2)', '#8e44ad', label_pos=(9.0, 14.5), curve=-0.2)
barrier((wg01_x + 1.3, 12.2), (wg2_x - 1.1, 12.2),
        'bar_softmax_corr_full', '#e67e22', label_pos=(12.0, 11.7))
barrier((wg2_x - 1.1, 12.8), (wg01_x + 1.3, 12.8),
        'bar_softmax_corr_empty', '#16a085', label_pos=(12.0, 13.3))
barrier((wg3_x + 1.1, 14.4), (wg2_x - 1.1, 17.1),
        'bar_o_full (last iter)', '#e74c3c', label_pos=(9.0, 16.0), curve=0.1)
barrier((wg2_x - 1.1, 17.9), (wg3_x + 1.1, 19.1),
        'bar_corr_epi_full', '#27ae60', label_pos=(9.0, 18.8), curve=-0.1)
barrier((wg3_x + 1.1, 19.9), (wg2_x - 1.1, 17.9),
        'bar_corr_epi_empty', '#16a085', label_pos=(9.0, 19.5), curve=-0.15)

# Time arrow
ax.annotate('', xy=(0.3, 20.5), xytext=(0.3, 1.2),
            arrowprops=dict(arrowstyle='->', color='#999999', lw=2))
ax.text(0.3, 11, 'time', ha='center', va='center', fontsize=10, color='#999999',
        rotation=90, style='italic')

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='#e74c3c', lw=2, label='MMA completion'),
    plt.Line2D([0], [0], color='#4a90d9', lw=2, label='TMA load ready'),
    plt.Line2D([0], [0], color='#8e44ad', lw=2, label='P ready + O rescaled'),
    plt.Line2D([0], [0], color='#e67e22', lw=2, label='Softmax -> Correction'),
    plt.Line2D([0], [0], color='#16a085', lw=2, label='Buffer free (empty)'),
    plt.Line2D([0], [0], color='#27ae60', lw=2, label='Epilogue ready'),
]
ax.legend(handles=legend_elements, loc='lower center', fontsize=8, framealpha=0.9,
          ncol=3, bbox_to_anchor=(0.5, -0.01))

ax.set_xlim(-0.5, 17.5)
ax.set_ylim(-0.5, 21.5)
ax.invert_yaxis()
ax.set_title('Flash Attention 4: Barrier Flow Between Warpgroups', fontsize=15, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('../flash_attention_barrier_flow.png', dpi=150, bbox_inches='tight')
print('Saved flash_attention_barrier_flow.png')
plt.close()
