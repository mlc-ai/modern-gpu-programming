"""Generate TMA Async Load Synchronization Flow diagram (chapter_gemm_async, Step 4)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

lanes = {
    'Elected\nThread': (2.5, '#f0f0f0'),
    'TMA\nHardware': (6.0, '#ffe0e0'),
    'mbarrier': (9.5, '#fff3cd'),
    'tcgen05\nMMA': (13.0, '#d5f5e3'),
}

for label, (x, color) in lanes.items():
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.9, -0.1), 1.8, 0.7, boxstyle='round,pad=0.05',
                 fc=color, ec='black', lw=2))
    ax.text(x, 0.25, label, ha='center', va='center', fontsize=11, fontweight='bold')

for label, (x, color) in lanes.items():
    ax.plot([x, x], [0.8, 11.5], color='#cccccc', lw=1.5, linestyle=':')

def action_arrow(y, x_from, x_to, label, color='#333333', lw=2):
    ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    mx = (x_from + x_to) / 2
    ax.text(mx, y - 0.25, label, ha='center', va='center', fontsize=9, fontweight='bold',
            color=color, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

def wait_bar(y, x, label, color='#e74c3c'):
    ax.plot([x - 0.3, x + 0.3], [y, y], color=color, lw=4, solid_capstyle='round')
    ax.text(x, y + 0.2, label, ha='center', va='bottom', fontsize=8, color=color, style='italic')

def step_label(y, num, text):
    ax.text(-1.2, y, '%d' % num, ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', bbox=dict(boxstyle='circle,pad=0.3', fc='#4a90d9', ec='black', lw=1.5))
    ax.text(-0.5, y, text, ha='left', va='center', fontsize=8, color='#555')

et_x = 2.5
tma_x = 6.0
bar_x = 9.5
mma_x = 13.0

y = 1.5
step_label(y, 1, 'Issue TMA')
action_arrow(y, et_x, tma_x, 'copy_async(A)', '#e74c3c')

y = 2.3
action_arrow(y, et_x, tma_x, 'copy_async(B)', '#e74c3c')

y = 3.2
step_label(y, 2, 'Set byte count')
action_arrow(y, et_x, bar_x, 'arrive.expect_tx(bytes)', '#e67e22')

y = 4.3
step_label(y, 3, 'HW transfer')
ax.add_patch(mpatches.FancyBboxPatch((tma_x - 0.55, 3.6), 1.1, 1.0, boxstyle='round,pad=0.05',
             fc='#ff6b6b', ec='black', lw=1.5, alpha=0.3))
ax.text(tma_x, 4.1, 'TMA transfer\n(GMEM->SMEM)', ha='center', va='center', fontsize=7, color='#cc0000', style='italic')

y = 4.8
action_arrow(y, tma_x, bar_x, 'arrive (auto)', '#27ae60')

y = 5.8
step_label(y, 4, 'Wait for data')
action_arrow(y, et_x, bar_x, 'try_wait(phase)', '#8e44ad')
wait_bar(5.8, et_x, 'blocked', '#8e44ad')

y = 6.6
action_arrow(y, bar_x, et_x, 'phase complete!', '#27ae60')

y = 7.8
step_label(y, 5, 'Make visible')
ax.text(et_x, y, 'fence.after_thread_sync()', ha='center', va='center', fontsize=8, fontweight='bold',
        color='#4a90d9',
        bbox=dict(boxstyle='round,pad=0.3', fc='#e8f4fd', ec='#4a90d9', lw=1.5))

y = 9.2
step_label(y, 6, 'Issue MMA')
action_arrow(y, et_x, mma_x, 'gemm_async + commit', '#4a90d9')

ax.add_patch(mpatches.FancyBboxPatch((mma_x - 0.55, 9.6), 1.1, 1.0, boxstyle='round,pad=0.05',
             fc='#27ae60', ec='black', lw=1.5, alpha=0.3))
ax.text(mma_x, 10.1, 'MMA compute\n(SMEM->TMEM)', ha='center', va='center', fontsize=7, color='#1a7a1a', style='italic')

ax.set_xlim(-2.0, 15.0)
ax.set_ylim(-0.5, 11.5)
ax.invert_yaxis()
ax.set_title('TMA Async Load: Synchronization Flow', fontsize=15, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('../tma_sync_flow.png', dpi=150, bbox_inches='tight')
print('Saved tma_sync_flow.png')
plt.close()
