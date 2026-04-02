"""Generate XOR Shuffle Reduction diagram (chapter_rmsnorm)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

N = 8
offsets = [4, 2, 1]
n_steps = len(offsets)

box_w, box_h = 1.2, 0.6
col_gap = 1.8
col_stride = box_w + col_gap
row_stride = 0.85

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

thread_colors = ['#4a90d9', '#e67e22', '#27ae60', '#8e44ad',
                 '#e74c3c', '#16a085', '#f39c12', '#2c3e50']

values = np.arange(1, N + 1, dtype=float)

def draw_col(x_center, vals, label, is_final=False):
    ax.text(x_center, -0.8, label, ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='#27ae60' if is_final else 'black')
    positions = {}
    for t in range(N):
        y = t * row_stride
        fc = '#d5f5e3' if is_final else '#f0f0f0'
        ec = '#27ae60' if is_final else thread_colors[t]
        ax.text(x_center, y, f"t{t}: {vals[t]:.0f}", ha='center', va='center',
                fontsize=11, fontweight='bold' if is_final else 'normal',
                bbox=dict(boxstyle='round,pad=0.3', fc=fc, ec=ec, lw=2))
        positions[t] = (x_center, y)
    return positions

pos_left = draw_col(0, values, "Initial")

for step_idx, offset in enumerate(offsets):
    new_values = values.copy()
    for t in range(N):
        new_values[t] = values[t] + values[t ^ offset]

    is_final = (step_idx == n_steps - 1)
    x_right = (step_idx + 1) * col_stride
    label = "Result" if is_final else f"Step {step_idx}"
    pos_right = draw_col(x_right, new_values, label, is_final)

    x_mid = (step_idx * col_stride + x_right) / 2
    ax.text(x_mid, -0.35, f"XOR {offset}", ha='center', va='center',
            fontsize=10, style='italic', color='#666666')

    x_src = pos_left[0][0] + box_w / 2
    x_dst = pos_right[0][0] - box_w / 2

    for t in range(N):
        partner = t ^ offset
        if partner > t:
            y_t = pos_left[t][1]
            y_p = pos_left[partner][1]
            y_t_r = pos_right[t][1]
            y_p_r = pos_right[partner][1]
            rad = 0.1 + 0.03 * abs(partner - t)
            ax.annotate("", xy=(x_dst, y_t_r), xytext=(x_src, y_p),
                        arrowprops=dict(arrowstyle='->', color=thread_colors[partner],
                                        lw=1.8, connectionstyle=f'arc3,rad={rad}'))
            ax.annotate("", xy=(x_dst, y_p_r), xytext=(x_src, y_t),
                        arrowprops=dict(arrowstyle='->', color=thread_colors[t],
                                        lw=1.8, connectionstyle=f'arc3,rad={rad}'))

    values = new_values
    pos_left = pos_right

ax.set_title(f"XOR Shuffle Reduction ({N} threads)\n"
             f"Each step: every thread adds its XOR partner's value. After 3 steps, all threads hold {int(values[0])}.",
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlim(-1.5, n_steps * col_stride + 1.5)
ax.set_ylim(-1.3, (N - 1) * row_stride + 0.8)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../shuffle_reduce.png', dpi=150, bbox_inches='tight')
print('Saved shuffle_reduce.png')
plt.close()
