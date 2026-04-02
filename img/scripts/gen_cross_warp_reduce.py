"""Generate Cross-Warp Reduction diagram (chapter_rmsnorm)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(18, 8))
ax.axis('off')

N_WARPS = 4
warp_colors = ['#4a90d9', '#e67e22', '#27ae60', '#8e44ad']
warp_sums = [120, 85, 200, 95]
grand_total = sum(warp_sums)
y_mid = (N_WARPS - 1) * 1.5 / 2

# Step 1: Each warp writes to SMEM
ax.text(0.5, -0.7, "Step 1: Write to SMEM", ha='center', fontsize=11, fontweight='bold')
for w in range(N_WARPS):
    y = w * 1.5
    ax.text(0, y, f"Warp {w}\nsum={warp_sums[w]}", ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc=warp_colors[w], ec='black', lw=1.5, alpha=0.3))
    ax.annotate("", xy=(1.5, y), xytext=(0.75, y),
                arrowprops=dict(arrowstyle='->', color=warp_colors[w], lw=2))

# SMEM boxes
x_smem = 2.0
ax.text(x_smem, -0.7, "SMEM", ha='center', fontsize=11, fontweight='bold', color='#555')
for w in range(N_WARPS):
    y = w * 1.5
    ax.text(x_smem, y, f"smem[{w}]={warp_sums[w]}", ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff3cd', ec='#ffc107', lw=1.5))

# Step 2: Barrier
x_bar1 = 4.0
ax.text(x_bar1, -0.7, "Step 2", ha='center', fontsize=11, fontweight='bold')
ax.text(x_bar1, y_mid, "bar.sync\n────────\nall warps\nwait here", ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', fc='#ffcccc', ec='#e74c3c', lw=2))

# Step 3: Warp 0 reads all from SMEM
x_read = 6.5
ax.text(x_read, -0.7, "Step 3: Warp 0 reads", ha='center', fontsize=11, fontweight='bold')
for w in range(N_WARPS):
    y = w * 1.5
    ax.text(x_read, y, f"t{w}: {warp_sums[w]}", ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#e8f4fd', ec='#4a90d9', lw=1.5))
    ax.annotate("", xy=(x_read - 0.55, y), xytext=(x_smem + 0.7, y),
                arrowprops=dict(arrowstyle='->', color='#4a90d9', lw=1.5))
ax.text(x_read, N_WARPS * 1.5, "Warp 0 only", ha='center', fontsize=9,
        style='italic', color='#666')

# Step 4: Shuffle XOR reduce
x_shuf = 9.5
ax.text(x_shuf, -0.7, "Step 4: Shuffle reduce", ha='center', fontsize=11, fontweight='bold')
ax.text(x_shuf, y_mid, f"Shuffle XOR\n(same pattern\nas before)\n\n→ {grand_total}",
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.6', fc='#d5f5e3', ec='#27ae60', lw=2))
ax.annotate("", xy=(x_shuf - 0.9, y_mid), xytext=(x_read + 0.55, y_mid),
            arrowprops=dict(arrowstyle='->', color='#4a90d9', lw=2))

# Step 5: Write grand total to smem[0]
x_write = 12.5
ax.text(x_write, -0.7, "Step 5: Write total", ha='center', fontsize=11, fontweight='bold')
ax.text(x_write, y_mid, f"smem[0]\n= {grand_total}", ha='center', va='center', fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', fc='#fff3cd', ec='#27ae60', lw=2))
ax.annotate("", xy=(x_write - 0.6, y_mid), xytext=(x_shuf + 0.9, y_mid),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

# Step 6: Barrier + all warps read
x_final = 15.5
ax.text(x_final, -0.7, "Step 6: Barrier + Read", ha='center', fontsize=11, fontweight='bold')
for w in range(N_WARPS):
    y = w * 1.5
    ax.text(x_final, y, f"Warp {w}\n= {grand_total}", ha='center', va='center', fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#d5f5e3', ec='#27ae60', lw=1.5))
    ax.annotate("", xy=(x_final - 0.7, y), xytext=(x_write + 0.6, y_mid),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.2,
                                connectionstyle=f'arc3,rad={0.1 * (w - 1.5)}'))

ax.set_title("Cross-Warp Reduction: Gather → Reduce → Broadcast",
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-1.5, 17)
ax.set_ylim(-1.3, N_WARPS * 1.5 + 0.5)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../cross_warp_reduce.png', dpi=150, bbox_inches='tight')
print('Saved cross_warp_reduce.png')
plt.close()
