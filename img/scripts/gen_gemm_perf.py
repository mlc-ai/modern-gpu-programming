"""Generate GEMM Optimization Journey performance chart (chapter_gemm_advanced)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

steps = ['Step 3\nSync tiled', 'Step 4\nTMA', 'Step 7\nWarp spec', 'Step 8\n2-CTA', 'Step 9\nMulti-cons.', 'cuBLAS']
times = [53.642159, 0.493814, 0.226613, 0.103529, 0.094139, 0.094139]
colors = ['#ff6b6b', '#ffa502', '#2ed573', '#1e90ff', '#5352ed', '#a0a0a0']

fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

# Vertical log-scale time chart. The table in the text carries speedup values;
# the figure focuses on the drop in runtime.
xpos = list(range(len(steps)))
ax.bar(xpos, times, color=colors, width=0.68)
ax.set_yscale('log')
ax.set_ylim(0.06, 120)
ax.set_xticks(xpos)
ax.set_xticklabels(steps)
ax.set_ylabel('Time (ms, log scale)')
ax.set_title('GEMM Optimization Journey (M=N=K=4096, fp16, B200)')
ax.grid(axis='y', which='major', linestyle='--', alpha=0.35)

for x, t in enumerate(times):
    time_label = f'{t:.3f} ms' if t < 0.2 else (f'{t:.2f} ms' if t < 10 else f'{t:.1f} ms')
    ax.text(x, t * 1.20, time_label, ha='center', va='bottom', fontsize=9, clip_on=False)

plt.savefig('../gemm_perf.png', dpi=150, bbox_inches='tight')
print('Saved gemm_perf.png')
plt.close()
