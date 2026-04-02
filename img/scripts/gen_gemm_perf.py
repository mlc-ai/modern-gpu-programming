"""Generate GEMM Optimization Journey performance chart (chapter_gemm_advanced)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

steps = ['Step 1\nSync', 'Step 4\nTMA', 'Step 7\nWarp Spec', 'Step 9\n2-CTA', 'cuBLAS']
times = [714, 5.8, 1.1, 0.9, 0.9]
colors = ['#ff6b6b', '#ffa502', '#2ed573', '#1e90ff', '#a0a0a0']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: log scale to show all steps
ax1.bar(steps, times, color=colors)
ax1.set_yscale('log')
ax1.set_ylabel('Time (ms, log scale)')
ax1.set_title('GEMM Optimization Journey (M=N=K=8192, fp16)')
for i, (t, s) in enumerate(zip(times, steps)):
    ax1.text(i, t * 1.3, f'{t} ms', ha='center', va='bottom', fontsize=9)

# Right: speedup
speedups = [1, 123, 649, 793, 793]
ax2.bar(steps, speedups, color=colors)
ax2.set_ylabel('Speedup vs Step 1')
ax2.set_title('Cumulative Speedup')
for i, s in enumerate(speedups):
    ax2.text(i, s + 20, f'{s}x', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../gemm_perf.png', dpi=150, bbox_inches='tight')
print('Saved gemm_perf.png')
plt.close()
