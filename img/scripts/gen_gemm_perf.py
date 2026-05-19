"""Generate GEMM Optimization Journey performance chart (chapter_gemm_advanced)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

steps = ['Step 1\nSync', 'Step 4\nTMA', 'Step 7\nWarp Spec', 'Step 9\n2-CTA', 'cuBLAS']
times = [70.0, 0.50, 0.23, 0.12, 0.11]
colors = ['#ff6b6b', '#ffa502', '#2ed573', '#1e90ff', '#a0a0a0']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: log scale to show all steps
ax1.bar(steps, times, color=colors)
ax1.set_yscale('log')
ax1.set_ylabel('Time (ms, log scale)')
ax1.set_title('GEMM Optimization Journey (M=N=K=4096, fp16, B200)')
for i, (t, s) in enumerate(zip(times, steps)):
    label = f'{t:.2f} ms' if t < 10 else f'{t:.0f} ms'
    ax1.text(i, t * 1.3, label, ha='center', va='bottom', fontsize=9)

# Right: speedup (cumulative vs Step 1)
speedups = [round(times[0] / t) for t in times]
ax2.bar(steps, speedups, color=colors)
ax2.set_ylabel('Speedup vs Step 1')
ax2.set_title('Cumulative Speedup')
for i, s in enumerate(speedups):
    ax2.text(i, s + max(speedups) * 0.02, f'{s}x', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../gemm_perf.png', dpi=150, bbox_inches='tight')
print('Saved gemm_perf.png')
plt.close()
