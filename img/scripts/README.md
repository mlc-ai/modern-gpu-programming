# Diagram Generation Scripts

These scripts generate the tutorial diagrams in `img/`. Run from this directory:

```bash
cd img/scripts
python gen_shuffle_reduce.py          # -> ../shuffle_reduce.png (ch5 RMSNorm)
python gen_cross_warp_reduce.py       # -> ../cross_warp_reduce.png (ch5 RMSNorm)
python gen_warp_specialization_timeline.py  # -> ../warp_specialization_timeline.png (ch8 Step 7)
python gen_tma_sync_flow.py           # -> ../tma_sync_flow.png (ch7 Step 4)
python gen_flash_attention_barrier_flow.py  # -> ../flash_attention_barrier_flow.png (ch9)
python gen_gemm_perf.py               # -> ../gemm_perf.png (ch8 Complete Journey)
```

Requires: `matplotlib`, `numpy`

Note: These images are also in `img/` as static files referenced by the tutorial markdown.
The scripts are kept here for reproducibility — if you need to tweak a diagram, edit the
script and re-run it.
