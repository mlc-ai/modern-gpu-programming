# Diagram Generation Scripts

These scripts generate the tutorial diagrams in `img/`. Run from this directory:

```bash
cd img/scripts
python gen_shuffle_reduce.py          # -> ../shuffle_reduce.png (RMSNorm practice kernel)
python gen_cross_warp_reduce.py       # -> ../cross_warp_reduce.png (RMSNorm practice kernel)
python gen_warp_specialization_timeline.py  # -> ../warp_specialization_timeline.png (legacy warp-specialization timeline)
python gen_tma_sync_flow.py           # -> ../tma_sync_flow.png (TMA synchronization)
python gen_flash_attention_barrier_flow.py  # -> ../flash_attention_main_handoff.png and ../flash_attention_softmax_correction.png
python gen_flash_attention_pipeline.py      # -> ../flash_attention_pipeline_v2.png (Flash Attention pipeline)
python gen_tmem_layout.py            # -> ../tmem_layout_v3.png (Flash Attention TMEM layout)
python gen_gemm_perf.py               # -> ../gemm_perf.png (GEMM optimization result)
python gen_ai_assisted_workflow.py    # -> ../ai_assisted_tirx_workflow.png (AI-assisted TIRX workflow)
```

Requires: `matplotlib`, `numpy`

Note: These images are also in `img/` as static files referenced by the tutorial markdown.
The scripts are kept here for reproducibility — if you need to tweak a diagram, edit the
script and re-run it.
