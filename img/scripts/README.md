# Diagram Generation Scripts

These scripts generate current and legacy tutorial diagrams. Run from this directory:

```bash
cd img/scripts
python gen_cross_warp_reduce.py              # -> ../cross_warp_reduce.png
python gen_flash_attention_barrier_flow.py   # -> ../flash_attention_main_handoff.png, ../flash_attention_softmax_correction.png
python gen_flash_attention_pipeline.py       # -> ../flash_attention_pipeline_v2.png
python gen_gemm_perf.py                      # -> ../gemm_perf.png
python gen_ldstmatrix.py                     # -> ../ldstmatrix.svg
python gen_memory_dataflow.py                # -> ../memory_dataflow.png
python gen_mma_layouts.py                    # -> ../mma_cg1_m128.svg, ../mma_cg1_m64.svg, ../mma_cg2_m256.svg, ../mma_cg2_m128.svg, ../mma_block_scaled.svg
python gen_roofline.py                       # -> ../roofline.png
python gen_sf_scale_vec.py                   # -> ../sf_scale_vec.svg
python gen_sf_tmem.py                        # -> ../sf_tmem.svg
python gen_shuffle_reduce.py                 # -> ../shuffle_reduce.png
python gen_smem_descriptor.py                # -> ../smem_descriptor.svg
python gen_swizzle_conflict.py               # -> ../swizzle_conflict.svg
python gen_tcgen05_ldst.py                   # -> ../tcgen05_ldst.svg
python gen_tma_sync_flow.py                  # -> ../tma_sync_flow.png
python gen_tmem_grid.py                      # -> ../tmem_grid.png
python gen_tmem_layout.py                    # -> ../tmem_layout_v3.png
python gen_warp_specialization_timeline.py   # -> ../warp_specialization_timeline.png
```

Requires: `matplotlib`, `numpy`

The images referenced by the current tutorial are checked into `img/`. Some scripts are kept only
for reproducibility of older or optional diagrams, so their outputs may not be checked in until
they are needed again.
