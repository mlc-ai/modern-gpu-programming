(chap_profiling)=
# Profiling and Debugging

```{note}
This chapter is being drafted. It will cover profiling and debugging TIRx kernels; for the performance model it builds on, see {ref}`chap_performance`.
```

```{note}
**Outline stub** — to be drafted (P2). See `OUTLINE.md`, Ch16.
```

**Goal:** measure and fix kernels.

## Reading the Generated Code
Inspecting generated CUDA / PTX / SASS.

## Nsight Compute Basics
Occupancy and stall analysis; the roofline in practice (ties back to {ref}`chap_performance`).

## Debugging Barriers and Correctness
A worked perf-debug example; the elected-commit bug case study.

*Feeds from:* existing `setup` (Inspect Generated CUDA) and `ai_assisted` (Case Study). New: the profiling workflow and a worked example.
