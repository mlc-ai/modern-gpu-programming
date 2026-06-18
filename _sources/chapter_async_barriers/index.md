(chap_async_barriers)=
# Async Coordination: mbarriers and Phases

The Tensor Core ({ref}`chap_tensor_cores`) and TMA ({ref}`chap_tma`) are both *asynchronous*:
issuing work is not the same as finishing it. To build a correct kernel, every time one engine
produces data another consumes, the handoff must be explicit. The primitive that makes those
handoffs safe — and reusable across pipeline iterations — is the **mbarrier**.

## The mbarrier

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: an mbarrier's counter + phase bit, and its init / arrive / wait APIs.*

An mbarrier is a hardware synchronization object stored in shared memory. It combines an
**arrival counter** with a **phase bit**, and its lifecycle is:

1. **Init** — set the expected number of arrivals; the barrier starts at phase 0.
2. **Arrive** — each arrival decrements the counter. There are three ways to arrive:
   - **TMA auto-arrive** — the hardware arrives once the expected byte count has landed (this is
     the `arrive.expect_tx` mechanism from {ref}`chap_tma`).
   - **`tcgen05` auto-arrive** — the hardware arrives once committed MMAs complete.
   - **Thread arrive** — a thread arrives explicitly, e.g. to signal that a shared buffer is free
     to reuse.
3. **Wait** — a consumer blocks until the barrier reaches the expected phase for this iteration,
   which means all required arrivals have happened.

The producer/consumer pattern is then direct: the producer (say, TMA) arrives when data is ready;
the consumer waits before touching it.

## Phase Tracking

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the phase bit flipping as a barrier is reused across pipeline iterations.*

The reason an mbarrier carries a **phase bit** is reuse. When all arrivals complete, the barrier
automatically **flips its phase** (0 → 1 → 0 → …). That lets a single barrier serve every
iteration of a pipelined loop: iteration 0 waits on phase 0, iteration 1 on phase 1, iteration 2
on phase 0 again, and so on.

So a pipelined kernel doesn't allocate a barrier per iteration — it keeps a small set of barriers
and tracks, in a register, *which phase* the current stage expects. This **stage + phase**
bookkeeping (a `PipelineState` in Part III does exactly this) is how a software pipeline reuses a
fixed pool of SMEM buffers and barriers across a long K-loop.

## Synchronization Rules

The whole model reduces to one rule: **whenever one path produces data and another consumes it,
make the handoff explicit.** The three handoffs that appear in every tensor-core kernel are:

- **Thread code → engine.** If threads write SMEM and a later MMA or TMA store reads it, insert a
  thread-level sync or fence first.
- **TMA → MMA.** If TMA fills a SMEM tile, the MMA must wait on the load's mbarrier before reading
  that tile.
- **MMA → epilogue.** If `tcgen05.mma` writes TMEM, the epilogue must wait for the MMA's completion
  barrier before reading the result.

The **TMA → MMA** handoff in motion — the TMA engine arrives on the barrier when its bytes land,
and the consumer's `try_wait` releases:

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; min-width:1320px; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: a TMA load signalling completion through an mbarrier. The `tcgen05` MMA → epilogue
handoff works the same way, with the Tensor Core arriving on the barrier instead of TMA.*

The same rule guards resource reuse: before TMEM or a SMEM buffer is freed or overwritten, every
participant that might still read it must have arrived. The GEMM chapters spell out the exact wait
and fence at each handoff; once you recognize the three patterns above, those kernels read as a
sequence of producer/consumer pairs rather than a wall of intrinsics.
