(chap_async_barriers)=
# Async Coordination: mbarriers

:::{admonition} Overview
:class: overview

- TMA and the Tensor Core are asynchronous, so issuing work is not the same as finishing it — consumers need an explicit completion signal.
- An mbarrier is that signal: producers arrive, consumers wait, and it tracks arrival counts and (for TMA) byte counts.
- Each barrier carries a *phase* that flips every round; waiting on the correct phase is what gates a consumer safely.
:::

The Tensor Core
({ref}`chap_tensor_cores`) and TMA ({ref}`chap_tma`) are both *asynchronous*, and on both of them
issuing work is not the same as finishing it. That asynchrony is what buys overlap, but it also
creates a hazard. The moment one engine produces data that another will consume — TMA filling a
tile that the Tensor Core will read, or the Tensor Core writing a result the epilogue will read —
the consumer has no built-in way to know the data has actually arrived. Every such handoff must be
made explicit, or the kernel races. The primitive that makes those handoffs safe, and reusable
across pipeline iterations, is the **mbarrier**.

## The mbarrier

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: an mbarrier's counter + phase bit, and its init / arrive / wait APIs.*

An mbarrier is a hardware synchronization object stored in shared memory: a counter that knows when
it has reached zero. It combines an **arrival counter** with a **phase bit**. Its lifecycle is the
three operations a kernel performs on it:

1. **Init** — set the expected number of arrivals; the barrier starts at phase 0.
2. **Arrive** — each arrival decrements the counter. There are three ways to arrive:
   - **TMA auto-arrive** — the hardware arrives once the expected byte count has landed (this is
     the `arrive.expect_tx` mechanism from {ref}`chap_tma`).
   - **`tcgen05` auto-arrive** — the hardware arrives once committed MMAs complete.
   - **Thread arrive** — a thread arrives explicitly, e.g. to signal that a shared buffer is free
     to reuse.
3. **Wait** — a consumer blocks until the barrier reaches the expected phase for this iteration,
   which means all required arrivals have happened.

The first two arrival paths come from the asynchronous engines of the previous
chapters: the same hardware that runs ahead also reports back through the mbarrier. This gives the
producer/consumer pattern directly — the producer (say, TMA) arrives
when its data is ready, and the consumer waits before touching it.

## Phase Tracking

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the phase bit flipping as a barrier is reused across pipeline iterations.*

The phase bit exists for reuse. A long
K-loop runs the same handoff hundreds of times, and allocating a fresh barrier for each iteration
would be wasteful and would not even fit in SMEM. The barrier is reused, and the phase bit
keeps successive reuses from being confused for one another. When all arrivals complete, the
barrier automatically **flips its phase** (0 → 1 → 0 → …), which lets a single barrier serve every
iteration of a pipelined loop: iteration 0 waits on phase 0, iteration 1 on phase 1, iteration 2 on
phase 0 again, and so on.

The consequence is that a pipelined kernel never allocates a barrier per iteration. It keeps a small
set of barriers and tracks, in a register, *which phase* the current stage expects. This
**stage + phase** bookkeeping is precisely how a software pipeline reuses a fixed pool of SMEM
buffers and barriers across a long K-loop ({ref}`chap_gemm_async`).

## Synchronization Rules

The whole model reduces to a single rule: **whenever one path produces data
and another consumes it, make the handoff explicit.** A tensor-core kernel only ever exhibits three
such handoffs:

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

The same rule extends to resource reuse, which is a handoff running the other way:
before TMEM or a SMEM buffer is freed or overwritten, every participant that might still read it
must first have arrived. The GEMM chapters spell out the exact wait and fence at each of these
handoffs; those kernels resolve into a sequence of producer/consumer pairs.
