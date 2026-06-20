(chap_async_barriers)=
# Async Coordination: mbarriers

:::{admonition} Overview
:class: overview

- TMA and the Tensor Core are asynchronous, so issuing work is not the same as finishing it — consumers need an explicit completion signal.
- An mbarrier is that signal: producers arrive, consumers wait, and it tracks arrival counts and (for TMA) byte counts.
- Each barrier carries a *phase* that flips every round; waiting on the correct phase is what gates a consumer safely.
:::

TMA ({ref}`chap_tma`) and the Tensor Core ({ref}`chap_tensor_cores`) are
*asynchronous*: issuing the work returns immediately, long before the work is done. That is what
lets a kernel overlap memory movement with compute, but it also means a consumer cannot infer that
its input is ready just because the producer instruction has been issued. If TMA is still filling a
tile when the Tensor Core reads it, the result is wrong; if the kernel waits on the wrong signal, it
deadlocks. Whenever one engine produces data that another path will consume — TMA filling a tile for
MMA, or MMA producing a result for the epilogue — the handoff needs an explicit completion signal.
The **mbarrier**, together with the *phase* it carries, is the mechanism that makes those handoffs
safe across repeated pipeline iterations. This chapter introduces the barrier itself, then phase
tracking, and finally the small set of synchronization rules a tensor-core kernel must obey
({ref}`chap_gemm_async`).

## The mbarrier

Start with the object itself. An mbarrier ("memory barrier") is the small piece of shared-memory
state that every handoff in this chapter is built on, so it is worth seeing exactly what it holds and
what a kernel can do to it. The demo below lays out that state — the arrival counter and the phase
bit — alongside the three APIs that touch it; watch how `init`, `arrive`, and `wait` each move the
counter and the bit.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: an mbarrier's counter + phase bit, and its init / arrive / wait APIs.*

An mbarrier is a hardware synchronization object that lives in shared memory. At heart it is just a
counter that knows when it has reached zero, paired with a single **phase bit**; together the
**arrival counter** and that phase bit are everything a kernel needs to coordinate a handoff. To see
how, it helps to walk through the three operations a kernel performs on a barrier over its lifetime.

1. **Init.** We tell the barrier how many arrivals to expect. It starts life at phase 0, counter
   loaded, waiting for the first round of work to report in.
2. **Arrive.** Each arrival brings the counter one step closer to zero. A barrier can be arrived at
   in three different ways, and the differences matter:
   - **TMA tx-count arrival.** Here `mbarrier.arrive.expect_tx(bytes)` records the expected byte
     (tx) count and also counts as the issuing thread's arrival. As the load runs, the TMA engine
     issues `complete-tx` while bytes land, and the barrier flips its phase only once BOTH the
     pending arrival count and the tx (byte) count have been satisfied. In other words, the
     `expect_tx` call is not a second ordinary "arrival" — it sets up a byte budget that the
     hardware drains on its own (see {ref}`chap_tma`).
   - **`tcgen05` commit arrival.** This one is not automatic. The arrival only happens once you issue
     an explicit `tcgen05.commit.mbarrier::arrive`, and it is the completion of that commit group
     that drives the barrier arrival. Forget the commit and the barrier never advances.
   - **Thread arrive.** A thread can also arrive directly, the plain way — for example to announce
     that a shared buffer it was using is now free to reuse.
3. **Wait.** Finally, a consumer blocks on the barrier until it reaches the phase expected for this
   iteration, which is the same as saying every required arrival has happened.

The first two of these arrival paths come straight from the asynchronous engines we met in the
previous chapters: the same hardware that runs ahead of the program also reports back through the
mbarrier. That is what gives us the producer/consumer pattern for free. The producer — TMA, say —
arrives once its data is ready, and the consumer simply waits before touching it.

## Phase Tracking

We have seen *that* a barrier carries a phase bit; now we turn to *why*. A single barrier has to serve
a loop that repeats the same handoff over and over, and the phase bit is what keeps those repetitions
apart. The demo below replays one barrier across several pipeline iterations; watch the phase bit flip
from 0 to 1 and back each time its arrivals complete, and notice that each iteration waits on the
opposite phase from the one before.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the phase bit flipping as a barrier is reused across pipeline iterations.*

Why does the barrier carry a phase bit at all? The answer is reuse. A long K-loop runs the very
same handoff hundreds of times over, and allocating a fresh barrier for each iteration would be both
wasteful and impossible — that many barriers would never fit in SMEM. So we reuse one barrier, and
the phase bit is what keeps one round of reuse from being mistaken for the next. Each time all of its
arrivals complete, the barrier automatically **flips its phase** (0 → 1 → 0 → …). A single barrier
can therefore serve every iteration of a pipelined loop: iteration 0 waits on phase 0, iteration 1
on phase 1, iteration 2 on phase 0 again, and so the pattern continues.

The practical consequence is that a pipelined kernel does not need one barrier per iteration.
Instead, it keeps a small fixed set of barriers and tracks, in registers, *which phase* each stage
is waiting for. That **stage + phase** bookkeeping is what lets a software pipeline reuse a fixed
pool of SMEM buffers and barriers across a long K-loop ({ref}`chap_gemm_async`).

## Synchronization Rules

With the barrier and its phase in hand, we can step back and ask how a whole kernel is wired
together. Despite the hardware details, the synchronization model comes down to one rule: **whenever
one path produces data and another consumes it, make the handoff explicit.** In practice, tensor-core kernels
mostly reuse the same three handoff patterns:

- **Thread code → engine.** When threads write SMEM and a later MMA or TMA store reads it, insert a
  thread-level sync or fence first, so the writes are visible before the engine goes looking for them.
- **TMA → MMA.** When TMA fills a SMEM tile, the MMA must wait on that load's mbarrier before it
  reads the tile.
- **MMA → epilogue.** When `tcgen05.mma` writes TMEM, the epilogue must wait on the MMA's completion
  barrier before it reads the result.

The interactive demo below shows the TMA → MMA handoff as a timeline. Click through it to see how
the mbarrier state changes and how those changes line up with the producer and consumer APIs. The
same pattern reappears later when the Tensor Core signals the epilogue.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; min-width:1320px; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: a TMA load signalling completion through an mbarrier. The `tcgen05` MMA → epilogue
handoff works the same way, with the Tensor Core arriving on the barrier instead of TMA.*

The same mechanism also governs **resource handoff**. A barrier is not only for passing data from a
producer to a consumer; it also signals that a SMEM or TMEM region has finished serving its current
consumers and can be reused by the next stage. That is why the later GEMM chapters are full of
waits, arrives, and fences around stage reuse. Read those sites as "this consumer is done, so this
buffer can be reused for the next stage," and the kernels become much easier to follow.
