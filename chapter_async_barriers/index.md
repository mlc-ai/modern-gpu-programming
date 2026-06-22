(chap_async_barriers)=
# Async Coordination: mbarriers

:::{admonition} Overview
:class: overview

- TMA and the Tensor Core are asynchronous, so issuing work is not the same as finishing it, and consumers need an explicit completion signal.
- An mbarrier is that signal: producers arrive, consumers wait, and it tracks arrival counts and (for TMA) byte counts.
- Each barrier carries a *phase* that flips every round; waiting on the correct phase is what gates a consumer safely.
:::

TMA ({ref}`chap_tma`) and Tensor Core ({ref}`chap_tensor_cores`) operations are asynchronous. When a kernel issues a TMA load or a `tcgen05` MMA, the issuing thread does not wait for the operation to finish. The instruction is only submitted to the hardware engine; the actual data movement or matrix operation continues in parallel with the rest of the program.

That is useful because it lets memory movement and compute overlap. It also means that program order is not enough to prove that data is ready. A later instruction may run before the earlier asynchronous operation has completed. If TMA is still writing a shared-memory tile when MMA starts reading it, the MMA reads incomplete data. If the epilogue reads TMEM before the Tensor Core has finished writing the accumulator, it reads the wrong value. If the kernel waits on the wrong condition, it may never make progress.

The kernel therefore needs an explicit completion signal at every asynchronous handoff. An `mbarrier` is that signal. A producer arrives on the barrier when its work is complete, and a consumer waits on the barrier before using the produced data. The same mechanism is used for TMA-to-MMA handoff, MMA-to-epilogue handoff, and buffer reuse across pipeline stages.

A barrier is not just a one-shot flag. It carries a phase bit, and that phase bit changes every time the barrier completes a round of arrivals. The phase is what lets one barrier be reused across many loop iterations without confusing the completion of one iteration with the completion of another.

## The mbarrier

An `mbarrier`, short for memory barrier, is a hardware synchronization object stored in shared memory. Conceptually, it contains two pieces of state: an arrival counter and a phase bit. The counter tells the barrier how many arrivals are still missing in the current round. The phase bit tells the kernel which round the barrier is currently in.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: an `mbarrier` state view showing the arrival counter, the phase bit, and the `init`, `arrive`, and `wait` operations; click a field to focus it.*

A barrier starts with initialization. During `init`, the kernel sets how many arrivals this barrier should expect. The barrier begins in phase 0 with its counter loaded to that expected arrival count. From that point on, the barrier is waiting for all required producers or users of a resource to report that they are done.

An arrival reduces the amount of work the barrier is still waiting for. Different parts of a kernel can arrive on a barrier in different ways, and the distinction matters.

For TMA loads, the usual arrival path is a tx-count arrival. An operation such as `mbarrier.arrive.expect_tx(bytes)` does two things. First, it counts as the issuing thread's arrival on the barrier. Second, it records the number of bytes that the TMA engine is expected to transfer. The barrier is not complete just because the issuing thread has arrived. It also waits for the TMA engine to drain the byte count as the transfer finishes. The phase flips only after both conditions are satisfied: the normal arrival count has reached zero, and the pending tx byte count has reached zero.

This is why `expect_tx` should not be read as "one more ordinary arrival." It sets up a byte budget for the asynchronous copy. The hardware later accounts for the actual copy completion through complete-tx updates. The barrier completes only when the arrivals and the byte transfer have both completed.

For Tensor Core work, the arrival path is different. A `tcgen05` MMA does not automatically advance a barrier just because the MMA was issued. The kernel must explicitly attach a barrier arrival to the commit path, for example with a `tcgen05.commit.mbarrier::arrive` operation. When that committed group completes, the Tensor Core side performs the barrier arrival. If the kernel forgets that commit arrival, the consumer waiting on the barrier will wait forever.

A normal thread can also arrive directly on a barrier. This is used when ordinary thread code is the producer, or when a set of threads is announcing that it has finished using a resource. For example, after a consumer finishes reading a shared-memory buffer, it can arrive on a barrier that tells the producer the buffer is free to reuse.

Waiting is the consumer side of the same protocol. A consumer waits until the barrier has completed the phase expected for the current iteration. Only then is it safe to read the data or reuse the resource protected by that barrier.

The important point is that asynchronous hardware does not only run ahead of the program; it also reports completion back through the barrier. TMA can signal that a shared-memory tile is ready. Tensor Core work can signal that TMEM results are ready. Ordinary threads can signal that a buffer is no longer in use. The barrier gives all of these cases the same producer-consumer shape: the producer arrives, the consumer waits.

## Phase Tracking

A barrier is usually not allocated for a single use. A pipelined K-loop may execute the same handoff hundreds of times, and allocating a new shared-memory barrier for every iteration would not be practical. Instead, the kernel keeps a small fixed set of barriers and reuses them as the loop advances.

The phase bit is what makes that reuse safe.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: a reused barrier across several pipeline iterations, showing the phase bit flipping after each completed round.*

Each time a barrier completes all arrivals for its current round, it flips phase: phase 0 becomes phase 1, phase 1 becomes phase 0, and so on. A wait operation checks the phase expected by the consumer. That expected phase is kept in a register by the kernel. After a stage has successfully waited for one round, the kernel toggles its local phase value before using the barrier for the next round.

This prevents the kernel from mistaking an old completion for a new one. Suppose a barrier was used for one TMA load and has already completed. If the next loop iteration reused the same barrier without tracking phase, a consumer could observe the previous completion and incorrectly assume the new load is ready. The phase bit separates those two rounds. Iteration 0 waits for one phase, iteration 1 waits for the opposite phase, iteration 2 waits for the first phase again, and the pattern continues.

In a real pipeline, the bookkeeping is usually per stage. The kernel has a fixed number of shared-memory stages, a matching fixed number of barriers, and a small set of phase values in registers. As the loop advances, each logical iteration maps onto one physical stage, and the phase value tells the wait operation which round of that physical barrier it is waiting for.

This is why the later GEMM code does not need one barrier per K tile ({ref}`chap_gemm_async`). It needs one barrier per reusable stage, plus phase tracking. The stage index selects the shared-memory buffer and barrier. The phase value distinguishes the current use of that stage from the previous one.

**Try with your agent**: Give it a two-stage pipeline and ask it to trace four iterations. For each iteration, list the stage index, the local phase value, when the barrier flips, and what would go wrong if the phase were not toggled before the stage is reused.

## Synchronization Rules

Once the barrier and phase mechanism are clear, the synchronization pattern in a tensor-core kernel is fairly mechanical. Every time one path produces data or releases a resource that another path will consume, the handoff must be made explicit.

There are three common cases.

The first case is thread code producing data for an asynchronous engine. If threads write shared memory and a later TMA store or MMA instruction reads that shared memory, the kernel must make the thread writes visible before the engine reads them. This requires the appropriate thread-level synchronization or fence. The exact instruction depends on the scope of the handoff, but the reason is always the same: the engine must not observe the shared-memory buffer before the producing threads have finished writing it.

The second case is TMA producing data for MMA. A TMA load fills a shared-memory tile asynchronously. The MMA path cannot infer that the tile is ready just because the TMA instruction was issued. The TMA operation must be associated with an `mbarrier`, and the MMA path must wait on that barrier before reading the tile.

The third case is MMA producing data for the epilogue. A `tcgen05` MMA writes its result into TMEM asynchronously. The epilogue cannot safely read the accumulator until the Tensor Core has completed the relevant work. The MMA commit path therefore arrives on a completion barrier, and the epilogue waits on that barrier before reading TMEM.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; min-width:1320px; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: a TMA load signaling completion through an `mbarrier`. The MMA path waits for the barrier before reading the shared-memory tile. The Tensor Core to epilogue handoff follows the same shape, except that the Tensor Core commit path performs the arrival instead of TMA.*

The same idea also applies to resource reuse. A barrier is not only a data-ready signal. It can also be a "resource is free" signal. A shared-memory stage cannot be overwritten until all consumers of the old tile are done with it. A TMEM region cannot be reused until the previous user has finished reading or writing it. In those cases, the arrival means "I am done with this resource," and the wait means "it is now safe to reuse this resource for the next stage."

This is the right way to read the synchronization in a pipelined GEMM kernel. The waits and arrives are not scattered around as defensive programming. Each one marks a concrete ownership transfer: a tile becomes ready, an accumulator becomes readable, or a buffer becomes reusable. Once those handoffs are identified, the control flow becomes much easier to follow.
