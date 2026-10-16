# 192 GiB HBM: what must change?

[Visual report with embedded diagrams](validation/hbm_capacity_design/report.pdf)

| V3: start with the tested recipe.<br>GPT-OSS: partitioning determines fit.<br>V4 Pro: the current execution needs a redesign. |
|---|


Removing 64 GiB cuts capacity by 25%; it does not imply a 25% cut in batch size. The practical question is which allocations overlap at the peak, and which configuration change removes them.

### Recorded peak reservation per GPU

Bars share a 256 GiB scale. The 192 GiB boundary is at the same position in every row. Green is room below that limit; red is recorded use above it.

V3 MXFP8 · 256 GPUs · batch 16 · sequence 4K

| 164.5 used | 27.5 | 64 removed |
|---|---|---|


GPT-OSS NVFP4 · 8 GPUs · batch 3 · sequence 8K

| 174.5 used | 17.5 | 64 removed |
|---|---|---|


GPT-OSS MXFP8 · 8 GPUs · batch 3 · sequence 8K

| 192 limit | +58.9 |  |
|---|---|---|


V4 Pro MXFP8 · 256 GPUs · batch 1 · sequence 4K

| 192 limit | +55.9 |  |
|---|---|---|


V3 has 27.5 GiB of counter-only room; mixed FP4 has 38.3 GiB. GPT-OSS NVFP4 has 17.5 GiB. These recipes do not yet show a need to shorten context or reduce batch.

GPT-OSS MXFP8 and V4 also exceed 192 GiB in live allocations, by 56.7 and 35.0 GiB. Reclaiming cached reservation alone cannot make those observed executions fit.

Scope: capacity-only comparison, with compute and bandwidth held constant. All memory is GiB/GPU, following the manager’s 192/256 GiB wording. If the SKU means decimal GB, the lower limit is 178.8 GiB: V3 MXFP8 room falls to 14.3 GiB and GPT-OSS NVFP4 room to 4.3 GiB. Archived counters omit some ranks/phases and external memory; V4 has NaN gradients.

## V3 — start with batch 16 and 4K context

671B · 256 GB300s · TP1 / PP1 / CP1 / EP64 / FSDP256 · full activation checkpointing · compiled training

| Recorded reservation is 164.5 GiB at batch 16, leaving 27.5 GiB for unmeasured device overhead. Validate this recipe before reducing batch or context. |
|---|


### What the real logs tell us

| 57.4 GiB<br>Batch-1 reservation | + | 107.1 GiB<br>Additional reservation at batch 16 |
|---|---|---|


Together: 164.5 GiB for MXFP8. This is a measured batch-sensitive decomposition—not a tensor-by-tensor allocation breakdown. Parameter/gradient/Adam storage, saved activations, communication, quantization and workspace are not separately measured in these logs. The prior 39.1 GiB state term was an assumption.

### How we estimate it, and the actual error

Interpolate total reserved memory between measured batches 1, 4 and 16, separately for each precision. Check batches 2 and 8 excluded from the fit. The coefficients cannot be reused for a new sequence length or mesh.

| Excluded batch / recipe | GPU | CPU | CPU − GPU |
|---|---|---|---|
| Batch 2 · MXFP8 | 61.53 | 63.28 | +1.75 / +2.85% |
| Batch 8 · MXFP8 | 105.06 | 104.84 | −0.22 / −0.21% |
| Batch 2 · mixed FP4 | 61.23 | 62.37 | +1.14 / +1.86% |
| Batch 8 · mixed FP4 | 98.69 | 99.42 | +0.73 / +0.74% |

### What changes if the complete device budget is too tight?

| Current<br>Batch 16 × accumulation 1<br>4K sequence · same mesh | → | Fallback to test<br>Batch 8 × accumulation 2<br>4K sequence · same mesh |
|---|---|---|


Both preserve global batch 4,096. The measured batch-8 runs reserve 105.1 GiB (MXFP8) / 98.7 GiB (mixed FP4), roughly 59.5 / 55.1 GiB less than batch 16. But those runs used accumulation 1; the proposed accumulation-2 peak and throughput are not validated.

Performance mechanism: a smaller microbatch can reduce matrix reuse and arithmetic intensity; accumulation restores training batch, not kernel size. No sequence or parallelism change is justified by the present V3 capacity result.

[Evidence: V3 allocation diagnosis and excluded-batch checks](https://github.com/elfiegg/titan-sim/blob/main/docs/MEMORY_GAP_DIAGNOSIS.md)

## GPT-OSS — model runtime buffers, then change the mesh

Allocation probe: 20B · 8 GB200s · TP1 / PP1 / CP2 / EP4 · batch 1 · sequence 8K · NVFP4

| The “~10% error” came from comparing training state with total allocation. Named runtime buffers reduce the matched post-step transfer error to +0.59%. This is not yet a peak predictor. |
|---|


### Real allocation breakdown: rank 0, end of step 2

State is 61.1 GiB: parameters 11.5 + gradient buffers 11.5 + master weights/Adam 28.6 + optimizer gradients 9.5. Storage ownership reconciles this census with the capture.

| Allocation component | GPU capture | CPU estimate* |
|---|---|---|
| Training state | 61.11 | 61.11 |
| Retained quantized caches | 4.74 | 4.74 |
| Dummy gradients + GEMM scratch | 1.00 | 1.00 |
| Other storage + block slack | 0.42 | 0.81 |
| Total current allocation | 67.27 | 67.66 |

*CPU recomputes state sharding and adds runtime components frozen from the 4-GPU run; the 8-GPU total is not an input. Cache ownership is identified; scratch labels use allocation stacks. Small differences largely reflect allocator block slack. Instrumented external HybridEP adds 0.11 GiB outside this total; other external memory is not fully counted.

| State-only estimate<br>61.11 vs 67.27 GiB<br>−6.15 GiB / −9.15% | → | Components included<br>67.66 vs 67.27 GiB<br>+0.40 GiB / +0.59% |
|---|---|---|


### What changes at 192 GiB?

The archived 8-GPU MXFP8 recipe (batch 3, CP1, EP1) peaks at 248.7 GiB allocated. An 8-GPU probe with batch 1, CP2 and EP4 keeps the 8K sequence and measures 115.3 GiB peak reserved over six steps. This is a measured low-memory starting point, not a matched-workload speed comparison.

| Archive: batch 3 × 1 × DP8<br>Global batch 24<br>CP1 / EP1 · sequence 8K | → | Candidate: batch 1 × 6 × DP4<br>Global batch 24<br>CP2 / EP4 · sequence 8K |
|---|---|---|


Accumulation 6 is untested. CP partitions token work and EP partitions experts; both add communication. Keep sequence length first. The archived NVFP4 recipe (EP2, batch 3) already has 17.5 GiB reserved-counter room; it does not require this change on current evidence.

[Evidence: GPT-OSS allocation captures and 4→8 GPU transfer](https://github.com/elfiegg/titan-sim/blob/main/docs/GPT_OSS_ALLOCATION_EXPERIMENT.md)

## V4 Pro — batch is already 1; redesign the peak

256 ranks · TP1 / PP4 / VPP4 / EP64 · batch 1 · sequence 4K · selective recomputation + attention offload

| At least 35.0 GiB of live allocation must disappear from the observed execution. Smaller microbatches are unavailable, and freeing cache alone is insufficient. |
|---|


### The after-step snapshot hides the capacity problem

After iteration 2: current allocated

| 175.1 used | 16.9 | 64 removed |
|---|---|---|


During the observed window: peak allocated

| 192 limit | +35.0 |  |
|---|---|---|


The live peak is 227.0 GiB versus 175.1 GiB after the step. Reservation is 235.5–247.9 GiB; these counters are not additive. Both runs develop NaN gradients, and reports precede completed graph capture.

### What is modeled—and what is missing

| Allocation group | What we can establish |
|---|---|
| Parameters, gradients, optimizer | No reconciled physical allocation breakdown for full Pro. |
| Saved tensors, quantization, scratch | Logged buffer shapes can be replayed, but ownership and simultaneous lifetimes are not established. |
| Allocator reservation | At this snapshot, reserved minus allocated is 60.4–72.8 GiB; this is not all fragmentation or a guaranteed saving. |
| Independent CPU peak vs GPU | CPU: unavailable · GPU: 227.0 GiB · delta: unavailable. |

The earlier 5.24% ratio error used the target GPU allocation as an input. It does not validate a CPU prediction of V4 memory.

### Configuration consequences

| Control | 192 GiB implication |
|---|---|
| Batch / sequence | Batch 1 cannot shrink. Keep 4K context unless changing the training requirement is acceptable; no validated 2K projection exists. |
| Same 256 GPUs | Measure the live peak, then test stronger recomputation/offload or TP changes. TP2 alone does not further partition experts when expert TP1 and EP64 remain fixed. |
| More pipeline partitions | At 256 GPUs, PP8/EP32 keeps PP×EP unchanged, so expert state is not halved. PP8/EP64 needs 512 GPUs and is an unvalidated candidate. |

Decision: no safe 192 GiB V4 configuration can be selected from the current evidence. Establish a numerically healthy baseline and measure the peak before assigning a throughput or GPU-count penalty.

[Evidence: allocation comparisons; full-Pro units resolved by the capacity source audit](https://github.com/elfiegg/titan-sim/blob/main/docs/validation/hbm_capacity_design/study.json)

## Methodology — explain bytes before projecting a recipe

| A useful estimator must answer two separate questions:<br>Which allocations exist and overlap?<br>Does the worst-rank peak fit after changing the configuration? |
|---|


| 1  MATCH THE EXECUTION | Model shapes + precision + batch/sequence + parallel groups + checkpoint/graph policy |
|---|---|
| ↓  2  ACCOUNT FOR MEMORY | Count unique state storage. Use captures for runtime buffers and their lifetimes. Use aggregate calibration only when components are unavailable. |
| ↓  3  CHECK AGAINST THE GPU | Compare the same counter, rank and phase. Freeze coefficients before evaluation. Delta = CPU estimate − GPU measurement. |
| ↓  4  CHANGE ONE DESIGN LEVER | Preserve sequence, precision and global batch; adjust microbatch/accumulation or mesh. Recompute overlap and test the new peak and throughput. |

### The component model

| Training state<br>Parameters · gradients · optimizer | + | Runtime allocations<br>Activations · quantized caches<br>Communication · scratch |
|---|---|---|


Apply storage aliasing and lifetimes, then allocator behavior. Add external pools and context usage only where they are not already counted. Device demand is the maximum simultaneous total across time and ranks—not the sum of each category’s independent maximum.

### What the present study can support

| Model | Supported today | Not established |
|---|---|---|
| V3 | Batch-dependent reserved-total interpolation; excluded-point error ≤2.85%. | Physical per-component breakdown; new sequence or mesh. |
| GPT-OSS | Physical post-step components; 4→8 GPU transfer +0.59% for NVFP4. | Full peak prediction; accumulation-6 alternative at fixed global batch. |
| V4 Pro | Observed capacity violation and some logged buffer geometry. | Independent complete CPU peak or a validated replacement recipe. |

Performance interpretation: if the same recipe fits and compute/bandwidth stay fixed, lower capacity alone need not slow it. Smaller kernels may lower arithmetic intensity; extra sharding adds communication; recomputation repeats work. These mechanisms need matched-workload timing, so this study does not report an invented average throughput loss.

Next validation: V3 batch16 unchanged, then batch8/accumulation2 if needed; GPT-OSS CP2/EP4 with accumulation6; V4 healthy full-Pro peak capture. Measure all-rank device usage through warmup, training and saving. Nemotron ~2.3T remains outside the conclusion until its configuration and matched runtime are verified.

[Reproducible measurements, input hashes and estimator deltas](https://github.com/elfiegg/titan-sim/blob/main/docs/validation/hbm_capacity_design/study.json)
