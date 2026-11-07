# Observe native physical memory

`titan_sim.memory_observer.MemoryObserver` records real CUDA allocator counters
around an unchanged native trainer or component. It requires the installed
training environment; importing the module does not import PyTorch. The observation
is evidence about the measured configuration and execution window, not a fit
prediction for another configuration.

## Caller integration

The caller owns CUDA device selection, process groups, trainer construction,
recipe arguments and cleanup. Use the same recipe and runtime provenance for all
ranks, and supply a globally unique physical-device identity (host plus physical
GPU identity, not just `cuda:0`). Keep such identifiers and generated results out
of commits. Each rank writes its own file; a coordinator merges them afterward.

```python
from titan_sim.memory_observer import MemoryObserver, merge_observations

# Device must already identify this rank's real local CUDA device.
observer = MemoryObserver(
    device=device, rank=rank, world_size=world_size,
    physical_device=physical_device_id,
    provenance={
        "recipe": original_recipe_identity,
        "runtime": pinned_runtime_identity,
        "source_commit": source_commit,
        "execution": {"warmup_steps": warmup_steps, "capture_steps": steps},
    },
    snapshot=True,
    observe_peak_resets=True,
)
try:
    with observer:
        with observer.phase("initialization"):
            trainer = build_original_trainer()
        observer.sample("initialized_state", engine=trainer.engine)
        # Bracket only operations that the original native training path executes.
        # Instrument native step callbacks to label warmup/capture separately.
        with observer.phase("capture", engine=trainer.engine):
            run_original_training(trainer)
finally:
    write_rank_json(observer.report())

# Coordinator: absent ranks remain explicit, duplicates/mismatches are rejected.
combined = merge_observations(rank_reports, expected_ranks=range(world_size))
```

The generic observer calls no trainer method itself. Native `capture_trainer`
can be placed inside a `capture` phase. Alternatively, its original `train_step`
can be wrapped with `warmup` or `capture` phases. Phases cannot nest. Building the
trainer before entering the observer leaves initialization unknown. A separate
`checkpoint_loading` phase is required to label that lifecycle operation;
training activation checkpointing is not checkpoint loading. A successful phase
label is a caller assertion that the complete operation was bracketed.
If construction resolves additional source/configuration identities, call
`observer.set_provenance(resolved_provenance)` afterward. Provenance is copied at
binding time, so mutating the original dictionary does not update the report.

When native capture catches its own error and returns a blocked workload, preserve
that workload's `capture_status` beside the observation: observer completion only
means its own context exited without an exception. On a propagated failure the
observer retains partial samples and reports failure; it does not swallow the
native exception. A failed phase never establishes lifecycle coverage.

## Reading the evidence

Each synchronized sample contains these distinct measurements:

| Field | Meaning | Limitation |
|---|---|---|
| `process_allocator.allocated_bytes` | Current allocations tracked by this process's PyTorch CUDA allocator | Excludes direct external CUDA allocations |
| `process_allocator.reserved_bytes` | Allocator reservations, including allocated memory | Never add allocated and reserved |
| `process_allocator.slack_bytes` | Reserved minus allocated at that boundary | Not peak reserved minus peak allocated |
| `process_allocator.peak_*` | Allocator high-water marks since the latest observer reset | Not a device-wide total or necessarily simultaneous peaks |
| `device_global_sample.free_bytes/total_bytes/used_bytes` | `mem_get_info` sample for the selected device | Includes other processes and misses between-sample peaks |
| `used_minus_process_reserved_bytes` | Arithmetic residual retained for audit | Not attributable to a library or to this process; readings are not atomic |
| `state_inventory` | Visible local backing storage for parameters, gradients, buffers and optimizer state | Boundary inventory, not full allocation history or an addition to allocator totals |

The observer resets peak counters on entry and at phase boundaries, preserving
previous samples before each reset. A short-lived allocator allocation between
samples is represented by the high-water mark even if freed before the next
sample. An external allocation between samples can be missed. The global sampled
peak therefore remains a sampled maximum with no finite physical upper bound.

State inventories inspect native `engine.model_parts` and
`engine.optim.optimizers` when available, recursively visiting optimizer state.
Local DTensor storage is inspected through `to_local()` without materializing a
global tensor. Aliases count once at each boundary and belong to the first
category encountered. CPU/offloaded and other-device tensor references are
reported as exclusions. Lazy optimizer state becomes visible only after the
native optimizer creates it. Unsupported state objects remain unresolved; no
optimizer is replaced or initialized by the observer.

## Runtime API support and measurement effects

The report records the installed Torch/CUDA versions, allocator backend and
runtime signatures. `snapshot=True` calls the public `memory_snapshot` API when
available and summarizes segments and active/inactive block bytes for the
selected device. Unsupported calls produce explicit unavailable evidence. The
summary is a boundary snapshot and contains no raw addresses or stack paths.
History is not enabled through private `_record_memory_history`; availability of
that private API is recorded, but a stable public ownership/restoration contract
is not assumed. Neither a snapshot nor history would attribute external CUDA
allocations. See PyTorch's [memory observation documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory)
and [public snapshot API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_snapshot.html).

Observation synchronizes the selected CUDA device at sample boundaries and resets
process-wide allocator peak counters. It changes measurement timing and counter
state, so use a dedicated memory-observation run; these durations are not timing
calibration. Do not run competing peak observers or let other code reset the
counters inside an observed phase without observation. With explicit
`observe_peak_resets=True` (default false), the context temporarily wraps public
`torch.cuda.reset_peak_memory_stats` and its `torch.cuda.memory` alias. Immediately
before a native reset of the observed device, it reads only allocator peak
counters, without synchronization or snapshots, then calls the original reset
with unchanged arguments. These records preserve whole-window and per-phase
maxima across native metrics resets. Observer-owned resets are excluded. Hook
errors become gaps while the original native reset still executes; original
reset errors propagate. Hooks are restored on exit, including failure, and
another observer cannot take ownership while the hook is active. Previously
cached reset references, private/C++ resets and concurrent calls can bypass this
observation and remain explicit limitations. The observer does not empty the cache, change
allocator settings, change the recipe, or add distributed collectives. Its first
synchronization can initialize CUDA; this pre-baseline CUDA/runtime initialization
is outside the labeled trainer-initialization phase.

## Combining ranks and using bounds

`merge_observations` validates schema, world size, recipe/runtime provenance,
duplicate ranks, missing ranks and supplied placement. A subset of the world
remains incomplete even if every requested subset rank is present. Differing
recipe/runtime identities fail rather than being silently combined. The aggregate
can compare hash-verified raw recipe dumps differing only in anchored Python
function-repr addresses using the same narrow comparison as preparation. All
other provenance fields must agree exactly. This retains each rank's raw hashes
and a comparison record; callable identity remains unverified, so aggregate
status stays incomplete. Semantic configuration changes are rejected. The aggregate
retains per-device sampled global maxima and the maximum of process allocator
peaks. It does not sum collocated rank peaks, since peaks may occur at different
times. Rank-local times are not synchronized global timestamps.

A measured process reservation peak is a lower bound on device use in that exact
run; it is not a complete device upper bound. A successful run demonstrates only
that the measured execution completed under its actual conditions. The output
always retains `physical_peak_upper_bytes: null`,
`supports_unseen_configuration_fit: false`, and aggregate
`memory_verdict: inconclusive`. It cannot be pasted into `manifest.memory.coverage`
to close storage, library, workspace, graph-pool or lifecycle prediction gaps.
Use the measurements to audit a prediction and identify missing categories;
validate any proposed predictive bound separately, retaining exact recipe,
source/runtime identity, placement, steps and failed/missing rank artifacts.
