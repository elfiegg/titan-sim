---
name: simulate-torchtitan
description: Prescreen native TorchTitan training recipes after runtime preparation for per-device memory sufficiency and projected throughput. Use fake or native capture, explicit execution and memory models, and reusable GPU calibration; report unsupported configurations and uncertainty.
---

# Simulate a TorchTitan recipe

Act as the simulation subagent after `prepare-runtime`. Return a reproducible
configuration assessment, per-physical-device memory verdict, timing estimate and
coverage report. Native recipe modules and overrides are inputs; no particular
model, cluster, GPU count or validated fixture is a default.

## Consume the handoff

Read the caller's pinned titan-sim checkout and instructions. Consume
`runtime-manifest.json`, `dependency-plan.json`, `capability-report.json`, their
evidence, the native recipe and overrides, the target hardware/topology, and the
supplied execution context. Preserve image/source identities, runtime differences,
tested scope and exclusions. The caller supplies command entry, visible paths,
allocation budget, status/log/stop operations and evidence access. This skill
does not choose a cluster, rebuild the runtime or implicitly submit GPU jobs.

Run `titan-sim runtime-check runtime-manifest.json --expected expected.json
--output runtime-check.json`. The expected file can bind the source commit,
image, exact recipe, target fields and required capability IDs. Inspect the
reported scope: artifact consistency is not a probe of the running container.
Verify the actual interpreter/image/source through the supplied context before
GPU work. Keep the qualified reduced case and the requested full prediction case
separate. A checker result for the reduced case does not qualify the full case;
binding the full recipe can correctly report a scope mismatch. Preserve that
mismatch while continuing read-only or CPU analysis where possible, with explicit
gaps. Do not relabel the handoff or silently weaken expected identities. A reduced
recipe qualification does not establish full-model fit.
Changed features or missing probes go back to runtime preparation with the exact
dependency/capability requirement and reproducer; continue independent cases.
Do not disable the requested feature to make simulation succeed. Stage remote evidence through
the caller's mechanism if the handoff checker cannot read it.

## Resolve the simulation case

Record the native module/config/overrides, input geometry, unique tokens per
optimizer update, precision of parameters/compute/gradients/optimizer state,
optimizer, checkpointing, compile/graph mode, communication implementation,
parallel dimensions and physical rank placement. Derive overlapping mesh axes
from the native recipe; never multiply printed DP/EP axes blindly.

Use the actual per-device usable capacity and topology. Published Blackwell or
Rubin specifications supply hardware facts and theoretical limits, not measured
kernel costs. Rack memory totals cannot establish per-GPU fit. Keep dense/sparse,
training/inference and precision rates distinct. Missing rates remain unknown;
specification-only scenarios must be labeled analytical and include explicit
efficiency, latency and contention assumptions. An analytical run also requires
explicit per-operation work/traffic quantities or separately justified costs;
setting hardware rates cannot price opaque captured operations. Without matching
Rubin measurements, do not relabel another architecture's catalog or report
specification-derived throughput as calibrated.

## Capture, prepare, calibrate, simulate

Read the pinned checkout's [GPU-assisted workflow](../../docs/gpu-assisted.md)
for local discovery and component/window measurement, or the
[native guide](../../docs/native-torchtitan.md) for full profiler observation.
Keep original model and optimizer execution.

When custom-library execution is missing from the prediction, follow the
[custom-library agent workflow](../../docs/custom-library-workflow.md). Build a
case packet from the actual invocation and prepared runtime; inspect native code,
choose a faithful local/distributed boundary, implement and qualify its provider,
then bind timing, scheduling and memory evidence. Unknown custom operations are
integration tasks, not terminal labels. Opaque state may require a larger native
region; hidden communication requires actual group execution. Never set replay
safety, no-communication or memory-coverage flags solely to pass validation.
Delegate independent cases with explicit file ownership and one GPU scheduler;
retain dependent inspection/provider/qualification steps within the same case.

- Choose fake capture for logical tensor configuration exploration when supported;
  enumerate distinct rank roles. Fake tensor mode does not guarantee a GPU-free
  process: native construction, CUDA initialization, data loading and other side
  effects still need a compatible, authorized execution context. Fake wall time
  is never GPU time. Native capture
  executes real training and its configured side effects; use it only inside the
  authorized allocation. Investigate unsupported custom operations through the
  custom-library workflow and retain their simulation gaps until qualified.
- Capture using `titan_sim.torchtitan:capture`, fresh trace directories and
  `workload.rank{rank}.json` paths. Keep raw traces and source/runtime identities.
- When authorized real GPUs are available, `execution: "local_cuda"` observes
  real local compute with the original target rank/world and explicit fake
  transport substitutions. Enumerate distinct roles; retain unsupported calls.
  Instrumented discovery time and memory do not establish training performance
  or fit. Retained native handles need original argument/runtime context and a
  replay state contract; see [local execution](../../docs/local-execution.md).
- Run `titan-sim prepare` across rank artifacts. Review direct GPU correlations,
  compiled-region boundaries, stream dependencies, rendezvous and storage
  lifetimes. Supply explicit evidence for semantics the traces cannot establish.
  A gap resolution records a checksum-bound integration assertion, not proof.
  Missing ranks and failed captures cannot be asserted away. See
  [preparation](../../docs/native-preparation.md).
- For complete native peer captures, [schedule recovery](../../docs/schedule-recovery.md)
  derives supported collective completion and wait dependencies from common run
  identity and exact storage lineage. Fake waits and unobserved native Work waits
  remain gaps. Cost binding can request this pass with `recover_schedule: true`.
- Calibrate with the bundled `titan_sim.native_runner:measure`. Trace mode imports
  instrumented GPU observations; execute mode times original callable regions
  supplied by a trusted provider. Match implementation, shapes, scalars, precision,
  graph mode, runtime and communication context. Reuse catalogs only for matching
  keys and hardware. Use a coordinator for aggregate output. See
  [calibration](../../docs/native-calibration.md). The scheduler enforces the
  authorized hard limit; initial GPU checks use five minutes unless overridden.
- Measure concurrent behavior through [native windows](../../docs/execution-windows.md)
  when isolated costs are insufficient. Preserve native shared buffers, streams
  and waits. Controlled windows are experiments; training binding needs checked
  source ordering/boundary evidence. Do not derive a universal overlap multiplier.
- Use `bind-cases` or `bind-window` to connect measurement workloads to exact
  source occurrences before attaching memory evidence. Inspect skipped/unbound
  entries. Host collective issue is not device completion, and later compiled
  calls are not automatically the same specialization. Keep local workloads and
  catalogs; centralized calibration storage is optional.
- Simulate on CPU with the prepared workload and matching costs. Strict mode
  checks timing/workload completeness. Also require `--require-memory-fit` when
  choosing a recipe to run, and require `physical_memory.memory_scope="recipe"`
  for a full-recipe decision. The flag alone also accepts a schedule-only fit;
  timing calibration alone cannot establish capacity.

## Decide capacity and timing separately

Follow the [memory contract](../../docs/native-memory.md). Combine simultaneously
live backing allocations on each physical GPU, including colocated ranks.
Cover persistent states, gathered weights, activation stashes, recomputation,
optimizer initialization/update, workspaces, communication buffers, graph pools,
allocator slack, external allocations and lifecycle peaks. Count aliases once;
allocator reserved memory already contains allocated memory. State whether the
assessment covers only the scheduled interval or the complete recipe lifecycle.
Use [memory evidence binding](../../docs/memory-evidence.md) for target-applicable
allocation generations, completion events and scoped coverage. Bind against the
final workload after region/window contraction. Native state/allocator samples
remain observations; conditional forecasts cannot pass unconditional fit.
Native allocator history can derive recorded generations and completed releases.
Use `allocation-evidence` for conditional envelopes only with matching source
applicability; coarse host intervals do not prove unavoidable live overlap.

Publish lower/upper peaks, capacity/budget, margin and limiting device. A credible
lower bound above a device budget means `estimated_not_fit`; a covered upper bound
at or below every device budget means `estimated_fit`; missing coverage or straddling bounds
means `inconclusive`. Distinguish exceeding a headroom-reduced budget from
exceeding the device's physical capacity. Producer-attested bounds remain conditional on their evidence
and assumptions. Never promote a logical tensor peak to physical device usage.

For timing, report the workload scope and useful-token normalization, measured
key coverage, unmodeled work, scheduling/contention assumptions and uncertainty.
Keep empirical sample spread distinct from scenario ranges and statistical
confidence. Unknown operation costs are blockers, not zero-cost operations.
If timing evidence is absent, do not invent durations merely to obtain a memory
verdict. Persistent-state and lifecycle lower bounds can still inform screening;
any schedule-dependent bounds need an explicit, justified scheduling envelope.
If required evidence is absent, return an actionable inconclusive assessment with
the exact missing operation, dependency, allocation or calibration case.

## Validate and return

Freeze workload, costs, assumptions and predictions before a held-out run. Compare
completed step time and all-device memory over the same lifecycle; separate
intrusive profiling from throughput measurement. A replay of the calibration
trace is not independent validation. Use multiple recipe families and layouts
over time; one validated model does not imply universal TorchTitan support.
Use [prediction validation](../../docs/prediction-validation.md) to freeze an exact
producer-recorded context before a trusted fresh control. Full-update comparison
requires complete timing coverage and target ranks; a component comparison cannot
qualify full-model accuracy. Keep wall/device timing and memory lifecycle scopes
consistent between prediction and control.

Return artifact locations/hashes, pinned identities, recipe/placement, capacity
and timing assessments, assumptions, unsupported features, evidence level,
validation errors and next required evidence. Preserve raw evidence and job IDs
in the caller's output directory. Keep case results out of this skill and the
generic implementation; validation fixtures belong in separate test evidence.
