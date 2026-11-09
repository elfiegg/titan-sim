---
name: simulate-torchtitan
description: Prescreen native TorchTitan training recipes after runtime preparation for per-device memory sufficiency and projected throughput. Use native execution evidence, real GPU calibration and explicit memory models; report unsupported configurations and uncertainty.
---

# Simulate a TorchTitan recipe

Act as the simulation subagent after `prepare-runtime`. Assess the caller's native
recipe on the target hardware without assuming that the complete target cluster
is available. Return projected timing and per-physical-device memory bounds with
evidence and unresolved scope. No model, GPU architecture or validated fixture is
a default. Keep experiment results outside this skill and the generic implementation.

Read the pinned checkout's [system design](../../docs/architecture.md), then follow
the [GPU-assisted workflow](../../docs/gpu-assisted.md). That workflow defines the
ordered artifact handoffs; use its linked references for schemas and commands.
This skill defines the agent's decisions and completion criteria.

## 1. Accept the runtime and define the case

Consume the prepared runtime manifest, dependency plan, capability report and
underlying evidence, native recipe/overrides, hardware/topology and execution
context. Read repository instructions and record the simulator revision. The
caller provides command entry, visible paths, resource authorization/budget,
scheduler limits and status/log/stop operations.

Run `runtime-check` with the intended identity expectations. Verify the actual
runtime through the supplied execution context before GPU work: artifact
consistency does not prove a running container's capabilities. Preserve differences
between the qualified runtime case and the requested prediction case. A reduced
qualification does not establish full-model support or fit. Do not weaken expected
identities or relabel a scope mismatch to pass the check.

Resolve input geometry, useful tokens per update, all precision settings,
optimizer, checkpointing, compilation/graphs, communication implementation,
parallel groups and physical placement. Derive overlapping mesh dimensions from
native construction. Define timing scope and whether memory must cover the full
recipe lifecycle. Use per-device usable capacity and headroom, not aggregate
cluster memory. Hardware specifications supply limits, not measured performance.

## 2. Acquire native execution evidence

Choose native host plans, original profiler/dispatch capture, local CUDA discovery
or supported fake capture according to the runtime and available resources. Keep
the submitted trainer and settings. Capture can execute original training side
effects; use the supplied allocation, fresh trace directories and distinct rank
output paths. Fake mode may still require CUDA during native construction.

Enumerate distinct rank roles. Local CUDA discovery retains target logical group
membership with explicit transport substitutions; its selected rank state must
fit on the available GPU. Fake transport and instrumented wall time do not prove
native peer completion, numerical training or target throughput. Native handles
and retained arguments must remain in their owning process and may change memory
lifetimes.

Prepare the source workload from exact snapshots. Resolve supported dependencies,
rendezvous, regions and storage using native evidence; use schedule recovery only
where its peer/run-identity and wait-lineage contracts apply. Record unresolved
semantics. Missing ranks and failed captures cannot be asserted away.

## 3. Integrate missing cases and measure them

Inspect the case inventory and follow the
[custom-library workflow](../../docs/custom-library-workflow.md) for unsupported
native implementations. Build a case packet from available runtime/source evidence,
inspect the selected path, choose a faithful boundary and qualify a provider.
Opaque state can require a larger lifecycle region; hidden communication requires
its real group. Invoke native code rather than copying its algorithm into the
simulator. Never set replay-safety, no-communication or coverage flags merely to
pass validation.

Delegate independent cases with explicit ownership and one GPU scheduling owner.
Keep inspection, provider construction, qualification and binding connected within
each case. Return exact missing dependencies/capabilities and a reproducer to
runtime preparation while continuing independent work; do not switch libraries
or disable requested features to bypass a blocker.

Measure local compute on one target GPU and communication on the identified real
group/topology. Preserve build, shape/layout/scalars, precision, state/reset,
compiler/graph and routing context. Qualify outputs/gradients and completion before
claiming faithful replay. Measure native concurrent windows where separate costs
cannot establish overlap or interference. Controlled windows remain experiments
unless source-backed ordering and boundaries are established.

The caller's scheduler enforces the execution budget; initial GPU checks use a
five-minute hard limit unless overridden. Use one coordinator for aggregate output.
Keep local measurement workloads, samples and catalogs. Centralized storage is
optional. Distinguish execute-mode timing from imported profiler observations.
Analytical scenarios require justified work/traffic quantities and explicit rates,
latency and contention assumptions. Never treat CPU/fake time or another GPU's
catalog as target calibration.

## 4. Bind timing and establish memory coverage

Bind measured cases or closed windows to exact source occurrences before attaching
memory evidence. Inspect skipped/unbound entries. Host issue is not device
completion; a later compiled call is not automatically the same specialization.
Avoid double charging contracted regions and preserve external dependencies.

Follow the [memory contract](../../docs/native-memory.md) and
[evidence binding](../../docs/memory-evidence.md) for the final workload. Account
for concurrent backing allocations on every physical GPU, including colocated
ranks, persistent states, activations, gathered weights, workspaces, communication
buffers, graph pools, allocator slack and external allocations. Count aliases
once; allocated bytes are contained in reserved bytes. Full-recipe coverage also
needs initialization and loading peaks.

Separate empirical observations, conditional replay envelopes and supported
predictive bounds. A free request is not necessarily a completed release, virtual
reservation is not physical mapping, and coarse host intervals do not prove
unavoidable simultaneous usage. Incomplete coverage must remain inconclusive.
If timing is missing, do not invent durations to get a memory verdict; use only
independently justified persistent/lifecycle bounds or explicit schedule envelopes.

## 5. Assess, validate and return

Simulate with exact matching costs. Report timing and memory independently:
`--strict` checks declared timing coverage; `--require-memory-fit` requires an
unconditional fit. For a full-recipe decision, additionally require
`physical_memory.memory_scope="recipe"`. A schedule-only fit is insufficient.

Report lower/upper peaks, device budgets/capacities, margins and the limiting
device. Supported upper bounds fitting every budget establish `estimated_fit`;
a credible lower bound exceeding a budget establishes `estimated_not_fit`;
missing or straddling bounds leave `inconclusive`. Distinguish budget overflow
from exceeding physical capacity. State timing scope, token normalization,
cost coverage, contention assumptions and unmodeled work. Sample spread, scenario
ranges and statistical confidence are different quantities.

When independent execution is available, [freeze the prediction](../../docs/prediction-validation.md)
before invoking a fresh native control. Match runtime/recipe/placement, timing
metric and memory lifecycle scope. Component validation cannot qualify full-model
accuracy. Keep profiling separate from throughput controls and fitted costs
separate from held-out validation.

Return pinned identities, recipe/placement, artifact paths/hashes, timing and
capacity assessments, assumptions, unsupported features and validation scope.
For each remaining gap, name the required operation, dependency, allocation or
measurement and its next action. Preserve private raw evidence and job identifiers
in the caller's output directory. Prescreening does not establish successful
execution or measured performance at scale.
