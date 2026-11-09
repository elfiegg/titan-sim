# Run a GPU-assisted recipe prescreen

Use this guide to assess a submitted TorchTitan recipe after `prepare-runtime`.
The result is a projected step time, useful-token throughput, per-device memory
verdict and a list of unsupported or uncertain behavior. The
[system design](architecture.md) explains the components; this guide follows the
work from the runtime handoff to the assessment.

The workflow uses real GPUs for native compute and communication measurements,
then CPU simulation for the target configuration. You need enough GPU capacity
for each selected measurement, though not necessarily the complete training
cluster. The tools do not allocate GPUs or automatically supply every provider,
distributed dependency or physical-memory contract.

## Before you start: establish the prediction case

Collect these inputs from the caller and prepared runtime:

| Input | What it establishes |
|---|---|
| Runtime manifest, dependency plan, capability report and evidence | Pinned image, source, framework/extensions and what was actually qualified |
| Native recipe and resolved overrides | Input geometry, optimizer, parameter/compute/gradient/state precision, checkpointing, compile/graph settings and communication implementation |
| Target parallelism and physical placement | Rank roles, group membership, device capacities and topology |
| Prediction scope | Scheduled interval or complete update, unique tokens per update, and schedule-only or full-recipe memory |
| Execution context | Command entry, accessible paths, GPU authorization/budget, scheduler limit and status/log/stop operations |

Check the staged handoff against the intended identities:

```sh
python -m titan_sim runtime-check runtime-manifest.json \
  --expected expected-runtime.json --output results/runtime-check.json
```

The [handoff reference](native-torchtitan.md#before-you-start) explains its files.
This checks artifact consistency; verify the actual running image/interpreter/source
through the supplied execution context before GPU work. Qualification of a reduced
case does not qualify a larger prediction case. Preserve any scope mismatch and
continue independent analysis; send exact missing capabilities to runtime preparation.

Use the original recipe throughout. In particular, do not disable compilation,
change precision or substitute a trainer to make capture succeed. Derive overlapping
parallel axes from native mesh construction rather than multiplying their labels.
Hardware specifications supply capacity and theoretical limits; calibrated timing
requires applicable GPU measurements. See [hardware inputs](hardware.md).

All filenames below are artifact templates. The native recipe, provider and
measurement configurations must be supplied using the linked references. The
[CPU example](../README.md#try-the-cpu-simulator) is the self-contained runnable demo.

## 1. Obtain the target execution structure

Choose evidence according to what the unchanged runtime exposes:

| Available evidence | Use | Output and next step |
|---|---|---|
| Native host schedules or stage graphs | [Export and lower plans](native-planning.md) | Schedule-derived workload with remaining gaps; obtain native implementation evidence for calibration |
| A target rank can execute locally with supported communication substitutions | [Local CUDA discovery](local-execution.md) | Rank-local source workload and native case inventory |
| Native execution is available on the required ranks | [Profiler or dispatch capture](native-torchtitan.md) | Per-rank workloads and raw traces for preparation |
| Only fake execution is feasible | [Fake dispatch capture](native-torchtitan.md#fake-and-eager-dispatch-capture) | Logical configuration evidence; GPU costs and physical-memory evidence still needed |

For local CUDA discovery, configure `execution: "local_cuda"`, `observer:
"dispatch"`, `target_rank`, `target_world_size`, original `recipe_args` and target
`hardware`, then launch in the authorized single-GPU environment:

```sh
python -m titan_sim capture --adapter titan_sim.torchtitan:capture \
  --config local-capture.json --output results/source.rank0.json
```

This runs actual local compute with fake target process groups and explicit
representative communication outputs. Repeat for distinct pipeline stages,
uneven shards and expert roles as needed. The selected rank's original trainer
and state must fit: discovery does not automatically stream an oversized rank.
Retained native callable handles must be measured in their owning process.

Preserve source snapshots, runtime identities and raw evidence. Instrumented
capture time and memory describe discovery, not the target training result.
Fake transport cannot demonstrate peer completion or numerical distributed training.

## 2. Prepare dependencies and select measurement cases

Use [workload preparation](native-preparation.md) to combine rank artifacts and
resolve supported correlations, regions, storage and ordering. For complete native
peer dispatch captures with a common run identity,
[schedule recovery](schedule-recovery.md) can derive supported functional collective
completion and wait dependencies:

```sh
python -m titan_sim recover-schedule results/native.rank*.json \
  --output results/recovered-workload.json
```

A local dispatch sequence is insufficient to infer every distributed wait or CUDA
stream dependency. Preserve unknown behavior as gaps and supply explicit evidence
where the bundled preparation cannot derive it. Missing ranks and failed captures
cannot be resolved by assertions.

The commands below follow the dispatch-case route; plan and profiler integrations
use their linked preparation and provider interfaces. Select cases from the exact
source snapshots you intend to bind later. Supported
[compute cases](operator-cases.md) are isolated native dense matmuls:

```sh
python -m titan_sim calibration-cases results/source.rank0.json \
  --kind compute --report results/compute-inventory.json \
  --output results/compute-workload.json
```

The output is a **measurement workload**, not a cost catalog. Inspect its skipped
inventory. For compiled or custom implementations, use retained native callables
or a qualified provider through the [custom-library workflow](custom-library-workflow.md).
Choose an enclosing native region when opaque state or fused behavior makes a
smaller case unsafe to replay. This integration step supplies the missing provider;
case selection does not invent one.

## 3. Calibrate on real GPUs

This is where durations are measured. Run `calibrate` inside the prepared runtime
on allocated target GPUs. Its execute-mode runner performs warmup and repeated
GPU timing of the provider's native implementation, then writes samples and
context into a **cost catalog**.

| Case | GPU execution | Configuration reference |
|---|---|---|
| Supported local operator | One target GPU, exact overload and metadata, representative finite inputs | [Operator measurement](operator-cases.md) |
| Compiled/custom local region | One target GPU, original callable and verified reset/context contract | [Live regions](local-execution.md), [native provider](native-calibration.md#execute-an-exact-native-region) |
| Communication | Identified real group and topology, actual buffers/protocol/implementation | [Communication cases](communication-cases.md) |
| Concurrent window | Actual native callables, shared state, streams, waits and completion joins | [Window measurement](execution-windows.md) |

For the selected compute workload, create `compute-measurement.json` using the
operator reference's execute-mode provider configuration and expected GPU identity:

```sh
python -m titan_sim calibrate results/compute-workload.json \
  --config compute-measurement.json --output results/compute-costs.json
```

The command uses the bundled native runner by default. The caller's launcher
provides GPUs and enforces the scheduler limit; use five minutes for initial
benchmarks unless another limit is authorized. One coordinator writes aggregate
output. Trace-import mode is a separate option that imports matched profiler GPU
observations and labels their instrumented origin.

Communication measurement must preserve group size, rank placement, dtype,
reduction, layout and relevant traffic distribution. A generic all-to-all does
not qualify a custom dispatch/combine implementation. If overlap matters, measure
a short native window: separate compute and communication timings cannot establish
interference. Controlled windows are experiments; training binding requires
source-backed ordering and boundary evidence. Automatic window reconstruction,
distributed CUDA graphs and subgroup-only window execution remain unsupported.

Keep measurement workloads, inventories and sample catalogs with the run. No
central storage service is required. Reuse a cost only when implementation and
execution context match; do not extrapolate a small communication group or a
GPU architecture silently.

## 4. Bind measurements to the original workload

The source workload identifies training occurrences. The measurement workload
identifies what was benchmarked. The catalog supplies the resulting durations.
`bind-cases` connects the first two so the simulator can look up the third:

```sh
python -m titan_sim bind-cases results/source.rank0.json \
  --measurement results/compute-workload.json \
  --report results/binding-audit.json --output results/bound-workload.json
```

Supply all required rank sources and repeat `--measurement` for additional cases.
Inspect bound, skipped and unbound occurrences. The binder checks full source
fingerprints and native descriptors; kernel-name matching alone is insufficient.
Communication costs require reconciled device-completion endpoints and measured
placement. A binding configuration can request supported native schedule recovery
with `recover_schedule: true` before assigning costs.

Use `bind-window` for closed source-backed windows. It replaces their operations
with one atomic measured event, avoiding double charging. Controlled windows
cannot replace training costs. Check external consumers and resource assumptions;
contraction does not prove internal physical-memory bounds.

Each binding produces a new workload snapshot. Complete timing bindings before
attaching memory evidence, so applicability refers to the final structure.

## 5. Establish physical-memory bounds

Follow [memory evidence production and binding](memory-evidence.md) to attach
allocation sizes, aliases, ownership, creation/release events and lifecycle bounds
to the final workload. Native state inspection, [memory observation](memory-observation.md)
and allocator histories can support this work. `allocation-evidence` derives
recorded generations and conditional replay envelopes from a native allocation
capture; it does not automatically establish predictive upper bounds.

```sh
python -m titan_sim bind-memory results/bound-workload.json \
  --bundle memory-evidence.json --context target-applicability.json \
  --output results/prediction-workload.json
```

Account for simultaneously live allocations on each physical GPU, including
colocated ranks, workspaces, communication buffers, graph pools, external memory
and allocator slack. Count aliases once; reserved bytes already contain allocated
bytes. Separate region peaks cannot simply be summed as a training peak.

For full-recipe fit, cover initialization and loading as well as training.
Observed maxima and conditional forecasts remain distinct from unconditional
bounds. Missing physical coverage leaves capacity inconclusive even when every
operation has a measured cost. See the [memory contract](native-memory.md).

## 6. Simulate and report the assessment

Supply the final workload and matching catalog. When combining catalogs from
multiple measurement runs, reject conflicting records for the same key and retain
original sample artifacts.

```sh
python -m titan_sim simulate results/prediction-workload.json \
  --calibration results/costs.json --strict --require-memory-fit \
  --output results/prediction.json
```

`results/costs.json` is that combined catalog, or a single applicable catalog when
one run covered the cases. `--strict` rejects unresolved timing/coverage gaps;
`--require-memory-fit` independently requires an unconditional physical fit.
For a full-recipe decision, also check `physical_memory.memory_scope="recipe"`.
For investigation, omit those flags and report the retained limitations explicitly.

Return timing scope and unique-token normalization, per-device bounds and budgets,
limiting devices, measurement coverage, assumptions, unresolved operations and
artifact identities. Explain which next observation or integration would resolve
an inconclusive result. Use [result interpretation](../README.md#read-the-result).

## 7. Validate accuracy when independent execution is available

[Freeze the prediction](prediction-validation.md) before a separate trusted control
run. Compare matching recipe/runtime/placement, completed-step timing and memory
lifecycle scope. Keep intrusive profiling separate from throughput measurement.
Full-update validation requires complete timing coverage and ranks; a local region
check cannot establish full-model prediction accuracy.

Validation fixtures test shared runtime behavior. Their results belong with run
evidence, while the generic skill and simulator remain independent of model names.
