# GPU-assisted recipe prescreening

Use this workflow after `prepare-runtime` to investigate a submitted native
TorchTitan recipe without allocating its complete training cluster. The target
recipe still determines sharding, pipeline roles, microbatches, precision,
checkpointing, compilation and communication. Real GPUs provide local compute
and communication measurements; the CPU simulator composes those costs using
explicit execution dependencies and allocation lifetimes.

The implementation provides the stages below. It is not yet an automatic
recipe-to-accurate-prediction pipeline: native window providers, some ordering
interpretation and complete allocation contracts still require integration.
Unsupported behavior is recorded rather than replaced with another trainer or
configuration. No central database or remote calibration API is required.

```text
Prepared runtime + unchanged recipe + target placement
                         |
           Native host plan / local CUDA discovery
                         |
          Source operations, boundaries, storage, gaps
                         |
             Select and retain native cases
                 /                   \
       One-GPU compute          Actual communication group
                 \                   /
                   Native window replay
                         |
          Exact source/cost binding + memory evidence
                         |
               CPU target-timeline simulation
                         |
          Throughput, per-device bounds, unresolved scope
```

## 1. Discover the target rank's local execution

The [local CUDA adapter](local-execution.md) constructs the original native
trainer with fake process groups carrying the target logical membership.
Supported communication calls produce explicit representative local inputs;
compute executes on a real GPU. This is a local execution experiment, not
numerically correct distributed training. Captured wall time includes discovery
instrumentation and is never used as predicted training time.

Set `execution: "local_cuda"`, `target_rank`, `target_world_size`, original
`recipe_args` and target `hardware` in the capture configuration. Run:

```sh
python -m titan_sim capture --adapter titan_sim.torchtitan:capture \
  --config local-capture.json --output results/source.rank0.json
```

Repeat for required distinct target roles. One observed rank does not establish
the behavior of other pipeline stages, uneven shards or expert assignments.
Sequential reuse of one GPU is possible only where the selected local state or
region fits. Whole-rank discovery currently constructs that rank's original
trainer and state; it does not automatically stream an oversized rank in pieces.

Use [native plan export](native-planning.md) when the runtime already exposes
schedule tables or graphs. Host schedules establish ordering constraints, not
kernel durations or contention. Local dispatch order alone cannot recover all
cross-rank completion dependencies, data-dependent routing or compiler overlap.

## 2. Measure identified native implementations

There are two local compute paths:

- [Dispatch cases](operator-cases.md) reconstruct supported native matrix
  operations from exact shape, stride, dtype and scalar contracts using finite
  synthetic inputs. They are isolated operator measurements.
- [Live native regions](local-execution.md#native-compiledcustom-callables)
  retain the actual callable returned by the original runtime and its observed
  arguments. Replay requires an explicit read-only or reset contract, matching
  execution context, and the same process. Opaque compiled boundaries do not
  automatically reveal their internal allocations or communication.

For dispatch cases, select the measurement workload, then time it on one GPU:

```sh
python -m titan_sim calibration-cases results/source.rank0.json \
  --kind compute --report results/compute-inventory.json \
  --output results/compute-workload.json
python -m titan_sim calibrate results/compute-workload.json \
  --config compute-measurement.json --output results/compute-costs.json
```

`compute-measurement.json` selects execute mode and the documented native provider;
it also verifies the expected target GPU. These commands do not allocate GPUs.
Use the authorized launcher and an initial five-minute scheduler limit.

[Communication cases](communication-cases.md) retain the identified group size,
native overload, buffers, reduction, peer roles and measured topology. Select
them from the necessary peer captures and execute on that many GPUs. Local
discovery records a **fake execution backend** and requested target backend;
only the later native communication benchmark produces transport timings.
HybridEP requires its actual dispatch/combine implementation and remains
unsupported by the bundled generic communication case provider.

## 3. Measure a short concurrent window when needed

An isolated compute duration plus an isolated communication duration does not
establish overlap. [Native windows](execution-windows.md) run selected actual
callables with declared shared state, streams, waits and completion joins.
A trusted native provider supplies these callables and their buffers. The runner
coordinates preparation failures and records per-rank timing samples.

Use `window-cases` to build the measurement workload and `calibrate` with
`titan_sim.execution_windows:region_provider` to measure it in an existing group.
Compare separate isolated, serial and concurrent manifests when investigating
interference. Their costs remain specific to those windows and contexts.

A controlled window tests caller-selected ordering and inputs. A source-backed
window additionally requires checked source artifacts and an explicit producer
assertion about actual stream dependencies and boundary semantics. Automatic
trace-to-window reconstruction is not implemented. Distributed CUDA graphs and
subgroup-only execution inside a larger measurement world remain unsupported.

## 4. Bind measurements to the original workload

A measurement workload describes what was benchmarked; its **cost catalog**
contains the resulting seconds and samples. `bind-cases` consumes the former,
checks its complete source fingerprint and native descriptors, and assigns its
exact cost keys to verified source occurrences:

```sh
python -m titan_sim bind-cases results/source.rank0.json \
  --measurement results/compute-workload.json \
  --report results/binding-audit.json --output results/bound-workload.json
```

Supply all relevant rank sources and repeat `--measurement` as needed. The audit
lists bound, skipped and unbound occurrences. It preserves source gaps and
rejects stale snapshots or duplicate ownership. A collective's blocking cost
requires a reconciled **device-completion endpoint**, not just its host issue
call. Topology must be explicitly bound to the measured placement.

Observed local callable boundaries can be contracted into one measured event
only when their membership and first captured argument specialization match.
Externally visible intermediate dependencies are rejected. Later invocations
are not automatically assumed equivalent.

`bind-window source.json --measurement window-workload.json --output bound.json`
accepts closed source-backed windows. It charges the whole window once as one
atomic event using all participant resources. Controlled windows cannot replace
training costs. This conservative representation loses internal scheduling
freedom and retains unresolved external contention and readiness assumptions.
After any contraction, projected physical storage is unverified until applicable
memory contracts establish its bounds.

Bind measurement cases/windows before memory evidence. Each binding derives a
new workload; downstream artifacts must use that exact snapshot. This identity
discipline prevents a measurement from silently attaching to a changed recipe.

## 5. Establish per-device memory coverage

[Memory evidence](memory-evidence.md) attaches allocation sizes, aliases,
generations, creation and completion events, placement and lifecycle bounds to
the final prediction workload. Native optimizer/state inspection and allocator
observations provide useful checks, but an observed maximum is not a proven
upper bound. Single-region peaks must not be summed as a training peak.

```sh
python -m titan_sim bind-memory results/bound-workload.json \
  --bundle memory-evidence.json --context target-applicability.json \
  --output results/prediction-workload.json
```

Coverage is per device, category and phase, including initialization and loading
for recipe-level fit. Missing workspaces, communication buffers, graph pools or
allocator coverage leave upper bounds unknown. Conditional forecasts are reported
separately and cannot satisfy an unconditional memory-fit requirement. Budget
overflow is distinguished from a lower bound exceeding physical GPU capacity.

## 6. Simulate, then independently validate

Pass the exact-key catalog to the CPU simulator. When combining catalogs, reject
conflicting entries for the same key; keep the original sample artifacts.

```sh
python -m titan_sim simulate results/prediction-workload.json \
  --calibration results/costs.json --strict --require-memory-fit \
  --output results/prediction.json
```

Strict mode intentionally rejects remaining execution gaps. Exploratory output
is available without `--strict`, with its unresolved limitations retained.
`--require-memory-fit` separately rejects inconclusive capacity. Useful-token
counts must count unique training tokens, not copies on model-parallel ranks.

Freeze prediction inputs before a held-out native training control. Compare
step time, per-device memory and fit decisions. Component replay, a successful
local discovery step or a controlled overlap window is narrower evidence than
full-model prediction accuracy. Validation models exercise shared runtime
features; they do not select model-specific simulator implementations.
