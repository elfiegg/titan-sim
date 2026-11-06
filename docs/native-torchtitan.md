# Native TorchTitan capture

`titan_sim.torchtitan:capture` loads the **original recipe**, builds its native
trainer, and observes selected calls to its original `train_step`. Model choice,
inputs, loss, optimizer, precision, activation checkpointing and parallelization
remain owned by TorchTitan. There is no Llama/Kimi-specific forward in this adapter.

This is an experimental capture implementation, not a claim that every TorchTitan
model/backend or every historical TorchTitan version has been validated. It also
does not make arbitrary recipes executable using fake tensors on a CPU.

## Where calibration happens

Native capture is the evidence-collection step. On CUDA it records real GPU
execution, but `ProfilerCapture` does **not** turn profiler events into a measured
cost catalog. GPU calibration happens later, inside the measurement runner passed
to `titan-sim calibrate --runner module:function`.

```text
Original TorchTitan recipe
  -> capture: execution/Chrome traces + rank-local workload
  -> match original executable regions; reconcile schedule and storage
  -> calibrate: measurement runner executes those regions on target GPUs
  -> measured cost catalog keyed by workload cost_key
  -> simulate: CPU timing/storage model using the catalog
  -> validate predictions against a separate, held-out training run
```

The generic calibration command and catalog checks are implemented. Automatic
native trace-to-kernel matching and a general native TorchTitan measurement runner
are **not** bundled. The example below captures evidence; it does not complete
this entire pipeline. See [Calibration and prediction](#calibration-and-prediction)
for the subsequent inputs, commands and remaining integration work.

## Inputs and invocation

Install titan-sim into the **existing pinned training environment**, without
replacing that environment's PyTorch. Make the original TorchTitan checkout,
recipe module, tokenizer, data and native extensions available exactly as for
training. The working directory and launch environment remain significant.

For example, create `capture.json`:

```json
{
  "execution": "native",
  "observer": "profiler",
  "recipe_args": ["--module", "my_recipes", "--config", "my_recipe"],
  "steps": 1,
  "warmup_steps": 1,
  "trace_dir": "results/capture-001/native-traces",
  "hardware": {"id": "capture-only", "capacity_bytes": 0},
  "runtime_provenance": {
    "torchtitan_revision": "record-your-pinned-revision",
    "container_digest": "record-your-training-container"
  }
}
```

`my_recipes` is the importable Python module containing your native TorchTitan
recipe, and `my_recipe` is its configuration function. Replace these two names in
`recipe_args` with the values you normally pass to TorchTitan's `--module` and
`--config`. The separate `--adapter titan_sim.torchtitan:capture` argument selects
titan-sim's capture implementation. The zero-capacity `capture-only` profile deliberately supplies no target
hardware assumption. Before calibration, put the actual target hardware description
in the workload: `calibrate` checks returned records against `workload.hardware.id`
and has no `--hardware` option. `simulate --hardware FILE` can override hardware
later, but an existing catalog must match that identity. Costs and graph semantics
still need completion; a hardware profile alone does not make this trace simulatable.

```sh
titan-sim capture --adapter titan_sim.torchtitan:capture \
  --config capture.json --output 'results/capture-001/workload.rank{rank}.json'
```

For distributed training use your ordinary authorized launcher/allocation. For
example, substitute the intended per-node process count in:

```sh
torchrun --nproc-per-node=NUM_LOCAL_GPUS -m titan_sim capture \
  --adapter titan_sim.torchtitan:capture --config capture.json \
  --output 'results/capture-001/workload.rank{rank}.json'
```

Each rank writes a separate workload and `native-traces/rankN/` directory. The CLI
requires `{rank}` when `WORLD_SIZE > 1`; the resolved logical rank names the file.
Use a fresh trace directory for each run. Existing rank directories are rejected
to avoid overwriting evidence. Multi-node launches must make the resulting rank
artifacts available for later reconciliation.

Native capture **really executes training**, including warmup. The adapter bounds
completed steps, not wall time, and does not suppress the recipe's checkpointing,
validation or data-loading side effects. Use an explicit scheduler time limit;
the initial GPU test policy is five minutes. The adapter does not request resources.
Avoid enabling a second native profiler in the same recipe: nested profiler
sessions are not supported. Adjust such a recipe explicitly and record the change.

## Outputs and implementation

| Component | Input | Output |
|---|---|---|
| `torchtitan.capture` | Native argument list, execution/observer choice, target metadata | Loads `ConfigLoader`, builds and closes the native trainer; returns `Workload` |
| `torchtitan.capture_trainer` | Already-built native trainer | Bounded original training loop, captured steps, restored instance methods |
| `profiler_capture.ProfilerCapture` | Native execution on the current rank | PyTorch execution-trace JSON and Chrome trace per selected step; capture evidence, not a calibrated cost catalog |
| `profiler_capture.project_trace` | Native execution trace | Conservative leaf-operation/storage projection into simulator records |
| `torch_capture.TorchCapture` | Explicit fake/eager execution | Dispatch operations, aliases, weak storage lifetimes and state metadata |
| User-supplied measurement runner | Prepared `Workload`, runner config, original executable implementations | Performs GPU warmup/timing and returns measured cost records |
| `adapters.calibrate` / `adapters.validate_catalog` | Runner reference, workload and config / returned catalog | Invokes the runner; checks hardware identity, finite durations, samples and checksums |
| `core.CostModel` / `core.simulate` | Workload, matching catalog, hardware and resource assumptions | CPU prediction using exact-key measured costs; strict mode rejects remaining gaps or uncalibrated costs |

The native profiler path does **not** install a Python dispatch mode, replace
operators, disable compilation, or rewrite a model. It uses PyTorch's
`ExecutionTraceObserver` with CPU/CUDA profiling activities. CUDA kernels and
runtime events are preserved in the Chrome artifact when CUDA profiling is
available. Matching these back to workload regions is still a separate task.

Workload output includes source/config fingerprints, software/runtime provenance,
rank/world size, operator signatures, parent execution-trace IDs, native input and
output descriptors, parameter placement, and explicit capture gaps. Raw trace
paths and checksums are recorded in the manifest. Retain those files alongside
the workload; the JSON projection does not replace them.

The execution-trace schema currently accepted is `1.1.1-chakra.0.0.4`, produced by
the tested PyTorch build. This is PyTorch's native JSON format; no Chakra protobuf
installation, conversion back to PyTorch, or Chakra simulator is required. Other
schemas fail explicitly until their importer is validated.

`capture_status=complete` means the requested training steps returned and operator
events were recorded. It does **not** mean all performance/storage semantics were
recovered. `blocked` and `incomplete` captures are written for diagnosis and make
the CLI return exit code 2. Structural validation still accepts a well-formed
partial workload; strict simulation rejects its declared gaps.

For an already-built custom trainer:

```python
from titan_sim.torchtitan import capture_trainer
from titan_sim.workload import write_workload

workload = capture_trainer(
    trainer, hardware=hardware, trace_dir="results/unique-run/traces",
    warmup_steps=1, steps=1,
)
write_workload("results/unique-run/workload.json", workload)
# The caller owns trainer and process-group cleanup in this API.
```

The supported trainer contract is `train()`, `train_step(...)`,
`should_continue_training()`, and `engine` with `device`/`model_parts`. The loader
path also needs `config.build()` and `trainer.close()`. Subclasses using this
contract retain their own step implementation. RL applications or alternative
entrypoints without the contract need an integration at their actual step boundary;
they are not silently converted to a standard text trainer.

## Calibration and prediction

### Prepare a workload that measurements can represent

Retain the raw capture. Use its Chrome correlations and execution-trace hierarchy
to identify the **original executable operator or compiled region** represented by
each cost key. One leaf trace operation is not necessarily one GPU kernel. Do not
assign a fused region's full duration to every leaf; represent its cost once and
preserve the associated dependencies. Kernel names alone are not sufficient
matching evidence.

For a distributed prediction, reconcile every rank's streams, waits, PP transfers
and collective participants into a workload with explicit dependencies and
resources. Rank 0 alone is not the distributed workload. This reconciliation and
compiled-region matching are integration work; there is no bundled command that
automatically produces the prepared workload from native traces.

In the templates below, `results/prepared/workload.json` means that separately
prepared artifact. Keep each unresolved limitation in its manifest. Never clear
`unresolved` merely to make strict simulation run. Calibration can collect a
partial catalog before reconciliation is complete, but it does not resolve those
gaps.

Set the target hardware before invoking calibration. For example, with a real
hardware description you supply in `target-hardware.json`:

```python
import json
from titan_sim.workload import read_workload, write_workload

workload = read_workload("results/prepared/workload.json")
workload.hardware = json.loads(open("target-hardware.json").read())
write_workload("results/prepared/workload.target.json", workload)
```

This changes target metadata only; it performs no graph reconciliation or
measurement. The hardware description must include a meaningful `id`, actual
per-device `capacity_bytes`, and the rate/resource assumptions used by the model.
The measurement runner must verify that its real devices and topology match the
intended target, rather than simply copying the requested ID.

### Measure costs on the target GPUs

This command is the calibration step:

```sh
titan-sim calibrate results/prepared/workload.target.json \
  --runner my_project.measure:measure_workload \
  --config measurement.json --output results/calibration-001/costs.json
```

`my_project.measure:measure_workload` is a **runner you provide**, not a bundled
native TorchTitan calibrator. Its Python contract is
`measure_workload(workload, config) -> dict`. `measurement.json` follows that
runner's own schema; the generic CLI supplies no universal warmup or timing
configuration.

The implementation path is
[`cli.main`](../titan_sim/cli.py) ->
[`adapters.calibrate`](../titan_sim/adapters.py) ->
**your runner's GPU measurements** ->
`validate_catalog` -> catalog JSON. The wrapper invokes and validates the runner;
it does not benchmark anything on its own.

The runner must:

- Reconstruct representative inputs and invoke the matched original implementations
  in the pinned training environment, preserving precision, layout, compilation,
  graph-replay mode and relevant scalar settings.
- Compile and warm up outside steady-state timing, then collect repeated GPU
  durations with a timing method that covers all relevant streams. Host dispatch
  wall time alone is not GPU completion time.
- For communication, coordinate the actual process groups, message sizes,
  reduction semantics and placement. Isolated collective latency does not establish
  overlap or contention during training.
- Return one record per measured `cost_key`, with `hardware_id`, representative
  `seconds`, nonempty raw `samples_seconds`, and `evidence_sha256`. The checksum is
  SHA-256 of `json.dumps(samples_seconds, separators=(",", ":")).encode()`.
  Record the timing method, source/runtime identity, shapes, warmup, sample count
  and statistic in additional metadata or retained evidence.

All durations are in **seconds**. See the
[calibration record contract](../README.md#calibration-record).
A sample checksum detects inconsistent records, not incorrect measurement
methodology. Repeated profiler durations are not automatically valid calibration:
establish executable-region matching and account for instrumentation overhead.

Run this inside an authorized GPU allocation with an explicit scheduler limit.
The command does not allocate GPUs. For a distributed runner, one coordinating
process writes the aggregate catalog; do not have all ranks write the same path.
The CLI reports `missing_cost_keys` and `unkeyed_nonsemantic_nodes`; an exit code
of zero can still accompany a partial catalog.

### Predict on CPU, then validate independently

After cost collection, use the matching catalog:

```sh
titan-sim simulate results/prepared/workload.target.json \
  --calibration results/calibration-001/costs.json --strict \
  --output results/prediction-001.json
```

[`core.CostModel`](../titan_sim/core.py) looks up each operation's exact cost key;
`simulate` applies those durations to declared dependencies and exclusive
resources. Neither runs training. Strict simulation rejects uncalibrated operation
costs and declared capture gaps. **The raw native projection still has those gaps,
even if every cost key has a measured duration.** A hardware profile and complete
timing catalog alone do not make it prediction-ready.

Timing calibration is also **not memory calibration**. The current catalog
contract supplies durations, not physical allocation sizes or lifetimes. A memory
prediction additionally needs reconciled backing storage, saved activations,
workspaces, graph pools, allocator behavior and native/device-wide allocations.
Observe those on representative hardware and incorporate justified storage
records or explicit allowances, without double-counting aliases or adding
allocated plus reserved memory. Preserve unknowns and cover startup separately
when assessing whether the complete recipe fits.

Freeze the prepared workload, catalog, assumptions and prediction before a
separate held-out training run. Compare predicted step time and worst-device peak
memory with that run's measurements. Do not fit calibration costs to the held-out
result and then report it as independent validation.

## Fake/eager capture

Set `execution: "fake"`, `observer: "dispatch"`, and include
`--comm-backend fake` in the native `recipe_args`. This enters `FakeTensorMode`
around original recipe construction/training and rejects real backends through
the instrumented process-group creation APIs. Use the launch variables required
by the pinned native fake backend (the inspected revision uses `NGPU` and, for
pipeline parallelism, `FAKE_PP_RANK`). Do not assume `WORLD_SIZE` alone configures
that backend or that one logical rank represents all ranks.

Fake mode does not emulate CUDA initialization, streams, data-dependent scalars,
custom operators without fake implementations, or the data loader. Those can
block the recipe and produce a partial diagnostic. It never substitutes fabricated
tensor values or fake wall-clock timings to manufacture a successful capture.

The dispatch observer is also available explicitly for native eager debugging.
It uses weak storage references, detects allocation reuse/resizing, and records
aliasing. Python dispatch modes can conflict with `torch.compile(fullgraph=True)`;
they are therefore **not the default native observer**.

## What is still needed for prediction

- The native projection retains leaf operations and their original nesting
  metadata. Its sequential host issue order is not GPU execution order. Streams,
  waits, pipeline dependencies, CUDA graph replay and all-rank communication must
  be reconciled before predicting distributed execution.
- Execution-trace tensor extents are not complete backing allocation sizes or
  release times. The projection retains observed extents to step end and records
  state separately; it never adds these overlapping descriptions. Strided spans,
  hidden buffers, allocator reservation and device-wide memory still need coverage.
- Native cost keys describe candidate operator/region signatures. They do not
  prove a one-to-one relationship with GPU kernels. Use Chrome correlations,
  original executable regions and separately measured calibration to establish
  that relationship. Profiling can perturb timing and tensor lifetimes.
- Startup, checkpointing and operations outside selected steps are outside this
  capture. A steady-state training trace cannot establish initialization fit.

These gaps remain in `unresolved`; strict simulation refuses to promote this
projection to a calibrated throughput or memory-fit claim.

## Validation scope

The optional PyTorch suite checks native-step numerical equivalence, warmup/step
limits, method restoration after failures, raw trace retention, Dynamo fullgraph
compatibility (eager backend), fake collective dispatch, tensor aliases, weak
lifetimes, storage resizing and signature reuse. The core suite remains runnable
without PyTorch. Run both with:

```sh
python -m unittest discover -s tests -v
```

Source integration was inspected against TorchTitan
`5f5fc93e5d225d9056b1a41fc917d5e9c6e15801`; local checks used PyTorch 2.15.0 on
macOS CPU. Full Llama/Qwen attention runs are blocked on this machine by native
FlexAttention/CPU compilation support. Native Llama3 and Qwen3 feed-forward
forward/backward captures matched uninstrumented outputs and gradients. Repeat
these block-level checks with the pinned TorchTitan source importable:

```sh
TITAN_SIM_TEST_TORCHTITAN=1 python -m unittest discover -s tests -p test_torchtitan_blocks.py -v
```

This is not full-model GPU, full trainer,
distributed pipeline, or CUDA-graph validation. Run those bounded checks in the
matching training container before claiming model/backend support.

