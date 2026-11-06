# Using titan-sim with TorchTitan

This guide explains how an existing TorchTitan training recipe can supply evidence
for a titan-sim memory and throughput prediction. Start with the
[program overview](../README.md#how-the-program-works) for the simulator's design.

The integration preserves the original recipe and trainer. TorchTitan still owns
the model, input data, loss, optimizer, precision, activation checkpointing and
parallelization. titan-sim observes training and translates evidence into its
framework-independent workload format. You do not write another model forward
pass for the simulator.

The native capture adapter is experimental. It can retain original training
traces, but turning those traces into a complete distributed prediction still
requires implementation matching, workload preparation and a measurement runner.
This guide identifies those boundaries before showing the commands.

## How the integration fits together

```text
Your recipe module + configuration function
                     |
       TorchTitan ConfigLoader -> native trainer
                     |
        titan-sim observes original train_step
                     |
          ProfilerCapture on each rank
                     |
     Execution trace + Chrome trace + workload
                     |
   Match executable regions, dependencies and storage
                     |
             Prepared workload -------------------+
                     |                            |
       Your GPU measurement runner                |
                     |                            |
             Measured cost catalog --------> CPU simulation
                                                  |
                                   Prediction + coverage/gaps
```

| Part | Responsibility | Current implementation |
|---|---|---|
| Recipe and trainer | Define and execute training | Your original TorchTitan environment |
| Capture adapter | Load the recipe, bound observed steps, retain source/config identity | Bundled `titan_sim.torchtitan:capture` |
| Observer | Record operations and tensor evidence during selected steps | Bundled `ProfilerCapture`; fake/eager dispatch alternative available |
| Workload preparation | Resolve compiled regions, all-rank dependencies and backing-storage lifetimes | Integration work; no automatic native preparation command |
| Measurement runner | Benchmark the matched original implementations on target GPUs | You provide it; no general native TorchTitan runner is bundled |
| Simulator | Apply measured costs to declared execution and storage semantics | Bundled CPU simulator |

`ProfilerCapture` wraps PyTorch's profiler and `ExecutionTraceObserver`. It records
CPU activity and, on CUDA, GPU activity, with shape and memory profiling enabled.
It does not replace operators, disable compilation or install a Python dispatch
mode. Its traces help identify what must be measured; it does not generate a
calibrated cost catalog. Actual calibration occurs in the measurement runner.

## Before you start

Use the existing pinned training environment. Install titan-sim there without
replacing PyTorch, and make the TorchTitan checkout, recipe module, tokenizer,
data and native extensions available as they are for ordinary training. Record
source revisions and the container/runtime identity. The working directory and
launch environment matter.

Choose how to collect evidence:

| Mode | Configuration | What it provides |
|---|---|---|
| Native profiler capture | `execution: "native"`, `observer: "profiler"` | Evidence from real training on the recipe's devices; default native path |
| Fake dispatch capture | `execution: "fake"`, `observer: "dispatch"`, native fake backend arguments | Logical operations and storage metadata for supported fake execution; no GPU timings |
| Native eager dispatch capture | `execution: "native"`, `observer: "dispatch"` | Dispatch and alias debugging; can conflict with compiled execution |

The walkthrough uses native profiler capture. **It executes training**, including
warmup and optimizer updates. It preserves configured data loading, validation
and checkpoint side effects. Use an authorized allocation and a scheduler time
limit; the initial GPU test policy is five minutes. The adapter bounds completed
steps, not wall time, and does not allocate resources. Disable any second profiler
in the recipe explicitly and record the change: nested profiler sessions are not
supported.

## 1. Capture your native recipe

Supply the same module and configuration names you normally give TorchTitan.
In this example, `my_recipes` is an importable Python module and `my_recipe` is
its configuration function. Replace both placeholders with your real recipe.
They are passed separately as `--module` and `--config`; neither is a simulator
adapter.

Create `capture.json`:

```json
{
  "execution": "native",
  "observer": "profiler",
  "recipe_args": ["--module", "my_recipes", "--config", "my_recipe"],
  "warmup_steps": 1,
  "steps": 1,
  "trace_dir": "results/capture-001/native-traces",
  "hardware": {"id": "capture-only", "capacity_bytes": 0},
  "runtime_provenance": {
    "torchtitan_revision": "record-your-pinned-revision",
    "container_digest": "record-your-training-container"
  }
}
```

This requests one warmup step followed by one profiled step, subject to the
trainer's own stop condition. The `capture-only` hardware value is a sentinel:
it supplies no target capacity and cannot support a memory-fit prediction.

Select the bundled adapter and launch capture:

```sh
titan-sim capture --adapter titan_sim.torchtitan:capture \
  --config capture.json --output 'results/capture-001/workload.rank{rank}.json'
```

For distributed training, use the recipe's ordinary launcher and allocation.
For example, substitute your per-node process count and supply any required
multi-node rendezvous arguments:

```sh
torchrun --nproc-per-node=NUM_LOCAL_GPUS -m titan_sim capture \
  --adapter titan_sim.torchtitan:capture --config capture.json \
  --output 'results/capture-001/workload.rank{rank}.json'
```

Every rank writes its own artifacts. `{rank}` is required when `WORLD_SIZE > 1`.
Use a fresh trace directory: existing rank directories are rejected to preserve
evidence. Gather all rank artifacts for later distributed preparation.

### Inspect the capture

For rank 0 and the first captured step, the output contains:

| Artifact | Contents and use |
|---|---|
| `workload.rank0.json` | Preliminary operation/storage records, source/config fingerprints, placement, trace checksums and unresolved gaps |
| `native-traces/rank0/step0.et.json` | PyTorch execution-trace hierarchy and native input/output descriptors |
| `native-traces/rank0/step0.chrome.json` | Profiling timeline, including CUDA kernels/runtime events when available |

Retain the raw traces alongside the workload. The JSON projection does not replace
them. `capture_status=complete` means the requested steps returned and operator
events were recorded, not that simulation semantics are complete. `blocked` or
`incomplete` captures retain diagnostic output and return CLI exit code 2.

You can check the artifact's structure on CPU:

```sh
titan-sim validate results/capture-001/workload.rank0.json
```

This checks graph/storage references and reports declared gaps. It does not
validate the real execution or establish that the model fits.

## 2. Prepare the prediction workload

This is the first integration boundary: native capture does not automatically
produce a prediction-ready distributed workload. Use the retained evidence to
resolve three things:

| Required semantics | Why the raw projection is insufficient |
|---|---|
| Executable regions and cost keys | A trace leaf is not necessarily a GPU kernel. Match original operators or compiled regions using hierarchy and Chrome correlations; charge a fused region once. |
| Distributed execution | Host issue order does not establish stream/event dependencies, pipeline waits or collective rendezvous. Reconcile every rank and its communication participants. |
| Storage lifetimes | Observed tensor extents are not complete backing allocations or free times. Resolve aliases, strided spans, saved activations, workspaces and retained graph storage. |

The projection retains observed extents to step end and records model state
separately without adding overlapping descriptions. It does not reconstruct
allocator reservation, native buffers or device-wide usage. Record unknowns in
`unresolved`; remove a gap only when evidence resolves it. Initialization and
checkpoint operations outside selected steps require separate coverage.

The remaining commands use `results/prepared/workload.json` to mean the artifact
you produce through this preparation. **There is no bundled command for this
step.** A partial workload can be used to collect costs, but it remains unsuitable
for strict prediction while its gaps are unresolved.

Set the actual target hardware before calibration. Given a hardware description
you supply in `target-hardware.json`:

```python
import json
from pathlib import Path
from titan_sim.workload import read_workload, write_workload

workload = read_workload("results/prepared/workload.json")
workload.hardware = json.loads(Path("target-hardware.json").read_text())
write_workload("results/prepared/workload.target.json", workload)
```

This edits metadata only. The description needs a meaningful hardware `id`, actual
per-device `capacity_bytes`, and any rate/resource assumptions used by your model.
`calibrate` checks against the workload's hardware ID and has no `--hardware`
option. The runner must verify that the real devices and topology match the target.

## 3. Measure component costs on GPUs

Calibration answers how long each matched implementation takes on the target.
It requires a measurement runner you provide, with this contract:

```python
def measure_workload(workload, config) -> dict:
    ...  # Measure original executable implementations and return a cost catalog.
```

The runner owns representative input construction, compilation, warmup, repeated
GPU timing and distributed coordination. Preserve precision, layout, scalar
settings, compilation and graph-replay mode. Time completion on all relevant
streams, not just host dispatch. Communication measurements must use the intended
groups, message sizes, reduction semantics and placement. Isolated timings alone
do not establish training overlap or contention.

With your runner in the importable module `my_project.measure`, invoke:

```sh
titan-sim calibrate results/prepared/workload.target.json \
  --runner my_project.measure:measure_workload \
  --config measurement.json --output results/calibration-001/costs.json
```

This is a template, not a bundled native benchmark. `measurement.json` follows
your runner's schema. The call path is `cli.main` -> `adapters.calibrate` ->
**your runner's measurements** -> catalog validation and writing. Neither the
CLI nor `ProfilerCapture` performs those measurements on the runner's behalf.

The returned catalog contains measured durations in seconds, keyed by workload
`cost_key`, with hardware identity, raw samples and sample checksums. Follow the
[calibration record contract](../README.md#calibration-record) and retain the
timing method, source/runtime identity, warmup, sample count and chosen statistic.
Profiler durations require implementation matching and assessment of profiling
overhead before they can be used as costs.

Run inside an authorized GPU allocation. For a distributed runner, one coordinating
process writes the aggregate catalog. Inspect `missing_cost_keys` and
`unkeyed_nonsemantic_nodes`: successful command completion can still leave a
partial catalog. A checksum validates consistency, not measurement quality.

## 4. Simulate and check the prediction

Once the prepared workload and matching catalog are available, prediction runs on
CPU:

```sh
titan-sim simulate results/prepared/workload.target.json \
  --calibration results/calibration-001/costs.json --strict \
  --output results/prediction-001.json
```

The cost resolver looks up exact keys. The simulator schedules events using
declared dependencies and exclusive resources, and tracks storage through its
declared consumers. It does not run the recipe. `simulate --hardware FILE` can
override the target description, but the catalog must match that hardware ID.

Strict mode rejects unresolved gaps and uncalibrated operation costs. **A complete
timing catalog does not make the raw native projection complete.** Memory fit
also depends on storage coverage, native allocations and allocator/graph behavior;
the timing catalog does not supply their sizes or lifetimes. Use measured evidence
or explicit allowances and preserve uncertainty. Allocated memory is contained in
reserved memory, so do not add them.

Read `status`, `cost_coverage`, `unresolved`, `memory_verdict` and worst-rank storage
alongside predicted step time. See [result interpretation](../README.md#read-the-result).
Freeze the workload, costs, assumptions and prediction, then compare with a
separate held-out training run. Measure throughput without intrusive memory
profiling, and compare memory over the relevant lifecycle. Do not fit costs to
that validation run and then claim an independent prediction.

## Alternative integration paths

### Fake and eager dispatch capture

For fake capture, select `execution: "fake"`, `observer: "dispatch"`, and include
`--comm-backend fake` in native `recipe_args`. The adapter enters `FakeTensorMode`
around original recipe construction/training and rejects real backends through
the instrumented process-group creation APIs. Use the launch variables required
by the pinned backend; the inspected TorchTitan revision uses `NGPU` and, for
pipeline parallelism, `FAKE_PP_RANK`. One selected logical rank does not establish
all-rank coverage.

Fake mode does not emulate CUDA initialization, streams, the data loader,
data-dependent scalars or custom operations without fake implementations. Such
failures produce partial diagnostics, not fabricated tensor values or timings.

The dispatch observer is also available for native eager debugging. It tracks
aliases, weak storage lifetimes and allocation resizing. Python dispatch modes
can conflict with `torch.compile(fullgraph=True)`, which is why this observer is
an explicit alternative to the default native profiler path.

### An already-built trainer

If your application constructs the trainer itself, call the lower-level API:

```python
from titan_sim.torchtitan import capture_trainer
from titan_sim.workload import write_workload

workload = capture_trainer(
    trainer, hardware=hardware, trace_dir="results/unique-run/traces",
    warmup_steps=1, steps=1,
)
write_workload("results/unique-run/workload.json", workload)
# The caller owns trainer and process-group cleanup.
```

The trainer must provide `train()`, `train_step(...)`,
`should_continue_training()`, and an `engine` with `device` and `model_parts`.
The loader-based adapter also needs `config.build()` and `trainer.close()`.
Subclasses retain their own step implementation. Applications without this
contract, including alternative RL entrypoints, need an integration at their
actual step boundary.

## Validation and compatibility

The importer currently accepts PyTorch execution-trace schema
`1.1.1-chakra.0.0.4`. This is native JSON; it needs neither Chakra protobuf
installation nor a Chakra simulator. Other schemas fail explicitly until their
importers are validated.

Source integration was inspected against TorchTitan
`5f5fc93e5d225d9056b1a41fc917d5e9c6e15801`. Recorded local checks used PyTorch 2.15.0
on macOS CPU. They cover native-step numerical equivalence, bounded steps, method
restoration, raw traces, Dynamo fullgraph compatibility with the eager backend,
fake collective dispatch, aliases and storage behavior. Native Llama3 and Qwen3
feed-forward captures matched uninstrumented outputs and gradients; full attention
runs were blocked by native FlexAttention/CPU compilation support.

Run the suite with:

```sh
python -m unittest discover -s tests -v
```

With the pinned TorchTitan source importable, repeat block-level checks with:

```sh
TITAN_SIM_TEST_TORCHTITAN=1 python -m unittest discover -s tests -p test_torchtitan_blocks.py -v
```

This evidence does not validate full-model GPU training, complete trainer
integration, distributed pipelines or CUDA-graph execution. Those require bounded
checks in the matching training container before claiming model/backend support.
