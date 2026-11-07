# Using titan-sim with TorchTitan

This guide explains how an existing TorchTitan training recipe can supply evidence
for a titan-sim memory and throughput prediction. Start with the
[program overview](../README.md#how-the-program-works) for the simulator's design.

The integration preserves the original recipe and trainer. TorchTitan still owns
the model, input data, loss, optimizer, precision, activation checkpointing and
parallelization. titan-sim observes training and translates evidence into its
framework-independent workload format. You do not write another model forward
pass for the simulator.

There are two evidence paths. This guide covers observing executed training steps.
The [native planning guide](native-planning.md) covers extracting already-built
host schedules and stage graphs, then deriving supported dependencies and logical
storage lifetimes. Plan export can also observe lazy graph preparation during
bounded native steps. Neither path changes the submitted recipe's settings.

The native capture adapter is experimental. It can retain original training
traces, but turning those traces into a complete distributed prediction still
requires evidence for unsupported implementation matching, synchronization and
allocation semantics. Bundled preparation and calibration tools handle the
explicitly supported cases.
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
       Bundled runner + native case provider      |
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
| Workload preparation | Resolve compiled regions, all-rank dependencies and backing-storage lifetimes | Bundled `prepare`; direct GPU correlations and explicit evidence assertions |
| Case selection | Identify operations that can be faithfully reconstructed for measurement | `calibration-cases` selects supported dense matmuls and collectives from dispatch captures, with explicit skips |
| Measurement runner | Benchmark the matched original implementations on target GPUs | Bundled trace importer and GPU timer; bundled providers for supported dispatch cases, custom providers for other native regions |
| Memory observer | Check actual allocator peaks and visible state in the unchanged runtime | Optional `MemoryObserver`; observations are separate from predictions |
| Simulator | Apply measured costs to declared execution and storage semantics | Bundled CPU simulator |

`ProfilerCapture` wraps PyTorch's profiler and `ExecutionTraceObserver`. It records
CPU activity and, on CUDA, GPU activity, with shape and memory profiling enabled.
It does not replace operators, disable compilation or install a Python dispatch
mode. Its traces help identify what must be measured; it does not generate a
calibrated cost catalog. Actual timing collection occurs in the measurement runner. It can import matched
GPU observations from the traces or execute supplied native regions repeatedly.

For an eager dispatch capture, [compute case selection](operator-cases.md) can
generate supported matrix-multiplication measurements for one GPU.
[Communication case selection](communication-cases.md) uses the identified group
size and peer captures to verify shared-buffer layouts before replay on multiple
GPUs. These are component measurements with local result files; no calibration
storage service is needed. Unsupported operations remain in the inventory.
Use a separate [memory observation](memory-observation.md) run to check native
peaks: tracing can change allocation lifetimes and its peak must not be treated
as the uninstrumented recipe's requirement.

## Before you start

This workflow runs after [prepare-runtime](https://gitlab-master.nvidia.com/elfieg/torchtitan-gpu-agent-suite/-/tree/main/skills/prepare-runtime).
The [simulation subagent skill](../skills/simulate-torchtitan/SKILL.md) consumes its
artifact handoff and the caller's execution context. It does not build an image,
choose a cluster or allocate GPUs. Stage the runtime manifest, dependency plan,
capability report and raw evidence where the chosen command environment can read them.

```sh
titan-sim runtime-check runtime-manifest.json --expected expected-runtime.json \
  --output results/runtime-check.json
```

`expected-runtime.json` is an explicit request binding, for example:

```json
{
  "codebase_commit": "replace-with-full-pinned-TorchTitan-commit",
  "image_identity": "sha256:replace-with-final-image-hash",
  "required_checks": ["recipe-training"]
}
```

Replace the example check ID with required IDs in your capability contract. You
can also bind the complete `recipe` object or selected `target` fields. New
features or changed precision/communication paths need relevant capability
qualification; a successful debug case alone does not qualify every recipe.
`runtime-check` checks schema, image/source/request consistency, required check
results and readable evidence, retaining hashes. Its `ready` status describes the
handoff artifacts, not the currently running process or full-scale readiness.
Verify the actual image, interpreter, code and hardware through the supplied
execution context. Passing `runtime_manifest`, `capability_report` and optional
`expected_runtime` in capture configuration repeats this check before recipe loading.

Use the existing pinned training environment. Install titan-sim there without
replacing PyTorch, and make the TorchTitan checkout, recipe module, tokenizer,
data and native extensions available as they are for ordinary training. Record
source revisions and the container/runtime identity. The working directory and
launch environment matter.

Choose how to collect evidence:

| Mode | Configuration | What it provides |
|---|---|---|
| Native profiler capture | `execution: "native"`, `observer: "profiler"` | Evidence from real training on the recipe's devices; default native path |
| Kineto-only profiler capture | Same, plus `execution_trace: false` | Retained Chrome GPU activities when ET is incompatible; occurrence-specific costs, no ET tensor/storage reconstruction |
| Fake dispatch capture | `execution: "fake"`, `observer: "dispatch"`, native fake backend arguments | Logical operations and storage metadata for supported fake execution; no GPU timings |
| Native eager dispatch capture | `execution: "native"`, `observer: "dispatch"` | Dispatch and alias debugging; can conflict with compiled execution |

The walkthrough uses native profiler capture. **It executes training**, including
warmup and optimizer updates. It preserves configured data loading, validation
and checkpoint side effects. Use an authorized allocation and a scheduler time
limit; the initial GPU test policy is five minutes. The adapter bounds completed
steps, not wall time, and does not allocate resources. Disable any second profiler
in the recipe explicitly and record the change: nested profiler sessions are not
supported.

Execution-trace support must be qualified separately from ordinary training.
Some native rematerialization paths use metadata placeholders that reject data
pointer access; an incompatible `ExecutionTraceObserver` can crash in native
code. An explicit `execution_trace: false` disables that observer while retaining
Kineto CPU/CUDA profiling and the original recipe. This is never an automatic
fallback. It preserves fewer semantics: GPU costs are bound to individual trace
occurrences and cannot be reused as shape/implementation matches across cases.
Projected storage remains unverified. Read the
[calibration limitations](native-calibration.md) before using those observations.

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

With `execution_trace: false`, only the Chrome trace is produced. The preliminary
workload contains GPU occurrences, without execution-trace tensor metadata.

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

Combine all captured ranks on CPU. `prepare` discovers checksum-recorded Chrome
traces and lowers unambiguous GPU correlations. An optional evidence file supplies
rank placement, the unique global token count, and execution/storage assertions:

```sh
titan-sim prepare results/capture-001/workload.rank*.json \
  --config preparation.json --output results/prepared/workload.json
```

Start `preparation.json` with facts from the actual recipe and target. For a
**two-rank example only**, where a completed update processes 4096 unique tokens:

```json
{
  "useful_tokens": 4096,
  "rank_placement": {"0": "gpu0", "1": "gpu1"}
}
```

Match placement IDs to your hardware description. Do not sum model-parallel
copies as unique tokens. The [preparation guide](native-preparation.md) explains
multi-launch matching, compiled-region assertions, communication rendezvous,
backing-storage reconciliation and audited gap resolutions.

Preparation does not infer CUDA event waits from host timestamps. An unsupported
correlation, missing rank, unknown allocation or failed capture stays visible.
Explicit integration assertions retain their evidence and do not become empirical
proof. Full physical-memory coverage needs the [memory contract](native-memory.md),
including graph pools, allocator slack, external buffers and lifecycle peaks.

Set the actual target identity and capacities before collecting costs. If capture
used the `capture-only` sentinel, replace the metadata explicitly:

```python
import json
from pathlib import Path
from titan_sim.workload import read_workload, write_workload

workload = read_workload("results/prepared/workload.json")
workload.hardware = json.loads(Path("target-hardware.json").read_text())
write_workload("results/prepared/workload.target.json", workload)
```

This metadata change cannot transfer recorded GPU timings to another architecture.
For trace-cost collection, prepare with the actual captured target identity from
the outset: exact trace keys include that identity. New hardware requires matching
measurements or an explicitly analytical model. See [hardware inputs](hardware.md).

## 3. Measure component costs on GPUs

The bundled `titan_sim.native_runner:measure` supports two paths:

| Path | What happens | Appropriate evidence claim |
|---|---|---|
| `mode: "trace"` | Read retained GPU events matched during preparation; no new GPU work | Instrumented observations of the captured run |
| `mode: "execute"` | Warm up and repeatedly time original native callables from a trusted provider | Measured component costs on the recorded devices/runtime |

For trace collection, create `measurement.json` with the exact captured hardware
ID and explicit acknowledgement of profiling overhead:

```json
{
  "mode": "trace",
  "hardware_id": "your-captured-target",
  "accept_profiler_overhead": true
}
```

Run the coordinator:

```sh
titan-sim calibrate results/prepared/workload.target.json \
  --runner titan_sim.native_runner:measure \
  --config measurement.json --output results/calibration-001/costs.json
```

For independent component timing, supply an executable-region provider in the
original training environment. The runner owns repeated CUDA timing; the provider
owns original implementation selection, representative inputs and compilation.
It supports existing all-rank process groups and single-rank graph replay.
Distributed graph replay and subgroup-only measurement rounds are currently
unsupported. See the [calibration guide](native-calibration.md) for runnable
provider and launcher examples.

The CLI calls `adapters.calibrate` → the selected runner → catalog validation and
writing. Inspect `missing_cost_keys` and `unkeyed_nonsemantic_nodes`: collection
can succeed with a partial catalog. A complete catalog does not resolve missing
execution or memory semantics. Trace costs remain instrumentation-dependent and
cannot be reused as held-out validation of the same run.

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
and `memory_unresolved` alongside predicted step time. Add
`--require-memory-fit` to return a nonzero exit status unless physical memory is
`estimated_fit`. Timing `--strict` alone does not establish memory completeness. See [result interpretation](../README.md#read-the-result).
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
