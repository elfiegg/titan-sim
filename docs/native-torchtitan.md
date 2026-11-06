# Native TorchTitan capture

`titan_sim.torchtitan:capture` loads the **original recipe**, builds its native
trainer, and observes selected calls to its original `train_step`. Model choice,
inputs, loss, optimizer, precision, activation checkpointing and parallelization
remain owned by TorchTitan. There is no Llama/Kimi-specific forward in this adapter.

This is an experimental capture implementation, not a claim that every TorchTitan
model/backend or every historical TorchTitan version has been validated. It also
does not make arbitrary recipes executable using fake tensors on a CPU.

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

`my_recipes:my_recipe` is your native TorchTitan recipe, not another simulator
adapter. The zero-capacity `capture-only` profile deliberately supplies no target
hardware assumption. Replace it with an actual hardware description, or supply
`simulate --hardware`, before attempting a prediction. Costs and graph semantics
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
| `profiler_capture.ProfilerCapture` | Native execution on the current rank | PyTorch execution-trace JSON and Chrome trace per selected step |
| `profiler_capture.project_trace` | Native execution trace | Conservative leaf-operation/storage projection into simulator records |
| `torch_capture.TorchCapture` | Explicit fake/eager execution | Dispatch operations, aliases, weak storage lifetimes and state metadata |

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
