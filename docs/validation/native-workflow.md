# Native workflow validation

This record distinguishes implementation checks from prediction accuracy.
The simulator and subagent skill consume native recipes; the model used here is
a validation fixture, not a special implementation or default configuration.

## Exercised scope

| Check | Evidence | Limit |
|---|---|---|
| Python tests | 114 tests: 113 passed, one optional TorchTitan source test skipped, using CPU PyTorch 2.15 | Includes synthetic timing contracts; does not make those synthetic samples GPU measurements |
| Native executable-region runner | Real BF16 matrix multiplication and FP32 all-reduce on eight GB200 GPUs, two nodes; three warmups and eight repetitions | Two generic measurement cases, not a training-throughput benchmark |
| Native recipe capture | Full Llama 3 8B validation recipe, eight GPUs, one ordinary warmup step plus one observed step; all ranks completed | Specific native trainer/runtime/feature combination only |
| Preparation → trace costs → simulation | 30,416 GPU occurrences across all eight rank artifacts; 30,416 exact occurrence-bound cost records; exploratory simulation completed | Costs came from the same captured run; no held-out accuracy claim |
| Physical memory result | `inconclusive`, with missing coverage reported | Tensor/storage coverage, allocator effects and lifecycle completeness were not established by this capture |

The GPU case used the existing prepared image with Torch
`2.16.0.dev20261030+cu132` and TorchTitan source identified by
`ff96670a26c7079bc1b9049f5b6942adf35567c1`. Source archive identity was retained.
The fixture preserved its native model, optimizer and rematerialization settings.
Native logs recorded finite loss and gradient norms. The simulator did not
substitute model operations or install another framework version.

Raw traces, checksums, launch scripts, image identities, scheduler outcomes and
numerical logs are retained in the validation artifact bundle. They are separate
from the generic skill. GPU attempts had five-minute scheduler hard limits.

## Compatibility and scaling findings

The default execution-trace observer failed on a rematerialization placeholder:
the native callback attempted storage-pointer access that the placeholder rejects,
then crashed. The preceding ordinary training step completed. A second run with
the explicit `execution_trace: false` setting completed all ranks using Kineto
profiling, preserving the original recipe. This establishes that compatibility
path for the tested runtime; it does not establish general ET support.

Kineto-only events carry artifact/index-bound keys. Their observed durations can
be imported, but they do not establish reusable operator shapes/scalars, complete
host costs, CUDA waits, communication rendezvous or backing-storage lifetimes.
Those limitations remain in the prepared workload and result.

Rank configuration comparison exposed process addresses in serialized Python
function representations. Preparation now permits only those narrowly identified
address differences, retains original hashes, and records an unresolved callable
identity gap. Different function names or substantive configuration differences
still fail comparison.

CPU cost collection initially retained every parsed rank trace. It now processes
one artifact at a time and references full source/device provenance by hashes,
avoiding repeated large recipe and device-property records. The final retained
capture completed local CPU processing after these changes.

Conservative observed storage extents are classified as unverified. They can
describe a modeled peak but cannot independently prove physical OOM. Unit tests
cover that distinction, colocated ranks, heterogeneous capacities, aliasing,
external allocations and separate initialization/checkpoint peaks.

## What remains unvalidated or unsupported

- Independent full-model prediction error, including near-capacity false-fit cases.
- Automatic recovery of every compiled region, stream wait, collective dependency,
  workspace and allocator allocation across arbitrary native recipes.
- Distributed CUDA-graph and subgroup-only executable measurement rounds.
- Empirical calibration on Vera Rubin; supplied specification records are not measurements.
- Universal support across TorchTitan versions, custom operations, precision modes,
  MoE routing policies and parallel layouts.

Use bounded, held-out validation for each additional scope. Do not convert this
one successful capture into a model whitelist or a claim that all recipes are
fully modeled.

## Native host-plan implementation checks

The subsequent host-plan integration adds `export-plan` and `lower-plan` without
changing the earlier GPU evidence. Its full suite runs 161 tests: 160 pass and
one optional TorchTitan source test skips in the installed CPU PyTorch 2.15
environment. Without PyTorch, 135 pass and 26 framework-dependent tests skip.

New checks exercise actual PyTorch schedule generation and communication action
lowering, actual FX/FakeTensor metadata, aliases, saved-value overlap across
microbatches, state and asynchronous completion lifetimes, joint action bindings,
nonfinite scalar serialization, artifact checksums and the CLI path. Native
preparation observation is checked for bounded steps, restored methods, preserved
updates and deferred observer errors. GraphTrainer-shaped object contracts in
these tests are not a full native GraphTrainer execution.

No additional GPU run or held-out prediction accuracy result is claimed for this
path. Its physical memory remains inconclusive because fake graph metadata does
not establish complete physical allocations. See the
[native planning guide](../native-planning.md) for supported inputs and remaining
integration work.

## Dispatch case and memory-observer implementation checks

The expanded suite runs 220 tests: 219 pass and one optional native-source test
skips with installed CPU PyTorch. With site packages disabled, 180 pass and 40
framework-dependent tests skip. New regressions cover native boxed reductions,
all-rank alias evidence, scalar collectives, lazy case allocation, exact selection
and interruption-safe restoration of memory-observer hooks.

Component calibration, repeatability checks and independent native-memory
observations produce local artifacts. Keep raw samples, allocation/source
identities and numerical experiment results with the caller's evidence bundle.
These are separate from the generic skill and implementation checks recorded here.
No central calibration storage service was introduced.

Testing different feature interactions checks the simulator implementation's
handling of submitted settings. It does not alter the configuration being
screened or require a recipe search on each invocation.
