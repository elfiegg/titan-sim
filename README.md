# titan-sim

titan-sim is an experimental tool for prescreening native TorchTitan training recipes.
It is intended to run as a simulation subagent after the training runtime has
been prepared and qualified.
It is designed to answer two questions before a full training run:

- Will the recipe's peak memory fit on **each device**, given its placement and schedule?
- What step time and throughput should we expect on the target hardware?

It combines a description of the training workload, measured component costs or
explicit analytical assumptions, and a CPU simulator. Predictions include their
coverage and unresolved limitations. They do not establish successful execution
or measured performance at scale.

## How the program works

A training recipe says how to train a model: inputs, precision, optimizer,
parallelism and other settings. The simulator needs a different description:
which operations run on which ranks (participating processes), what they depend
on, which resources they use, and when their backing storage must stay alive.

Two native paths connect these descriptions. A **plan exporter** reads schedules
and stage graphs that the selected runtime has already constructed; plan lowering
derives supported dependencies and logical storage lifetimes. A **capture adapter**
observes actual execution; trace preparation resolves its supported evidence.
Both preserve unsupported semantics as explicit gaps. A **measurement runner**
benchmarks the corresponding implementations on the target hardware. The
**simulator** uses the resulting costs and workload to calculate event times and
per-rank storage peaks.

```text
prepare-runtime handoff + native TorchTitan recipe
                   |
       Native plans or observed execution
                   |
       Export/lower or capture/prepare
                   |
             Prepared workload ----------------------+
                   |                                 |
        Measurement runner on GPUs                   |
                   |                                 |
          Measured cost catalog ---------------------+
                                                     |
Target hardware + resource assumptions ------> CPU simulator
                                                     |
                               Timing, memory, coverage and gaps
                                                     |
                               Compare with independent training
```

Capture, calibration and simulation have different responsibilities:

| Component | Question it answers | Produces |
|---|---|---|
| Plan export and lowering | What native schedule, stage graphs and logical storage lifetimes are already available? | A workload derived from supported native artifacts, with remaining gaps |
| Capture adapter | What does this recipe execute, and what evidence can we observe? | Workload records and framework-specific evidence |
| Workload preparation | How do those observations map to executable regions, distributed dependencies and storage lifetimes? | A workload with explicit semantics and remaining gaps |
| Measurement runner | How long do these implementations take on the target hardware? | A catalog of measured durations |
| CPU simulator | What happens when those operations follow this schedule and share these declared resources? | Event timing, per-rank peak storage and coverage |
| Independent validation | How well did the prediction match a separate real run? | Prediction errors and evidence about model limitations |

The simulator is independent of the training framework. It reads data; it does not
import a model or discover its kernels. Adapters and runners are explicit Python
functions executed in the environments their frameworks require. The core remains
framework-independent, while the supported integration workflow targets native
TorchTitan recipes without copying model implementations into the simulator.

### The data connecting the components

A **workload** contains operations, storage records, a manifest and a hardware
description. An operation occurrence has a node ID, rank, dependencies and
attributes. Its `cost_key` identifies a measurement case that multiple occurrences
can reuse. The key must distinguish relevant implementation, shape, layout,
precision, scalar settings, software and communication context.

A **cost catalog** maps those keys to measured durations and supporting samples.
Calibration is the act of collecting these measurements. Fake execution can
provide shapes or logical communication calls, but its wall time is not GPU time.
Likewise, a profiler trace is evidence for identifying implementations; it is not
automatically a calibrated catalog.

Memory uses **storage records**, not timing samples. A record identifies a unique
backing allocation, its size, producer and consumers. The simulator retains it
through its declared users, counting aliases once. Predicting physical device
usage also requires coverage of workspaces, communication buffers, graph pools
and allocator behavior. Calibrated timing alone cannot establish memory fit.

## What is available today

| Capability | Current state |
|---|---|
| Workload JSON, structural checks, adapter/runner invocation | Implemented |
| Exact-key measured cost lookup and CPU event/storage simulation | Implemented |
| Native host-plan extraction and lowering | Supported native action tables, FX graphs, alias/state metadata and microbatch lifetimes; one pipeline-group merge; no universal offline graph construction or complete physical-memory claim |
| Native TorchTitan capture | Original training-step observation; ET+Kineto by default, explicit Kineto-only mode for ET-incompatible runtimes; raw evidence retained |
| Native trace-to-GPU matching and rank preparation | Bundled `prepare`: checksum-verified direct correlations, rank merging and evidence-backed region/dependency/rendezvous assertions; ambiguous correlations and unknown waits remain gaps |
| Local CUDA discovery | Original trainer with target logical fake groups and explicit communication substitutions; actual local compute; unsupported boundaries fail closed |
| Native schedule recovery | Complete peer dispatch captures with shared run identity can recover supported functional collective completion/wait dependencies; fake waits and missing stream semantics remain gaps |
| Native GPU measurement runner | Bundled trace-cost importer and callable-region timer; live native handles with explicit state/context contracts; existing all-rank process groups supported, distributed graph/subgroup-only rounds unsupported |
| Identified compute/communication calibration | Native dense matmuls, supported collectives, uniform all-to-all and two-rank-source-group P2P; exact source/cost binding; explicit skips and local artifacts |
| Concurrent native windows | Provider-owned state, CUDA streams/events and all-rank preflight; closed source-backed windows bind atomically; automatic trace-to-window recovery remains unimplemented |
| Per-device physical-memory bounds | Placement-aware lifetime envelopes, heterogeneous capacities, aliases, external allocations and lifecycle peaks; complete coverage requires explicit evidence |
| Real native memory observation | Allocator peaks, boundary snapshots and visible state inventories, including optional native peak-reset tracking; observations do not establish unseen-configuration fit |
| Target-applicable memory evidence | Source/placement/phase-bound allocation contracts, aliases and coverage; conditional forecasts separated from unconditional fit; complete contract production still needs integration |
| Native allocation history | Generation/release reconstruction and conditional replay envelopes from recorded allocator evidence; process-allocator coverage is distinct from physical predictive coverage |
| Prediction comparison | Frozen exact-scope predictions and trusted fresh-control invocation; partial captures cannot qualify as full-update accuracy |
| Full-model GPU/distributed accuracy across recipes | Requires held-out validation for each claimed scope; universal configuration support is not established |

The repository-owned [simulation skill](skills/simulate-torchtitan/SKILL.md) consumes
the [prepare-runtime handoff](docs/native-torchtitan.md#before-you-start).
You can run the framework-free example below, simulate an existing compatible
workload, or develop a capture/calibration integration. For the workflow using
one GPU for local discovery and compute, then actual groups for communication,
start with [GPU-assisted prescreening](docs/gpu-assisted.md). For full native
profiler observation, use the [native integration guide](docs/native-torchtitan.md). The bundled pipeline preserves
unsupported semantics as gaps; it does not automatically recover every compiled
region, stream dependency or allocation from a trace.
For already-constructed native schedules and graph metadata, use the
[native planning workflow](docs/native-planning.md). Its observation mode preserves
the original recipe; it does not convert eager training to GraphTrainer.

## Get started

CPU simulation requires Python 3.10+ and no mandatory third-party dependencies.
Install from a pinned checkout and read [AGENTS.md](AGENTS.md) before executing a
workload:

```sh
git clone https://github.com/elfiegg/titan-sim.git
cd titan-sim
git rev-parse HEAD
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
titan-sim --help
```

`python -m titan_sim` is equivalent to `titan-sim`. Native capture and GPU
calibration run in the original pinned training environment, preserving its
PyTorch, CUDA and custom extensions. The program does not allocate cluster
resources or impose scheduler time limits.

### Run a small example

This example constructs two synthetic operations and uses declared analytical
rates. It demonstrates the file flow without requiring PyTorch or GPUs:

```sh
titan-sim capture --adapter examples.minimal:capture \
  --config examples/minimal.json --output results/minimal/workload.json

titan-sim validate results/minimal/workload.json

titan-sim simulate results/minimal/workload.json \
  --output results/minimal/result.json
```

The result is exploratory and its memory verdict is `inconclusive`, because the
example declares missing workspace coverage. Adding `--strict` rejects it. This
is not a model benchmark. Example modules are importable from the checkout and
are not installed as part of the simulator wheel.

### Use your own workload

The CLI exposes these operations:

| Command | Required inputs | Output | Where it runs |
|---|---|---|---|
| `runtime-check` | Runtime manifest, capability evidence and expected identities | Readiness/gaps and artifact hashes | CPU |
| `prepare` | Rank workloads and optional preparation evidence | Combined workload with explicit semantics/gaps | CPU |
| `recover-schedule` | Complete native peer captures with matching run identity | Supported collective completion and wait dependencies, with remaining gaps | CPU |
| `export-plan` | Prepared native objects through a trusted adapter, or native recipe export configuration | Versioned schedule/graph plan | Original framework environment; observation may execute native steps |
| `lower-plan` | Native pipeline-group plans, hardware and explicit bindings | Workload with schedule and logical storage lifetimes | CPU |
| `calibration-cases` | Native dispatch workload; actual placement for communication | Isolated measurement workload and optional eligible/skipped inventory | CPU |
| `window-cases` | Source workload, native provider and stream/dependency manifest | Exact-context native window measurement workload | CPU; measurement uses the existing GPU group |
| `bind-cases` | Exact source snapshots and native measurement workloads | Derived workload and occurrence-binding audit | CPU |
| `bind-window` | Exact source and a closed source-backed window workload | One measured atomic event with all participant resources | CPU |
| `bind-memory` | Final workload, allocation evidence bundle and exact applicability | Workload with audited observations and predictive contracts | CPU |
| `allocation-evidence` | Raw native allocation capture, final workload and exact applicability | Observed generation audit and conditional replay evidence | CPU; capture uses the original runtime |
| `freeze-prediction` | Workload, catalog and producer-recorded validation context | Immutable prediction identity and result | CPU |
| `validate-prediction` | Frozen prediction and trusted fresh-control runner | Exact-scope timing/memory comparison | Single coordinator; runner owns native execution |
| `capture` | Adapter reference and adapter-owned configuration | Workload JSON | Adapter's framework environment |
| `validate` | Workload JSON | Structural counts and declared gaps | CPU |
| `calibrate` | Workload, measurement runner and runner-owned configuration | Cost catalog and unmatched-key report | Runner's measurement environment, usually GPUs |
| `simulate` | Workload, hardware assumptions and optional matching catalog | Timing, storage peaks and coverage | CPU |

The following calibration/prediction commands are templates. They assume you have
prepared `workload.json` and implemented the named runner. `my_project` is not a
bundled module:

```sh
titan-sim calibrate workload.json \
  --runner my_project.measure:measure_workload \
  --config measurement.json --output results/costs.json

titan-sim simulate workload.json --calibration results/costs.json \
  --strict --output results/prediction.json
```

The bundled runner supports trace import and executable-region timing; see
[calibration](docs/native-calibration.md). A provider owns native region construction
and compilation; the runner performs warmup, GPU timing and supported distributed coordination.
For supported captured cases, use the bundled [dense-matmul provider](docs/operator-cases.md)
on one GPU or the [communication provider](docs/communication-cases.md) on the
identified group size. [Memory observation](docs/memory-observation.md) is a separate
instrumented run. Central calibration storage is optional; retained local
workloads, inventories and sample catalogs are sufficient for these operations.
The CLI invokes it and checks its returned records. A coordinating process should
write the aggregate catalog; distributed ranks must not compete for one output
path. GPU execution needs an authorized allocation and an explicit scheduler
limit; use five minutes for initial benchmarks unless another limit is requested.

Put the intended hardware identity and per-device capacity in the workload before
calibration. `simulate --hardware FILE` can replace the hardware description, but
the catalog must match that identity. The CLI checks the identity string; the
runner is responsible for verifying the actual devices, topology and compatible
implementations. No GPU profile is silently selected.

## Read the result

Interpret these fields together:

| Field | Meaning |
|---|---|
| `seconds`, `useful_tokens_per_second` | Predicted duration and throughput for the declared workload scope; throughput requires a correct useful-token count |
| `modeled_peak_bytes`, `worst_rank`, `peak_categories_bytes` | Per-rank storage peaks under the supplied lifetime model |
| `memory_verdict`, `physical_memory` | Per-device lower/upper envelope and capacity verdict; missing placement or memory coverage makes fit inconclusive |
| `memory_unresolved` | Missing physical-memory evidence, distinct from timing/workload gaps |
| `cost_coverage` | Which costs came from exact measured keys, analytical assumptions or historical proxies |
| `cost_evidence`, `cost_limitations` | Whether exact costs are profiler observations, executable-region measurements or external records, with retained measurement limitations |
| `unresolved` | Limitations declared by the producer or integration |
| `status` | Whether the result used complete declared coverage and calibrated costs, or remains exploratory |

`validate` checks graph and storage references. It does not establish real
execution, complete capture or memory sufficiency. `simulate` additionally checks
communication rendezvous. `--strict` rejects declared gaps and uncalibrated
nonsemantic operation costs; it cannot discover every omission in a producer's
model. A `calibrated_prediction` qualifies the declared timing model; it can still have
inconclusive physical memory. Use `--require-memory-fit` to require a physical
`estimated_fit` result. See the [memory contract](docs/native-memory.md).

The storage peak is not automatically CUDA allocator-reserved or device-wide
memory. Per-rank peaks describe per-device usage only when the placement model
assigns one rank to each device. Multiple ranks sharing a device require a model
of their combined live allocations. Allocated memory is contained in reserved
memory; never add them. Include initialization and checkpoint-loading peaks when
assessing the complete recipe.

Commands print JSON. `--output` writes the relevant artifact; simulation stdout
omits the detailed event timeline retained in its output file. Runtime/contract
errors return exit code 2. A native capture marked `blocked` or `incomplete` also
returns 2 while retaining a diagnostic workload. Argument syntax errors use
argparse's standard error message.

## Integrate a framework

Adapters and runners are trusted, importable `module:function` references:

```python
from titan_sim.workload import Workload

def build_workload(config: dict) -> Workload:
    ...  # Observe the framework or construct explicit workload records.

def measure_workload(workload: Workload, config: dict) -> dict:
    ...  # Measure matched implementations and return a cost catalog.
```

The capture function is selected by `capture --adapter`; the measurement function
by `calibrate --runner`. Their configuration schemas belong to those functions.
Loading workload JSON for validation or simulation does not execute a recorded
adapter or import its framework.

The core Python interface is:

```python
from titan_sim.core import simulate
from titan_sim.workload import Workload

workload = Workload(
    nodes=nodes,
    storage=storage,
    manifest=manifest,
    hardware=hardware,
)
result = simulate(*workload.arguments(), calibration=catalog, strict=True)
```

`Node` supports compute, send, receive and collective events. `Storage` represents
backing allocations. The manifest records ranks, useful tokens, source/runtime
provenance and unresolved gaps. Preserve evidence of implementation matching and
capture adaptations; a configuration hash alone is insufficient provenance.

### Calibration record

```text
catalog[cost_key] = {
    hardware_id: identity matching workload.hardware.id,
    seconds: representative measured duration,
    samples_seconds: nonempty list of raw durations,
    evidence_sha256: SHA256 of compact JSON encoding of that sample list,
    ... implementation, timing-method and provenance metadata ...
}
```

The checksum uses `json.dumps(samples_seconds, separators=(",", ":")).encode()`.
It detects inconsistent records, not an incorrect benchmark. Catalog validation
checks hardware identity, finite nonnegative durations and sample checksums.
Partial catalogs are accepted during collection; inspect `missing_cost_keys` and
`unkeyed_nonsemantic_nodes` even when `calibrate` returns zero.

Freeze the workload, costs and assumptions before comparing the prediction with
a separate held-out run. Do not fit component costs to that run and then report
the comparison as independent validation.

### Implementation map

| Source | Responsibility |
|---|---|
| [`cli.py`](titan_sim/cli.py) | Commands, artifact paths and result reporting |
| [`adapters.py`](titan_sim/adapters.py) | Explicit function invocation and catalog validation |
| [`workload.py`](titan_sim/workload.py) | Workload contract and JSON persistence |
| [`runtime.py`](titan_sim/runtime.py) | Prepare-runtime artifact consistency and request identity checks |
| [`prepare.py`](titan_sim/prepare.py) | Native evidence matching, rank merging and explicit semantics |
| [`native_plan.py`](titan_sim/native_plan.py), [`plan_capture.py`](titan_sim/plan_capture.py) | Native plan inspection, artifact validation and bounded preparation observation |
| [`plan_lowering.py`](titan_sim/plan_lowering.py), [`plan_memory.py`](titan_sim/plan_memory.py) | Native action dependencies and graph-backed logical storage lifetimes |
| [`native_runner.py`](titan_sim/native_runner.py) | Trace observations and repeated native GPU measurements |
| [`operator_cases.py`](titan_sim/operator_cases.py), [`communication_cases.py`](titan_sim/communication_cases.py) | Eligible native overload cases and replay providers with exact metadata/runtime context |
| [`local_execution.py`](titan_sim/local_execution.py), [`executable_regions.py`](titan_sim/executable_regions.py) | Target-rank CUDA discovery, explicit communication substitution and live native callable retention |
| [`case_binding.py`](titan_sim/case_binding.py), [`identity.py`](titan_sim/identity.py) | Exact source snapshots, native measurement correspondence and occurrence audits |
| [`schedule_recovery.py`](titan_sim/schedule_recovery.py) | Native peer collective reconciliation and exact functional wait lineage |
| [`execution_windows.py`](titan_sim/execution_windows.py) | Native concurrent measurement and conservative closed-window binding |
| [`memory_observer.py`](titan_sim/memory_observer.py) | Native process allocator peaks, sampled device use and state inventories |
| [`memory_evidence.py`](titan_sim/memory_evidence.py) | Target-applicable allocation contracts, scoped coverage and separate empirical observations |
| [`allocation_capture.py`](titan_sim/allocation_capture.py), [`allocation_contracts.py`](titan_sim/allocation_contracts.py) | Native allocator history/state capture, generations and conditional lifetime evidence |
| [`prediction_validation.py`](titan_sim/prediction_validation.py) | Frozen predictions and exact-scope native control comparison |
| [`memory.py`](titan_sim/memory.py) | Physical placement, allocation/lifecycle envelopes and fit verdicts |
| [`hardware.py`](titan_sim/hardware.py) | Explicit capacity/rate validation; specification assumptions |
| [`torchtitan.py`](titan_sim/torchtitan.py) | Native recipe loading and bounded trainer observation |
| [`profiler_capture.py`](titan_sim/profiler_capture.py) | Native profiler evidence and preliminary workload projection |
| [`torch_capture.py`](titan_sim/torch_capture.py) | Explicit fake/eager dispatch observation |
| [`core.py`](titan_sim/core.py) | Exact-key cost lookup, event scheduling and storage lifetimes |
| [`bundle.py`](titan_sim/bundle.py) | Optional Chakra transport |

## Scope and validation

The [native workflow validation record](docs/validation/native-workflow.md)
separates real GPU capture/measurement checks from held-out prediction accuracy
and lists the remaining unsupported scope.

Durations use seconds, sizes use bytes, and rates use FLOP/s or bytes/s. The JSON
schema is `recipe-prescreen-workload`, version 1. Hardware supports uniform capacity or explicit per-device capacities and placement. The simulator models declared exclusive resources and
communication rendezvous. Bandwidth sharing, compute/communication interference,
dynamic scheduling need extensions or explicit
approximations.

Run the tests with:

```sh
python -m unittest discover -s tests -v
```

The core suite works without PyTorch; native tests require it. The
[TorchTitan guide](docs/native-torchtitan.md#validation-and-compatibility) describes
the native integration's tested scope. The
[recipe experiments](examples/recipe_prescreen/README.md) retain narrower capture,
HybridEP and GB200 calibration evidence. A completed four-GB200 experiment measured
39 operation configurations; an independent two-GEMM chain differed from its
component-cost prediction by +3.2% eager and +4.0% with CUDA graph replay. These
results do not establish full-model accuracy or complete physical-memory coverage.

JSON is the default transport. Install `.[chakra]` only for the optional bundle
API; it does not provide kernel discovery or arbitrary Chakra-simulator
compatibility. Historical prototype interfaces, including `--case`, remain in
Git history and require their original pinned revisions.
