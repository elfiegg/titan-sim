# titan-sim

titan-sim is an experimental tool for prescreening distributed training recipes.
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

A **capture adapter** connects these descriptions. It observes the framework or
constructs a workload explicitly. Framework-specific preparation then resolves
the execution and storage details needed by the model. A **measurement runner**
benchmarks the corresponding implementations on the target hardware. The
**simulator** uses the resulting costs and workload to calculate event times and
per-rank storage peaks.

```text
Training recipe + original framework
                   |
             Capture adapter
                   |
        Captured workload + evidence
                   |
      Resolve execution/storage semantics
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
| Capture adapter | What does this recipe execute, and what evidence can we observe? | Workload records and framework-specific evidence |
| Workload preparation | How do those observations map to executable regions, distributed dependencies and storage lifetimes? | A workload with explicit semantics and remaining gaps |
| Measurement runner | How long do these implementations take on the target hardware? | A catalog of measured durations |
| CPU simulator | What happens when those operations follow this schedule and share these declared resources? | Event timing, per-rank peak storage and coverage |
| Independent validation | How well did the prediction match a separate real run? | Prediction errors and evidence about model limitations |

The simulator is independent of the training framework. It reads data; it does not
import a model or discover its kernels. Adapters and runners are explicit Python
functions executed in the environments their frameworks require. This separation
lets different frameworks use the same simulator without copying their model
implementations into it.

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
| Native TorchTitan capture | Experimental adapter that observes original training steps and preserves raw traces |
| Native trace-to-compiled-region matching and distributed workload preparation | Requires integration work; not automatic |
| General native TorchTitan GPU measurement runner | Not bundled; provide a runner or a compatible existing catalog |
| Complete physical-memory modeling and full-model GPU/distributed validation | Incomplete |

You can run the framework-free example below, simulate an existing compatible
workload, or develop a capture/calibration integration. For TorchTitan, start with
the [native integration guide](docs/native-torchtitan.md). Its bundled capture path
does not yet provide an automatic recipe-to-prediction pipeline.

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

The CLI exposes four operations:

| Command | Required inputs | Output | Where it runs |
|---|---|---|---|
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

The runner performs compilation, warmup, GPU timing and distributed coordination.
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
| `memory_verdict` | Fit assessment against the hardware capacity and reserve; unresolved coverage makes it inconclusive |
| `cost_coverage` | Which costs came from exact measured keys, analytical assumptions or historical proxies |
| `unresolved` | Limitations declared by the producer or integration |
| `status` | Whether the result used complete declared coverage and calibrated costs, or remains exploratory |

`validate` checks graph and storage references. It does not establish real
execution, complete capture or memory sufficiency. `simulate` additionally checks
communication rendezvous. `--strict` rejects declared gaps and uncalibrated
nonsemantic operation costs; it cannot discover every omission in a producer's
model. A `calibrated_prediction` is still a prediction, not a successful training
run.

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
| [`torchtitan.py`](titan_sim/torchtitan.py) | Native recipe loading and bounded trainer observation |
| [`profiler_capture.py`](titan_sim/profiler_capture.py) | Native profiler evidence and preliminary workload projection |
| [`torch_capture.py`](titan_sim/torch_capture.py) | Explicit fake/eager dispatch observation |
| [`core.py`](titan_sim/core.py) | Exact-key cost lookup, event scheduling and storage lifetimes |
| [`bundle.py`](titan_sim/bundle.py) | Optional Chakra transport |

## Scope and validation

Durations use seconds, sizes use bytes, and rates use FLOP/s or bytes/s. The JSON
schema is `recipe-prescreen-workload`, version 1. Hardware capacity is currently
uniform per device. The simulator models declared exclusive resources and
communication rendezvous. Bandwidth sharing, compute/communication interference,
heterogeneous capacities and dynamic scheduling need extensions or explicit
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
