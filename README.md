# titan-sim

A workload-neutral interface for capture, calibration and CPU simulation of
execution time and per-device peak storage. Framework-specific code belongs in
explicit adapters; models and communication libraries are not CLI modes.

**Status:** generic adapter invocation, workload persistence, structural validation,
exact-key cost lookup and the event/storage simulator are implemented. Automatic
native TorchTitan recipe capture and compiled implementation matching remain
unfinished. The [recipe experiments](examples/recipe_prescreen/README.md) are
examples with narrower evidence, not built-in support for every workload.

## Motivation

Prescreening should help evaluate candidate training recipes before allocating a
large cluster. It needs to account for per-device placement and peak simultaneous
storage rather than aggregate capacity, and estimate step time under explicit
scheduling and communication assumptions.

Fake execution can help capture operations, dependencies and tensor metadata.
It does not measure GPU durations, achieved overlap, native workspaces or network
contention. Calibration supplies measured component costs; simulation applies
those costs to the workload's dependencies and resource constraints. Unsupported
operations and missing memory coverage must stay visible. Prescreening does not
establish measured performance or successful execution at scale.

The simulator should not care whether a workload comes from TorchTitan, another
framework, a synthetic generator or a manually constructed graph. Producers must
supply a compatible workload description. A generic interface does **not** mean
all frameworks or execution semantics are already supported.

## Basic structure: how agents use it

```text
Recipe/configuration --> Capture adapter --> Workload
                                               |
                               +---------------+---------------+
                               |                               |
                         Measurement runner               CPU simulator
                               |                               ^
                               +------ Measured catalog -------+
                                                               |
                                                 Timing, memory, coverage/gaps
```

The repository is code that agents install or invoke from a pinned checkout. It
is not a hosted service or a cluster allocation. Read [AGENTS.md](AGENTS.md) before
running a workload. Record the repository revision and upstream/container versions.

### Install

Python 3.10+; CPU simulation has no mandatory third-party dependencies:

```sh
git clone https://github.com/elfiegg/titan-sim.git
cd titan-sim
git rev-parse HEAD                  # record/pin the revision
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
titan-sim --help
python -m unittest discover -s tests -v
```

`python -m titan_sim` is equivalent to `titan-sim` and works from the checkout.
The simulator does not import PyTorch, TorchTitan, CUDA, model examples or protobuf.
Capture and measurement adapters run in their own required software environment.

## Commands available today

| Command | Input | Output | Runtime |
|---|---|---|---|
| `capture` | Explicit Python adapter and adapter-owned JSON config | Workload JSON | Whatever the adapter requires; native capture usually needs the pinned training environment |
| `validate` | Workload JSON | Structural counts and declared gaps | CPU Python |
| `calibrate` | Workload, explicit Python measurement runner and runner-owned JSON config | Cost catalog, missing cost keys and unkeyed operation count | Runner-dependent; actual GPU calibration needs authorized target hardware |
| `simulate` | Workload and optional matching cost catalog/hardware override | Timing, per-rank modeled peak storage and coverage | CPU Python |

There is no model selector or `--case` argument. Kimi and HybridEP were prototype
experiments; they have moved to [examples/recipe_prescreen](examples/recipe_prescreen).
Existing workloads remain readable; the old `--case` command syntax has changed.

### Runnable, framework-free example

From the repository checkout:

```sh
titan-sim capture --adapter examples.minimal:capture \
  --config examples/minimal.json --output results/minimal/workload.json

titan-sim validate results/minimal/workload.json

titan-sim simulate results/minimal/workload.json \
  --output results/minimal/result.json
```

This example generates two synthetic operations; it does not capture a real model.
It explicitly declares analytical rates and missing workspace coverage, so its
memory verdict is `inconclusive`. Adding `--strict` rejects it. Example modules
are available in the checkout and are not installed as part of the simulator wheel.

### Bring a framework adapter and measurement runner

The following are invocation templates: `my_project` refers to functions you
provide, not a bundled TorchTitan implementation.

```sh
titan-sim capture --adapter my_project.capture:build_workload \
  --config recipe.json --output results/workload.json

titan-sim calibrate results/workload.json \
  --runner my_project.measure:measure_workload \
  --config measurement.json --output results/costs.json

titan-sim simulate results/workload.json \
  --calibration results/costs.json --strict --output results/prediction.json
```

Adapters are explicit trusted Python imports (`module:function`), not executable
code embedded in workload JSON. Their modules must be installed or importable on
`PYTHONPATH`. Loading a workload for validation/simulation does not import its
recorded adapter or execute native model code.

A measurement runner owns representative input construction, compilation/warmup,
GPU timing, distributed coordination and result aggregation. The generic command
invokes it and validates the returned catalog; it does not discover kernels,
choose hardware, submit Slurm jobs or impose time limits. Invoke the catalog-writing
command once from the runner's coordinating process, not on every distributed rank
with the same output path. Use authorized allocations and explicit scheduler limits;
the initial recipe examples use five minutes.

### What agents prepare and report

Prepare the resolved recipe/scenario, upstream and software versions, a compatible
capture adapter, declared rank placement/schedule, hardware identity/capacity,
and a measurement runner or existing matched catalog. No default GB200 profile is
silently applied by the generic CLI. `simulate --hardware FILE` explicitly replaces
the embedded hardware description; a mismatched calibration is rejected.

Commands print JSON. `capture --output` writes workload JSON, `calibrate --output`
writes catalog JSON, and `simulate --output` writes the full result including event
times. The simulation summary printed to stdout omits the detailed event timeline.
Runtime/contract errors return exit code 2 with a JSON error on stderr; argument
syntax errors use argparse's ordinary error message.

Report `status`, `cost_coverage`, `unresolved`, worst-rank storage and
`memory_verdict` with any timing estimate. `validate` checks DAG/storage references;
it does not establish valid real execution, communication matching, calibrated
costs or capture completeness. Simulation additionally checks communication
rendezvous. `--strict` rejects declared gaps and uncalibrated costs, but does not
independently prove that the adapter captured every operation or allocation.

Allocated storage is included in reserved memory: never add them. A logical storage
peak is not automatically a CUDA allocator-reserved or physical device-wide peak.

## Design structure

### Stable component boundaries

| Component | Implementation | Input | Output / responsibility |
|---|---|---|---|
| Capture adapter | User/framework Python function | JSON config | `Workload`; owns native capture semantics and fidelity/provenance |
| Adapter invocation | `titan_sim.adapters` | Explicit function reference and config | Calls adapter; records reference/config digest; validates return contract |
| Workload contract | `titan_sim.workload.Workload`, `core.Node`, `core.Storage` | Producer records | Framework-independent Python structures |
| Persistence | `titan_sim.workload` | Python records or JSON | Versioned JSON serialization/reconstruction; no native execution |
| Measurement runner | User/framework Python function | `Workload` and JSON config | Catalog of measured costs, with samples and hardware identity |
| Catalog validation | `titan_sim.adapters.validate_catalog` | Catalog and expected hardware identity | Checks shapes of records, finite durations, sample checksums and hardware match |
| Cost resolver | `titan_sim.core.CostModel` | Operation signature, hardware and catalog | Matched measured cost, or explicitly labelled analytical/historical fallback |
| Simulator | `titan_sim.core.simulate` | Workload, resource assumptions and catalog | Event timing, per-rank storage peaks, verdict and cost coverage |
| Optional transport | `titan_sim.bundle` | Same records | Chakra files with custom attributes/sidecars; not kernel discovery |

### Adapter contracts

```python
from titan_sim.workload import Workload

# Capture adapter: supported by `titan-sim capture`.
def build_workload(config: dict) -> Workload:
    ...  # execute/instrument the intended framework or generate declared events

# Measurement runner: supported by `titan-sim calibrate`.
def measure_workload(workload: Workload, config: dict) -> dict:
    ...  # run original implementations and return records keyed by cost signature
```

The signatures are generic; there are no hardcoded recipe names or GPU counts.
The framework adapter/runner supplies the implementation-specific behavior.
A complete native TorchTitan implementation of these contracts is not yet bundled.

The workload contains:

```python
Workload(
    nodes=[...],       # Node occurrence IDs, operation names, ranks, deps, attributes
    storage=[...],     # unique backing allocation IDs, bytes, producer/consumers
    manifest={...},    # ranks, scope/provenance, useful tokens, unresolved gaps
    hardware={...},    # identity, capacity and explicit rate/resource assumptions
)
```

The direct Python simulator interface remains:

```python
result = simulate(*workload.arguments(), calibration=catalog, strict=True)
```

`Node` supports `compute`, `send`, `recv` and `collective` events. Operation names
are arbitrary; attributes provide cost keys, shape/layout information, resource
IDs and communication peers/participants. A cost key describes a calibration case;
a node ID identifies an occurrence. The producer must choose signatures that include
relevant implementation, precision, shape/layout, scalar settings, software and
communication context. Kernel names alone are insufficient.

`Storage` identifies backing allocations so aliases are not counted twice. It
records rank, byte size, producer, consumers, persistence and category. The engine
allocates at the declared producer and retains storage through all declared users.
A model-specific adapter must supply hidden workspaces and buffer lifetimes as
well as tensors if it wants to make complete memory predictions.

The manifest must list ranks and declared gaps; adapters should also record source
revision, container/software identity, capture adaptations and compilation mode.
The wrapper's config digest is not a substitute for that provenance.

### Calibration record

```text
catalog[cost_key] = {
    hardware_id: identity matching workload.hardware.id,
    seconds: representative measured duration,
    samples_seconds: nonempty list of raw durations,
    evidence_sha256: SHA256 of compact JSON encoding of that sample list,
    ... additional implementation/provenance/method metadata ...
}
```

The sample checksum detects inconsistency; it does not prove measurements were
collected correctly. Implementation compatibility and coverage remain the runner's
responsibility. An empty/partial catalog is allowed for collection; the command
reports unmatched keys and unkeyed nonsemantic nodes. Strict simulation rejects
unmeasured nonsemantic operation costs. Held-out validation data must not be used
as fitted component costs.

All durations use **seconds**; sizes use **bytes**; rates use FLOP/s or bytes/s.
The JSON schema is `recipe-prescreen-workload`, version 1. Current hardware capacity
is a uniform per-device value. The engine uses declared exclusive resources and
communication rendezvous; bandwidth sharing, realistic compute/communication
interference, heterogeneous capacities and dynamic scheduling need extensions or
explicit approximations. Unsupported semantics must be declared, not silently
claimed to be supported by the generic interface.

### Chakra is optional

The default path uses Python records and JSON. Install `.[chakra]` only for the
optional `titan_sim.bundle` API. Its schema/license/provenance are in
`titan_sim/vendor`. Custom attributes/sidecars require a compatible consumer;
arbitrary Chakra simulator compatibility is not asserted. Serialization does not
select kernels or reconstruct executable PyTorch code.

## Examples, evidence and next work

[Recipe-prescreen examples](examples/recipe_prescreen/README.md) retain the earlier
small FakeTensor/FPG capture, HybridEP fake-shape probe, generated Kimi schedule,
GB200 benchmark cases and held-out validation importer. Their model names and
four-/32-rank constraints are local to those experiments.

The completed four-GB200 experiment measured 39 operation configurations. An
independent two-GEMM chain differed from the component-cost prediction by +3.2%
eager and +4.0% CUDA graph replay. This is narrow composition evidence, not a general
accuracy claim. Full Kimi capture, automatic native implementation matching,
complete physical memory lifetimes, and full-scale overlap/contention remain gaps.

The next workload adapter should capture one native model block, retain the original
executable calibration regions, and validate predicted time and memory against an
independent run. Extend to distributed scheduling after that connection works.
A generic CLI is an interface milestone, not proof of full-workload support.

This revision supersedes the initial `--case` CLI. Earlier implementations and
experiments remain available in Git history; pin a commit when reproducing them.
