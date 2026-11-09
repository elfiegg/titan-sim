# Command and integration reference

Use the [workflow](gpu-assisted.md) to run a prescreen and the
[architecture](architecture.md) to understand how its artifacts connect.
This page is a lookup reference for commands, extension interfaces and source files.
Run `python -m titan_sim COMMAND --help` for exact arguments.

## Commands

| Command | Required inputs | Output | Where it runs |
|---|---|---|---|
| `runtime-check` | Runtime manifest, capability evidence and expected identities | Readiness/gaps and artifact hashes | CPU |
| `export-plan` | Prepared native objects through a trusted adapter, or native recipe export configuration | Versioned schedule/graph plan | Original framework environment; observation may execute native steps |
| `lower-plan` | Native pipeline-group plans, hardware and explicit bindings | Workload with schedule and logical storage lifetimes | CPU |
| `capture` | Adapter reference and adapter-owned configuration | Workload JSON | Adapter's framework environment |
| `prepare` | Rank workloads and optional preparation evidence | Combined workload with explicit semantics/gaps | CPU |
| `recover-schedule` | Complete native peer captures with matching run identity | Supported collective completion and wait dependencies, with remaining gaps | CPU |
| `calibration-cases` | Native dispatch workload; actual placement for communication | Isolated measurement workload and optional eligible/skipped inventory | CPU |
| `window-cases` | Source workload, native provider and stream/dependency manifest | Exact-context native window measurement workload | CPU; measurement uses the existing GPU group |
| `calibrate` | Workload, measurement runner and runner-owned configuration | Cost catalog and unmatched-key report | Runner's measurement environment, usually GPUs |
| `bind-cases` | Exact source snapshots and native measurement workloads | Derived workload and occurrence-binding audit | CPU |
| `bind-window` | Exact source and a closed source-backed window workload | One measured atomic event with all participant resources | CPU |
| `allocation-evidence` | Raw native allocation capture, final workload and exact applicability | Observed generation audit and conditional replay evidence | CPU; capture uses the original runtime |
| `bind-memory` | Final workload, allocation evidence bundle and exact applicability | Workload with audited observations and predictive contracts | CPU |
| `validate` | Workload JSON | Structural counts and declared gaps | CPU |
| `simulate` | Workload, hardware assumptions and optional matching catalog | Timing, storage peaks and coverage | CPU |
| `freeze-prediction` | Workload, catalog and producer-recorded validation context | Immutable prediction identity and result | CPU |
| `validate-prediction` | Frozen prediction and trusted fresh-control runner | Exact-scope timing/memory comparison | Single coordinator; runner owns native execution |

Commands print JSON. `--output` writes the artifact; simulation stdout omits the
full event timeline retained in that file. Runtime/contract errors return exit
code 2. Native capture also returns 2 for a blocked or incomplete capture while
retaining its diagnostic workload. Argument errors use argparse's messages.

Durations use seconds, sizes use bytes, and rates use FLOP/s or bytes/s. The
workload JSON schema is `recipe-prescreen-workload`, version 1. Loading JSON for
validation or simulation does not import the recorded framework or execute its adapter.

## Python extension interfaces

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
| [`cli.py`](../titan_sim/cli.py) | Commands, artifact paths and result reporting |
| [`adapters.py`](../titan_sim/adapters.py) | Explicit function invocation and catalog validation |
| [`workload.py`](../titan_sim/workload.py) | Workload contract and JSON persistence |
| [`runtime.py`](../titan_sim/runtime.py) | Prepare-runtime artifact consistency and request identity checks |
| [`prepare.py`](../titan_sim/prepare.py) | Native evidence matching, rank merging and explicit semantics |
| [`native_plan.py`](../titan_sim/native_plan.py), [`plan_capture.py`](../titan_sim/plan_capture.py) | Native plan inspection, artifact validation and bounded preparation observation |
| [`plan_lowering.py`](../titan_sim/plan_lowering.py), [`plan_memory.py`](../titan_sim/plan_memory.py) | Native action dependencies and graph-backed logical storage lifetimes |
| [`native_runner.py`](../titan_sim/native_runner.py) | Trace observations and repeated native GPU measurements |
| [`operator_cases.py`](../titan_sim/operator_cases.py), [`communication_cases.py`](../titan_sim/communication_cases.py) | Eligible native overload cases and replay providers with exact metadata/runtime context |
| [`local_execution.py`](../titan_sim/local_execution.py), [`executable_regions.py`](../titan_sim/executable_regions.py) | Target-rank CUDA discovery, explicit communication substitution and live native callable retention |
| [`case_binding.py`](../titan_sim/case_binding.py), [`identity.py`](../titan_sim/identity.py) | Exact source snapshots, native measurement correspondence and occurrence audits |
| [`schedule_recovery.py`](../titan_sim/schedule_recovery.py) | Native peer collective reconciliation and exact functional wait lineage |
| [`execution_windows.py`](../titan_sim/execution_windows.py) | Native concurrent measurement and conservative closed-window binding |
| [`memory_observer.py`](../titan_sim/memory_observer.py) | Native process allocator peaks, sampled device use and state inventories |
| [`memory_evidence.py`](../titan_sim/memory_evidence.py) | Target-applicable allocation contracts, scoped coverage and separate empirical observations |
| [`allocation_capture.py`](../titan_sim/allocation_capture.py), [`allocation_contracts.py`](../titan_sim/allocation_contracts.py) | Native allocator history/state capture, generations and conditional lifetime evidence |
| [`prediction_validation.py`](../titan_sim/prediction_validation.py) | Frozen predictions and exact-scope native control comparison |
| [`memory.py`](../titan_sim/memory.py) | Physical placement, allocation/lifecycle envelopes and fit verdicts |
| [`hardware.py`](../titan_sim/hardware.py) | Explicit capacity/rate validation; specification assumptions |
| [`torchtitan.py`](../titan_sim/torchtitan.py) | Native recipe loading and bounded trainer observation |
| [`profiler_capture.py`](../titan_sim/profiler_capture.py) | Native profiler evidence and preliminary workload projection |
| [`torch_capture.py`](../titan_sim/torch_capture.py) | Explicit fake/eager dispatch observation |
| [`core.py`](../titan_sim/core.py) | Exact-key cost lookup, event scheduling and storage lifetimes |
| [`bundle.py`](../titan_sim/bundle.py) | Optional Chakra transport |

## Development and optional transport

Run `python -m unittest discover -s tests -v` for CPU changes. Core tests need no
PyTorch; native tests require a compatible pinned framework environment. See
[validation scope](validation/native-workflow.md) for the distinction between
component qualification and independent prediction accuracy.

JSON is the default transport. Install `.[chakra]` only for the optional bundle
API; it does not provide kernel discovery or arbitrary Chakra-simulator
compatibility. Historical interfaces, including `--case`, require their original
pinned revisions. Narrower experimental fixtures live under
[recipe experiments](../examples/recipe_prescreen/README.md).
