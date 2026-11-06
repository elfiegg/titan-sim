# Native cost collection

The bundled runner `titan_sim.native_runner:measure` collects two distinct kinds
of evidence. It does not discover executable model regions or rebuild a model.
Use the original pinned training environment for execution and retain its source,
PyTorch, CUDA, extension and container identities.

| Mode | Input | Recorded measurement |
|---|---|---|
| `trace` | Prepared workload and retained native Chrome traces | GPU activity durations observed during instrumented training |
| `execute` | Prepared exact region descriptors and a trusted callable provider | Repeated CUDA-event timings of the supplied native implementations |

Both produce exact-key catalogs. Neither constitutes independent validation of a
full training prediction. Trace costs come from the captured execution itself;
executable-region costs still need a separate, unfitted full-recipe comparison.
The runner records `independent_validation: false` in both cases.

## Import retained GPU activity

First [prepare the rank-local captures](native-preparation.md). Preparation binds
GPU events to execution-trace operations by explicit correlation and records the
artifact checksum and event index. A host operation or execution-trace leaf does
not become a compiled kernel merely because it has a duration.

Create `trace-costs.json`, replacing the hardware identity with the one already
recorded in the workload:

```json
{
  "mode": "trace",
  "hardware_id": "your-captured-target",
  "accept_profiler_overhead": true
}
```

Run once in the coordinating process; reading retained traces needs no GPU:

```sh
python -m titan_sim calibrate prepared.json \
  --runner titan_sim.native_runner:measure \
  --config trace-costs.json --output costs.json
python -m titan_sim simulate prepared.json --calibration costs.json \
  --output prediction.json
```

The runner verifies file checksums, event categories, event names, launch fields,
exact implementation/context keys and finite GPU durations. Supported Chrome
categories are `kernel`, `gpu_memcpy` and `gpu_memset`. Repeated occurrences with
the same exact key contribute raw samples; their representative cost is the
median. The same retained event cannot be charged twice. `displayTimeUnit` is a
viewer setting; Chrome event durations are converted from microseconds to seconds.

Unmatched host projections stay absent from the partial catalog. Inspect
`missing_cost_keys` and `unkeyed_nonsemantic_nodes` in calibration output. No
matched GPU events is an error. A workload containing explicit executable-region
bindings requires execute mode for those bindings; trace mode does not fabricate
samples for them. Preparation gaps remain unresolved after collecting costs.

Trace catalog records reference the input workload's full provenance through
`provenance.source_provenance_location` (`input workload /manifest/provenance`)
and `provenance.source_provenance_sha256`. Retain that input workload unchanged
alongside the catalog and raw trace artifacts. The SHA256 is computed once from
`json.dumps(workload.manifest.get("provenance", {}), sort_keys=True,
separators=(",", ":"), allow_nan=False).encode()`. Records may also include short
PyTorch/CUDA version strings. They do not repeat the full recipe configuration
per GPU occurrence. Sample checksums and individual trace artifact references
remain separate. Trace parsing retains one artifact tree at a time; memory also
includes the prepared workload, compact catalog and largest parsed trace.

If artifacts moved, add `artifact_paths`, a mapping from their full SHA256 strings
to their current paths. Checksums remain mandatory. Trace hardware identity is
based on the captured workload and the caller's target attestation; retained
`deviceProperties` remain in the original trace. Where present, each sample audit
references them with `device_properties_location: "/deviceProperties"` and
`device_properties_sha256`, using the same canonical JSON hash algorithm as
source provenance. Full device property objects are not repeated per sample.
Importing traces does not
independently query the original GPUs. Profiler overhead and training contention
remain part of these observations.

## Chrome-only capture when execution traces are incompatible

Set `"execution_trace": false` explicitly in the native profiler capture
configuration to omit PyTorch's `ExecutionTraceObserver`. The default remains
`true`; there is no automatic retry or silent mode change. This mode preserves
the native recipe and its activation rematerialization while collecting Kineto
CPU/CUDA activity with shape and memory profiling enabled. It avoids ET tensor
storage access, which can be incompatible with placeholder/rematerialized tensors.
It does not establish that every native profiler/backend combination is supported.

Each completed GPU activity becomes an occurrence-only binding. Its exact context
includes the retained artifact SHA256, event index, rank, step, source/runtime
provenance digest and hardware identity, with `occurrence_only: true`. Even equal
kernel names receive distinct keys across occurrences. These costs describe that
captured activity; they cannot be reused as a general shape/scalar calibration.
`prepare` preserves verified direct bindings without lowering them a second time.

This mode has no ET tensor allocation/lifetime projection. Source-to-kernel
matching, physical storage coverage and cross-stream/communication dependencies
remain explicit gaps. Both profiler modes mark projected storage evidence as
unverified; collecting a complete timing catalog cannot turn it into physical
memory-fit evidence.

## Execute an exact native region

An executable node carries this binding in its attributes:

```python
binding = {
    "kind": "executable_region",
    "implementation": implementation,
    "context": context,
}
cost_key = native_cost_key(implementation, context)
```

Import `native_cost_key` from `titan_sim.native_runner` or `titan_sim.prepare`.
The key hashes compact, sorted JSON of both dictionaries. Include every setting
that can change the implementation or its cost: code/build identity, source and
runtime versions, shapes, strides, precision and scalar arguments, compilation,
graph mode, hardware, and communication group/placement where relevant. Changing
these dictionaries changes the key. A checksum or matching descriptor verifies
identity consistency; the trusted provider remains responsible for faithfully
executing that implementation with representative inputs.

`region_provider(workload, provider_config)` returns:

```python
{
    "provenance": {"source_revision": "pinned-original-source"},
    "regions": [{
        "cost_key": cost_key,
        "implementation": implementation,
        "context": context,
        "run": zero_argument_callable,
        "stream_semantics": "current_stream",
        "before_sample": optional_zero_argument_reset,
        "supports_cuda_graph": False,
    }],
}
```

Supply exactly one region for each distinct requested key on the local rank.
The provider owns compilation, original executable selection, tensor allocation,
state, input values and cleanup. `run()` must enqueue work on the current CUDA
stream, or join every auxiliary stream back to it before returning. A callable
that leaves unjoined work on other streams violates the timing contract.
`before_sample()` resets state outside timing and also runs before each warmup.
Avoid destructive repeated training updates unless they are part of the intended
case. A provider exception during collective execution can still require the
launcher to terminate all ranks.

The runner checks actual device name and memory against `expected_device`; it
also accepts `compute_capability: [major, minor]` there. These properties are a
hardware consistency check, not a topology or performance qualification. Defaults
are five warmup calls and twenty measured repetitions. At least one warmup and
two repetitions are required. CUDA events time completion; Python wall time is
not used as GPU duration.

### Runnable matrix multiplication fixture

This small fixture exercises the provider interface using native `torch.mm`.
It is a validation fixture, not a model default, a training model or an automatic
substitute for a compiled TorchTitan kernel. Save it as `mm_fixture.py` in your
working directory, with titan-sim importable in the existing CUDA environment:

```python
import hashlib
import json
from pathlib import Path
import torch
from titan_sim.core import Node
from titan_sim.native_runner import native_cost_key
from titan_sim.workload import Workload, write_workload


def descriptor(graph=False):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    hardware_id = f"mm-validation:{props.name}:{props.major}.{props.minor}:{props.total_memory}"
    implementation = {
        "callable": "torch.mm(out=output)",
        "fixture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    context = {
        "hardware_id": hardware_id,
        "shape": [128, 128, 128], "dtype": "float32",
        "input_strides": [[128, 1], [128, 1]],
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_graph": graph,
    }
    return implementation, context


def provide(workload, config):
    implementation, context = descriptor(bool(config.get("cuda_graph", False)))
    key = native_cost_key(implementation, context)
    requested = {n.attrs.get("cost_key") for n in workload.nodes}
    if requested != {key}:
        raise ValueError("fixture source/runtime/context differs from prepared case")
    a = torch.ones((128, 128), device="cuda", dtype=torch.float32)
    b = torch.ones_like(a)
    out = torch.empty_like(a)
    def run():
        torch.mm(a, b, out=out)
    return {
        "provenance": {"fixture": "native torch.mm; not a model benchmark"},
        "regions": [{"cost_key": key, "implementation": implementation,
                     "context": context, "run": run,
                     "stream_semantics": "current_stream", "supports_cuda_graph": True}],
    }


if __name__ == "__main__":
    implementation, context = descriptor()
    key = native_cost_key(implementation, context)
    props = torch.cuda.get_device_properties(0)
    binding = dict(kind="executable_region", implementation=implementation, context=context)
    workload = Workload(
        [Node(0, "native torch.mm fixture", 0, attrs=dict(cost_key=key, native_measurement=binding))],
        [], dict(ranks=[0], world_size=1,
                 unresolved=["Fixture does not model complete physical memory or training."]),
        dict(id=context["hardware_id"], capacity_bytes=props.total_memory),
    )
    write_workload("mm-workload.json", workload)
    Path("mm-measure.json").write_text(json.dumps({
        "mode": "execute", "hardware_id": workload.hardware["id"],
        "region_provider": "mm_fixture:provide", "device": "cuda:0",
        "expected_device": {"name": props.name, "total_memory": props.total_memory},
        "warmup": 5, "repeats": 20, "cuda_graph": False,
    }))
```

Inside an authorized GPU allocation, run:

```sh
python mm_fixture.py
python -m titan_sim calibrate mm-workload.json --config mm-measure.json \
  --output mm-costs.json
```

The catalog contains actual timings only if these commands execute successfully
on CUDA. The fixture documentation and CPU unit tests themselves provide no GPU
measurement evidence. The fixture deliberately retains a memory/training gap.

For single-rank graph replay, set `context.cuda_graph` when generating the
workload, set configuration `cuda_graph: true`, and pass
`provider_config: {"cuda_graph": true}`. Regenerate the workload so its exact key
changes. Graph mode requires `supports_cuda_graph: true`, stable captured storage,
no `before_sample`, and graph-safe native operations. Warmup and graph capture are
outside the measured replay samples. Distributed CUDA graph execution is rejected.

## Distributed executable rounds

The runner uses an **already initialized** process group and never creates one.
Each process receives the same aggregate workload containing every rank. Providers
return only that process's local keys. Every rank must supply the same ordered
`round_id` sequence and identical JSON `round_context` dictionaries, including
`participants: [0, ..., world_size-1]`. Include the operation, group, shape, dtype
and reduction settings in that context. The per-rank implementation/context
still has to match its workload key exactly.

For example, corresponding local descriptors may add:

```python
region["round_id"] = "all_reduce-shape-128-fp32"
region["round_context"] = {
    "participants": list(range(torch.distributed.get_world_size())),
    "operation": "all_reduce", "group": "world", "shape": [128],
    "dtype": "float32", "reduction": "sum",
}
```

The provider supplies the actual native collective callable and representative
buffers. Each round must involve all ranks; subgroup-only rounds are rejected.
Barriers run outside each warmup and timed sample. After measurement, the runner
collects each rank's sample vector and uses the maximum duration at each sample
index for that round. It retains the per-rank vectors. Only the designated
coordinator receives the merged catalog; other ranks receive `{}`.

A minimal explicit entry wrapper, saved as `distributed_measure.py`, initializes
the process group and suppresses competing output writes:

```python
import datetime
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from titan_sim.native_runner import measure
from titan_sim.workload import read_workload

workload_path, config_path, output_path = sys.argv[1:]
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
try:
    workload = read_workload(workload_path)
    config = json.loads(Path(config_path).read_text())
    catalog = measure(workload, config)
    if dist.get_rank() == config.get("coordinator_rank", 0):
        Path(output_path).write_text(json.dumps(catalog, allow_nan=False) + "\n")
finally:
    dist.destroy_process_group()
```

Use the allocation's ordinary launcher and substitute its authorized process
count, rendezvous options and a real provider configuration:

```sh
torchrun --nproc-per-node=NUM_LOCAL_GPUS distributed_measure.py \
  prepared.json execute-costs.json costs.json
```

Within a working process group, the runner shares preflight and provider contract
errors before measurement barriers, and checks agreement on warmup/sample counts,
mode, provider and coordinator. This cannot recover a failed CUDA context, broken
process-group transport, a process that never entered the runner, or a hanging
native callable. Preserve launcher failure propagation and the explicit process
group timeout. Initial benchmarks also need an external five-minute scheduler
hard limit unless a different limit was authorized; neither this wrapper nor the
measurement runner enforces the scheduler deadline.

Read `status`, `cost_coverage`, `unresolved` and `memory_verdict` together after
simulation. Complete measured-key coverage does not remove missing dependencies,
physical-memory evidence, initialization peaks or independent validation work.
