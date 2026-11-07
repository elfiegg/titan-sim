# Measure identified native collectives

The native dispatch observer records supported collective overloads, tensor
metadata and resolved process-group membership. `communication_cases` turns
those records into isolated measurement cases. It does not choose a new training
configuration, infer collective groups from kernel names, or extrapolate one
group size to another.

Supported overloads are native c10d base all-gather, base reduce-scatter and
single-tensor all-reduce, plus their `_c10d_functional` equivalents. Group and
reduction ScriptObjects are unboxed for observation and correctly boxed for
native replay. Unsupported operations remain in the skipped-case report.

## Select the communication that actually occurred

Start with a native dispatch workload (`observer: "dispatch"`). The observer
records communication metadata without waiting or creating process groups.
Failure to inspect a collective records a gap and lets the original call run.

```sh
python -m titan_sim calibration-cases results/native.rank0.json \
  --kind communication --world-size 8 --topology measured-placement.json \
  --peer-workload results/native.rank1.json --peer-workload results/native.rank2.json \
  --peer-workload results/native.rank3.json --peer-workload results/native.rank4.json \
  --peer-workload results/native.rank5.json --peer-workload results/native.rank6.json \
  --peer-workload results/native.rank7.json \
  --report results/communication-inventory.json \
  --output results/communication-workload.json
```

In Python, use `prepare_communication_cases(primary, peer_workloads=peers)` and
`communication_case_workload(primary, case_ids, topology=topology,
peer_workloads=peers)`. Supply artifacts from the **same capture run**. Available
capture IDs/job IDs and framework/source identities must agree. When a run ID
was not recorded, the case explicitly reports that same-run identity remains a
caller assertion; matching tensor signatures cannot prove it.

Native in-place all-gather is supported when each participant's input aliases
its own group-rank slice of the output buffer. Base reduce-scatter supports the
inverse layout. Every participant must supply matching shape, dtype, group and
alias evidence; rank-zero evidence alone is insufficient. The provider allocates
one shared base and reconstructs the local view at
`measurement_group_rank * slice_elements`, retaining literal original offsets,
storage IDs and per-rank provenance in the inventory. Other alias/offset patterns
remain unsupported. This does not reconstruct an all-rank training schedule.

Eight is an illustrative selection: use an actual identified group size. If no
captured group has that size, selection fails. Launch the measurement on exactly
that many participating GPUs. A case from a subgroup can run on a standalone
measurement group of the same size; measurement rank `i` corresponds to the
source participant at index `i`. This does not implement subgroup-only rounds
inside a larger measurement world.

The topology JSON records the actual intended placement: GPU identities, ranks,
nodes and relevant fabric information. Preserve allocation evidence. The same
object must be supplied as the provider configuration's `topology`; it is part
of the cost key. It is a caller assertion, not an automatic NVLink/fabric probe.
Actual measurement placement can differ from the original capture; the result
is scoped to the measured placement and does not prove original-topology timing.

Cases retain native schema, reduction, asynchronous mode, shapes, dtypes,
participants, runtime/source provenance, NCCL version and selected NCCL settings.
Repeated occurrences are audit records outside the cost identity. Native
`PREMUL_SUM` is preserved using its captured finite host scalar factor and the
original native reduction constructor; it is never replaced by a separate scale
or SUM operation. Device/tensor factors remain unsupported. Old captures that
lost an opaque reduction factor must be captured again; no factor is inferred
from the group size. Scalar tensors (`shape: []`) contain one element and are
supported for all-reduce; zero-length tensor dimensions are rejected. Additional
schema defaults, including `config=None`, are retained. Unsupported nondefault
extra arguments are rejected rather than silently dropped. Unknown
dtypes, sparse-index calls, DTensor metadata, unsupported aliases, noncontiguous
inputs and incompatible collective shapes are rejected. Payload fields are
logical input/output bytes, not inferred network bus traffic.

## Execute using the pinned GPU runtime

Use `titan_sim.communication_cases:region_provider` with the bundled execute
runner. The application initializes its NCCL world and assigns local CUDA devices
before calling the runner. Every rank participates in the same ordered cases;
only the coordinator writes the catalog.

```python
from titan_sim.native_runner import measure

config = {
    "mode": "execute",
    "hardware_id": workload.hardware["id"],
    "expected_device": expected_device,
    "region_provider": "titan_sim.communication_cases:region_provider",
    "provider_config": {"topology": measured_topology},
    "warmup": 5,
    "repeats": 20,
}
catalog = measure(workload, config)
```

The provider runs the captured native overload, joins native work completion onto
the timing stream and initializes/resets inputs to ones outside timing. All-rank
barriers also occur outside timing. Only the current case retains tensor buffers;
previous-case buffers are released before allocating the next. Preparation/reset
errors are shared across ranks before any timed collective starts, so one-rank
allocation failure cannot leave peers entering an unmatched call. Each sample uses the slowest participant's
CUDA-event duration. NCCL/runtime differences are rejected. CUDA-graph replay is
unsupported for these distributed cases.

These are isolated native communication measurements using synthetic values.
They do not establish training overlap, contention, transport equivalence across
allocations or full-model throughput. Keep the timing distribution and the
matching workload alongside the catalog. A second independent process execution
can test repeatability; it is not a held-out full-training accuracy result.

Central storage is unnecessary: the workload, inventory, samples, topology and
provenance are ordinary local artifacts. Do not relabel them as costs for a
different group size, precision, runtime or placement.
