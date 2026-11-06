# Physical device memory assessment

`simulate` reports physical memory bounds under the declared placement and event
schedule. It can assess heterogeneous devices and multiple ranks sharing a device.
It does not discover allocations, measure allocator behavior, or verify that a
capture covers the full recipe. Missing coverage produces an inconclusive fit
assessment even when every observed tensor fits.

The public Python entry point is:

```python
from titan_sim.memory import assess_memory

assessment = assess_memory(nodes, storage, timing, hardware, manifest)
```

`nodes` and `storage` contain the existing `Node` and `Storage` objects. `timing`
maps each node ID, as an integer or JSON string, to `(start_seconds, end_seconds)`.
The normal `simulate` path supplies its scheduled event times automatically.

## Result and verdict

The simulation result's `memory_verdict` is the physical assessment. Its
`physical_memory` object contains the complete assessment, including `per_device`,
`memory_scope`, `memory_unresolved`, and `memory_assumptions`. The simulation
result also exposes `memory_unresolved` at the top level for summary consumers.

| Per-device field | Meaning |
|---|---|
| `ranks`, `placement_verified` | Ranks assigned by the explicit placement map; the flag means placement was declared, not independently measured |
| `capacity_bytes`, `budget_bytes` | Declared capacity and capacity after reserve headroom |
| `peak_lower_bytes` | Peak lower bound from the supplied allocation and lifecycle evidence |
| `peak_upper_bytes` | Physical upper bound; `null` when evidence cannot bound it |
| `modeled_peak_lower_bytes`, `modeled_peak_upper_bytes` | Peaks of the declared model, retaining unverified storage projections; these are not physical bounds when storage evidence is unverified |
| `scheduled_peak_lower_bytes`, `scheduled_peak_upper_bytes` | Scheduled allocation bounds before lifecycle peaks are considered |
| `peak_lower_time_seconds`, `peak_upper_time_seconds` | Modeled peak time, or `null` for a lifecycle peak or an unbounded declared allocation; a modeled upper time can remain finite when missing global coverage makes the physical upper bound unknown |
| `peak_lower_categories_bytes`, `peak_upper_categories_bytes` | Categories at the corresponding peak; category maxima at different times are not summed |
| `peak_lower_phase` | `schedule`, `initialization`, or `checkpoint_loading` |

The verdict uses each device's budget separately:

- `estimated_fit`: every device has a finite upper bound at or below its budget.
- `estimated_not_fit`: with explicit placement, at least one known lower bound
  exceeds its budget. This can be established despite missing upper coverage.
- `inconclusive`: neither condition is established, including a budget between
  lower and upper bounds, missing placement, or unknown allocation coverage.

An overflow verdict concerns the budget, which can be lower than physical
capacity. The default reserve fraction is 0.1; set it explicitly when a different
policy is intended. Reserve headroom reduces the budget and is separate from
modeled allocator slack.

The existing `modeled_peak_bytes`, `worst_rank`, `peak_time_seconds`, and
`peak_categories_bytes` remain logical **per-rank** storage reports.
`logical_memory_verdict` retains the corresponding uniform-capacity comparison;
it is inconclusive when hardware only declares individual device capacities.
These fields cannot substitute for the physical result, particularly when storage
is an unverified native projection. Likewise,
`status=calibrated_prediction` concerns declared workload coverage and operation
costs; it does not establish physical memory coverage. Inspect `memory_verdict`
and `physical_memory.memory_unresolved` even after a strict simulation succeeds.
`--require-memory-fit` requires an `estimated_fit` verdict, whose scope must still
be read from `physical_memory.memory_scope`.

## Placement and heterogeneous capacities

Device identifiers are user-defined physical placement keys. There are no
model-specific or GPU-family capacity defaults.

```json
{
  "hardware": {
    "id": "explicit-target",
    "reserve_fraction": 0.1,
    "devices": {
      "device-a": {"capacity_bytes": 80000000000},
      "device-b": {"capacity_bytes": 120000000000, "reserve_fraction": 0.05}
    }
  },
  "manifest": {
    "ranks": [0, 1, 2],
    "rank_placement": {"0": "device-a", "1": "device-a", "2": "device-b"}
  }
}
```

This is an illustrative capacity declaration, not a hardware specification or
measurement. Alternatively, `hardware.capacity_bytes` supplies uniform capacity
for all devices named by `rank_placement`. Sizes must be nonnegative integer byte
counts; fractions and times must be finite and valid.

Placement must map every declared rank exactly once. The normal scheduler uses
`gpu:<device>` as the default exclusive resource, so collocated ranks serialize
unless their nodes explicitly supply different resources. Explicit streams or
other resource assumptions are the producer's responsibility. Physical memory
uses the resulting overlap; it never adds independent per-rank peak values.

Without placement, uniform hardware produces provisional separate-rank device
reports. With named heterogeneous hardware, unassigned allocations appear under
`unplaced-rank:<rank>` with unknown capacity. Both cases remain inconclusive.
`placement_verified=false` exposes this limitation.

## Coverage is a producer attestation

Put the memory contract in `manifest.memory`. Its `coverage` object accepts only
these boolean keys:

```json
{
  "scope": "recipe",
  "storage_evidence": "unverified",
  "coverage": {
    "storage": false,
    "external_allocations": false,
    "workspaces": false,
    "communication_buffers": false,
    "graph_pools": false,
    "allocator_slack": false,
    "initialization": false,
    "checkpoint_loading": false
  },
  "allocations": [],
  "lifecycle_peaks": [],
  "unresolved": []
}
```

`memory.storage_evidence` separately classifies backing identities, sizes, and
lifetimes. It accepts `exact` or `unverified`; omitted values default to `exact`
for explicitly authored producer workloads. Older native profiler captures lacking
the field are recognized by their observer identity, including through retained
`source_manifests`, and default to `unverified`. An explicit `exact` promotion
retains precedence. This default is a producer assertion,
not automatic validation. Native profiler projections are marked `unverified`:
observed tensor extents and conservatively retained lifetimes do not prove either
actual concurrent allocation or a complete physical bound.

With `unverified` storage, its projected sizes remain in `modeled_peak_lower_bytes`
and `modeled_peak_upper_bytes`, but contribute zero to credible
`peak_lower_bytes`; `peak_upper_bytes` remains unknown. Thus a projected peak above
the budget cannot by itself prove `estimated_not_fit`. Independent, evidenced
external-allocation or lifecycle lower bounds can still establish overflow. An
external allocation referencing `storage_id` supplies a credible lower only over
its separately attested interval, without inheriting the projection's lifetime.

Setting coverage flags true does not promote unverified storage to exact. An
integration must establish the backing allocations and lifetimes and explicitly
record that promotion with supporting evidence during preparation. The memory
calculator consumes that assertion; it does not independently verify the evidence.
The returned `storage_evidence` field makes the selected classification visible.

A missing or false required flag leaves the physical upper bound unknown. A true
flag asserts that the category is fully represented or known not to occur in the
stated scope. It does not mean the simulator has independently inspected or
measured that category. Do not turn flags on simply because a trace contains no
such allocation. Preserve capture provenance, source/runtime identities,
measurement artifacts, and the reasoning supporting each completeness assertion
in the workload's evidence records. This module checks the allocation contract;
it does not authenticate that supporting evidence.

`manifest.unresolved` and `memory.unresolved` also prevent a finite physical upper
claim. An explicitly unbounded allocation affects its own device; missing global
coverage affects all devices. Additional measurements are needed to close those
gaps; timing calibration alone does not close them.

The default scope is `recipe`, requiring all eight flags. Explicit
`memory.scope="schedule"` requires only the first six. A scheduled-interval fit
**does not validate initialization or checkpoint loading**. Supplied lifecycle
peaks are still considered if present, even under schedule scope.

## Storage, aliases, and external allocations

Under an exact producer contract, a `Storage` is a unique backing allocation. It starts when its producer is
issued, or at time zero when there is no producer. It remains through the last
registered user, including its producer. Persistent storage, and storage with no
registered users, remains through the simulated schedule's end.

Prefer one storage record shared by tensor views. If an integration retained
separate records for one backing allocation, declare aliases explicitly:

```json
{
  "storage_aliases": {"gradient-view": "gradient-buffer"}
}
```

Both names must already exist in `storage`. Aliases must have the same byte size
and physical device; their lifetimes are unioned and counted once. Aliases may
cross ranks only when those ranks share a physical device. Cycles and mismatched
sizes/devices are rejected. Backing-allocation reuse across separate generations
must retain the correct lifetime records; names alone do not prove aliasing.

Additional `memory.allocations` entries cover memory outside those storage
records:

```json
[
  {
    "id": "operation-workspace",
    "rank": 0,
    "category": "workspaces",
    "lower_bytes": 1048576,
    "upper_bytes": 2097152,
    "start_seconds": 0.1,
    "end_seconds": 0.2
  },
  {
    "id": "allocator-unused-reservation",
    "device": "device-a",
    "category": "allocator_slack",
    "bytes": 4194304
  }
]
```

Each entry needs a unique `id`, exactly one of `rank` or `device`, and a category
from `external_allocations`, `workspaces`, `communication_buffers`, `graph_pools`,
or `allocator_slack`. Use either exact `bytes` or both `lower_bytes` and
`upper_bytes`. An explicit `upper_bytes: null` records an unknown upper bound.
Omitted start/end default to zero and the schedule's end. Explicit intervals may
also describe allocations outside that interval.

Allocated bytes are contained in allocator-reserved bytes. Model only the unused
reservation as `allocator_slack = reserved - allocated`; never add the full
reserved value to storage. The category `allocator_reserved` is rejected. A
reserve headroom policy does not establish this allocation coverage.

If an external allocation is already represented by a storage record, add
`"storage_id": "existing-storage-id"` and matching exact size/device. Its
interval extends that backing allocation's lifetime without counting it twice.
The storage's category remains the category used in the peak breakdown.

Ordinary intervals release storage before allocations beginning at the same
timestamp. Zero-duration events can leave boundary ordering ambiguous. The
reported lower and upper envelopes account conservatively for that ambiguity,
including zero-duration final users. Consequently, a single exact allocation
size does not always imply identical lower and upper peak values.

## Initialization and checkpoint-loading peaks

Lifecycle peaks are **whole-device totals**, including any storage, reservations,
and external allocations present during that phase. They are maxed against the
scheduled peak; they are never added to it.

```json
{
  "lifecycle_peaks": [
    {
      "phase": "initialization",
      "device": "device-a",
      "lower_bytes": 50000000000,
      "upper_bytes": 55000000000
    },
    {
      "phase": "checkpoint_loading",
      "device": "device-a",
      "lower_bytes": 60000000000,
      "upper_bytes": null
    }
  ]
}
```

The only phase names are `initialization` and `checkpoint_loading`. A `rank` can
replace `device` only when it is the sole rank on that device. Independent rank
peaks do not establish a combined collocated peak, so that form is rejected for
collocated ranks. Coverage flags still need to attest that the declared records
bound the relevant lifecycle scope on every device, including phases known not
to occur.

## Small overlap example

This CPU-only example supplies synthetic times and known sizes. Its coverage
flags are assertions for this constructed example, not evidence about a real
training recipe.

```python
from titan_sim.core import Node, Storage
from titan_sim.memory import ALLOCATION_CATEGORIES, assess_memory

nodes = [Node(0, "first", 0), Node(1, "second", 1)]
storage = [Storage("a", 0, 60, 0), Storage("b", 1, 70, 1)]
hardware = {"devices": {"device": {"capacity_bytes": 200}}, "reserve_fraction": 0}
manifest = {
    "ranks": [0, 1],
    "rank_placement": {"0": "device", "1": "device"},
    "memory": {
        "scope": "schedule",
        "coverage": dict.fromkeys(ALLOCATION_CATEGORIES, True),
    },
}

overlap = assess_memory(nodes, storage, {"0": (0, 2), "1": (1, 3)}, hardware, manifest)
separate = assess_memory(nodes, storage, {"0": (0, 2), "1": (2, 3)}, hardware, manifest)
assert overlap["per_device"]["device"]["peak_upper_bytes"] == 130
assert separate["per_device"]["device"]["peak_upper_bytes"] == 70
```

Both examples fit the declared scheduled-interval budget. Neither claims that
an actual model initializes, loads, or trains successfully on a GPU.
