# Bind target-applicable memory evidence

`titan_sim.memory_evidence` stores native observations and explicit producer
contracts in a versioned local JSON bundle. `bind_memory_evidence` attaches the
bundle to a copy of `Workload.manifest.memory`; normal `assess_memory` and
`simulate` consume it. Binding and each assessment verify the bundle checksum,
full original workload fingerprint, and requested applicability. The original
workload, storage and unresolved gaps remain intact.

The fingerprint uses `titan_sim.identity.workload_fingerprint`: serialized node
and storage dataclasses, manifest and hardware, encoded as sorted compact JSON
with nonfinite numbers rejected. Changing source nodes, placement, configuration,
hardware, memory assumptions or existing gaps invalidates the binding. Checksums
detect stale or inconsistent artifacts; they do not authenticate a producer or
prove a completeness statement.

## Applicability and contracts

The caller supplies these six explicit applicability fields:

```json
{
  "source": {"commit": "pinned-source", "recipe_sha256": "retained-recipe-digest"},
  "target": {"hardware_id": "declared-target", "configuration_sha256": "retained-config-digest"},
  "runtime": {"environment_sha256": "pinned-runtime-digest"},
  "placement": {"0": "physical-device-a"},
  "phases": ["schedule", "initialization", "checkpoint_loading"],
  "implementation": {"graph_sha256": "retained-native-graph-digest"}
}
```

These are illustrative identifiers, not measured evidence. Placement must equal
the workload's physical rank placement. Phases must include `schedule`; recipe
scope additionally requires `initialization` and `checkpoint_loading`. Every
field must be nonempty. All six fields compare exactly with the caller's binding
context. The source/target/runtime/implementation objects are caller-defined
identities, not a fuzzy compatibility policy; the full workload hash separately
binds every workload field. No scaling from another configuration is inferred.

`evidence_refs` contains `{id, sha256}` records identifying retained local
artifacts. A `producer_contracts` record requires `id`, `producer`, `statement`,
`evidence_ref`, and `applicability_sha256` computed with `identity.digest` over
the complete applicability object. Every allocation and coverage record must
reference both a contract and evidence. Referenced artifact bytes are retained
and audited by the producer; this module validates reference identity and digest
format, and does not open arbitrary filesystem paths.

```python
from titan_sim.memory_evidence import build_evidence_bundle, bind_memory_evidence

bundle = build_evidence_bundle(
    workload, applicability=context,
    producer_contracts=contracts, evidence_refs=references,
    allocations=allocation_contracts, coverage=coverage_statements,
    observations=native_observations,
)
bound_workload = bind_memory_evidence(workload, bundle, applicability=context)
```

A previously bound workload cannot be rebound. Start from the original workload
when extending evidence, so the new bundle has a single unambiguous source.

## Allocation identity and lifetime

An allocation contract has this form:

```json
{
  "allocation_id": "native-optimizer-storage",
  "generation": 0,
  "device": "physical-device-a",
  "phase": "schedule",
  "category": "storage",
  "kind": "exact",
  "lower_bytes": 4096,
  "upper_bytes": 4096,
  "creation_node_id": 10,
  "completion_node_ids": [14, 17],
  "persistent": false,
  "storage_ids": ["existing-storage", "existing-view"],
  "contract_id": "native-allocation-contract",
  "evidence_ref": "retained-allocation-evidence"
}
```

An exact contract has equal finite byte bounds. A `conditional` contract may
have unequal bounds or explicit `null` unknowns and must list its `conditions`.
Conditional assumptions remain visible in the result as
`conditional_assumptions`, `memory_bound_kind="conditional"`,
`conditional_memory_verdict`, and `conditional_per_device`. When any conditions
are unverified, the top-level `memory_verdict` is `inconclusive` and affected
devices have unknown unconditional upper bounds. Conditional allocations do not
contribute to unconditional credible lower bounds, so they cannot prove budget
overflow or physical-capacity OOM either. `--require-memory-fit` cannot accept a
conditional forecast as an unconditional fit. No condition is experimentally
verified by this binder. Unknown lower or upper bounds never become a claimed zero.

`allocation_id` plus `generation` identifies one allocation instance. Reuse of
an address requires another generation; a snapshot's storage IDs are not
allocation generations. IDs in `storage_ids` must already exist, have identical
exact sizes and physical placement, and identify only one generation. They are
aliases of a single backing allocation, counted once. Existing aliases cannot
be split into separate generations. An empty list represents an additional
allocation outside projected storage. Existing `manifest.memory.allocations`
are preserved; do not repeat them as new evidence allocations. Identifiable ID
collisions are rejected. This API does not yet merge an additional allocation
contract into a preexisting nonstorage allocation. Categories follow the existing memory
contract: storage, external allocations, workspaces, communication buffers,
graph pools and allocator slack. Reserved memory must not be added to allocated
memory; allocator slack means reserved minus allocated.

Creation uses the specified node's issue time, or schedule start for explicit
`null`. Release waits for **all completion nodes**, including asynchronous device
completion, and at least creation completion. Existing storage producers and
consumers remain lifetime constraints. Persistent allocations remain through the
schedule. Missing completion on a nonpersistent allocation leaves its retention upper
bound open through the schedule and records an unresolved gap. Its credible
lower bound covers only guaranteed creation completion and any preexisting
storage users; unknown retention cannot prove overlap with later allocations. A zero-duration completion keeps a
conservative upper envelope at the boundary; it cannot establish reuse order.

Exact evidence promotes an unverified projected allocation only when its
contract covers the entire existing projected lifetime. A narrower attestation
can provide a lower bound over its own interval without promoting the longer
projection. All existing capture, schedule, optimizer and memory gaps remain
blockers even when allocation contracts are present.

Optional `phase_bounds` entries provide whole-device initialization or
checkpoint-loading envelopes. Each requires `device`, `phase`,
`scope="whole_device"`, `kind="exact"`, finite ordered `lower_bytes` and
`upper_bytes`, `contract_id` and `evidence_ref`. Here `exact` identifies an unconditioned producer bound for this exact context;
it does not require equal phase lower/upper bounds. These are explicit producer
bounds, not measured maxima. They are maxed against scheduled peaks, never added
to them. Independent rank peaks do not establish a collocated device bound.

## Scoped coverage and verdicts

Each `coverage` row identifies `device`, `phase`, `category`, boolean `complete`,
`contract_id`, `evidence_ref`, and a completeness `statement`. Allocation
categories require an explicit statement for **every requested phase on every
device**. Initialization/checkpoint-loading category flags additionally require
their own named phase. A statement may assert that a category is bounded by the
represented records or cannot occur; absence from a trace proves neither.

`coverage_audit` shows each required device/category/phase cell. Missing or false
cells keep that device's physical upper bound unknown.
`compatibility_coverage` derives the old category booleans only when all required
cells are complete. A generic global boolean cannot close partial device or
phase coverage. Existing global gaps continue to affect every device.

`memory_verdict` retains the existing budget-based values. Per-device
`budget_outcome` distinguishes `overflow`, `within_budget` and `unknown`.
`physical_capacity_outcome` separately reports `oom_lower_bound`,
`within_capacity` or `unknown`. Exceeding headroom policy while staying within
physical capacity is a budget overflow, not proof of a physical-capacity OOM.
These are model/contract verdicts, not observations of a CUDA allocation failure.

## Capture adapters and native optimizer state

`observation_from_report(report)` consumes the existing `MemoryObserver.report()`
JSON and verifies its provenance digest. It retains native state inventories,
component/window phase peaks, samples, runtime and unresolved limitations under
`kind="observation"`, with a report digest. Its physical upper bound is always
`null`, and `supports_unseen_configuration_fit` is always false. Binding a report
never inserts its measured maximum into predicted allocations or coverage.

`inspect_native_state(models=[...], optimizers=[...])` inspects the original
native objects without initializing or replacing optimizer state. It recursively
visits parameters, gradients, buffers, and optimizer state, and deduplicates live
local backing storage across categories and tensor views. DTensor inspection
uses `to_local()`; CPU/offloaded allocations retain their device labels. Lazy
optimizer state appears only after the native optimizer has created it. Unknown
state objects and meta storage remain unresolved. No fixed parameter multiplier,
model branch or synthetic optimizer is used.

The returned inventory is an observation valid only at that sample. Its local
IDs contain no raw address and do not identify lifetimes or allocation
generations. It is not additive to allocator totals and cannot fill missing
optimizer-state contracts in a native plan by itself.

CPU tests exercise stale hashes, applicability mismatch, partial devices/phases,
observation-only inconclusiveness, storage aliases/generations, asynchronous and
zero-duration completion, retained global gaps, lifecycle bounds, and budget vs
capacity outcomes. A CPU PyTorch test inspects actual Adam state before/after its
native first update. These tests do not establish full-model GPU memory bounds.

## Producing native allocator evidence

`NativeAllocationCapture` in `titan_sim.allocation_capture` reads actual native
state and CUDA allocator `_snapshot()` data. It does not synchronize CUDA,
initialize lazy optimizer state, retain tensors, replace native calls, or change
compiler/allocator policy. Its optional history recorder changes diagnostic
recording only. Use it in a fresh, dedicated observation process:

```python
from titan_sim.allocation_capture import NativeAllocationCapture
from titan_sim.allocation_contracts import (
    analyze_allocation_capture, produce_allocation_evidence,
)

capture = NativeAllocationCapture(
    device_index=local_device, physical_device=placement[str(rank)], rank=rank,
    provenance={**provenance, 'applicability': context},
)
with capture.record_history(max_entries=100000, dedicated_process=True):
    with capture.region(native_node_id, models=model_parts, optimizers=optimizers):
        original_native_call()
raw = capture.report()
audit = analyze_allocation_capture(raw)
bundle = produce_allocation_evidence(workload, raw, applicability=context)
```

`dedicated_process=True` attests fresh recorder ownership. The implementation
checks for existing history and rejects it before enabling its recorder. Native
APIs cannot reliably inspect/restore every prior recorder setting; recording
therefore remains enabled until the dedicated process exits. Do not use this
mode inside somebody else's profiler session. The bounded recorder requests all
allocation/free actions. Entry and exit snapshots are automatic; a `region`
records boundaries around the original call, including an exit sample on error.
No implicit terminal synchronization is performed. A validation harness may use
an explicit, separately identified terminal synchronization boundary.

Raw evidence includes addresses, native stack/source metadata and state paths;
retain it locally with the other private artifacts. Public examples and tests
must use synthetic identifiers. Compact analysis replaces allocator addresses
with local allocation IDs and generations. The analysis still contains local
state evidence and is not a public export sanitizer.

The CPU analyzer reconstructs the native allocator's allocation, free request
and completed-free events. A free request does **not** release its backing block:
`active_awaiting_free` blocks remain charged until `free_completed`, and address
reuse starts a new generation only after that event. It reconciles each history
prefix with the next snapshot's live block sizes and pending-free states. Lost
prefixes, bounded-history saturation, unknown actions, unavailable snapshots,
incomplete execution and native state outside the allocator remain explicit
issues. Imported history with unknown recorder settings cannot establish that
transient allocations were recorded.

Newer native snapshots call pending stream retention `active_pending_free`; it
has the same requirement for an actual completed-free event. Expandable allocator
`segment_map`/`segment_unmap` events change physical mapped reservations, which are
reconciled with the snapshot's segment `total_size`. The much larger
`expandable_reservation_size` is a virtual address reservation and is never
charged as physical GPU memory. These mappings do not create tensor allocations
and do not release active blocks by themselves.

`recorded_allocator_coverage` describes only this recorded process allocator
domain and observed phases. It is separate from predictive physical coverage.
The analyzer reports actual sampled state bytes, including alias deduplication,
CPU/offloaded storage and newly created optimizer state. A complete history can
establish the recorded active-block peak. Neither result bounds an unseen native
invocation: mutable state, lazy optimizer creation, data-dependent control flow,
external CUDA allocators, communication/library workspaces, graph/private-pool
growth, allocator reservations, and unobserved lifecycle phases remain open.
An allocation surviving the last snapshot is retained **at that boundary**; it
is not proven permanent.

The evidence producer derives allocation sizes and native generations from the
raw records, and derives release boundaries from actual `free_completed` events.
Its capture provenance must contain the exact requested `applicability` context;
an observation from another source, target, runtime or implementation is rejected.
It emits conditional replay envelopes, with explicit repetition and boundary
correspondence conditions. Missing node correspondence keeps completion unknown.
Coarse region lifetimes provide upper envelopes only, with zero lower bytes:
allocations freed and reused inside one region need not coexist. An allocation
present at capture entry and retained through the last snapshot can retain its
known byte lower bound under the explicit replay condition.
All predictive physical coverage rows remain false, and no sampled maximum
becomes an unconditional bound. Thus this producer cannot establish unconditional
memory fit, including when its recorded allocator audit is complete. It does not
close the general native physical-bound production gap.

Existing workload storage must have native identity correspondence before any
allocation envelopes are added. Optional `storage_bindings` maps each local
workload storage ID to `{sample_index, state_id}` in the native state inventory.
The producer checks actual backing bytes and allocation generation, counts views
once, and represents allocator rounding separately from backing storage. Equal-byte
alias bindings currently require native storage present at entry and retained at
exit: projected transient storage needs finer lifetime correspondence and remains
observation-only. Missing correspondence retains observations only; it does not add
allocator totals to projected storage. Multi-phase captures remain observations because this adapter
does not yet map initialization/checkpoint lifetimes into scheduled nodes.
Existing additional memory allocations also require an identity merge that this
producer does not implement; such workloads retain observations without adding
potentially duplicate allocations.
