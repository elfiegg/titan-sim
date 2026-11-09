# Native schedule recovery

`titan_sim.schedule_recovery.recover_schedule(workloads)` accepts original
rank-local `Workload` objects or workload JSON paths and returns a new `Workload`.
It runs on CPU without importing PyTorch or executing a recipe. The input
objects remain unchanged. The returned `manifest.schedule_recovery` contains
source workload fingerprints, candidate and recovered rendezvous, source node
references, native wait references, and rejected evidence. Remaining limitations
also appear in `manifest.unresolved` and continue to block strict simulation.

```python
from titan_sim.schedule_recovery import recover_schedule
from titan_sim.workload import write_workload

workload = recover_schedule(["rank0.json", "rank1.json"])
write_workload("reconciled.json", workload)
```

## What is recovered

The current implementation consumes complete dispatch captures with resolved
native communication descriptors. Every declared world rank must be supplied;
each process group must have the same number of collective occurrences on every
participant. All captures must share an explicit `capture_run_id`, recorded in
the manifest or provenance. A scheduler job ID alone does not prove that rank
artifacts came from the same execution. Runtime, recipe and source identities
are checked by the existing preparation and communication evidence validators.

Collectives are paired by process-group identity and occurrence order proven by
the captured dependency graph. Node numbers, list position, wall timestamps and
shape similarity are not ordering evidence. Any pair of consecutive group
occurrences without an explicit dependency path makes that group's ordering
ambiguous. Peer operation, native schema, tensor layout and dtype, reduction,
group membership, runtime context, and captured step must agree. Uniform native
all-reduce, gather, reduce-scatter and all-to-all descriptors are supported.
Uneven peer shards and nonuniform all-to-all splits remain unsupported.

For native `_c10d_functional` operations, successful `wait_tensor` calls must
match the exact collective output storage generation, tensor layout and dtype
on every participant. The collective must precede the wait in the dependency
graph. An output use preceding the wait, ambiguous waits, changed storage,
contradictory output shape, or missing wait prevents lowering that occurrence.
This does not follow arbitrary aliases or infer CUDA stream synchronization.

Each verified occurrence becomes:

- The original dispatch node, retained as a semantic issue event with its
  original dependency edges and a separate `source_cost_key`.
- One collective completion node per participant, sharing a rendezvous identity
  and depending on its local issue. Its
  `communication_completion="device"` identifies the cost-binding endpoint.
- Native semantic wait nodes that depend on the corresponding completion.
  Independent host issue work between issue and wait does not gain a completion
  dependency.

Completion nodes include `source_rank`, `source_node_id` (the original rank
capture node ID), `source_workload_sha256`, and `source_issue_node_id` (the merged
issue ID). They retain the original communication descriptor. A fresh completion
cost key prevents a host dispatch measurement from becoming device-completion
timing. `completion_measured` is false: recovering a dependency does not measure
its duration. A measurement binder must verify the source fingerprint and native
collective signature before attaching an exact device-completion cost.

Output allocation producers stay at the original issue boundary; delaying them
until global rendezvous could undercount memory while a peer is late. Exact
input/output storage references must exist with consistent extents and a unique
collective output producer. Every referenced allocation gains a completion
consumer, including allocations whose captured consumer list was empty. This
is still logical lifetime evidence, with the original physical-memory limitations.

## What remains unresolved

Local CUDA discovery and fake process groups may produce candidate rendezvous
metadata when all peer artifacts are supplied. Their local representative values,
no-op work handles and returned waits never establish native communication
completion. Missing rank artifacts are never replicated from rank zero.

Native `c10d` Work-returning calls need a separate observer for the actual Work
wait and stream semantics. Host return, including `async_op=False`, is not used
as a device-completion event. P2P matching and opaque/custom communication
descriptors are unsupported in this recovery pass.

Existing native pipeline host schedules should use `export-plan` / `lower-plan`.
That path already represents its supported issue and wait actions. This pass
does not reconstruct schedules from microbatch counts, combine independent
pipeline groups into a global mesh, or reinterpret prepared Chrome GPU events
as original dispatch evidence.

CUDA streams, event waits, stream contention, CPU launch/wait costs, compiled
correspondence, numerical equivalence, external allocations and physical memory
bounds remain separate evidence requirements. Existing source gaps are retained.
The default simulator resources remain conservative; the recovered graph alone
does not prove overlap or a calibrated full-training prediction. Malformed
artifacts, conflicting provenance and dependency/rendezvous cycles raise errors;
unsupported evidence returns explicit gaps.
