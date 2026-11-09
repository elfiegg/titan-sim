# Native host-plan scheduling

`lower_plan(plan, hardware, config=None)` reads the extracted
`titan_sim.native_plan.v1` JSON contract and returns a framework-independent
`Workload`. It expands the submitted native action table and its attached FX
region graphs. It does not construct a different schedule, change microbatch
counts, choose a pipeline algorithm, or run a model.

The native extractor owns schedule generation: this module requires the actual
`pipeline_order_with_comms`. A compute-only `pipeline_order` is rejected because
it does not establish transfer or wait placement. Source list positions identify
actions, including positions containing `None`; empty positions are annotations,
not time delays or cross-rank synchronization.

## Supported action semantics

| Native action | Lowering |
| --- | --- |
| `F`, `B`, `I`, `W` | Attached forward, full backward, input backward, and weight backward FX regions; rank compute order, stage/microbatch data dependencies |
| `SEND_F`, `RECV_F`, `SEND_B`, `RECV_B` | Asynchronous transfer completions, paired by direction, producer stage, consumer stage, and microbatch |
| `WAIT_SEND_F`, `WAIT_SEND_B` | Completion dependency on the matching previously issued send, with subsequent work gated by the wait |
| `UNSHARD` | Attached native unshard region issued as a prefetch; dependent stage compute waits for completion |
| `RESHARD` | Zero-duration native lifecycle marker after stage use; later unshard depends on that release |
| `REDUCE_GRAD`, `WAIT_REDUCE_GRAD` | Attached native reduction region after all stage backwards; explicit completion wait |
| Native `BACKWARD`, `BACKWARD_WITH_REDUCE_GRAD`, `BACKWARD_WEIGHT_WITH_REDUCE_GRAD` | Exact matching native graph-call role and corresponding backward dependencies |
| `FULL_FORWARD_BACKWARD`, `FORWARD_BACKWARD`, `FORWARD_BACKWARD_NOGRADACCUM`, `FORWARD_BACKWARD_FIRST_WITH_UNSHARD`, `FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD` | Exact matching native graph-call role, supported for one pipeline stage |

Compound `OVERLAP_F_B`, opaque/custom actions, missing region graphs, and
unsupported node kinds fail explicitly. A fused native graph is expanded as its
own graph; it is not replaced with separate invented forward/backward calls.
Reduction, communication, optimizer work, or stream behavior absent from the
native plan is not synthesized. An `UNSHARD` action without a corresponding graph
fails even if another runtime might treat that action as a no-op.

Rank tables must cover the declared world size, stages must have explicit unique
ownership, and each stage/microbatch must contain forward plus either full
backward or both split backward actions. Single-stage joint actions fulfill both
forward and backward obligations. Unknown peers, duplicate actions, missing
receives, unmatched waits, mismatched payloads, and cycles after communication
rendezvous contraction are rejected.

The extractor labels rank IDs as **pipeline-group ranks**. Their presence does
not establish global DP/TP/CP/EP mesh coverage. Lowering retains an unresolved gap
for that namespace; a single group's action table must not be presented as a
complete multi-group workload. Explicit global placement and full native rank
evidence remain the integration's responsibility.

## Graphs, transfers, and resource identities

Graph placeholder, attribute, and output nodes are semantic boundaries. Actual
FX instructions carry graph data dependencies and compute-stream order. Every
occurrence retains its native action, graph, node, stage, and microbatch identity.
The source graph and provenance participate in cost identity together with the
operator arguments, output tensor metadata, implementation binding, and
communication context. Different scalar arguments or runtime provenance cannot
silently reuse the same cost key.

Pipeline payloads are derived when the native split-graph calling convention is
available: forward's first `num_user_outputs`, full backward's
`num_input_grad_values` after `num_param_grad_values`, or input backward's first
`num_input_grad_values`. Each transported flat value must have one tensor leaf
with known logical bytes. Aliased outputs still represent separate transported
values; backing-storage deduplication is a memory concern. Unknown sizes require
explicit matching native transport evidence. They never default to zero.

Asynchronous dispatch is distinct from completion. Sends depend on their actual
producer; receives can be posted before their consumer is ready. Explicit waits
gate following work. Unshard prefetch completion gates only the relevant stage,
plus declared resources. No dependency is inferred from a timestamp. CPU launch,
data loading, and host synchronization durations remain unresolved.

Resources supplied in bindings are preserved. Otherwise the declared exclusive
resource model uses `gpu:<device>` for compute, `pp:<device>` for pipeline
communication, and `collective:<device>` for collectives. These are explicit
model assumptions, not discovered CUDA streams or measured interference. Device
IDs come from optional rank placement, or rank identity in its absence. Missing
placement does not become a physical memory-fit claim.

## Cost and communication bindings

Optional configuration supplies `node_attrs[graph_id][fx_node_name]` and
`action_attrs[rank_string][source_index_string]`. Node bindings may carry
analytical `flops`/`bytes`, implementation/context metadata, and resources. An
upstream cost key is retained inside the complete native context, not used to
bypass that context. Schedule identities and graph dependencies cannot be
overridden. Unused bindings fail to catch stale source indices.

Pipeline action bindings can supply `payload_bytes` when native metadata is
insufficient. A value contradicting native metadata fails. Collective graph
nodes require `kind: collective`, exact `participants`, `group`, `operation`,
`payload_bytes`, and a `rendezvous` identity shared by all native participants.
Repeated collective occurrences require distinct identities. Rendezvous strings
may substitute `{microbatch}`, `{stage}`, and `{action}`; these tokens use native
occurrence values. Native functional wait nodes may be explicitly marked
`native_wait: true, zero_cost_semantic: true`; other compute instructions cannot
be made semantic through bindings. Distributed targets without corresponding
collective/wait semantics fail.

An FX operator is not a selected compiled kernel. The workload always retains
that unresolved implementation-matching limitation even when analytical costs
are supplied. Unbound instructions expose exact cost keys for further native
measurement; simulation requires matching costs or explicit analytical
quantities. The lowerer never labels invented samples as measurements.

## Storage and validation scope

After lowering, `plan_memory.lower_graph_memory(workload, plan)` attaches native
graph-backed logical allocations and calling-convention lifetimes. Lowering
retains `manifest.native_plan.graph_invocations` and per-node source identities
for inspection. This does not establish physical allocator or workspace coverage.

The CPU tests exercise native PyTorch's `ScheduleLoopedBFS` action generator,
its native communication/wait passes, and actual `make_fx` graph extraction,
followed by event simulation. Framework-free tests include an analytically
checkable two-stage training fixture (four one-second compute operations and
two one-second transfers, six seconds total), joint graph actions, unshard
ordering, exact cost contexts, and malformed-plan rejection. They do not execute
a TorchTitan GPU trainer or validate end-to-end GPU prediction accuracy.
