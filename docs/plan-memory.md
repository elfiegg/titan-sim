# Memory from a native execution plan

`lower_graph_memory(workload, plan)` adds logical storage records to a workload
expanded from a versioned native plan. It returns the same workload after updating
`storage`, `manifest.memory` and `manifest.unresolved`. The implementation is
`titan_sim.plan_memory`; it uses no model names, layer formulas or checkpointing
policy approximations. The submitted native runtime owns graph partitioning,
recomputation, offloading, FSDP graph extraction and action ordering.

The native extractor reads FX tensor metadata, including backing-storage identity,
backing byte extent, logical tensor extent, device, shape, stride and offset.
For example, a four-element slice of a 32-element float32 backing allocation keeps
128 backing bytes live, not just its 16 logical bytes. Aliased tensors within a
graph invocation share one storage record. Separate microbatch invocations create
separate temporary allocation generations even when they reuse the same fake
trace objects. Graph constants and initialized native state persist across calls.
CPU-resident tensors are excluded from GPU storage totals; this does not establish
host-memory capacity or offload transfer coverage.

The input calling convention comes from the native plan's stage metadata:

- Forward parameter and buffer inputs bind to initialized native state, or to the
  outputs of the corresponding unshard action.
- The forward output slice identified by `num_user_outputs` and
  `num_saved_for_backward` binds to backward placeholders using the native
  partition's `saved_for_backward_names`. The binding includes stage, rank and
  microbatch identity.
- Split input/weight backward preserves its saved values through weight backward.
- Native gradient accumulator indices bind subsequent backward invocations to
  their preceding gradient outputs, including in-place aliases. Reduction graphs
  select gradients by the native reduction-input names.
- Scheduled joint graphs use native `call_specs`, including first-microbatch
  unshard outputs and subsequent gradient accumulator inputs. Plain joint graphs
  bind their static state prefix when its native count matches the initialized
  parameter and buffer values and no extra graph/optimizer state needs binding.

Opaque conventions, missing state, symbolic sizes, unknown backing identities and
incompatible bindings remain explicit gaps. Constants in graph output tuples keep
their positional slots. Native tensor-leaf paths preserve sparse state positions.

Each `Storage` records its producer and all device users. A view extends the same
backing allocation's lifetime. A `RESHARD` action ends runtime ownership of its
unsharded parameter generation; it does not erase outstanding device uses or
saved-value aliases. The memory envelope releases storage only after all those
registered completion events. There is no use of Python object deletion as GPU
completion evidence.

The lowerer supplies workload nodes with `plan_action_id`, `plan_graph_id`,
`plan_graph_node`, `stage_index`, `microbatch_index` and `computation_type`
attributes. `completion_node_id` may identify a separate completion event for an
asynchronous operation. When unavailable, the operation's scheduled end is the
modeled completion, subject to the lowerer's unresolved execution gaps.
The lower-level `derive_plan_memory(plan, invocations, nodes)` API accepts explicit
per-node completion mappings and cross-invocation placeholder bindings. A binding
references an earlier invocation ID and FX node name; it does not guess by shape.

Transport and cache release for graph user outputs, input gradients, losses and
opaque outputs are conservatively retained through the schedule end. Unbound
inputs are modeled from schedule start. These choices can overstate the logical
peak. Missing workspace/state coverage can understate physical usage. Neither
choice supports a physical capacity conclusion. Precise release and input
ownership evidence remain necessary to narrow those lifetimes.

The resulting memory scope is `schedule`, storage evidence is `unverified`, and
all physical coverage flags remain false. FX fake-storage extents are not proof
of compiled buffer assignment, reuse, allocation generations or allocator
reservation. Optimizer execution/state, workspaces, external allocations,
communication buffers, graph pools, allocator slack, initialization and checkpoint
loading require additional evidence. Plan extraction gaps propagate to memory and
the workload. A projected peak over capacity does not establish an actual OOM;
a projected peak below capacity does not establish that training fits.

CPU tests cover alias extents, saved-value overlap across microbatches, persistent
state deduplication, gradient accumulation, graph constants, sparse state slots,
and completion after host reshard. An optional installed-Torch test traces real
FX operators under `FakeTensorMode` and checks storage identities and byte extents
without executing a GPU kernel.
