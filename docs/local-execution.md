# Local CUDA discovery and live native regions

`capture --adapter titan_sim.torchtitan:capture` accepts an explicit third mode,
`execution: "local_cuda"`. It requires `target_rank`, `target_world_size`, unchanged
native `recipe_args`, `hardware`, and a CUDA training environment. The current
observer is `dispatch`. Existing `native` and `fake` modes retain their behavior.

The native recipe builds its original trainer with its original precision,
rematerialization and compilation settings. Scoped process-group interception
creates native fake-backend groups with the target rank and membership. It records
the requested backend and actual substituted backend. It never starts transport
or creates a smaller real training mesh. Actual compute uses real CUDA tensors.

Supported gather, reduce-scatter and all-reduce dispatch boundaries execute the
native fake overload for its Work-handle contract, then materialize finite local
representatives. Gather repeats the local shard; reduce-scatter selects the local
rank's slice; all-reduce retains the local value. These are explicit substitutions,
not numerical distributed training. Nonfinite collective inputs, unsupported
collectives and mismatched target-group initialization fail closed. Wait and
barrier boundaries are recorded as transport-free. Every substitution retains
operation/schema, group membership and original tensor contracts.

The result remains rank-local, with unresolved communication, ordering, physical
memory and compiled-correspondence limitations. Instrumentation and substituted
inputs can change compiler behavior. A completed discovery step establishes only
that this supported local experiment ran. It does not establish successful
training or selected kernels at target scale.

## Native compiled/custom callables

The pinned TorchTitan `LocalCompileConfig._compile` hook exposes the actual native
callable returned by its original compiler. The observer invokes that hook without
changing its backend or compile options, records native boundary calls and their
source dispatch occurrence IDs, and leaves correspondence to uninstrumented
training **unverified**. Dispatch-visible custom overloads likewise retain their
original callable handle and schema. A callable boundary becomes an explicit source node even when its kernels are opaque. Bypassed native hooks are not recovered.

`retain_executable_arguments: true` additionally retains live callable handles and
observed argument objects for captured source occurrences in `local_execution.LIVE_CAPTURES[token]`. The token
is returned in `manifest.live_capture_token`. Argument metadata records shapes,
strides, offsets, aliases and state contracts. Retained objects extend tensor
lifetimes, so the capture explicitly cannot serve as unmodified memory evidence.
Without this option, metadata is retained but live arguments are released. Unknown
mutation/state contracts prevent automatic replay. Handles do not survive process
exit and are never deserialized from JSON.

For a boundary with an established replay contract, use
`executable_regions.LiveRegionRegistry.register(original_callable, args=...,
kwargs=..., implementation=..., context=..., source_workload_sha256=...,
source_node_ids=..., state_contract=..., before_sample=..., replay_safe=True)`.
The context must include `hardware_id`; registration captures the current runtime,
math, grad and autocast settings and the provider rejects changes. Mutable state
requires a reset callback outside timing; read-only state must be
explicit. The registry retains the actual callable, arguments, aliases and source
links, rather than recreating a callable from an FX graph.

`registry.calibration_workload(hardware=...)` creates a one-device isolated
measurement workload. Pass it to `native_runner.measure` in the same process with
`mode: "execute"`, `region_provider: "titan_sim.executable_regions:region_provider"`,
and the runner's ordinary explicit device/hardware settings. The provider checks
exact identity, hardware/runtime context and the retained argument contract before timing.
The caller descriptor and state-safety assertion remain explicit trust inputs. Call
`registry.close()` to release handles and arguments. State reset, representative
values and auxiliary-stream completion remain the trusted integration's contract.
This timing is native isolated callable execution; it does not prove distributed
step performance or FX-to-generated-kernel equivalence.

The qualified pinned runtime rejects `torch.compile(fullgraph=True)` while a
non-infrastructure `TorchDispatchMode` is active. The local dispatch mode therefore
cannot claim general compiled-recipe support. `observe_native_compile(...,
boundary_only=True)` offers a separate explicit observation mode: it records an
actual native callable invocation and its input/output contracts without entering
a dispatch observer. Its source node is opaque, and internal communication,
intermediate storage and generated-kernel identity remain unverified. This option
does not silently switch the local discovery adapter's observation mode.

`registry.register_observed(record, native_callable_index=...,
source_workload_sha256=..., context=..., state_contract=..., replay_safe=True)`
consumes an actual retained source invocation and checks its pre-invocation
argument contract and runtime/math/grad/autocast identity. Metadata-mutating
inputs must match their original contract before registration; post-call shape
or stride changes cannot silently become the replay inputs.
`occurrence` selects an entry in the aligned `source_occurrences`,
`source_argument_contracts`, `source_runtime_identities` and live `source_arguments`
lists. The source workload
must retain the corresponding native boundary inventory for later binding. This
avoids manually substituting an unrelated function for an observed boundary.

Current source binding supports the first observed native compile-hook boundary.
Dispatch-visible custom overload handles and their contracts can be retained, but
`bind-cases` does not yet bind those handles as observed native callable regions.
Keeping an actual custom handle is not proof that it has a safe calibrated replay.
