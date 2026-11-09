# Native execution windows

A window measures a bounded set of actual native callables together. It preserves
provider-owned inputs and state throughout the window, launches each occurrence
on its declared CUDA stream, records dependency events, and joins every declared
stream onto the runner's timing stream before the end event. Returned outputs stay
live until the next sample reset. No interference multiplier is fitted or applied.

This is a measurement interface, not automatic native model reconstruction.
A trusted provider constructs the original implementations and retains shared
buffers and native state. Compilation belongs to that provider; communication
must remain in a compiled region when the native region contains communication.
Independent isolated-case buffers cannot stand in for training dataflow.

## Prepare and measure

`execution_window_workload(source, manifest)` builds an exact-key measurement
workload. `validate_window_manifest(source, manifest)` checks its source identity,
occurrence membership, local dependency graph, streams, live-in boundary and
communication coordination contract. The corresponding CLI is:

```sh
python -m titan_sim window-cases prepared-source.json \
  --config window-manifest.json --output window-workload.json
```

The source nodes must already have exact `native_measurement` implementation and
context descriptors. The manifest selects **occurrences**, not deduplicated cost
keys. Changing order, streams, dependencies, input policy, source, provider code,
runtime or topology changes the window key.

A version 1 manifest contains these fields:

```python
{
    "schema": "native-execution-window", "version": 1,
    "window_id": "native-region-window",
    "source_workload_sha256": workload_fingerprint(source),
    "provider": "native_adapter:provide_window",
    "provider_source_sha256": "sha256 of the resolved provider Python file",
    "participants": [0, 1],
    "physical_topology": {"rank_placement": "explicit actual placement"},
    "runtime_context": {"runtime": "exact native software/settings identity"},
    "input_state_policy": {
        "lifetime": "entire_window", "reset": "before_sample",
        "values": "describe actual or synthetic inputs",
        "sharing": "describe native aliases and state retention"
    },
    "completion": "join_all_streams",
    "qualification": "controlled_native_window",
    "ordering_evidence": {
        "kind": "caller_assertion", "description": "explicit controlled order"
    },
    "communication_contract": {
        "kind": "matched_native_collectives",
        "description": "same ordered native collective cases on every rank"
    },
    "live_in_dependencies": [],
    "steps": [
        {"occurrence_id": 12, "rank": 0, "stream": "compute", "deps": []},
        {"occurrence_id": 13, "rank": 0, "stream": "communication", "deps": []},
        {"occurrence_id": 24, "rank": 1, "stream": "compute", "deps": []},
        {"occurrence_id": 25, "rank": 1, "stream": "communication", "deps": []}
    ]
}
```

These illustrative node IDs must exist in the actual source. Independent streams
have no implicit dependency. A `deps` entry adds an event wait on that occurrence's
stream. Source local dependencies cannot be dropped; external source predecessors
must appear exactly in `live_in_dependencies`. Cross-rank source dependencies
currently require a separate native boundary integration and are rejected.

Use `titan_sim.native_runner:measure` in execute mode with
`region_provider: "titan_sim.execution_windows:region_provider"` and
`provider_config.physical_topology` equal to the manifest topology. The existing
native runner performs CUDA-event timing and all-rank sample aggregation. Its
catalog retains the complete window context, raw samples and per-rank samples.
Window evidence is identified by `context.measurement_scope: execution_window`;
it is not an isolated operator cost even though it uses the generic executable
region timer. Single-occurrence windows can provide explicitly labeled isolated
comparisons. Separate serial and overlapped manifests yield separate measured
contexts, not a reusable contention factor.

Execution uses the caller's initialized process group. Every existing rank must
participate; subgroups and distributed CUDA graphs are unsupported. The launcher
owns device setup, process-group timeout, rank failure propagation and the
scheduler hard limit. Initial GPU checks require an authorized allocation and a
five-minute hard limit. Only the coordinator writes results.

## Trusted native provider contract

The manifest's callable is invoked as `provide_window(source, manifest)`. It
returns local occurrences only:

```python
{
    "source_workload_sha256": manifest["source_workload_sha256"],
    "input_state_policy": manifest["input_state_policy"],
    "runtime_context": manifest["runtime_context"],
    "communication_contract": manifest["communication_contract"],
    "preparation_is_local": True,
    "provenance": {"native_implementation": "retained code/build identities"},
    "before_window": optional_local_reset_callable,
    "regions": [{
        "occurrence_id": source_node.id,
        "implementation": source_node.attrs["native_measurement"]["implementation"],
        "context": source_node.attrs["native_measurement"]["context"],
        "run": native_callable,
        "before_sample": optional_local_reset_callable,
        "stream_semantics": "current_stream"
    }]
}
```

Construction, `before_window`, and per-occurrence reset callbacks **must not call
collectives**. Their local errors are gathered before any rank enters native
communication. The `preparation_is_local` flag is an explicit provider attestation;
it cannot detect a provider violating this rule. Callables may run actual native
collectives during `run`. Provider-internal auxiliary streams must join the
current stream before that callable returns. The composer joins all additional
streams that it creates. Native calls that hang, failed CUDA contexts, or ranks
that never enter the runner still require launcher failure handling.

Identifiable native NCCL cases must have identical ordered case signatures on
all ranks. This validates available peer/group/shape descriptors before replay.
Custom native/compiled regions require `communication_contract.kind:
provider_owned` and an explicit description. Matching manifest hashes cannot
prove that arbitrary custom native callables coordinate correctly; that remains
the trusted provider's responsibility. Runtime context is likewise attested by
the provider, while the runner independently checks its configured CUDA device
properties. Neither check independently establishes physical network topology.

## Source-backed ordering and conservative binding

A controlled window with synthetic independent operands and caller-selected
streams qualifies only that measured setup. `bind_execution_window` rejects it
as a training cost replacement.

A `source_backed_native_window` requires `ordering_evidence.kind:
captured_stream_events`, a retained `artifact_path`, and its SHA256. The JSON
artifact uses schema `native-window-ordering-assertion`, version 1, with:

- `source_workload_sha256`, exact manifest `steps`, and
  `input_state_policy_sha256`;
- a named `producer` and nonempty `source_artifacts`, each with a retained `path`
  and verified `sha256`;
- for binding, `boundary_contract` containing `entry: all_live_inputs_ready`,
  `exit: all_streams_joined`, and
  `external_consumers_wait_for_window_completion: true`.

The artifact is a **producer assertion tied to checked source artifacts**, not
an automatic parser or independent proof of raw trace semantics. The producer
must interpret actual native stream events, waits, buffer/state dependencies and
entry/exit boundaries faithfully. Hashes establish consistency, not correctness.

`bind_execution_window(source, measurement)` replaces selected occurrences with
one atomic event that owns the union of their participant resources and uses the
whole-window duration once. It retains all incoming dependencies, redirects all
outgoing dependencies to completion, and rejects contractions that create a
cycle. It conservatively extends internal storage lifetimes to the atomic
boundary and marks physical storage evidence unverified. Bind windows before
memory evidence. Existing unresolved gaps remain, including the assumption that
all participants are ready and external interference is unqualified.

This deliberately sacrifices intermediate scheduling freedom. It cannot model
outside consumers that begin before the native window completes. It does not
establish a complete training schedule, unseen-configuration memory fit, or
held-out full-model performance accuracy.
