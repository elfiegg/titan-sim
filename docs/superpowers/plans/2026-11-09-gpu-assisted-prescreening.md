# GPU-assisted TorchTitan prescreening implementation plan

**Goal:** Connect native local execution, isolated and concurrent GPU measurements, and per-device memory evidence to traceable training predictions.

**Architecture:** Preserve target distributed semantics while discovering real local execution. Keep communication boundaries explicit, measure native compute and communication separately and in selected windows, and bind applicable evidence to an immutable source workload. Extend existing workload, runner and memory contracts rather than introduce a second simulator.

**Execution:** User approved the consolidated design and subagent implementation. Use independent file ownership and coordinated integration. Central storage remains deferred. Native numerical results, cluster identifiers and raw artifacts remain outside the repository.

## Shared contracts

- `Workload` remains the CPU input. Source identity is a canonical SHA256 over its serialized nodes, storage, manifest and hardware.
- Measurement workloads retain their source occurrence identities. Bindings carry source checksum, exact case key, member IDs, evidence and applicability; no name-only or shape-only matching.
- Preserve original worklists. Binding creates a derived workload and an audit report. Missing costs and unsupported semantics remain gaps.
- Regions and windows must have explicit entry/completion boundaries. Do not hide intermediate external dependencies or double-charge contained kernels.
- Memory evidence distinguishes observations, analytical estimates and explicit conditional bounds. Observed maxima never close an unknown upper bound.
- Single-GPU local discovery retains the target logical groups/rank roles. It is not a world-size-one training substitution, nor proof of unchanged compiled communication behavior.

## Task 1 — Native local discovery and executable regions

Owner: native execution agent. Own new `local_execution.py`, `executable_regions.py`, corresponding focused tests and `docs/local-execution.md`; request ownership before changing existing files.

- [x] Inspect the pinned native runtime and identify executable boundaries and fake-group support.
- [x] Implement explicit real-compute/local-rank discovery with recorded communication substitutions, input policy and original target identity. Reject unsupported paths; never change compilation, checkpointing, precision or trainer to obtain a successful trace.
- [x] Retain native callable handles and exact argument/alias/state contracts for supported local regions, with live-process versus reloadable capability explicit.
- [x] Provide executable measurement workloads and runner-compatible callables with source occurrence links.
- [x] Test cleanup, failed substitutions, value-dependent inputs, callable identity and compilation boundaries; run an authorized bounded real-GPU validation.

## Task 2 — Native communication and distributed windows

Owner: communication/window agent. Own new `execution_windows.py`, `communication_replay.py` if needed, focused tests and `docs/execution-windows.md`; may own `communication_cases.py` and its tests after notifying root.

- [x] Extend supported native communication cases with actual P2P and all-to-all contracts where the installed runtime exposes them. Custom EP providers must replay the actual backend; unsupported HybridEP must remain explicit.
- [x] Implement a window specification from source occurrences, native provider references, streams, dependencies, live inputs and completion fences.
- [x] Replay real compute/communication windows in an existing distributed group, with coordinated preflight and timed completion of all auxiliary streams.
- [x] Preserve isolated and window sample distributions and topology. Never infer a universal interference multiplier or substitute missing native providers.
- [x] Test event/wait ordering, rank agreement, missing participants, nested/overlapping member rejection, failure cleanup and context mismatch. Supply a real-GPU harness to the single scheduler owner.

## Task 3 — Target-applicable memory evidence

Owner: memory agent. Own new `memory_evidence.py`, focused tests and `docs/memory-evidence.md`; request ownership before modifying existing memory modules.

- [x] Add a versioned local evidence bundle bound to source, target roles, device placement, phase and implementation.
- [x] Bind explicit allocation generations/aliases and completion events to existing physical-memory envelopes without double counting allocator totals or observed state.
- [x] Derive localized coverage obligations and retain unknown upper bounds. Distinguish budget overflow from physical-capacity rejection.
- [x] Accept native state inventories and component/window observations as observations; explicit supported allocation bounds require applicability and evidence.
- [x] Test stale identities, incomplete devices/phases, measured-versus-bounded evidence, asynchronous release and false-fit rejection.

## Task 4 — Binding and workflow integration

Owner: root. Own `case_binding.py`, CLI changes, integration tests, overview documentation and version changes.

- [x] Bind existing compute and collective cases to exact source occurrences with canonical source checksums, validated descriptors and immutable source inputs.
- [x] Preserve collective rendezvous and correct completion scope. Reject missing participant evidence, duplicate ownership, incompatible runtime/placement and changed occurrence signatures.
- [x] Connect new native region, window and memory interfaces after agent API handoff; avoid speculative parallel schema changes.
- [x] Add CLI entry points and one coherent native workflow guide, including explicit unsupported-case output.
- [x] Run CPU suite and framework-free checks, independent specification/code review, then bounded GPU integration checks through one scheduler owner.
- [ ] Held-out full-training prediction accuracy remains a separate validation task. Component and controlled-window checks do not complete it; freeze any future full prediction before its independent control.

## Implemented scope and remaining work

The stages and exact binding interfaces are implemented for their documented supported cases. Automatic trace-to-window semantic recovery, full global mesh scheduling, arbitrary custom/compiled-region binding, HybridEP, distributed graph replay and automatic complete physical-allocation contracts remain gaps. Local discovery still requires the selected rank state to fit. The public design must preserve these boundaries. Numerical execution results are retained only in the caller output directory.

## Validation and delivery

Use the existing prepared runtime and unchanged recipe as the first fixture; model names must not select simulator behavior. CPU tests must exercise malformed and partial evidence as well as supported paths. Real GPU checks must retain exact frozen simulator/runtime/source identities and use initial five-minute hard limits. Root integrates and publishes implementation-only changes to the existing draft PR; experimental measurements stay in the caller's local report.
