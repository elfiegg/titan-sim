# Integrate a custom native implementation

Use this procedure when case selection in the [prescreening workflow](gpu-assisted.md)
finds an operation or region that the bundled providers cannot measure faithfully.
It applies to custom operators, Python/C++ extensions, compiler-generated code,
communication libraries and opaque native state.

The integration connects the existing source workload to the existing measurement,
scheduling and memory interfaces described in the [system design](architecture.md).
The agent inspects what the unchanged recipe actually invokes, chooses an executable
boundary, qualifies a native provider and binds its evidence back to that source.
The deliverable is usable provider code and evidence, or a precise missing runtime
capability or simulator interface with a next action.

The coordinator supplies the prepared runtime, source occurrences and execution
budget. Keep providers and case results in the caller's run directory until
qualified for broader reuse. The generic skill records this method; it does not
accumulate library/model branches or experiment results.

## What the coordinator gives an integration agent

Create a case packet for each distinct implementation/context that is missing
from the prediction. This is an agent handoff artifact, not a new simulator JSON
schema or an automatically generated CLI output. Include:

| Packet field | Required content |
|---|---|
| Source case | Original workload fingerprint, rank/occurrence IDs, native recipe and resolved configuration |
| Implementation | Observed callable/schema or enclosing native entry point; source/build identity and available code paths |
| Runtime | Prepared image/interpreter, loaded extension locations and versions, capability evidence, relevant execution settings |
| Target | Rank roles, group membership, physical placement, shapes/layouts/dtypes, routing or other value-dependent workload assumptions |
| Execution access | Caller-provided command entry, visible paths, authorized GPU budget, hard limit, status/log/stop operations |
| Open questions | Exact missing replay, completion, alias, allocation, cost or source-binding semantics |
| Deliverables | Provider code, bounded probe, source correspondence, measurements, memory evidence and unresolved scope |

Populate this packet from the existing runtime handoff, repository and retained
captures before requesting information from the user. Record missing fields and
continue inspection that does not depend on them; a packet is a working artifact,
not a prerequisite questionnaire.

Do not infer a Python distribution name from an operator namespace. Record how
the callable resolves in the actual prepared process. An installed package that
the recipe never selects is not a required measurement case.

## 1. Discover the selected native path

Inspect the resolved recipe and actual runtime imports, dispatcher schemas,
compile hooks and captured occurrences. Read the original wrapper, extension
registration, native tests and call sites to establish which implementation is
selected. A Python wrapper may hide transport, internal streams, persistent
buffers, fused backward work or workspace allocation.

Start with read-only inspection and the smallest relevant capability probe. A
successful import is weaker evidence than constructing and executing the selected
path. Capture dependencies that are absent from Python dispatch through an
appropriate enclosing native boundary or profiler evidence; do not assume they
do not exist because the dispatch inventory is silent.

If a dependency cannot load, return a precise runtime-preparation handoff:
required module/symbol and native build, actual interpreter/image, import or
loader error, affected recipe feature, and the bounded capability check needed
after preparation. Continue independent cases. Do not install a replacement or
change the selected implementation to make this case appear supported. Repair
through the existing runtime-preparation workflow when that work is authorized;
a missing dependency is not a reason to redesign the simulator around that
particular library.

## 2. Choose the smallest faithful replay boundary

Classify from source and observed behavior, not names:

| Native behavior | Measurement boundary |
|---|---|
| Local callable with supported tensor/scalar contracts | Retain and replay the original callable on one GPU |
| Opaque state passed between calls | Keep the native owner and state alive; use an enclosing region covering its required lifecycle |
| Collective or P2P path | Use the actual native implementation with the identified real group and topology |
| Fused compute/communication or overlap-sensitive region | Preserve the region and measure a distributed window with its native buffers, streams and waits |
| Compiled execution incompatible with dispatch observation | Use supported boundary-only observation or native profiler evidence; leave unavailable internals explicit |

Expand the boundary when the smaller operation cannot be replayed faithfully.
Do not replace an opaque handle with a fabricated tensor or split fused
communication out before compilation. If the supported simulator interfaces
cannot express the required boundary, implement and test a generic interface
extension or return the exact missing interface; do not weaken validation.

## 3. Establish the executable contract

The integration agent records evidence for each of these obligations before
claiming replay readiness:

- **Inputs and outputs:** shape, stride, dtype, device, aliases, scalar options,
  pre-call state and any value-dependent routing, sparsity or early-exit behavior.
- **State:** owner and lifetime of opaque objects, parameter/optimizer mutation,
  RNG/dropout behavior, reset or reconstruction between samples, and cleanup.
- **Execution:** original compiler options and callable, dispatcher/autograd/
  autocast context, graph mode and relevant runtime/build settings.
- **Communication:** participants, peer roles, protocol, buffer layout and
  completion semantics. Local replay requires evidence for no transport, not
  merely a convenient `communication: "none"` assertion.
- **Streams:** where work is enqueued and which native events/waits establish
  completion. Auxiliary work must join the timer's stream for complete timing;
  any measurement-only join must be distinguished from the original schedule.
- **Memory:** persistent state, transient workspaces, communication/graph pools,
  allocation ownership and release boundaries, including allocations outside
  PyTorch's allocator.

Separate observed facts, source-derived semantics, trusted provider assertions
and analytical assumptions in the case packet. Schema alias annotations, tensor
version counters and successful finite outputs are useful checks; none alone
establish safe replay or numerical fidelity.

## 4. Implement and qualify the provider

Reuse [live callable retention](local-execution.md) when its contracts apply.
Otherwise implement the original implementation's lifecycle through the existing
[native region provider](native-calibration.md#execute-an-exact-native-region) or
[window provider](execution-windows.md#trusted-native-provider-contract). The
provider invokes native code; it does not reproduce the algorithm in simulator
code. Use bounded retention so a small view does not unintentionally retain the
whole model's intermediate storage.

Before timing, run a bounded qualification that checks outputs and gradients
against an independent native invocation or available native correctness oracle,
using appropriate tolerances and representative inputs. For stateful operations,
check repeated reset/replay behavior. For communication, check every peer's
outputs and completion and the lifecycle through backward where applicable.
When no suitable oracle exists, report that limitation; finite outputs alone
must not be reported as numerical-equivalence validation.

Then measure the actual region on one GPU or its required group, preserving
implementation and context. Retain sample distributions and actual placement.
Keep allocator instrumentation separate from throughput controls. Inputs can be
representative rather than original only when their effect on implementation,
routing, memory and cost is recorded. A best-case routing case cannot silently
price arbitrary training traffic.

## 5. Bind the evidence to the prediction

Use `bind-cases` for supported observed boundaries and exact occurrences. For a
provider outside that binder's supported cases, use the documented
[preparation integration](native-preparation.md#supply-semantics-from-an-integration)
with explicit region/dependency evidence, or extend the binder with tests. Native
windows need source-backed ordering and boundary evidence before training binding.
Do not insert a key by matching a kernel name or tensor shape alone.

Cost identity must distinguish native build, runtime/math/compile settings,
input contracts, state policy, graph mode, group/topology and workload-dependent
behavior. Keep actual executable handles in their owning process; serializing
their metadata does not make them reloadable. Verify that contracting a region
does not double-charge its internal operations or hide an early external consumer.

Track memory separately using [native allocation evidence](memory-evidence.md).
For external allocations, seek native workspace-size queries, buffer capacities,
pool accounting or source-backed allocation contracts. Missing instrumentation
leaves physical coverage unknown. Allocator observations and conditional replay
envelopes are not unconditional upper bounds for a new recipe.

## 6. Return usable work and precise remaining scope

Return the case packet with provider/probe paths, frozen identities, successful
and failed checks, measurements and source bindings. Report each dimension
separately: runtime capability, replay fidelity, cost coverage, scheduling,
memory coverage and independent validation. A measured kernel may still lack a
usable training dependency or physical-memory bound.

Mark cost binding complete when the original invocation is identified, its native
provider is qualified for the declared context, and costs bind to verified source
occurrences. Report scheduling and memory readiness independently, with their
own evidence or unresolved obligations. Full-recipe prescreening remains incomplete if any required case
still lacks costs or execution semantics. Run a full-update accuracy comparison
only when [prediction freezing](prediction-validation.md) accepts that scope.

## Delegate by integration responsibility

The coordinator can assign independent implementation cases in parallel when
their files and native state do not overlap. Use one owner for GPU scheduling.
Within a case, inspection precedes provider construction, which precedes timing
and source binding; these dependent steps must not become disconnected tasks.
For complex state/communication paths, assign an independent reviewer to challenge
the executable contract and completion/memory evidence before broad reuse.

A task prompt should identify the case packet and ask the agent to:

> Resolve the observed native invocation in the prepared runtime. Inspect its
> implementation and select a faithful local or distributed replay boundary.
> Implement the native provider and bounded qualification, using the caller's
> execution budget. Return source/cost correspondence, memory and scheduling
> evidence, and precise unsupported scope. Preserve the native recipe and
> implementation. Coordinate interface changes and GPU launches with the owner.

The agent should produce code and evidence when feasible, rather than ending
at “custom operator unsupported.” A genuine blocker should name the missing
dependency, native semantic information, resource or simulator interface and
the concrete next action, while other independent work continues.
