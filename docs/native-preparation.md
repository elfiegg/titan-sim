# Prepare captured native workloads

`prepare` turns rank-local captures into one workload for calibration and
simulation. It combines ranks, gives their nodes and backing-storage records
unique IDs, and lowers GPU events that can be matched to captured operations.
Where capture does not establish execution or memory semantics, preparation
keeps the limitation visible or accepts a specific integration assertion.

This step runs on CPU without importing PyTorch. Native capture still needs the
original training environment; see the [native guide](native-torchtitan.md).
Preparing a trace does not run a model, launch a GPU job, or establish a memory fit.

## Start with captured rank files

Keep each workload JSON and its recorded trace artifacts together. Native
capture records absolute artifact paths and SHA256 checksums. Supply one
workload per rank, containing that rank's captured steps:

```sh
python -m titan_sim prepare results/capture/rank*.json \
  --output results/prepared.json

python -m titan_sim validate results/prepared.json
```

The input glob must include every rank in the captured world. Preparation
rejects duplicate ranks, incompatible hardware, and conflicting recorded recipe
or runtime identities. Missing ranks produce an unresolved coverage gap. Node
and storage IDs may repeat across input files; preparation remaps their graph
and lifetime references without combining unrelated allocations.

Registered artifacts named `stepN.chrome.json` are discovered automatically.
Preparation verifies their checksums, associates them with their captured rank
and step, and attempts GPU correlation. To use relocated artifacts, give their
new paths explicitly; their content must still match the capture checksums:

```json
{
  "chrome_traces": [
    {"rank": 0, "step": 0, "path": "staged/rank0/step0.chrome.json"},
    {"rank": 1, "step": 0, "path": "staged/rank1/step0.chrome.json"}
  ],
  "useful_tokens": 0
}
```

Save this as `prepare.json` and add `--config prepare.json` to the command.
Paths are resolved from the current working directory. Set `chrome_traces` to
`[]` when using explicit execution-region assertions instead of GPU-event
lowering. One source operation with several GPU launches has no unique endpoint
for a source-node assertion; that combination is rejected.

`useful_tokens` is the global count for the entire modeled interval. Count each
useful training token once: tensor/pipeline/expert-parallel copies do not create
additional useful tokens. Include every captured step and the intended
data-parallel contribution. The merger does not sum rank counts, because their
meaning may differ by adapter. Use zero to leave throughput unspecified until
the integration supplies a defensible count.

## Collect costs from matched GPU events

For captured GPU timings, create a measurement configuration using the exact
hardware identity already recorded in the prepared workload:

```sh
python - <<'PY'
import json
from pathlib import Path
workload = json.loads(Path('results/prepared.json').read_text())
config = {
    'mode': 'trace',
    'hardware_id': workload['hardware']['id'],
    'accept_profiler_overhead': True,
}
Path('results/trace-costs-config.json').write_text(json.dumps(config))
PY

python -m titan_sim calibrate results/prepared.json \
  --config results/trace-costs-config.json --output results/costs.json
```

The bundled runner reads GPU event durations from the verified artifacts.
These are instrumented observations from the captured run. They are not an
independent validation run. The explicit acceptance flag records that limitation.
The runner can instead benchmark executable regions supplied by a trusted
integration; see [native calibration](native-calibration.md).

Inspect `missing_cost_keys` and `unkeyed_nonsemantic_nodes` in calibration output.
Matched keys can be collected into a partial catalog, but a remaining opaque
operation still needs an implementation and a cost. Simulation may reject such
a workload even without `--strict`; preparation does not invent a fallback cost.
Once its operations have costs, run:

```sh
python -m titan_sim simulate results/prepared.json \
  --calibration results/costs.json --output results/prediction.json
```

Read `status`, `cost_coverage`, `unresolved`, and `memory_verdict` together.
Use `--strict` only after the workload's declared limitations and all
nonsemantic operation costs are resolved. Physical memory coverage remains a
separate requirement; `--require-memory-fit` requires its estimated-fit verdict.

## What automatic GPU matching establishes

The supported correlation chain is:

```text
execution-trace operation rf_id
    -> Chrome CPU Record function id / External id
    -> Chrome GPU External id
```

Each unambiguously matched complete `kernel`, `gpu_memcpy`, or `gpu_memset`
event becomes a GPU node. Its cost identity includes the kernel/activity name,
recorded launch parameters, original operation signature, and hardware ID.
Repeated occurrences reuse a key only when these descriptors match. Measurement
also checks the bound artifact, event index, and launch descriptor. A kernel
name alone is insufficient.

Same-stream execution order is retained, including across the captured steps.
CPU issue order and timestamp proximity do not establish cross-stream
dependencies. CUDA waits, events, communication dependencies and contention need
additional evidence. Unknown correlations remain unsupported and visible.
This matcher does not generally reconstruct compiler fusion boundaries, CUDA
graph replay internals, hidden allocations, or collective protocol structure.

Projected tensor extents remain approximate storage evidence. When lowering
their producing operation, preparation conservatively makes those spans live
from the beginning and retains their mapped users. It does not infer actual
allocation or free events from GPU durations.

## Supply semantics from an integration

An integration can provide JSON assertions or call the same Python API:

```python
from titan_sim.prepare import prepare
from titan_sim.workload import write_workload

prepared = prepare(['results/capture/rank0.json'], config={
    'chrome_traces': [],
    'useful_tokens': 0,
})
write_workload('results/prepared.json', prepared)
```

`prepare` also accepts `Workload` objects and leaves input objects unchanged.
Assertions reference **original** node IDs as `{"rank": 0, "id": 17}`; IDs in
the prepared output may differ. The following fields are optional:

| Configuration field | Meaning |
|---|---|
| `dependencies` | Replace a node's complete incoming dependency list with explicit node references. |
| `regions` | Contract verified same-rank compute nodes into one executable region with implementation and context descriptors. |
| `communications` | Bind explicit collective participants or a send/receive pair to a group and sequence. |
| `allocations` | Replace specified observed storage records with a verified backing allocation and its lifetime users. |
| `memory`, `rank_placement` | Supply the physical-memory coverage and device placement contract. |
| `resolutions` | Record an individually scoped assertion resolving an exact existing gap. |

Dependency, region, communication, and allocation assertions require a common
`evidence` object with a nonempty `source` and a 64-character SHA256 identity.
That identity is retained as supplied; it is not independent proof that an
assertion describes the native implementation correctly.

For example, this region assertion is a schema example. Replace the node IDs,
implementation/context descriptors, and evidence identity with your integration's
actual values before use:

```json
{
  "chrome_traces": [],
  "evidence": {
    "source": "integration-region-manifest",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "regions": [{
    "nodes": [{"rank": 0, "id": 17}, {"rank": 0, "id": 18}],
    "name": "native_region",
    "implementation": {"artifact_sha256": "actual-implementation-digest"},
    "context": {"shape": [8, 16], "dtype": "float32", "cuda_graph": false}
  }],
  "dependencies": [{"node": {"rank": 0, "id": 17}, "deps": []}]
}
```

An empty dependency list asserts that the operation has no incoming dependency;
it must not be used merely to remove an inconvenient host-order edge. Region
descriptors must distinguish relevant source/runtime versions, shapes, layouts,
precision, scalar settings and execution mode. Their exact-key callable comes
from the measurement provider; preparation never synthesizes a replacement
forward pass.

A communication entry contains `kind` (`collective` or `send_recv`), `nodes`,
`group`, nonnegative `sequence`, `operation`, `participants`, and `payload_bytes`.
For `send_recv`, list the sender first and receiver second. Participant ranks
must match the nodes; captured signatures cannot be contradicted. Communication
context becomes part of its cost key, so changing group, peers, operation, or
payload cannot silently reuse the earlier measurement.

An allocation entry contains `rank`, `id`, `source_storage_ids`, `size`,
`producer` (node reference or null), `consumers`, and optional `persistent` and
`category`. Source IDs identify observed records on that rank. Multiple views
can become one backing allocation; overlapping assertions are rejected.
Physical-memory alias references follow the ID remapping. Unqualified original
storage IDs must be globally unique; an allocation's `memory.allocations`
`storage_id` can instead use `{"rank": 0, "id": "original-storage-id"}`.
See [the preparation API](../titan_sim/prepare.py) for the full accepted shapes
and [memory accounting](native-memory.md) for its separate coverage contract.

## Resolve a declared gap without hiding the evidence

Partial assertions do not automatically erase a capture's warnings. An
integration that has resolved a particular limitation can supply:

```json
{
  "resolutions": [{
    "gap": "exact string copied from manifest.unresolved",
    "reason": "Specific explanation of how the integration established coverage.",
    "scope": "integration_assertion",
    "evidence": {
      "path": "evidence/coverage.json",
      "sha256": "actual-64-character-sha256-of-that-file"
    }
  }]
}
```

Preparation checks that the named gap exists and the evidence file matches its
checksum. It retains the resolution under `manifest.preparation.resolved_gaps`
with `validation: artifact_checksum_only` and `empirical_verification: false`.
The resulting coverage rests on the integration's assertion. Unknown gaps remain
unless specifically addressed; there is no global clear switch. Missing ranks
and blocked/incomplete source captures require new capture evidence and cannot
be erased this way.

Fake or dispatch-only captures can use the same rank merging and explicit
assertions, but they do not contain real GPU activity durations. They therefore
cannot use the Chrome trace cost path. Unsupported custom operations, compiler
regions, distributed protocols, and memory coverage need a compatible native
integration before the simulation can claim calibrated coverage.
