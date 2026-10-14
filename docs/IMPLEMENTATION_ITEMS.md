# Implementation worklist

The target comparison is full DeepSeek-V3 671B, the user-selected revision and PP=1 / EP=64 / TP=1 / FSDP-auto / FullAC policy,
256 GB200 GPUs, and the newest public NVIDIA PyTorch container resolved in
`profiles/versions.json`. Synthetic fixed-length token batches and random model
initialization replace dataset/tokenizer setup for the memory experiment; model,
optimizer, attention, rematerialization, and parallel policies must be recorded.

Each item has one owner and a bounded file boundary. An implementation item is
not validated merely because a module imports or a reduced model runs.

| ID | Work item | Files / owner | Acceptance |
|---|---|---|---|
| ENV-01 | Resolve newest public ARM64 image and exact PyTorch/CUDA versions | Cluster agent; private run ledger, public versions manifest | Registry tag/digest evidence and compute-node import |
| ENV-02 | Pin user-selected TorchTitan revision and compatible dependencies | Integration lead | Commit, clean source diff, exact environment |
| CAP-01 | Metadata dispatch capture, storage identity, aliasing, lifetime | Capture agent: capture.py | No numeric model execution; view/copy/lifetime tests |
| CAP-02 | Actual DeepSeek-V3 forward/backward/optimizer smoke | Capture agent: torchtitan_adapter.py, recipes.py | Successful CPU metadata run with every override visible |
| CAP-04 | CPU-only native device/stream guard compatible with autograd | Native-runtime agent; isolated extension and fake_device.py | Forward/backward with GPUs hidden; native and Python stream identity/order agree; no silent no-op guard |
| CAP-05 | Explicit initialization and steady-state capture scopes | Capture agent: torchtitan_adapter.py, provenance and comparison contracts | Preserve buffer setup; unknown values explicit; excluded startup phases cannot enter full-run comparisons |
| CAP-03 | Full recipe, rank-local FSDP/EP metadata execution | Capture agent and integration lead sequentially | Actual 671B local shards, optimizer, and recomputation; no reduced-model substitution |
| RUN-01 | Event schema, causal and stream scheduler | Runtime agent: events.py, runtime.py | Dependency/stream/host-wait tests; cycle diagnostics |
| MEM-01 | Storage lifetime, asynchronous safe release, per-rank trajectory | Runtime agent | Aliases counted once; outstanding users retain storage |
| MEM-02 | Allocator pools, reservation, external memory | Follow-up memory agent | Match defined allocator policy; separate uncertainty and external allocations |
| COST-01 | Hardware profiles and roofline duration provider | Runtime agent: costs.py | Correct units, dtype path, zero-work ops, unsupported-mode rejection |
| COST-02 | Calibrated lookup and explicit fallback coverage | Runtime agent | Strict misses rejected; hybrid provenance reported |
| DIST-01 | Fake ProcessGroup, Work/future and collective ordering | Follow-up distributed agent | Backend-specific host/stream waits; all participants matched |
| DIST-02 | Actual pipeline and P2P schedule execution | Follow-up distributed agent | Upstream schedules, stage imbalance, buffer lifetime and deadlock tests |
| DIST-03 | Connect all-rank model capture and FSDP to the metadata backend | Distributed integration agent | Real DeviceMesh/FSDP hooks; capture compute/storage across processes; registered P2P; no rank-zero extrapolation |
| GPU-01 | Actual TrainingEngine synthetic benchmark harness | Integration lead: gpu_compare.py | Original model/optimizer policies, >=10 steps, initialized optimizer states |
| GPU-02 | GPU memory history and phase trajectory collection | Integration lead | Valid completed allocation trace; unsynchronized and synchronized observations labeled |
| VAL-01 | Compare CPU projection with independent GPU observations | Integration lead | Same config and phase; peak and trajectory errors; no copying actual allocations into prediction |
| REP-01 | JSON/CSV/timeline exports and comparison plots | Integration lead: reporting.py, cli.py | Reproducible reports; scope/provenance visible |
| COMP-01 | Compiled-region manifests and graph replay | Follow-up compiler agent | Actual fused allocation/lifetime plan; no eager-as-compiled claims |
| COMP-02 | Dynamic compiler branches and full training-region capture | Compiler integration agent | Explicit conditional allocation paths, backward/optimizer regions and unsupported-operation coverage |
| QUANT-01 | Capture native quantized training operations | Quantization integration agent | Actual backend payload/scale layout, conversions, backward/optimizer, amax communication and scratch; independent allocator comparison |
| MEM-03 | Native caching and CUDA async allocator fidelity | Allocator integration agent | Versioned policy, stream arenas and reuse, expandable segments, graph pool destruction, measured fragmentation comparisons |
| MOE-01 | Routing scenario/replay and expert load distributions | Capture/distributed owners sequentially | Explicit source of routing values and equal comparison workload |
| HOST-01 | Host overhead and optional wall-clock pacing | Follow-up runtime agent | Metadata/tracing overhead excluded; real blocking semantics preserved |
| CAL-01 | GPU kernel/collective calibration collection | Follow-up calibration agent | Version/shape keys and held-out accuracy results |
| CAL-02 | Topology and concurrent-work validation | Calibration integration agent | Independent collective timings, compute/communication contention, launch overhead and held-out full-step errors |
| QA-01 | Independent specification and code-quality reviews | Review agents | No open correctness findings; CPU/GPU checks scoped |
| PUB-01 | GitHub repository and reproducible public artifacts | Integration lead | No credentials/private cluster data; published commit and reports |

## Integration contract

Capture documents contain `schema_version`, `source`, `metadata`, `events`, and
coverage/limitations. Events have `event_id`, `kind`, `rank`, `stream`, `name`,
`deps`, dtype/FLOP/traffic fields, and backing-storage reads/writes where applicable.

Runtime entry point: `simulate(events, hardware, timing="roofline", calibrations=None)`.
It returns `timeline`, `memory`, `summary`, and explicit fidelity/coverage information.
Actual GPU observations are separate documents; they are never relabeled projections.

## Validation levels

1. Pure CPU semantic tests.
2. Actual upstream reduced DeepSeek-V3 metadata and GPU comparison.
3. Full 671B projection with target local sharding.
4. Full 671B / 256-GB200 actual comparison.

Levels 1-3 do not establish level 4. Incomplete items remain visible in the release
scope and final report.

## Current implementation status

Implemented foundations: CAP-01, RUN-01, MEM-01, COST-01, COST-02, REP-01.
These have semantic tests; fidelity still depends on the supplied event stream.
ENV-01 registry/runtime versions are verified; cached-image byte-for-byte OCI provenance remains unverified.
ENV-02 now selects the requested native-API revision; historical compatibility
adapters are opt-in and are not used for target validation.
GPU-01/GPU-02 and CAP-02/CAP-03 have entry points but are not target-validated.
HOST-01 now intercepts Python-level live submissions and host waits using a
coordinated virtual clock; native interception and independent host clocks remain.
DIST-01 now has collective participant matching, a simulated ProcessGroup and
Work/Future interface, and a registered metadata-only CPU multiprocess backend.
MOE-01 has a declared balanced-split scenario, not measured routing replay.
CAP-05 has explicit post-initialization sampling omission and full-comparison
rejection; a scoped steady-state comparison contract remains outstanding.
CAP-04 has diagnostic probes only. VAL-01 now has gated named-phase/explicit
correspondence exports; it has no matched DeepSeek projection/observation pair.
MEM-02 now has ideal and configurable caching allocator scenarios; exact CUDA
allocator equivalence remains unproven. DIST-02 lowers an actual upstream
interleaved 1F1B schedule with declared stage costs and buffers. COMP-01 supports
static compiler plans and graph replay, with separate explicit quantization
layouts. CAL-01 has a real CUDA-event collection and held-out comparison tool.
These component implementations do not complete their full-model acceptance
checks. VAL-01 still needs a matched full-model CPU/GPU pair.
QA-01 is iterative; review findings require fixes and new verification before
scope can be promoted. PUB-01 publishes this scope explicitly.

See [validation record](VALIDATION.md) for completed evidence and the remaining
256-GPU comparison gates.

## Supplied runtime-log analysis

The [runtime memory analysis](RUNTIME_MEMORY_ANALYSIS.md) adds two implemented
foundations: a log observation importer with source hashes, explicit metric
semantics and configuration comparison gates; and a phase-based component
budget with rank ownership, storage aliasing, host/GPU separation and capacity
scenarios. Logged buffer geometry remains observation evidence and cannot be
submitted as an independent predicted component.

This log-analysis work provides tests and conditional examples for partial
memory budgets. Subsequent [execution extensions](EXECUTION_EXTENSIONS.md) add
allocator, pipeline, compiler, and runtime components. Full-model fidelity still
requires actual allocation/free and stream lifetimes, upstream execution across
ranks, matching compiler policies, and independent GPU observations.
