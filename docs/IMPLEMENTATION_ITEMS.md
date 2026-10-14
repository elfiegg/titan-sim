# Implementation worklist

The target comparison is full DeepSeek-V3 671B, the pinned upstream default recipe,
256 GB200 GPUs, and the newest public NVIDIA PyTorch container resolved in
`profiles/versions.json`. Synthetic fixed-length token batches and random model
initialization replace dataset/tokenizer setup for the memory experiment; model,
optimizer, attention, rematerialization, and parallel policies must be recorded.

Each item has one owner and a bounded file boundary. An implementation item is
not validated merely because a module imports or a reduced model runs.

| ID | Work item | Files / owner | Acceptance |
|---|---|---|---|
| ENV-01 | Resolve newest public ARM64 image and exact PyTorch/CUDA versions | Cluster agent; private run ledger, public versions manifest | Registry tag/digest evidence and compute-node import |
| ENV-02 | Pin TorchTitan tip-of-tree and compatible dependency revisions | Integration lead | Commit, clean source diff, exact environment |
| CAP-01 | Metadata dispatch capture, storage identity, aliasing, lifetime | Capture agent: capture.py | No numeric model execution; view/copy/lifetime tests |
| CAP-02 | Actual DeepSeek-V3 forward/backward/optimizer smoke | Capture agent: torchtitan_adapter.py, recipes.py | Successful CPU metadata run with every override visible |
| CAP-03 | Full recipe, rank-local FSDP/EP metadata execution | Capture agent and integration lead sequentially | Actual 671B local shards, optimizer, and recomputation; no reduced-model substitution |
| RUN-01 | Event schema, causal and stream scheduler | Runtime agent: events.py, runtime.py | Dependency/stream/host-wait tests; cycle diagnostics |
| MEM-01 | Storage lifetime, asynchronous safe release, per-rank trajectory | Runtime agent | Aliases counted once; outstanding users retain storage |
| MEM-02 | Allocator pools, reservation, external memory | Follow-up memory agent | Match defined allocator policy; separate uncertainty and external allocations |
| COST-01 | Hardware profiles and roofline duration provider | Runtime agent: costs.py | Correct units, dtype path, zero-work ops, unsupported-mode rejection |
| COST-02 | Calibrated lookup and explicit fallback coverage | Runtime agent | Strict misses rejected; hybrid provenance reported |
| DIST-01 | Fake ProcessGroup, Work/future and collective ordering | Follow-up distributed agent | Backend-specific host/stream waits; all participants matched |
| DIST-02 | Actual pipeline and P2P schedule execution | Follow-up distributed agent | Upstream schedules, stage imbalance, buffer lifetime and deadlock tests |
| GPU-01 | Actual TrainingEngine synthetic benchmark harness | Integration lead: gpu_compare.py | Original model/optimizer policies, >=10 steps, initialized optimizer states |
| GPU-02 | GPU memory history and phase trajectory collection | Integration lead | Valid completed allocation trace; unsynchronized and synchronized observations labeled |
| VAL-01 | Compare CPU projection with independent GPU observations | Integration lead | Same config and phase; peak and trajectory errors; no copying actual allocations into prediction |
| REP-01 | JSON/CSV/timeline exports and comparison plots | Integration lead: reporting.py, cli.py | Reproducible reports; scope/provenance visible |
| COMP-01 | Compiled-region manifests and graph replay | Follow-up compiler agent | Actual fused allocation/lifetime plan; no eager-as-compiled claims |
| MOE-01 | Routing scenario/replay and expert load distributions | Capture/distributed owners sequentially | Explicit source of routing values and equal comparison workload |
| HOST-01 | Host overhead and optional wall-clock pacing | Follow-up runtime agent | Metadata/tracing overhead excluded; real blocking semantics preserved |
| CAL-01 | GPU kernel/collective calibration collection | Follow-up calibration agent | Version/shape keys and held-out accuracy results |
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
ENV-01 is verified. ENV-02 has documented compatibility gaps and guarded adapters.
GPU-01/GPU-02 and CAP-02/CAP-03 have entry points but are not target-validated.
HOST-01 has a saved-timeline pacing demonstration only, not live interception.
DIST-01 has partial control emulation, not collective participant simulation.
MOE-01 has a declared balanced-split scenario, not measured routing replay.
MEM-02, DIST-02, COMP-01, CAL-01, and VAL-01 remain substantial independent work.
QA-01 is iterative; review findings require fixes and new verification before
scope can be promoted. PUB-01 publishes this scope explicitly.

See [validation record](VALIDATION.md) for completed evidence and the remaining
256-GPU comparison gates.
