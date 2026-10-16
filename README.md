# titan-sim

CPU event simulation and metadata capture for PyTorch TorchTitan training.

The target experiment is DeepSeek-V3 671B at the user-selected TorchTitan
revision on 256 GB200 GPUs: PP=1, EP=64, TP=1, FSDP auto, and full activation
checkpointing (`FullAC.Config`). It uses the
newest public NVIDIA PyTorch container resolved in `profiles/versions.json`.
This is experimental software. Full target validation is recorded separately;
a passing event-engine test is not a demonstrated 671B training projection.

For a human-readable explanation of the design, governing calculations,
TorchTitan fidelity, and evidence limits, read the
[implementation analysis](docs/IMPLEMENTATION_ANALYSIS.md).

The [allocation coverage report](docs/ALLOCATION_COVERAGE_REPORT.md) recomputes
model-by-model memory deltas, separates calibrated totals from state subtotals
and conditional checks, and links the
[implementation plan](docs/plans/2026-10-15-allocation-coverage.md) for complete
allocation sizes, lifetimes, dependencies and matched GPU validation.

[Original DeepSeek CPU step capture](docs/CPU_STEP_CAPTURE.md) now completes
unsharded model forward, backward, original FullAC recomputation and direct
AdamW using an explicit FlexAttention metadata contract. Both the reduced model
and full 671B parameter dimensions execute with 16 tokens on CPU. This is a
model-operator milestone: distributed production TrainingEngine capture,
CUDA compilation and physical GPU memory validation remain incomplete.
An explicit CPU/AOT diagnostic also completes the original reduced-model
TrainingEngine step with degree-one FSDP and with rank 0 of degree-two FSDP;
its backend/device adaptations and all-rank limits are recorded separately.

The newer [execution extensions](docs/EXECUTION_EXTENSIONS.md) describe the CPU
distributed backend, upstream pipeline schedule lowering, live host waits,
compiler/quantization plans, and allocator/network scenarios, with their current
coverage boundaries. Full-model validation remains separate.

The [log-calibrated memory model](docs/LOG_CALIBRATED_MEMORY.md) adds empirical
V3 runtime allowances and V4 allocator ratios from the supplied GPU logs. It runs
on CPU, preserves calibration scope, and keeps fitted counter estimates separate
from independently simulated storage and required-capacity claims.

[Expanded-log validation](docs/EXPANDED_LOG_VALIDATION.md) tests those frozen
factors against a separate batch sweep. It finds 18–20% underestimation at
batch 1 and up to 5.24% variation in the conditional V4 reservation estimate;
the original factors remain unchanged and should not be treated as capacity bounds.

The [memory-gap diagnosis](docs/MEMORY_GAP_DIAGNOSIS.md) identifies the V3 counter
as peak reserved memory at the logged historical revision, adds bounded batch
interpolation, and reduces the largest retrospective non-fitting-point error
from 12.80% to 2.85%. It also isolates V4's 5.24% discrepancy to reservation
differences despite equal allocated counters. These are empirical estimates,
not full-model execution or required-capacity validation.

[Additional model/configuration checks](docs/OTHER_MODEL_VALIDATION.md) apply
the method to V3 16B with separate calibration (up to 2.20% excluded-batch error)
and recover legacy allocator counters from two different 15-layer V4 hybrid
configurations. Their reservation-ratio comparison reaches 0.56% maximum error,
conditional on measured current allocation; neither result establishes universal
cross-model prediction accuracy.

[GPT-OSS CPU cross-validation](docs/GPT_OSS_CPU_VALIDATION.md) now matches the
logged synthetic vocabulary and every reported per-rank parameter count across
five Megatron runs. Independent estimates include the FP32 optimizer gradient
shards retained at the reporting point and leave 2.2–17.5% of current allocation
unexplained. A retrospective factor transfer still misses larger-run current
memory by 5.6–7.4%, illustrating the limits of a single multiplier. Training peaks remain
unmodeled; the actual CPU dispatch probe stops at FlexAttention backward support.

The [GPT-OSS allocation experiment](docs/GPT_OSS_ALLOCATION_EXPERIMENT.md)
completed eight six-step runs on four/eight GB200s with MXFP8/NVFP4. Snapshots
confirm optimizer-gradient release and identify retained quantized caches,
allocator slack, dummy gradients and GEMM workspaces. A retrospective transfer
of fixed overhead components from four to eight GPUs predicts current allocation
within 0.6%; this does not validate a full CPU training peak model.

## What it models

- Unique tensor backing storage, aliases, frees, and storage retained by in-flight work.
- A separate program-order tensor ownership trajectory when host submission timing
  is unavailable. This is not CUDA allocator `memory_reserved()`.
- Stream/dependency ordering, device synchronization, and per-rank compute and
  communication lanes. Overlap is idealized and requires explicit collective dependencies.
- Dense hardware roofline lower bounds, exact operation/shape/dtype calibration,
  and hybrid lookup with visible fallback coverage.
- Actual upstream model operations under PyTorch FakeTensor metadata execution.
  Unsupported operations and data-dependent control stop the trace explicitly.
- Independent GPU allocator observations through the original TrainingEngine,
  with synthetic fixed-length documents and random initialization.

The event engine does not need PyTorch or a GPU. Full upstream capture depends on
PyTorch internals and its native CUDA control-plane requirements. Eager operator
capture cannot reproduce a compiled allocator plan. Missing scratch storage,
NCCL allocations, routing imbalance, allocator policy, and execution timings limit
memory precision. Hardware peak FLOP/s is a ceiling, not achieved utilization.

## Install and run

Python 3.10 or newer:

```sh
python -m pip install -e .
python -m unittest discover -s tests

titan-sim trace.json --hardware profiles/gb200.json --output report
```

A trace is an object containing `events` and optional `metadata`. Event times use
microseconds; capacities and traffic use bytes; hardware rates use FLOP/s and
bytes/s. See `tests/test_runtime.py` for small executable event examples.

Use `--timing calibrated --calibrations measurements.json` to reject missing
measurements, or `--timing hybrid` to report roofline fallbacks. A measurement
must include operation name, shapes, dtype, duration, and provenance. Measurements
are specific to a software/hardware environment; users must supply a matched set.

Reports contain JSON, CSV, and SVG. Captured diagnostic `capture_wall_us` never
becomes simulated GPU time. Without explicit `host_time_us` on every event,
scheduled capacity verdicts are unavailable; use the labeled logical trajectory.

## Upstream integration

Clone the commit in `profiles/versions.json` and install that checkout's
requirements into an environment using the specified NVIDIA PyTorch build.
Keep the original container PyTorch version; check it after dependency installation.
Put the checkout and this package on `PYTHONPATH`.

```sh
python -m titan_sim.launch torchtitan_adapter \
  --torchtitan-root /path/to/torchtitan --output cpu-smoke.json

python -m titan_sim.launch project_engine \
  --torchtitan-root /path/to/torchtitan --expert-parallel-degree 64 --output cpu-full.json

# Run under a suitable torchrun/Slurm launch with the requested world size.
python -m titan_sim.launch gpu_compare --flavor 671B --expert-parallel-degree 64 --steps 10 --output observations
```

The smoke path is an explicitly unsharded operator-coverage experiment with BF16
model parameters and the original operations' dtype conversions.
It is not the production recipe. `project_engine` attempts the selected recipe with the requested parallelism
and FullAC overrides while preserving its compiler policy. `--routing-scenario balanced` is a
conditional equal-split scenario, not a prediction of routing decisions.
`--capture-scope post_initialization` explicitly omits random parameter sampling
while preserving buffer initialization. Parameter values remain unknown, and
initialization runtime/startup memory peaks are excluded. Strict capture remains
the default. Neither initialization option bypasses unsupported native CUDA/autograd behavior.
For driverless CPU model execution, explicitly add `--attention-contract flex_metadata
--full-ac --steps 2` to the smoke command. That separate attention contract disables
its compiler wrappers and retains the original HOP/autograd metadata kernels;
it does not establish CUDA compiled storage equivalence.

`launch` uses native container APIs by default. The original compatibility
wrappers remain available only through explicit `--legacy-import-compat` for
the superseded TorchTitan revision; they are not enabled for this target.

The selected profile resolves to dense FSDP=256 and expert FSDP=4 with EP=64.
The code verifies resolved topology and uses upstream FullAC, including its
registered-effect handling and RNG-preservation defaults. FlexAttention and the
upstream local compile regions remain enabled. See
`profiles/deepseek-v3-256gb200.json` for the complete target.

See [implementation items](docs/IMPLEMENTATION_ITEMS.md) for independent agent
work units, acceptance checks, and remaining fidelity work.

The [runtime memory analysis](docs/RUNTIME_MEMORY_ANALYSIS.md) audits the supplied
DeepSeek-V3 and V4 logs, separates measured counters from reconstructed buffer
payloads, and explains the remaining parallel capture and validation gaps.
Two additional CPU tools support that analysis:

```sh
python -m titan_sim.log_observations training.log --output observations.json
python -m titan_sim.memory_budget examples/memory/deepseek-v3-batch8-conditional.json \
  --hardware profiles/memory/gb200-observed-1098308.json \
  --hardware profiles/memory/gb300-observed-rspring.json --output budget.json
```

The component budget tracks declared lifetimes, rank-local storage aliases,
and host versus GPU buffers. It is a separate analytical input path: it does
not generate a TorchTitan pipeline schedule. Incomplete coverage keeps required
memory unknown, even when the known subtotal is below device capacity.
Hardware capacity changes do not change declared tensor byte sizes.

To reproduce the audited reports with the original private logs and the optional
`plots` dependency installed:

```sh
python scripts/analyze_runtime_logs.py --log-root /path/to/extracted-bundle \
  --additional-log-dir /path/to/deepseek-1098308 --output results/log-audit
```

This script verifies the reviewed source hashes before attaching configuration
annotations. Raw logs are not included in the repository.

The [multi-node validation guide](docs/MULTINODE_VALIDATION.md) includes the
self-running container checks, full 64-node launcher, bounded allocator history,
and independent comparison export commands.

Current target-validation limits and historical failures are recorded in
[validation record](docs/VALIDATION.md).

For captured projections, calibrated/hybrid CLI runs require a calibration
`context` matching `torch_version`, `torchtitan_commit`, and the hardware profile
`name` (stored as `hardware`). Captured operation signatures include tensor
layout/strides and scalar arguments. Host/device transfer timing additionally
requires exact calibration or an explicitly supplied effective host-link
bandwidth; HBM bandwidth is not used for that link.

Run optional upstream configuration integration tests with:

```sh
TORCHTITAN_ROOT=/path/to/torchtitan python -m unittest discover -s tests
```

An explicitly reduced debug integration run uses `--flavor debugmodel
--expert-parallel-degree 1 --seq-len 128 --tokens 1024`; its results do not
validate the 256-rank production configuration.
