# titan-sim

CPU event simulation and metadata capture for PyTorch TorchTitan training.

The target experiment is DeepSeek-V3 671B at the user-selected TorchTitan
revision on 256 GB200 GPUs: PP=1, EP=64, TP=1, FSDP auto, and full activation
checkpointing (`FullAC.Config`). It uses the
newest public NVIDIA PyTorch container resolved in `profiles/versions.json`.
This is experimental software. Full target validation is recorded separately;
a passing event-engine test is not a demonstrated 671B training projection.

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

The smoke path is an explicitly unsharded eager BF16 operator-coverage experiment.
It is not the production recipe. `project_engine` attempts the selected recipe with the requested parallelism
and FullAC overrides while preserving its compiler policy. `--routing-scenario balanced` is a
conditional equal-split scenario, not a prediction of routing decisions.

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
