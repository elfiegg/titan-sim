# titan-sim

CPU event simulation and metadata capture for PyTorch TorchTitan training.

The target experiment is upstream DeepSeek-V3 671B on 256 GB200 GPUs, using the
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
  --torchtitan-root /path/to/torchtitan --output cpu-full.json

# Run under a suitable torchrun/Slurm launch with the requested world size.
python -m titan_sim.launch gpu_compare --flavor 671B --steps 10 --output observations
```

The smoke path is an explicitly unsharded eager BF16 operator-coverage experiment.
It is not the production recipe. `project_engine` attempts the original recipe,
including its parallel and compiler policies. `--routing-scenario balanced` is a
conditional equal-split scenario, not a prediction of routing decisions.

`launch` enables a narrowly scoped import compatibility context: the public
container lacks two pipeline analysis exports that TorchTitan eagerly imports
for optional Dist-MoE code. Raising placeholders permit unused imports only;
they cannot execute those features. Any applied compatibility must appear in
run metadata. This does not upgrade PyTorch or modify the upstream checkout.

See [implementation items](docs/IMPLEMENTATION_ITEMS.md) for independent agent
work units, acceptance checks, and remaining fidelity work.

The FSDP compatibility wrapper also permits the newer optional
`param_dtype_override_fn` keyword only when it is `None`; a non-None callback
raises. Current target-validation limits and observed failures are recorded in
[validation record](docs/VALIDATION.md).

For captured projections, calibrated/hybrid CLI runs require a calibration
`context` matching `torch_version`, `torchtitan_commit`, and the hardware profile
`name` (stored as `hardware`). Captured operation signatures include tensor
layout/strides and scalar arguments. Host/device transfer timing additionally
requires exact calibration or an explicitly supplied effective host-link
bandwidth; HBM bandwidth is not used for that link.
