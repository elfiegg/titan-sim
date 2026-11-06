# Agent entry points

Pin and record this repository's commit SHA. Read README.md and run
`python -m titan_sim --help`. The public operations are **runtime-check, capture,
prepare, validate, calibrate, simulate**. There is no `--case` or built-in model selection.

- CPU simulation and structural validation require Python 3.10+ only. Do not
  install or replace PyTorch for these operations.
- `capture --adapter module:function --config config.json --output workload.json`
  invokes an explicit trusted Python adapter. The adapter returns `Workload`.
  It owns framework dependencies, semantics and source/container provenance.
- `calibrate workload.json --runner module:function --config measurement.json
  --output costs.json` invokes a trusted measurement runner with `(workload, config)`.
  The runner owns native implementation matching and distributed measurement;
  the generic CLI validates the returned catalog. Do not invent measured samples.
- `validate workload.json` checks structural references and reports declared gaps.
  It does not establish real execution or complete capture.
- `simulate workload.json --calibration costs.json --strict --output result.json`
  uses exact cost keys and rejects declared gaps or uncalibrated costs.
- The experimental native TorchTitan adapter is `titan_sim.torchtitan:capture`.
  Read `docs/native-torchtitan.md`. Native execution defaults to PyTorch's profiler;
  explicit fake/eager dispatch capture has narrower compatibility. Neither path
  establishes universal model support, complete distributed lowering, or automatic
  native compiled-kernel calibration. Do not present these gaps as implemented.
- Native capture executes the original recipe and its configured side effects.
  Require explicit execution mode, use a fresh trace directory, preserve original
  model code, and write one output per rank with `{rank}` in the output path.
- Model-specific prototypes remain in `examples/recipe_prescreen/README.md`.
- The repository-owned `skills/simulate-torchtitan/SKILL.md` consumes the upstream
  prepare-runtime handoff and caller-owned execution context. Runtime artifact
  readiness is not a live container probe or full-scale recipe qualification.
- `prepare` reconciles rank artifacts, direct GPU correlations and explicit
  execution/storage assertions. Missing ranks, failed capture and unsupported
  semantics remain blockers. Evidence-backed gap resolutions are labeled
  integration assertions, not automatic empirical verification.
- The bundled runner `titan_sim.native_runner:measure` imports matched profiler
  GPU observations or times trusted native callable regions. Preserve that
  distinction and exact context identities. Existing all-rank process groups are
  supported; distributed graph/subgroup-only measurements are not.
- `memory_verdict` now reports physical-device envelopes; `logical_memory_verdict`
  retains the simpler storage-only assessment. Report `memory_unresolved` too.
  `--require-memory-fit` checks capacity independently of timing `--strict`.
  Do not mark all coverage flags true without supporting evidence.

Use a compatible pinned training environment for native capture/calibration and
preserve its installed PyTorch/CUDA/custom-op versions. GPU measurement requires
explicitly authorized allocations; a repo invocation does not authorize cluster
jobs. Use a five-minute scheduler hard limit for initial benchmarks unless the
user requests another limit. The Python harness does not enforce that limit.
Do not run GPU calibration on a login node. Use one coordinating process to write
an aggregate catalog, not competing distributed ranks writing the same output.

Report `status`, `cost_coverage`, `unresolved` and `memory_verdict`. Fake-execution
wall time is not GPU time. Do not add allocated and reserved bytes or promote
incomplete logical storage coverage to a physical memory-fit claim. Hardware and
sample checksums are necessary checks, not proof of native implementation fidelity.
Keep held-out validation separate from fitted costs.

Keep credentials, private paths, GPU identifiers, raw logs, environments and
generated results out of commits. Do not modify external skill files as a side
effect of running this repository. Run `python -m unittest discover -s tests -v`
for CPU changes and report actual GPU validation scope separately.

Durations use seconds, sizes use bytes. Pin an older commit to reproduce the
superseded `--case` CLI or the earlier microsecond-based event implementation.
