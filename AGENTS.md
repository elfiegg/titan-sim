# Using titan-sim

1. Pin and record this repository's commit SHA. Read README.md and run
   `python -m titan_sim --help` before choosing a command.
2. Start with `python -m titan_sim --case checks`. Simulation and generated
   scenarios need only Python 3.10+; do not install or replace PyTorch for them.
3. Select an implemented capture case explicitly. `dense` is a small example;
   `hybridep` is a source-extracted fake-shape probe; `kimi` is a generated
   analytical stage graph. None captures a complete native Kimi training step.
4. Native capture belongs in a pinned training environment. Set
   `TORCHTITAN_PROTOTYPE_SOURCE` to a matching upstream checkout. Preserve the
   container's PyTorch/CUDA/custom-op versions. Record failures and adaptations.
5. Calibration requires authorized GPU resources. Use the site's launcher and a
   five-minute scheduler hard limit unless the user requests another limit. The
   benchmark does not allocate resources or enforce a timeout itself. Never run
   GPU calibration on a login node. `CALIBRATION_PHASES=compute` skips communication;
   `communication` skips compute. The current importer accepts four- or 32-rank runs.
6. Keep raw logs, private paths, credentials, GPU identifiers and generated results
   out of commits. Do not modify external skill files as a side effect of this work.
7. Check `unresolved`, `cost_coverage`, `status`, and `memory_verdict` in outputs.
   Exploratory success is not successful recipe execution. `--strict` rejects
   declared gaps and uncalibrated costs; it does not independently prove capture
   completeness. Fake-execution wall time is never GPU time. Do not add allocated
   and reserved bytes or treat component memory as a full-model peak.
8. Match operation signatures, implementation settings, software, hardware and
   topology before reusing calibration. Held-out validation measurements must not
   enter the fitted cost catalog. Unknown operations remain unresolved.

Run `python -m unittest discover -s tests -v` for CPU changes. Report actual GPU
validation scope separately. The simulator uses seconds and bytes. The previous
repository implementation used a different event API and microsecond units;
historical traces and calibration files are not automatically compatible.

The proposed general `TorchTitanAdapter.capture(recipe, scenario)` and automatic
native calibration-target generation are not implemented. Do not present them as
available tools. See the README's component table and development sequence.
