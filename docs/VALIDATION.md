# Validation record

This initial implementation is a foundation, not a validated full DeepSeek-V3
runtime predictor. No 256-GPU training accuracy result is claimed.

## Current selected validation target

The user replaced the initial tip-of-tree revision with
`607852350b41b4b280d8b8de6340f4847ed71c5e` and requested PP=1, EP=64, TP=1,
automatic FSDP sharding, and full activation checkpointing. The shared recipe
builder applies upstream `FullAC.Config`, calls the original model sharding and
configuration validation, and preserves FlexAttention and local compilation.
Expected dense FSDP/input DP degree is 256; expert FSDP degree is 4.
Both projection and GPU adapters use this shared configuration.

Configuration integration tests load the actual selected upstream recipe,
validate the FullAC and compile policies, and resolve dense FSDP=256 / expert
FSDP=4 through upstream ParallelismContext. The local suite passes 96 tests
with these upstream integration checks enabled (macOS PyTorch 2.14.1).

A local metadata smoke attempt reaches model initialization, then stops at
PyTorch 2.14.1 `trunc_normal_`: its `mask.any()` branch requires tensor values
and raises `DataDependentOutputException`. The same data-dependent implementation was confirmed by source inspection
in the exact cached NVIDIA image; a target runtime probe remains pending.
Strict capture still stops rather than inventing parameter values. An explicit
`--capture-scope post_initialization` option omits parameter random sampling,
preserves upstream buffer initialization, and excludes initialization timing and
startup peak from its claims. This permits full-model shape construction; it does
not establish forward/backward, sharded execution, or a complete trajectory.

The new revision does not contain the optional pipeline API imports or the
new FSDP callback argument that caused the earlier failures. Native API
execution is now the default; previous compatibility wrappers are opt-in.
This source inspection is not a successful training run.

The environment and failures below describe the **initial revision** unless
explicitly stated otherwise; they are retained as historical evidence.

## Environment established

- Initial TorchTitan tip-of-tree pinned at `dbfba1da8ba1cd5688f32d16321a32aaf9132a43`.
- Newest public monthly image resolved from registry tags:
  `nvcr.io/nvidia/pytorch:26.09-py3`; no 26.10 tag was present at resolution.
- Compute-node import confirmed PyTorch `2.14.0a0+b2c75dd062.nv26.09`, CUDA 13.4,
  NCCL 2.31.2, and NVIDIA GB200 devices.
- Four-rank NCCL all-reduce completed successfully on one GB200 node.
- Observed per-device PyTorch capacity: 197,897,617,408 bytes (184.31 GiB).
- The initial default production recipe resolved to 671B flavor, 16,384-token context,
  65,536 tokens per microbatch per data-parallel rank, EP=2, PP=1, and automatic
  data-parallel sharding. The experiment targets world size 256.

Container digest pins are in `profiles/versions.json`. These observations
establish software/hardware identity and communication availability; they do
not establish training memory or speed accuracy.

## Historical compatibility findings at the initial revision

The initial TorchTitan revision and newest public NVIDIA PyTorch image did not run together
without adaptation:

1. TorchTitan eagerly imports `analyze_pipeline_activation_liveness` and
   `PipelineStageInfo`, absent from this PyTorch build. The default recipe does
   not use the optional Dist-MoE pipeline path. `compat.py` installs raising
   placeholders for missing names only. A direct import probe reached the
   original TrainingEngine and production recipe successfully.
2. The reduced GPU run reached FSDP initialization, then failed because this
   PyTorch `MixedPrecisionPolicy` lacks `param_dtype_override_fn`. The upstream
   default passes `None`. A local guarded adapter accepts only that no-op value;
   a non-None callback fails. This latest adapter has not yet been revalidated on
   the target GPU environment.
3. The initial fake-CPU upstream smoke reached model construction, then failed
   during a parameter dtype conversion because upstream tensor swapping rejects
   existing weak references. The smoke adapter now constructs BF16 parameters directly. A full-engine
   wrapper invalidates only FakeTensor converter memo entries before native
   parameter swapping, preserving the remaining native checks. Both changes
   have local unit coverage but await target-environment validation.

Default-path FSDP mesh APIs, DeviceMesh concatenation, flash-attention selection,
and variable-length attention imports were verified present. No computation is
substituted for a missing implementation.

## What tests establish

The earlier local verification passed 96 tests with no skips on PyTorch 2.14.1.
Editable package installation and the installed command-line example also passed.
The comparison exporter passed a separate synthetic JSON/CSV/SVG check; this
checks presentation only and is not a DeepSeek-V3 result.
The upstream TorchTitan checkout remains unmodified.

Pure scheduler tests exercise causality, host synchronization, aliases, delayed
reclamation, rank separation, timing coverage, and invalid input rejection.
PyTorch CPU tests exercise real dispatch/autograd metadata, tensor storage
identity, alias lifetime, saved activations, optimizer state, and scoped runtime
controls. Report tests distinguish theoretical ceilings from measured points.

The local test environment uses public macOS PyTorch 2.14.1. Passing tests there
are separate evidence from the specified NVIDIA container. Initial tests in
that container exposed the native Stream API difference; the updated Python
control shim subsequently passed the selected-container 96-test run described below.

## Outstanding target validation

The user explicitly approved the authored-source transfer, and it succeeded.
A finite, self-running development validation script installs the selected
upstream requirements while constraining the container PyTorch version, records
the environment, runs unit and FullAC effect tests, records CPU probe failures
separately, then attempts ten-step single-GPU and four-GPU EP debug experiments.
The first attempt with this script failed during dependency installation before
unit tests or GPU training. The runtime version string used as the pip constraint
differed from the installed NVIDIA wheel metadata version. The installer now
uses distribution metadata for the constraint and retains its separate exact
runtime assertion; a local regression test covers the mismatch. The dependent
full run was cancelled as designed.

The replacement attempt confirmed that dependency correction and recorded
TorchTitan `607852350b41b4b280d8b8de6340f4847ed71c5e`, runtime PyTorch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA 13.4, NCCL 2.31.2, and four GB200 devices
with 197,897,617,408 bytes each. All 96 simulator tests then present passed.
The next upstream test, `test_full_ac_does_not_recompute_registered_effects`,
failed: its registered effect executed twice instead of once after backward.
The validation script stopped before CPU capture probes and GPU debug training;
the dependent full run was cancelled. This was not an out-of-memory failure.

The selected FullAC policy requests recomputation and relies on the checkpoint
runtime to preserve registered effects. A standalone CPU diagnostic reproduces
effect replay without simulator imports. Its explicit-save diagnostic control
avoids replay, and both cases produce the expected gradients. No upstream policy
was changed. This establishes a checkpoint/effect-contract incompatibility,
not demonstrated DeepSeek gradient corruption. See
[local diagnostic](validation/checkpoint_effect_probe_local.json) and
[reproduction script](../scripts/probe_checkpoint_effects.py). The local and
[inspected NVIDIA checkpoint source](validation/checkpoint_source_nv2609.json)
have identical SHA-256 hashes; source inspection remains separate from runtime
execution evidence.

Still required before a DeepSeek-V3 accuracy report:

- Re-run all tests and both reduced adapters in the specified container.
- Complete metadata execution of the full selected sharded recipe, including
  FullAC checkpointing and the actual compilation policy.
- Resolve native fake-CUDA autograd on a CPU-only host, data-dependent routing,
  and actual asynchronous FSDP/collective control semantics.
- Run the selected 671B configuration for at least 10 steps on 256 GB200 GPUs.
- Compare independently produced projection and allocator observations at
  matched phase boundaries and with an identical policy/routing fingerprint.
- Report rank distribution, steady-state and optimizer-initialization peaks,
  allocator reservation, error, unsupported allocations, and timing coverage.

The included `examples/two_streams.json` is a synthetic scheduler example.
Its charts must never be presented as DeepSeek-V3 results.

Full-model shape evidence and remaining CPU execution blockers are detailed in
[CPU component probes](validation/CPU_COMPONENT_PROBES.md).

## Supplied historical runtime logs

Latest local verification with the selected upstream checkout enabled: **129
tests passed, no skips**, on macOS PyTorch 2.14.1. This includes 14 observation
parser tests and 19 component-budget tests. The five-run analysis CLI completed,
source-hash mismatch rejection was checked, and all three generated figures
were visually inspected. The two declared budgets remained incomplete and
returned unknown required memory on both capacity profiles, as intended.

The [runtime memory analysis](RUNTIME_MEMORY_ANALYSIS.md) audits five supplied
DeepSeek-V3/V4 workloads before using their observations. Its importer preserves
source hashes, rank uncertainty, units, and counter windows. Explicit stash
payloads are reconstructed from logged geometry; separate conditional budgets
use declared shapes and lifetimes without fitting observed totals. All five
historical runs fail the matching-context gate for the selected target.
There is still no independent full-model CPU/GPU memory-error result.
