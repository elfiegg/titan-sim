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
FSDP=4 through upstream ParallelismContext. The local suite passes 73 tests
with these upstream integration checks enabled (macOS PyTorch 2.14.1).

A local metadata smoke attempt reaches model initialization, then stops at
PyTorch 2.14.1 `trunc_normal_`: its `mask.any()` branch requires tensor values
and raises `DataDependentOutputException`. This is a local FakeTensor coverage
finding, not evidence of a failure in the requested NVIDIA container. No
initialization workaround or model substitution has been applied.

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

Latest local verification: 66 tests passed with no skips on PyTorch 2.14.1.
Editable package installation and the installed command-line example also passed.
The upstream TorchTitan checkout remains unmodified.

Pure scheduler tests exercise causality, host synchronization, aliases, delayed
reclamation, rank separation, timing coverage, and invalid input rejection.
PyTorch CPU tests exercise real dispatch/autograd metadata, tensor storage
identity, alias lifetime, saved activations, optimizer state, and scoped runtime
controls. Report tests distinguish theoretical ceilings from measured points.

The local test environment uses public macOS PyTorch 2.14.1. Passing tests there
are separate evidence from the specified NVIDIA container. Initial tests in
that container exposed the native Stream API difference; the updated Python
control shim subsequently passed local tests and awaits target revalidation.

## Outstanding target validation

The latest authored source transfer to the development cluster is pending
explicit approval after automatic approval review rejected the transfer.

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
