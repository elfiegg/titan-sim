# Checkpoint execution and storage capture

Pinned TorchTitan `607852350b41b4b280d8b8de6340f4847ed71c5e` implements
`FullAC` with native non-reentrant checkpoint wrappers and selective checkpoint
contexts. Its `_full_ac_policy` always returns `PREFER_RECOMPUTE`. The policy
expects PyTorch to preserve registered effects. The simulator must retain that
policy, early stopping, RNG preservation, and the configured compilation regions.

`titan_sim.checkpoint.checkpoint_phase_tracking()` observes entry and exit of
the original PyTorch selective checkpoint modes. `execution_phase` distinguishes
their original forward from recomputation and the caller's backward/optimizer
phase. These annotations supplement the step phase; they do not change policy,
dispatch, mode classes, caches, or saved-tensor hooks. Native early-stop exits
restore the previous phase. Installation uses process-scoped patches and does
not support overlapping capture sessions in different Python threads.

Module backward hooks alone cannot identify recomputation: the module executes
another forward during backward. On PyTorch 2.14.1, the policy is evaluated only
in the original forward; the backward mode consults saved policy decisions.
Consequently, observing `ctx.is_recompute` in the policy is also insufficient.

Saved-tensor hook entries are not a complete inventory of checkpoint storage.
Non-reentrant checkpointing installs its own inner hooks, and selective
checkpoint caches can retain values outside normal autograd saves. In the
pinned runtime, these caches bypass outer hooks unless explicitly opted in.
Capture must leave the default unchanged and follow actual StorageImpl
lifetimes. A regression test checks that a MUST_SAVE matmul output keeps the
same storage during replay and that its storage expires after backward.

Compiled forward outputs can include tensors saved for backward. Their storage
must survive the forward region; a compiled plan must declare those values in
`retained_buffers`, then pass their physical storage IDs to the backward plan
as external storages. Compiler tracing tensors and FX logical intermediates
are not GPU allocations. The existing compiled manifest validator requires a
hash-bound allocation artifact before accepting `plan_source=compiler_artifact`
and rejects reuse that overwrites retained outputs. Phase annotations alone do
not establish compiler workspace, fusion, or physical saved-storage coverage.

The local registered-effect diagnostic still fails the expected preservation
contract: effect calls change from one after forward to two after backward,
while the explicit MUST_SAVE control executes once. Both identity examples
produce correct gradients. This is a real regression-test failure, but it does
not establish that a DeepSeek-V3 training step fails: the diagnostic uses its
own ordered-effect custom operator. The selected model source has no such
registration. Runtime libraries may still introduce effects, so actual model
execution remains a separate check. Do not bypass the failing test by changing
FullAC or represent this diagnostic as model gradient corruption.

Local tests exercise replay, gradients, AdamW state, storage lifetime, nested
contexts, error cleanup, and Dynamo/AOTAutograd compatibility with the CPU
`aot_eager` backend. The last test preserves compiled checkpoint mode contracts;
it is not evidence of GPU Inductor allocation equivalence.
