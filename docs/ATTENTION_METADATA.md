# Original FlexAttention metadata contract

`titan_sim.attention_metadata.flex_attention_metadata_scope()` enables original
TorchTitan DeepSeek attention on CPU FakeTensors with numerical fallback
disabled. It is opt-in and distinct from the native CUDA training path.

The scope retains the original DeepSeek MLA module, upstream mask construction,
PyTorch FlexAttention higher-order operator, autograd function and saved tensors,
and registered forward/backward FakeTensor implementations. It disables the two
upstream compilation wrappers and uses PyTorch's direct-HOP debugging entry.
The public device validator checks detached fake views so its numerical-CPU
backward restriction does not reject the existing fake backward kernel. Original
inputs keep their autograd edges. Real inputs and fallback-enabled modes fail
before entering the attention operation. All patched attributes are restored on
exit, including when an exception occurs. Because these are module attributes,
callers must serialize the scope.

This preserves exposed Q/K/V, output, FP32 logsumexp, FP32 maximum-score and mask
storage metadata. Tests check that Q/K/V are saved, that logsumexp and maximum
scores remain live until backward, that those temporary storages release after
backward, and that all three input gradients have their original shapes. The
captured forward and backward operations are `flex_attention` and
`flex_attention_backward`; there is no dense attention replacement.

Compilation is disabled, so Python mask closure tensors are not lifted into
HOP buffer arguments. Their read dependencies remain unobserved. Nonidentity
`score_mod` callbacks are rejected: without compiler buffer lifting, a captured
differentiable score tensor would otherwise lose its gradient edge. The original
DeepSeek identity-score path is supported. Inductor fusion, compiled kernel
scratch, numerical values, CUDA scheduling, and compiled storage lifetimes are
outside the contract. HOP events explicitly have unknown workspace and an
unsupported calibration signature. They cannot establish CUDA memory or timing
equivalence.

The original unsharded debug and 671B DeepSeek MoE also run forward/backward without an
operator adapter: routing, auxiliary loss, local dispatcher behavior,
`aten._grouped_mm`, shared experts and deterministic scatter use their existing
upstream implementations. Expert counts remain unknown FakeTensors. No routing
counts or EP split sizes are fabricated. An isolated grouped GEMM backward probe
must use a contiguous gradient seed, such as `out.backward(torch.empty_like(out))`;
`out.sum().backward()` supplies a zero-stride expanded gradient that the original
grouped GEMM metadata implementation rejects.

The 671B MoE check retains its original 7,168-wide input, 256 experts, top-8
selection and group-limited routing (eight groups, four selected groups). All
five MoE parameter tensors receive gradient metadata.

Run the integration checks from the repository root with the pinned TorchTitan
checkout and PyTorch 2.14 environment:

```bash
TORCHTITAN_ROOT=/path/to/torchtitan python -m unittest tests.test_attention_metadata -v
```

On TorchTitan `607852350b41b4b280d8b8de6340f4847ed71c5e` and macOS PyTorch
2.14.1, the original debug and 671B models, chunked cross-entropy loss, backward
and direct AdamW completed under this scope. Applying original
`FullAC.Config().build()` also completed: the debug model executed six forward
and six recomputed attention calls; the 671B model executed 61 forward and 61
recomputed attention calls. The 671B probe used 16 tokens and sequence length
16. The reproducible CLI run recorded 193,205 storage/operation events; see
[the step summary](../results/cpu-step/full-671b-fullac-step.summary.json).
These are unsharded BF16-parameter, post-initialization operator diagnostics; they
do not validate the production quantization, FSDP/EP, or 256-GPU recipe.
