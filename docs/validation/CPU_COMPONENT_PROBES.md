# CPU metadata component probes

The [recorded 671B construction probe](cpu-671b-construction.json) completed on
macOS ARM64 with PyTorch 2.14.1 and TorchTitan
`607852350b41b4b280d8b8de6340f4847ed71c5e`. It built 671,026,404,352 parameter
elements and initialized 235 buffers using CPU FakeTensors. This is an
unsharded component test. It does not validate FSDP, EP, forward/backward,
optimizer execution, or a memory trajectory for the 256-GB200 target.

The explicit `post_initialization` scope omits random sampling within upstream
parameter initializer callbacks. It preserves their other operations and all
buffer initialization. This probe omitted random sampling inside 603 parameter initializer callbacks,
while retaining their other operations. It omitted 722 sampling calls: one normal sampler and 721 truncated-normal samplers.
Parameter values remain unknown; initialization time and peak memory are
excluded. Strict capture remains the default. The current comparison gate
rejects this scope until an explicitly scoped steady-state comparison contract
is available.

Reproduce the component probe from the repository root with the pinned
TorchTitan checkout on `PYTHONPATH`:

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from titan_sim.recipes import build_smoke_model_config
from titan_sim.torchtitan_adapter import post_initialization_scope

config = build_smoke_model_config(flavor="671B", seq_len=16384)
with FakeTensorMode(allow_fallback_kernels=False), torch.device("cpu"):
    model = config.build()
    with post_initialization_scope() as coverage:
        model.init_states(buffer_device=torch.device("cpu"))
    print(sum(parameter.numel() for parameter in model.parameters()))
    print(len(list(model.buffers())))
    print(coverage)
```

Three separate limitations were observed or identified:

- Strict initialization fails on a FakeTensor value read in
  `torch.nn.init.trunc_normal_`: rejection sampling tests `mask.any()`. The same
  implementation was read directly from the cached NVIDIA PyTorch 26.09
  container (`torch 2.14.0a0+b2c75dd062.nv26.09`). Its data-dependent iteration
  count cannot be recovered from parameter shapes.
- With GPUs hidden, Python CUDA controls can be emulated but native autograd
  still requests a C++ CUDA device guard. The local CPU build aborts during
  forward autograd-edge construction. The selected NVIDIA source routes this
  guard through native GPU-count and stream-priority queries. NVIDIA execution
  verification remains pending.
- An isolated local call to upstream compiled Flex mask construction crashed
  after a FakeTensor data-pointer warning. This was a separate component probe,
  not part of the successful construction result. No compiler or mask-policy
  replacement was applied.

A separate scratch C++ guard experiment advanced local fake-CUDA execution into
backward, where the CPU build hit an autograd stream assertion. It is not enabled
in the package and has no validated cross-stream semantics.
