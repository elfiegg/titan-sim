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

A packaged native guard feasibility diagnostic now reproduces that boundary:

```bash
python -m titan_sim.native_guard
# In the pinned NVIDIA CUDA-enabled container, with no GPU execution:
python -m titan_sim.native_guard --allow-cuda-build
```

The wrapper compiles a C++20 extension in a temporary directory and loads it only
in a fresh child process with `CUDA_VISIBLE_DEVICES` empty. CUDA-enabled builds
require the second command's explicit option. The child atomically saves and
replaces the C10 CUDA guard, then restores it in `finally`; the caller's guard is
never changed. Native virtual streams retain thread-local stream identity and
native events record producer/consumer dependency edges. Completion queries,
synchronization, elapsed time and native handle access remain unsupported.
A generic C10 CUDA allocator trap rejects allocation and copying; this is not a
trap for every internal CUDA caching-allocator entry point. Hidden GPUs and
FakeTensor fallback being disabled remain required boundaries.

The [local structured result](native-metadata-guard-probe.json) records successful
compilation, virtual-stream/event self-tests, rejection of a real CUDA allocation,
and fake-CUDA tensor creation and forward edge construction. Backward remains
blocked on local PyTorch 2.14.1. Its compiled accelerator is MPS:
[`Node::stream()`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/node.h)
selects input metadata matching `at::getAccelerator()`, so CUDA metadata has no
canonical parent stream. A guard replacement cannot change this accelerator
selection. The failing engine assertion checks that the incoming gradient and
parent node both have streams.

The [NVIDIA-container result from job 3045179](extensions-3045179/native-guard.json)
now establishes a separate target boundary. PyTorch
`2.14.0a0+b2c75dd062.nv26.09` selects CUDA, builds the extension, passes virtual
stream/event and generic allocator-trap checks, rejects a real CUDA allocation,
and completes fake tensor creation and forward. Backward fails with
`hasPrimaryContext expects a valid device index, but got device_index=0`.
The original guard is restored. This is a structured `blocked` result even
though the diagnostic process exits with code zero.

CUDA primary-context queries still use native accelerator hooks and the real
visible-device count. Replacing the C10 guard does not replace those hooks.
The inspected hook interface exposes lookup and registration but no supported
scoped hook override for an already initialized CUDA build. No safe bounded
DeviceGuard-only fix was established. Further work would require a separately
reviewed metadata accelerator/hooks path in a pinned PyTorch build (or an
upstream-supported extension point), with explicit autograd stream and event
semantics. This experiment stops at that framework boundary; it does not
promise or implement a patched vendor build, suppress the assertion, or mark
native backward as supported.

Even successful elementary default/multistream backward would establish only a
component capability: native event edges are not yet integrated into the
simulator's event scheduler, and full TorchTitan/FSDP/EP execution remains a
separate validation gate. The diagnostic does not install a training backend,
change numerical kernels, or suppress autograd assertions.
