"""Explicit CPU dispatch diagnostic for source code requesting CUDA tensors.

This is not a CUDA backend. CPU FakeTensors let the native CPU autograd engine
run without installing a device guard or changing accelerator hooks. Device
normalization changes dispatch, device branches, and potentially copy/alias
behavior; these traces cannot establish CUDA memory or compiler fidelity.
"""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def cpu_metadata_execution():
    """Run a bounded diagnostic with CPU FakeTensors and recorded device edits.

    Construct the model and inputs inside this scope. Existing accelerator
    tensors are rejected before calling the original operation. CUDA factories,
    ``Tensor.to`` and ``Tensor.cuda`` are normalized through TorchFunctionMode,
    above native autograd. Unintercepted native extension calls are outside this
    contract. Numerical fallback and automatic real-input conversion are disabled.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.overrides import TorchFunctionMode
    from torch.utils._pytree import tree_leaves

    report = dict(
        scope="cpu_dispatch_component_diagnostic",
        execution_device="cpu",
        tensor_values="unknown",
        native_autograd="cpu",
        numerical_fallback=False,
        cuda_dispatch_equivalent=False,
        cuda_memory_equivalent=False,
        cuda_stream_semantics=False,
        backward_delegation="original backward APIs with public redispatch_function and device normalization active",
        backward_calls=0,
        device_normalizations=[],
        limitations=[
            "CPU dispatch and device-dependent Python branches may differ from CUDA.",
            "Normalized device transfers can change copy and alias behavior.",
            "CUDA compiler, allocator, stream and accelerator-hook behavior is not exercised.",
            "Native extension calls bypassing TorchFunctionMode are outside this contract.",
        ],
    )

    class DeviceMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            for tensor in tree_leaves((args, kwargs)):
                if isinstance(tensor, torch.Tensor) and tensor.device.type not in {"cpu", "meta"}:
                    raise NotImplementedError(
                        "CPU metadata execution requires model and inputs constructed "
                        "inside its scope; existing accelerator tensors are unsupported"
                    )

            if func is torch.Tensor.backward or func is torch.autograd.backward:
                tensors = tree_leaves(args[0] if args else kwargs.get("tensors"))
                if not tensors or not all(isinstance(tensor, FakeTensor) and
                                          tensor.device.type == "cpu" for tensor in tensors):
                    raise NotImplementedError("CPU metadata backward requires a CPU FakeTensor")
                # Public redispatch skips only this override, leaving the
                # original argument validation and autograd behavior intact.
                # Both APIs need it: Tensor.backward calls autograd.backward,
                # and either popped mode would hide FSDP's callback factories.
                redispatch = getattr(torch.overrides, "redispatch_function", None)
                if redispatch is None:
                    raise NotImplementedError("CPU metadata backward requires torch.overrides.redispatch_function")
                if func is torch.autograd.backward:
                    report["backward_calls"] += 1
                with self:
                    return redispatch(func, types, args, kwargs)

            def normalize(device):
                if device is None:
                    return device
                parsed = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
                if parsed.type not in {"cpu", "meta", "cuda"}:
                    raise NotImplementedError(f"Unsupported metadata device request: {parsed}")
                if parsed.type != "cuda":
                    return device
                report["device_normalizations"].append(dict(
                    operation=getattr(func, "__qualname__", str(func)),
                    requested_device=str(parsed), execution_device="cpu",
                ))
                return torch.device("cpu")

            if func is torch.Tensor.cuda:
                device = kwargs.pop("device", args[1] if len(args) > 1 else None)
                if device is not None and not isinstance(device, int) and torch.device(device).type != "cuda":
                    raise ValueError("Tensor.cuda requires a CUDA device")
                normalize("cuda" if device is None else device)
                non_blocking = kwargs.pop("non_blocking", args[2] if len(args) > 2 else False)
                # Tensor.cuda has no dtype/copy overload; retain its memory-format
                # option while explicitly changing its target to the CPU.
                return args[0].to(device="cpu", non_blocking=non_blocking, **kwargs)
            if "device" in kwargs:
                kwargs["device"] = normalize(kwargs["device"])
            if func is torch.Tensor.to and len(args) > 1 and isinstance(args[1], (str, torch.device, int)):
                args = (args[0], normalize(args[1]), *args[2:])
            return func(*args, **kwargs)

    with FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=False):
        with torch.device("cpu"), DeviceMode():
            yield report
