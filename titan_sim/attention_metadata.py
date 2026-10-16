"""Explicit FlexAttention metadata execution using PyTorch's original HOP.

This scope is an operator contract, not a CPU implementation of CUDA kernels.
The original public FlexAttention validation, mask construction, higher-order
operator, autograd saves, and backward metadata kernels remain in use. Only
compilation and the public CPU-training device guard are adapted. Inductor
fusion, kernel scratch, numerical values, and device scheduling are excluded.
"""
from __future__ import annotations

import contextlib
import importlib
from unittest.mock import patch


@contextlib.contextmanager
def flex_attention_metadata_scope():
    """Run original upstream FlexAttention on CPU FakeTensors, explicitly.

    Enter inside a fallback-disabled FakeTensorMode (or cpu_metadata_execution).
    Callers must serialize this scope: upstream compiler wrappers are class and
    module attributes, restored even when an operation raises. Real tensors are
    rejected before entering the original attention function.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    from torchtitan.models.common import attention

    flex = importlib.import_module("torch.nn.attention.flex_attention")
    original_validate_device = flex._validate_device
    report = dict(
        scope="flex_attention_metadata_contract",
        implementation="original_pytorch_flex_hop_and_autograd",
        compiler_backend="disabled_original_flex_hop_metadata",
        cpu_training_guard="validated_on_detached_fake_views",
        cuda_dispatch_equivalent=False,
        compiled_storage_lifetimes_equivalent=False,
        attention_calls=0,
        mask_calls=0,
        score_mod="identity_only",
        callback_tensor_reads="unobserved_without_compiler_buffer_lifting",
        excluded=["Inductor fusion", "compiled kernel scratch", "numerical values",
                  "CUDA dispatch and scheduling", "mask callback tensor read dependencies"],
    )

    def require_fake_cpu(*tensors):
        if not all(isinstance(tensor, FakeTensor) and tensor.device.type == "cpu"
                   and not tensor.fake_mode.allow_fallback_kernels
                   for tensor in tensors):
            raise RuntimeError(
                "FlexAttention metadata contract requires CPU FakeTensors "
                "with fallback kernels disabled"
            )

    def validate_device(query, key, value):
        require_fake_cpu(query, key, value)
        # The original validator rejects CPU requires_grad because there is no
        # numerical CPU backward kernel. Fake HOP backward exists independently.
        # Detach only these validation views; original inputs retain autograd.
        original_validate_device(query.detach(), key.detach(), value.detach())

    def run_attention(query, key, value, **kwargs):
        require_fake_cpu(query, key, value)
        if kwargs.get("score_mod") is not None:
            raise NotImplementedError(
                "FlexAttention metadata contract supports identity score_mod only; "
                "captured score tensors require compiler buffer lifting"
            )
        report["attention_calls"] += 1
        return flex.flex_attention(query, key, value, **kwargs)

    def run_mask(*args, **kwargs):
        # create_block_mask also receives Python sizes and a mask callback.
        # Require an active fake mode before allowing any allocation.
        from torch._guards import detect_fake_mode
        mode = detect_fake_mode()
        if mode is None or mode.allow_fallback_kernels:
            raise RuntimeError("Mask metadata contract requires fallback-disabled FakeTensorMode")
        if torch.device(kwargs.get("device", "cuda")).type != "cpu":
            raise RuntimeError("Mask metadata contract requires explicit CPU device")
        report["mask_calls"] += 1
        return flex.create_block_mask(*args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(flex, "_validate_device", validate_device))
        # TensorCapture is a non-infrastructure dispatch mode, which Dynamo
        # refuses to compile through. This upstream debugging entry calls the
        # *same* HOP directly; its FakeTensor kernels avoid numerical fallback.
        # It is not a numerical unfused/dense attention implementation here.
        stack.enter_context(patch.object(flex, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", True))
        stack.enter_context(patch.object(attention.FlexInnerAttention,
                                        "_compiled_flex_attn", staticmethod(run_attention)))
        stack.enter_context(patch.object(attention, "_compiled_create_block_mask", run_mask))
        yield report
