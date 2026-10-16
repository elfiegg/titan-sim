"""Run the upstream model/loss/optimizer on fake CPU tensors.

This unsharded probe establishes operator coverage, optionally including original
FullAC. It does not claim production FSDP, compiler, or 256-rank fidelity. The
production adapter should observe the original TrainingEngine separately.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import traceback
from unittest.mock import patch

from .recipes import build_smoke_model_config, import_torchtitan


@contextlib.contextmanager
def post_initialization_scope():
    """Exclude random parameter sampling while retaining upstream state setup.

    This is a steady-state projection boundary, not a replacement initializer.
    Only random sampling inside upstream parameter callbacks is omitted;
    deterministic parameter initialization and all buffer initialization run.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    from torchtitan.protocols.module import Module

    original = Module._init_param
    report = dict(scope="post_initialization", random_parameter_values="unknown",
                  initialization_time_included=False, initialization_peak_included=False,
                  skipped_parameter_initializers=0, initializers={})

    def initialize(module, name, parameter):
        sampled = False

        def omit_sampling(label):
            def sample(tensor, *args, **kwargs):
                nonlocal sampled
                local = getattr(tensor, "_local_tensor", tensor)
                if not isinstance(local, FakeTensor):
                    raise RuntimeError("post_initialization scope only supports FakeTensor parameters")
                sampled = True
                report["initializers"][label] = report["initializers"].get(label, 0) + 1
                return tensor
            return sample

        # Composite initializers (e.g. stacked gate/up weights) call samplers
        # on parameter views. Keep those callbacks and view operations intact.
        with contextlib.ExitStack() as stack:
            for label in ("trunc_normal_", "normal_", "uniform_"):
                stack.enter_context(patch.object(torch.nn.init, "_no_grad_" + label,
                                                 omit_sampling(label)))
            result = original(module, name, parameter)
        if sampled:
            report["skipped_parameter_initializers"] += 1
        return result

    with patch.object(Module, "_init_param", initialize):
        yield report


def capture_smoke(*, torchtitan_root, flavor="debugmodel", seq_len=16, tokens=16,
                  fake_device="cpu", include_optimizer=True, capture_scope="strict",
                  attention_contract="native", full_ac=False, steps=1):
    if attention_contract not in {"native", "flex_metadata"}:
        raise ValueError("attention_contract must be native or flex_metadata")
    if attention_contract == "flex_metadata" and fake_device != "cpu":
        raise ValueError("The explicit FlexAttention contract requires CPU FakeTensors")
    if steps < 1:
        raise ValueError("steps must be positive")
    if capture_scope not in {"strict", "post_initialization"}:
        raise ValueError("capture_scope must be strict or post_initialization")
    if fake_device not in {"cpu", "meta"}:
        raise ValueError("CPU-only capture supports cpu/meta; fake CUDA backward may initialize a native CUDA device guard")
    if seq_len <= 0 or tokens <= 0 or tokens % seq_len or tokens % 8:
        raise ValueError("tokens must be positive and divisible by seq_len and upstream loss's 8 chunks")
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .capture import TensorCapture

    revision = import_torchtitan(torchtitan_root)

    capture = TensorCapture(source="cpu_projection", modeled_device="cuda:0")
    metadata = dict(status="running", torchtitan_commit=revision,
                    torch_version=torch.__version__, scenario="unsharded_eager_operator_coverage",
                    flavor=flavor, seq_len=seq_len, tokens=tokens, fake_device=fake_device,
                    capture_scope=capture_scope,
                    attention_contract=attention_contract, steps=steps,
                    activation_checkpoint="upstream_FullAC" if full_ac else "none",
                    completed_steps=0, production_training_capture=False,
                    cuda_memory_equivalent=False,
                    overrides=["BF16 model parameters; original upstream operation dtypes retained; no FP32 master parameters.",
                               "No FSDP, EP or production local compiler transforms.",
                               "Synthetic fixed-length positions passed through upstream FlexAttention mask construction.",
                               "Direct AdamW without trainer optimizer hooks or gradient clipping."],
                    routing="Upstream routing; no fabricated expert counts or replacement kernels")
    config = build_smoke_model_config(flavor=flavor, seq_len=seq_len)
    try:
        with FakeTensorMode(allow_fallback_kernels=False), torch.device(fake_device), capture:
            with contextlib.ExitStack() as contracts:
                if attention_contract == "flex_metadata":
                    from .attention_metadata import flex_attention_metadata_scope
                    metadata["attention_coverage"] = contracts.enter_context(flex_attention_metadata_scope())
                return _capture_model_steps(capture=capture, metadata=metadata, config=config,
                                            fake_device=fake_device, include_optimizer=include_optimizer,
                                            capture_scope=capture_scope, full_ac=full_ac, steps=steps,
                                            flavor=flavor, seq_len=seq_len, tokens=tokens)
    except Exception as error:
        metadata.update(status="blocked", error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
    return capture.document(metadata=metadata)


def _capture_model_steps(*, capture, metadata, config, fake_device, include_optimizer,
                         capture_scope, full_ac, steps, flavor, seq_len, tokens):
    import torch
    from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
    from torchtitan.components.optim import AdamW
    from torchtitan.tools.utils import set_default_dtype
    from torchtitan.models.common.aux_loss import AuxLoss

    try:
        initialization = (post_initialization_scope() if capture_scope == "post_initialization"
                          else contextlib.nullcontext())
        with capture.phase("initialization"), initialization as initialization_report:
            if initialization_report is not None:
                metadata["initialization_coverage"] = initialization_report
            with set_default_dtype(torch.bfloat16):
                model = config.build()
            model.init_states(buffer_device=torch.device(fake_device))
            model.train()
            metadata["model_parameter_count"] = sum(p.numel() for p in model.parameters())
            metadata["trainable_parameter_tensors"] = sum(p.requires_grad for p in model.parameters())
            metadata["step_checks"] = []
            if full_ac:
                from torchtitan.distributed.activation_checkpoint import FullAC
                FullAC.Config().build().apply(model)
            capture.register_model(model)
            loss_fn = ChunkedLossWrapper.Config(
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=config.vocab_size)
            ).build()
            loss_fn.set_lm_head(model.lm_head)
            model._skip_lm_head = True
            optimizer = AdamW.Config(pattern=r".*", lr=8e-4 if flavor == "debugmodel" else 2.2e-4).build(params=model.parameters())
            input_tokens = torch.zeros(tokens, dtype=torch.int64)
            labels = torch.zeros_like(input_tokens)
            positions = torch.arange(tokens, dtype=torch.int64) % seq_len
            padding_mask = torch.zeros(tokens, dtype=torch.bool)
            denominator = torch.full((), float(tokens), dtype=torch.float32)
            AuxLoss.set_step_denominator(denominator)
        with capture.phase("attention_mask"):
            attention_masks = model.get_attention_masks(
                positions, padding_mask=padding_mask,
                max_context_length=seq_len,
            )
        for step in range(steps):
            with capture.phase(f"forward_step_{step}"):
                prediction = model(input_tokens, positions=positions,
                                   attention_masks=attention_masks, padding_mask=padding_mask)
            # The original chunked loss runs lm_head backward per chunk.
            with capture.phase(f"loss_and_head_backward_step_{step}"):
                loss, _ = loss_fn(prediction, labels, denominator)
            with capture.phase(f"backward_step_{step}"):
                loss.backward()
                capture.register_model(model)
                check = {"step": step,
                         "parameter_tensors_with_grad": sum(p.grad is not None for p in model.parameters()),
                         "optimizer_completed": False}
            if include_optimizer:
                with capture.phase(f"optimizer_step_{step}"):
                    optimizer.step()
                    capture.register_optimizer(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    check["optimizer_completed"] = True
            metadata["step_checks"].append(check)
            metadata["completed_steps"] += 1
            del prediction, loss
            capture.sweep()
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="blocked", error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
    return capture.document(metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchtitan-root", required=True)
    parser.add_argument("--flavor", default="debugmodel", choices=["debugmodel", "16B", "236B", "671B"])
    parser.add_argument("--seq-len", default=16, type=int)
    parser.add_argument("--tokens", default=16, type=int)
    parser.add_argument("--fake-device", default="cpu", choices=["cpu", "meta"])
    parser.add_argument("--capture-scope", default="strict", choices=["strict", "post_initialization"])
    parser.add_argument("--attention-contract", default="native", choices=["native", "flex_metadata"],
                        help="Opt in to original Flex HOP metadata without CUDA compilation")
    parser.add_argument("--full-ac", action="store_true", help="Apply the unchanged upstream FullAC policy")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    document = capture_smoke(torchtitan_root=args.torchtitan_root, flavor=args.flavor,
                             seq_len=args.seq_len, tokens=args.tokens, fake_device=args.fake_device,
                             capture_scope=args.capture_scope, attention_contract=args.attention_contract,
                             full_ac=args.full_ac, steps=args.steps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document["metadata"], indent=2))
    if document["metadata"]["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
