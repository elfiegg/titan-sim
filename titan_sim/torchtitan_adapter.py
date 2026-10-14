"""Run the upstream model/loss/optimizer on fake CPU tensors.

This unsharded eager probe establishes operator coverage. It does not claim
production FSDP, activation-checkpoint, compiler, or 256-rank fidelity. The production
adapter should place TensorCapture around the unchanged TrainingEngine instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from .recipes import build_smoke_model_config, import_torchtitan


def capture_smoke(*, torchtitan_root, flavor="debugmodel", seq_len=16, tokens=16,
                  fake_device="cpu", include_optimizer=True):
    if fake_device not in {"cpu", "meta"}:
        raise ValueError("CPU-only capture supports cpu/meta; fake CUDA backward may initialize a native CUDA device guard")
    if seq_len <= 0 or tokens <= 0 or tokens % seq_len or tokens % 8:
        raise ValueError("tokens must be positive and divisible by seq_len and upstream loss's 8 chunks")
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .capture import TensorCapture

    revision = import_torchtitan(torchtitan_root)
    from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
    from torchtitan.components.optim import AdamW
    from torchtitan.tools.utils import set_default_dtype
    from torchtitan.models.common.aux_loss import AuxLoss

    capture = TensorCapture(source="cpu_projection", modeled_device="cuda:0")
    metadata = dict(status="running", torchtitan_commit=revision,
                    torch_version=torch.__version__, scenario="unsharded_eager_operator_coverage",
                    flavor=flavor, seq_len=seq_len, tokens=tokens, fake_device=fake_device,
                    overrides=["All parameters and compute BF16; production retains FP32 master parameters.",
                               "No FSDP, EP, full activation checkpoint policy or local compiler transforms.",
                               "Synthetic fixed-length positions passed through upstream FlexAttention mask construction.",
                               "Direct AdamW without trainer optimizer hooks or gradient clipping."],
                    routing="Upstream routing; no fabricated expert counts or replacement kernels")
    config = build_smoke_model_config(flavor=flavor, seq_len=seq_len)
    try:
        with FakeTensorMode(allow_fallback_kernels=False), torch.device(fake_device), capture:
            with capture.phase("initialization"):
                with set_default_dtype(torch.bfloat16):
                    model = config.build()
                model.init_states(buffer_device=torch.device(fake_device))
                model.train()
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
            with capture.phase("forward"):
                prediction = model(input_tokens, positions=positions,
                                   attention_masks=attention_masks, padding_mask=padding_mask)
            # The upstream chunked loss itself runs lm_head backward per chunk.
            with capture.phase("loss_and_head_backward"):
                loss, _ = loss_fn(prediction, labels, denominator)
            with capture.phase("backward"):
                loss.backward()
                capture.register_model(model)
            if include_optimizer:
                with capture.phase("optimizer"):
                    optimizer.step()
                    capture.register_optimizer(optimizer)
                    optimizer.zero_grad(set_to_none=True)
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    document = capture_smoke(torchtitan_root=args.torchtitan_root, flavor=args.flavor,
                             seq_len=args.seq_len, tokens=args.tokens, fake_device=args.fake_device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document["metadata"], indent=2))
    if document["metadata"]["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
