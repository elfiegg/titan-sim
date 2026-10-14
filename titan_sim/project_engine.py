"""Attempt the pinned validation TrainingEngine under metadata-only execution.

Failures produce partial traces and a nonzero exit. No unsupported operator,
parallelism, compiler region or data-dependent routing is silently removed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import traceback
from unittest.mock import patch

from .recipes import DEFAULT_RECIPE, import_torchtitan, load_validation_recipe


@contextlib.contextmanager
def _balanced_routing_scenario(*, tokens, capture):
    """Supply only host split counts, preserving every upstream dispatch tensor op."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

    original = AllToAllTokenDispatcher._sync_token_count_exchange
    original_tolist = FakeTensor.tolist

    def synchronize(self, num_local_tokens_per_expert_E,
                    num_global_tokens_per_local_expert_EP_e, ep_size):
        total = tokens * self.top_k
        if total % ep_size:
            raise ValueError("Balanced routing requires T*K divisible by EP degree")
        calls = 0

        def scenario_tolist(tensor):
            nonlocal calls
            if isinstance(tensor, FakeTensor) and tensor.device.type == "cpu" and tuple(tensor.shape) == (ep_size,):
                calls += 1
                return [total // ep_size] * ep_size
            return original_tolist(tensor)

        # This revision materializes split lists inside _sync_token_count_exchange.
        # Preserve the original wait/reduce/copy operations and substitute only
        # its two value-dependent host list reads for this explicit scenario.
        with patch.object(FakeTensor, "tolist", scenario_tolist):
            result = original(self, num_local_tokens_per_expert_E,
                              num_global_tokens_per_local_expert_EP_e, ep_size)
        if calls != 2:
            raise RuntimeError(f"Expected two EP split-list reads, observed {calls}")
        capture._event("compute", "scenario.balanced_ep_splits", category="routing",
                       device="cpu", stream="__host_control__", metadata_only=True,
                       split_sizes=[total // ep_size] * ep_size, scenario="balanced")
        return result

    with patch.object(AllToAllTokenDispatcher, "_sync_token_count_exchange", synchronize):
        yield


def project_engine(*, torchtitan_root, output_dir, world_size=256, flavor="671B",
                   seq_len=None, tokens=None, expert_parallel_degree=64, routing_scenario="strict", steps=1,
                   probe=True):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .capture import TensorCapture
    from .fake_device import FakeCudaRuntime, probe_fake_cuda_autograd, fake_tensor_parameter_swaps

    if routing_scenario not in {"strict", "balanced"}:
        raise ValueError("routing_scenario must be strict or balanced")
    if world_size < 1 or steps < 1 or expert_parallel_degree < 1:
        raise ValueError("world_size, steps and expert_parallel_degree must be positive")
    revision = import_torchtitan(torchtitan_root)
    capture = TensorCapture(source="cpu_projection")
    metadata = dict(status="running", torchtitan_commit=revision, torch_version=torch.__version__,
                    recipe=DEFAULT_RECIPE, world_size=world_size, rank=0,
                    import_compatibility=globals().get("IMPORT_COMPATIBILITY", {}),
                    flavor=flavor, routing_scenario=routing_scenario,
                    execution="upstream TrainingEngine with fake tensor kernels and fake process group",
                    overrides={"comm.backend": "fake", "synthetic_documents": "fixed-length",
                               "sequence_length": seq_len, "tokens": tokens,
                               "expert_parallel_degree": expert_parallel_degree,
                               "pipeline_parallel_degree": 1, "tensor_parallel_degree": 1,
                               "data_parallel_shard_degree": -1, "activation_checkpoint": "full"},
                    limitations=["Native fake-CUDA autograd may require CUDA driver/device support even though tensor kernels execute only metadata.",
                                 "Python CUDA stream/event shim preserves control calls but does not model native allocator or kernel workspace.",
                                 "Rank 0 represents a logical rank; all rank placements must be captured to predict rank imbalance."])
    if probe:
        metadata["native_autograd_probe"] = probe_fake_cuda_autograd()
        if metadata["native_autograd_probe"]["status"] != "complete":
            metadata["status"] = "blocked"
            return capture.document(metadata=metadata)
    # These target properties are declared architecture inputs, not hardware
    # observations; only dispatch metadata contributes to projected storage.
    properties = dict(name="logical NVIDIA GB200", major=10, minor=0,
                      total_memory=197_897_617_408, multi_processor_count=152)
    metadata["declared_target_properties"] = properties
    runtime = FakeCudaRuntime(capture, properties=properties)
    engine = None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Import compiler/optimizer internals before replacing CUDA classes:
        # Dynamo registers guards against the native stream classes at import.
        import torch._dynamo
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, NGPU=str(world_size), LOCAL_RANK="0", RANK="0", WORLD_SIZE="1"))
            stack.enter_context(runtime)
            from torchtitan.components.data.types import TokenizedTrainingMicrobatch
            from torchtitan.training_engine import TrainingEngine
            config = load_validation_recipe(flavor=flavor, seq_len=seq_len, tokens=tokens,
                                            expert_parallel_degree=expert_parallel_degree)
            if flavor == "debugmodel":
                metadata["recipe"] = "torchtitan_recipes.tests.models.deepseek_v3.deepseek_v3_debugmodel"
            if config.parallelism.pipeline_parallel_degree != 1:
                raise NotImplementedError("This capture entry point requires PP=1")
            if world_size % expert_parallel_degree:
                raise ValueError("world_size must be divisible by expert_parallel_degree")
            config.comm.backend = "fake"
            seq_len = config.training.max_context_length
            tokens = config.training.num_tokens_per_microbatch_per_dp_rank
            if tokens % seq_len:
                raise ValueError("Fixed-document scenario requires tokens divisible by context length")
            metadata.update(sequence_length=seq_len, tokens_per_microbatch_per_dp_rank=tokens)
            from .provenance import execution_policy
            metadata.update(execution_policy(config, routing_scenario=routing_scenario))
            if routing_scenario == "balanced":
                stack.enter_context(_balanced_routing_scenario(tokens=tokens, capture=capture))
                metadata["limitations"].append("Conditional on equal tokens sent to every EP peer on every layer and step; does not predict routing imbalance.")
            stack.enter_context(FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True))
            metadata["fake_parameter_swap_compatibility"] = stack.enter_context(fake_tensor_parameter_swaps())
            stack.enter_context(capture)
            with capture.phase("initialization"):
                engine = TrainingEngine(config, model_config=config.model,
                                        max_num_documents=config.dataloader.max_num_documents,
                                        output_dir=str(output_dir))
                engine.initialize(compile_config=config.compile, hf_assets_path=config.hf_assets_path)
                from .recipes import resolved_validation_topology
                metadata["resolved_topology"] = resolved_validation_topology(
                    engine.parallelism_context, world_size=world_size,
                    expert_parallel_degree=expert_parallel_degree,
                )
                metadata["model_parameter_count"] = engine.model_param_count
                for model in engine.model_parts:
                    capture.register_model(model)
            dp = engine.parallelism_context.dp_replicate * engine.parallelism_context.dp_shard
            global_tokens = tokens * dp
            if config.training.num_tokens_per_train_step not in {-1, global_tokens}:
                raise NotImplementedError("This capture entry point requires one microbatch per optimizer step")
            for step in range(steps):
                with capture.phase(f"input_step_{step}"):
                    batch = TokenizedTrainingMicrobatch(
                        input=torch.zeros(tokens, dtype=torch.int64, device="cpu"),
                        labels=torch.zeros(tokens, dtype=torch.int64, device="cpu"),
                        positions=torch.arange(seq_len, device="cpu").repeat(tokens // seq_len),
                        padding_mask=torch.zeros(tokens, dtype=torch.bool, device="cpu"),
                        num_valid_tokens=tokens,
                    )
                    for tensor in (batch.input, batch.labels, batch.positions, batch.padding_mask):
                        capture.register_tensor(tensor, category="inputs", modeled_device="cpu")
                with capture.phase(f"forward_backward_step_{step}"):
                    engine.forward_backward(microbatch_groups=[[batch]], global_valid_tokens=global_tokens)
                    for model in engine.model_parts:
                        capture.register_model(model)
                with capture.phase(f"optimizer_step_{step}"):
                    engine.optim_step()
                    for optimizer in engine.optim.optimizers:
                        capture.register_optimizer(optimizer)
            metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="blocked", error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    metadata["runtime_control_calls"] = runtime.actions
    return capture.document(metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchtitan-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--world-size", default=256, type=int)
    parser.add_argument("--flavor", default="671B", choices=["671B", "debugmodel"])
    parser.add_argument("--expert-parallel-degree", default=64, type=int,
                        help="Use 64 for validation; explicitly select 1 for an unsharded debug probe")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--routing-scenario", default="strict", choices=["strict", "balanced"])
    args = parser.parse_args()
    document = project_engine(torchtitan_root=args.torchtitan_root, output_dir=args.output.parent,
                              world_size=args.world_size, flavor=args.flavor, seq_len=args.seq_len,
                              tokens=args.tokens, expert_parallel_degree=args.expert_parallel_degree,
                              routing_scenario=args.routing_scenario, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: document["metadata"][key]
                      for key in ("status", "model_parameter_count", "error_type", "error")
                      if key in document["metadata"]}, indent=2))
    print(f"Trace: {args.output}")
    if document["metadata"]["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
