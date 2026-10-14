"""Collect independent allocator observations from the upstream TrainingEngine.

Use with torchrun inside the pinned NVIDIA container. Target model execution is
unchanged; fixed-length synthetic input replaces dataset and tokenizer setup.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class MemoryObserver:
    """Observe allocator counters without synchronizing at each module hook."""

    def __init__(self, torch: Any, output: Path, rank: int, enabled: bool):
        self.torch = torch
        self.output = output
        self.rank = rank
        self.enabled = enabled
        self.started = time.perf_counter()
        self.step = 0
        self.rows: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def sample(self, phase: str, module: str = "") -> None:
        if not self.enabled:
            return
        cuda = self.torch.cuda
        self.rows.append({
            "time_us": (time.perf_counter() - self.started) * 1e6,
            "step": self.step,
            "rank": self.rank,
            "phase": phase,
            "module": module,
            "allocated_bytes": cuda.memory_allocated(),
            "reserved_bytes": cuda.memory_reserved(),
            "peak_allocated_bytes": cuda.max_memory_allocated(),
            "peak_reserved_bytes": cuda.max_memory_reserved(),
        })

    def attach(self, engine: Any) -> None:
        if not self.enabled:
            return
        for part_index, part in enumerate(engine.model_parts):
            for name, module in part.named_modules():
                # Layer boundaries keep hook overhead bounded on a 671B model.
                if name.startswith("layers.") and name.count(".") == 1:
                    fqn = f"part{part_index}.{name}"
                    self.handles.append(module.register_forward_pre_hook(
                        lambda mod, args, label=fqn: self.sample("forward_pre", label)
                    ))
                    self.handles.append(module.register_forward_hook(
                        lambda mod, args, result, label=fqn: self.sample("forward_post", label)
                    ))

    def save(self, status: str, metadata: dict[str, Any], error: str | None = None) -> None:
        if not self.enabled:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "source": "gpu_observation",
            "status": status,
            "metadata": metadata,
            "samples": self.rows,
            "error": error,
            "limitations": [
                "Module samples are host-observed allocator counters without per-hook GPU synchronization.",
                "Recomputed forward hooks are included; phase labels do not assert unique backward boundaries.",
                "CUDA allocator counters exclude allocations outside the PyTorch allocator.",
            ],
        }
        (self.output / f"observed-rank{self.rank}.json").write_text(json.dumps(document, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flavor", choices=["debugmodel", "671B"], default="671B")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--expert-parallel-degree", type=int, default=64, help="Use 1 for the explicitly reduced debug experiment")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--observe-ranks", default="0")
    parser.add_argument("--history-step", type=int, default=3)
    parser.add_argument("--history-entries", type=int, default=250000)
    parser.add_argument("--dump-history", action="store_true")
    args = parser.parse_args()
    if args.steps < 10:
        raise ValueError("Use at least 10 steps to include optimizer initialization and later iterations.")

    import torch
    from torchtitan.components.data.types import TokenizedTrainingMicrobatch
    from torchtitan.models.common.config_utils import decoder_vocab_size
    from torchtitan.observability.logging import init_logger
    from torchtitan.training_engine import TrainingEngine

    init_logger()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    from .recipes import load_validation_recipe, TORCHTITAN_COMMIT, resolved_validation_topology
    config = load_validation_recipe(flavor=args.flavor, seq_len=args.seq_len,
                                    tokens=args.tokens, expert_parallel_degree=args.expert_parallel_degree)
    config.training.steps = args.steps
    seq_len = config.training.max_context_length
    tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    if tokens % seq_len:
        raise ValueError("Synthetic fixed-length documents require tokens divisible by sequence length.")
    args.output.mkdir(parents=True, exist_ok=True)
    config_doc = _json_value(config)
    if rank == 0:
        (args.output / "resolved-config.json").write_text(json.dumps(config_doc, indent=2))
    import torchtitan
    titan_root = Path(torchtitan.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(titan_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != TORCHTITAN_COMMIT:
        raise ValueError(f"Expected TorchTitan {TORCHTITAN_COMMIT}, found {commit}")
    metadata = {
        "import_compatibility": globals().get("IMPORT_COMPATIBILITY", {}),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "torchtitan_commit": commit,
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_capacity_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
        "world_size": world,
        "rank": rank,
        "flavor": args.flavor,
        "sequence_length": seq_len,
        "tokens_per_microbatch_per_dp_rank": tokens,
        "steps": args.steps,
        "input": "synthetic fixed-length documents, random initialization, seed 42",
        "recipe": "torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b" if args.flavor == "671B" else "torchtitan_recipes.tests.models.deepseek_v3.deepseek_v3_debugmodel",
        "overrides": {"steps": args.steps, "sequence_length": args.seq_len, "tokens": args.tokens, "expert_parallel_degree": args.expert_parallel_degree, "activation_checkpoint": "FullAC", "pipeline_parallel_degree": 1, "tensor_parallel_degree": 1, "data_parallel_shard_degree": -1, "disable_cuda_graphs": config.training.disable_cuda_graphs},
        "execution": "upstream TrainingEngine with synthetic microbatches",
    }
    from .provenance import execution_policy
    metadata.update(execution_policy(config, routing_scenario="natural_seed_42"))
    observe_ranks = {int(x) for x in args.observe_ranks.split(",")}
    observer = MemoryObserver(torch, args.output, rank, rank in observe_ranks)
    engine = None
    history_enabled = False
    try:
        torch.manual_seed(42)
        config.debug.seed = 42
        observer.sample("before_initialize")
        engine = TrainingEngine(config, model_config=config.model,
                                max_num_documents=config.dataloader.max_num_documents,
                                output_dir=str(args.output))
        metadata["resolved_topology"] = resolved_validation_topology(engine.parallelism_context, world_size=world, expert_parallel_degree=args.expert_parallel_degree)
        engine.initialize(compile_config=config.compile, hf_assets_path=config.hf_assets_path)
        observer.attach(engine)
        torch.cuda.synchronize()
        observer.sample("after_initialize")
        metadata["model_parameter_count"] = engine.model_param_count
        metadata["parallelism"] = _json_value(config.parallelism)
        dp_degree = engine.parallelism_context.get_mesh("dp").size() if engine.parallelism_context.dp_enabled else 1
        num_pp = config.parallelism.num_pp_microbatches if engine.parallelism_context.pp_enabled else 1
        per_group_tokens = tokens * num_pp * dp_degree
        global_tokens = config.training.num_tokens_per_train_step
        if global_tokens < 0:
            global_tokens = per_group_tokens
        if global_tokens % per_group_tokens:
            raise ValueError("Global token budget is not divisible by synthetic microbatch group.")
        accumulation = global_tokens // per_group_tokens
        vocab = decoder_vocab_size(config.model)
        dp_rank = engine.parallelism_context.get_mesh("dp").get_local_rank() if engine.parallelism_context.dp_enabled else 0
        generator = torch.Generator(device="cpu").manual_seed(42 + dp_rank)

        def make_batch() -> Any:
            input_tokens = torch.randint(vocab, (tokens,), generator=generator)
            return TokenizedTrainingMicrobatch(
                input=input_tokens,
                labels=torch.randint(vocab, (tokens,), generator=generator),
                positions=torch.arange(seq_len).repeat(tokens // seq_len),
                padding_mask=torch.zeros(tokens, dtype=torch.bool),
                num_valid_tokens=tokens,
            )

        for step in range(1, args.steps + 1):
            observer.step = step
            if observer.enabled and step == args.history_step:
                torch.cuda.memory._record_memory_history(max_entries=args.history_entries)
                history_enabled = True
            torch.cuda.reset_peak_memory_stats()
            groups = [[make_batch() for _ in range(num_pp)] for _ in range(accumulation)]
            observer.sample("step_start")
            started = time.perf_counter()
            engine.forward_backward(
                microbatch_groups=groups,
                global_valid_tokens=global_tokens,
            )
            observer.sample("after_forward_backward")
            grad_norm = engine.optim_step()
            observer.sample("after_optimizer")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            observer.sample("step_end_synchronized")
            if observer.enabled:
                observer.rows[-1]["step_time_s"] = elapsed
                observer.rows[-1]["loss"] = float(engine.loss.item())
                observer.rows[-1]["grad_norm"] = float(grad_norm.item())
                observer.save("running", metadata)
            if rank == 0:
                print(json.dumps({"step": step, "seconds": elapsed,
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)
            if history_enabled:
                if args.dump_history:
                    torch.cuda.memory._dump_snapshot(str(args.output / f"allocator-rank{rank}-step{step}.pickle"))
                history = torch.cuda.memory._snapshot()
                traces = history.get("device_traces", [])
                # This JSON omits stacks and addresses, retaining allocator events.
                trace_rows = []
                for device, trace in enumerate(traces):
                    for index, event in enumerate(trace):
                        trace_rows.append({"device": device, "index": index,
                                           **{k: v for k, v in event.items() if k in {"action", "size", "stream", "time_us"}}})
                (args.output / f"allocator-events-rank{rank}-step{step}.json").write_text(json.dumps(trace_rows))
                torch.cuda.memory._record_memory_history(enabled=None)
                history_enabled = False
        observer.save("complete", metadata)
    except BaseException as error:
        observer.sample("failure")
        observer.save("failed", metadata, f"{type(error).__name__}: {error}")
        if observer.enabled and args.dump_history:
            try:
                torch.cuda.memory._dump_snapshot(str(args.output / f"failure-rank{rank}.pickle"))
            except RuntimeError:
                pass
        raise
    finally:
        for handle in observer.handles:
            handle.remove()
        if engine is not None:
            engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
