#!/usr/bin/env python3
"""Bounded, randomly initialized GPT-OSS training with allocation observations."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp8_mx", "nvfp4"), required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--detailed-ranks", default="0,1")
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 4 or args.world_size % 4:
        raise ValueError("Need at least four steps and a world size divisible by EP=4")
    sys.path.insert(0, str(args.bridge_root / "scripts/performance"))
    import torch
    from configs.gpt_oss.gpt_oss_llm_pretrain import gpt_oss_20b_pretrain_config_gb300
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.callbacks import CallbackManager
    from titan_sim.allocation_probe import AllocationRecorder, make_bridge_callback

    # Use the exact public recipe family from the archived 72-GPU runs. Physical
    # hardware, world size and global batch are independently recorded below.
    config = gpt_oss_20b_pretrain_config_gb300(precision=args.precision, mock=True, config_variant="v2")
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.context_parallel_size = 2
    config.model.expert_model_parallel_size = 4
    config.model.expert_tensor_parallel_size = 1
    config.model.sequence_parallel = False
    config.train.global_batch_size = args.world_size // 2
    config.train.micro_batch_size = 1
    config.train.train_iters = args.steps
    config.validation.eval_iters = 0
    config.validation.eval_interval = 1000000
    config.dataset.seq_length = 8192
    config.dataset.num_workers = 0
    config.dataset.persistent_workers = False
    config.dataset.fast_cache_load = False
    config.dataset.defer_npy_index_mmap = False
    config.dataset.path_to_cache = str(args.output / "dataset-cache")
    config.checkpoint.save = None
    config.checkpoint.load = None
    config.checkpoint.save_interval = None
    config.checkpoint.save_tokenizer_assets = False
    config.logger.log_interval = 1
    config.logger.wandb_project = None
    config.logger.tensorboard_dir = None
    config.scheduler.lr_warmup_iters = 0
    config.scheduler.lr_decay_iters = args.steps
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    expected = {"num_layers": 24, "hidden_size": 2880, "num_moe_experts": 32,
                "moe_router_topk": 4, "num_attention_heads": 64, "num_query_groups": 8,
                "kv_channels": 64, "vocab_size": 128256}
    for name, value in expected.items():
        if getattr(config.model, name) != value:
            raise ValueError(f"Wrong model identity: {name}")
    manifest = {"scope": f"new {args.world_size}-GPU diagnostic using archived 72-GPU recipe family; not an exact archived-run replay",
        "world_size": args.world_size, "global_batch": args.world_size // 2,
        "micro_batch": 1, "sequence_length": 8192, "tp": 1, "cp": 2, "ep": 4, "pp": 1,
        "precision": args.precision, "steps": args.steps, "model": expected,
        "container": "nvcr.io/nvidia/pytorch:26.09-py3",
        "bridge_revision": "fcbb6031103d0ca845c1a54d4fee55ecfcca17b6",
        "mcore_revision": "d0b3b775498a8ae6870075f1a5fc32bc260aa1b3",
        "changes_from_archive": ["world size and global batch", "GPU SKU and container runtime",
            "six steps", "random initialization and mock inputs", "no checkpoint/evaluation",
            "dataset workers zero; cached-file fast paths disabled for fresh mock data",
            "learning-rate warmup zero", "instrumentation"],
        "snapshot_ranks": args.detailed_ranks, "torch": torch.__version__, "cuda": torch.version.cuda,
        "training_complete": False}
    for package in ("transformer-engine", "transformers", "deep-ep"):
        try: manifest[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: manifest[package] = None
    if rank == 0:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.config_only:
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != args.world_size:
        raise ValueError("Launch world size differs from requested workload")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if torch.__version__ != "2.14.0a0+b2c75dd062.nv26.09":
        raise ValueError("Refusing an unintended replacement of the container's PyTorch")
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
    if not HAVE_HYBRIDEP:
        raise RuntimeError("Required HybridEP backend unavailable; no silent dispatcher fallback")
    manifest.update(gpu=torch.cuda.get_device_name(), device_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    detailed = tuple(int(v) for v in args.detailed_ranks.split(",") if v)
    recorder = AllocationRecorder(args.output, rank=rank, detailed_ranks=detailed)
    callback = make_bridge_callback(recorder)
    # Parameter counts and the effective finalized policy are checked before step 1.
    original_start = callback.on_train_start
    def start(context):
        original_start(context)
        cfg = context.state.cfg
        if (cfg.ddp.use_megatron_fsdp or cfg.ddp.grad_reduce_in_fp32
                or not cfg.optimizer.use_distributed_optimizer
                or cfg.optimizer.use_precision_aware_optimizer):
            raise ValueError("Effective storage policy differs from the independent CPU projection")
        if any(p.dtype != torch.bfloat16 for model in context.model for p in model.parameters()):
            raise ValueError("Expected BF16 model parameter storage")
        local_count = sum(p.numel() for model in context.model for p in model.parameters())
        if local_count != 6157545024:
            raise ValueError(f"Per-rank parameter mismatch: {local_count}")
        manifest["local_parameter_elements"] = local_count
        manifest["effective_ddp"] = str(cfg.ddp)
        manifest["effective_optimizer"] = str(cfg.optimizer)
        manifest["effective_mixed_precision"] = str(cfg.mixed_precision)
        manifest["effective_model"] = {key: str(getattr(context.state.cfg.model, key)) for key in (
            "params_dtype", "fp8", "fp8_recipe", "fp4", "fp8_param", "fp4_param",
            "recompute_granularity", "cuda_graph_impl", "cuda_graph_warmup_steps",
            "moe_flex_dispatcher_backend", "num_layers_at_end_in_bf16")}
        if rank == 0:
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    callbacks = CallbackManager()
    callbacks.register("on_train_start", start)
    callbacks.register("on_train_step_start", callback.on_train_step_start)
    callbacks.register("on_train_step_end", callback.on_train_step_end)
    recorder.start()
    recorder.install_megatron_hooks()
    try:
        pretrain(config=config, forward_step_func=forward_step, callbacks=callbacks)
        manifest["training_complete"] = True
    except Exception as exc:
        manifest.update(error_type=type(exc).__name__, error=str(exc))
        recorder.record("failure", snapshot=True)
        raise
    finally:
        recorder.close()
        if rank == 0:
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
