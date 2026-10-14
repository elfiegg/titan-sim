"""Pinned upstream entry points; no copied model dimensions or training policies."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

TORCHTITAN_COMMIT = "607852350b41b4b280d8b8de6340f4847ed71c5e"
DEFAULT_RECIPE = "torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b"


def import_torchtitan(checkout: str | Path, *, verify_commit: bool = True) -> str:
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if verify_commit and revision != TORCHTITAN_COMMIT:
        raise ValueError(f"Expected TorchTitan {TORCHTITAN_COMMIT}, found {revision}")
    if not (checkout / "torchtitan" / "__init__.py").is_file():
        raise ValueError(f"No torchtitan package at {checkout}")
    sys.path.insert(0, str(checkout))
    return revision


def load_default_recipe():
    """Return the upstream production recipe with every default intact."""
    module, name = DEFAULT_RECIPE.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)()


def build_smoke_model_config(*, flavor: str = "debugmodel", seq_len: int = 16):
    """Unsharded eager metadata probe, explicitly distinct from production."""
    from torchtitan.models.deepseek_v3 import build_model_config

    return build_model_config(flavor, seq_len=seq_len, attn_backend="flex")


def load_validation_recipe(*, flavor="671B", seq_len=None, tokens=None, expert_parallel_degree=64):
    """Build the user-selected recipe and apply the original Trainer preparation."""
    import copy
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.config.validation import validate_model_training_config
    if flavor == "671B":
        from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b
        config = deepseek_v3_671b(seq_len=seq_len)
    elif flavor == "debugmodel":
        from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel
        config = deepseek_v3_debugmodel(seq_len=seq_len or 128)
    else:
        raise ValueError("Unsupported validation flavor")
    parallel = config.parallelism
    parallel.pipeline_parallel_degree = 1
    parallel.tensor_parallel_degree = 1
    parallel.expert_parallel_degree = expert_parallel_degree
    parallel.data_parallel_shard_degree = -1
    config.activation_checkpoint = FullAC.Config()
    if tokens is not None:
        config.training.num_tokens_per_microbatch_per_dp_rank = tokens
    # TrainingEngine expects Trainer to have already resolved model sharding.
    config.model = copy.deepcopy(config.model)
    config.model.set_sharding_(parallel)
    validate_model_training_config(
        config.model, parallelism=parallel, training=config.training,
        debug=config.debug, activation_checkpoint=config.activation_checkpoint,
        local_compile_config=config.compile,
        max_num_documents=config.dataloader.max_num_documents,
    )
    return config


def resolved_validation_topology(context, *, world_size, expert_parallel_degree):
    """Check the actual upstream context against the selected validation layout."""
    expected = dict(pp=1, tp=1, cp=1, dp_replicate=1,
                    dp_shard=world_size, ep=expert_parallel_degree)
    mismatches = {key: (getattr(context, key), value) for key, value in expected.items()
                  if getattr(context, key) != value}
    if world_size % expert_parallel_degree or mismatches:
        raise ValueError(f"Validation mesh does not match requested layout: {mismatches}")
    return dict(dense_fsdp=context.dp_shard, expert_parallel=context.ep,
                expert_fsdp=context.dp_shard * context.cp * context.tp // context.ep,
                input_data_parallel=context.dp_replicate * context.dp_shard)
