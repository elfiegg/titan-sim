"""CPU storage census for GPT-OSS with Megatron distributed Adam.

This is an unpadded persistent-state subtotal, not a full training peak model.
Observed memory counters are deliberately not an input to the predictor.
"""
from dataclasses import replace
from math import prod

from .runtime import logical_memory_events


def model_config(vocab_size, sequence_length):
    from torchtitan.models.gpt_oss import build_model_config

    config = build_model_config("20b", seq_len=sequence_length, attn_backend="flex")
    config.vocab_size = vocab_size
    config.tok_embeddings = replace(config.tok_embeddings, num_embeddings=vocab_size)
    config.lm_head = replace(config.lm_head, out_features=vocab_size)
    return config


def parameter_inventory(vocab_size=128256, sequence_length=8192):
    """Instantiate upstream parameters on meta; no values or GPU are allocated."""
    import torch

    with torch.device("meta"):
        model = model_config(vocab_size, sequence_length).build()
    rows = []
    for name, param in model.named_parameters():
        if "routed_experts" in name:
            placement = "expert"
        elif ("router.gate" in name or "norm.weight" in name
              or name.endswith("attention.wo.bias")):
            placement = "dense_replicated_tp"
        elif (name in ("tok_embeddings.weight", "lm_head.weight")
              or ".attention.qkv_linear.wqkv." in name
              or name.endswith(("attention.wo.weight", "attention.sinks"))):
            placement = "dense_sharded_tp"
        else:
            raise ValueError(f"Unreviewed parameter placement: {name}")
        rows.append({"name": name, "shape": list(param.shape),
                     "elements": param.numel(), "placement": placement})
    return rows


def project_state(inventory, *, world_size, tp, cp, ep, pp=1, etp=1):
    """Replay idealized persistent state for one representative rank.

    BF16 parameters and full BF16 gradient buffers are replicated within their
    data groups. FP32 master weights and two FP32 Adam moments are sharded.
    CP joins the dense optimizer group; experts use a distinct EP/ETP mesh.
    FP32 optimizer gradient shards survive the step and are freed at next zero_grad.
    Optimizer shards use ceil(mean elements), excluding bucket layout/padding.
    """
    for value in (world_size, tp, cp, ep, pp, etp):
        if type(value) is not int or value < 1:
            raise ValueError("Parallel dimensions must be positive integers")
    if pp != 1 or etp != 1 or tp not in (1, 2) or 32 % ep:
        raise ValueError("Reviewed scope: PP=ETP=1, TP=1 or 2, EP divides 32")
    if world_size % (tp * cp) or world_size % ep:
        raise ValueError("Invalid dense/expert mesh")
    local = {"dense": 0, "expert": 0}
    for row in inventory:
        n = row["elements"]
        if type(n) is not int or n <= 0 or prod(row["shape"]) != n:
            raise ValueError("Invalid parameter inventory")
        placement = row["placement"]
        if placement not in ("expert", "dense_sharded_tp", "dense_replicated_tp"):
            raise ValueError("Unknown placement")
        divisor = ep if placement == "expert" else tp if placement == "dense_sharded_tp" else 1
        if n % divisor:
            raise ValueError("Nonintegral parameter partition")
        local["expert" if placement == "expert" else "dense"] += n // divisor
    groups = {"dense": world_size // tp, "expert": world_size // ep}
    components, events = {}, []
    for phase, category, width, sharded in (
        ("model_and_ddp_setup", "parameters", 2, False),
        ("model_and_ddp_setup", "gradients", 2, False),
        ("optimizer_initialized", "master_weights", 4, True),
        ("first_optimizer_gradient_copy", "optimizer_gradients", 4, True),
        ("first_optimizer_step", "adam_first_moment", 4, True),
        ("first_optimizer_step", "adam_second_moment", 4, True),
    ):
        for group, n in local.items():
            elements = (n + groups[group] - 1) // groups[group] if sharded else n
            key = f"{group}.{category}"
            components[key] = elements * width
            events.append({"event_id": key, "kind": "allocate", "rank": 0,
                           "storage_id": key, "size_bytes": elements * width,
                           "category": category,
                           "metadata": {"phase": phase, "device": "cuda:0",
                                        "provenance": "CPU meta inventory and reviewed storage policy"}})
    # Model the next iteration's gradient release and re-creation. No fabricated
    # forward/backward activation events or GPU durations are inserted.
    for action, phase in (("release", "iteration_2_zero_grad"), ("allocate", "iteration_2_optimizer_gradient_copy")):
        for group in local:
            key = f"{group}.optimizer_gradients"
            events.append({"event_id": f"{phase}.{group}", "kind": action,
                           "rank": 0, "storage_id": key, "size_bytes": components[key],
                           "category": "optimizer_gradients",
                           "metadata": {"phase": phase, "device": "cuda:0"}})
    trajectory = logical_memory_events(events)
    return {"source": "cpu_projection", "scope": "unpadded persistent-state subtotal",
            "full_training_prediction": False, "local_parameter_elements": sum(local.values()),
            "local_elements_by_group": local, "optimizer_group_sizes": groups,
            "components_bytes": components, "events": events, "trajectory": trajectory,
            "persistent_state_bytes": sum(v for k, v in components.items() if not k.endswith("optimizer_gradients")),
            "end_step_state_bytes": trajectory[-1]["allocated_bytes"],
            "unknown_components": ["DDP bucket padding and optimizer shard imbalance",
                "quantized weight caches, transposes and scales", "saved activations and recomputation",
                "routing, dispatch and communication buffers", "kernel and compiler workspaces",
                "CUDA graph pools", "allocator cache and fragmentation", "external CUDA/NCCL allocations"],
            "capacity_verdict": "unknown", "runtime_prediction": None}
