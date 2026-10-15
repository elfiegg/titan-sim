"""GPU evidence collection and conservative CPU allocation attribution.

Snapshots contain allocator metadata and stacks, never tensor values. Exact
ownership is separate from stack-based candidate labels and untracked memory.
"""
from collections import defaultdict
from functools import wraps
import json
import os
from pathlib import Path
import pickle


def summarize_external_allocations(events):
    """Replay only the instrumented HybridEP allocator, not all CUDA memory."""
    live, peak, issues = {}, 0, []
    for event in events:
        address = event["address"]
        if event["event"].startswith("allocate_"):
            if address in live:
                issues.append("address allocated twice without a recorded free")
            physical = event["physical_bytes"] if event["event"] == "allocate_fabric" else None
            live[address] = (event["requested_bytes"], physical)
        elif event["event"] == "free":
            if address not in live:
                issues.append("free without a recorded allocation")
            live.pop(address, None)
        else:
            issues.append("unknown allocation event")
        peak = max(peak, sum(value[1] if value[1] is not None else value[0] for value in live.values()))
    requested = sum(value[0] for value in live.values())
    fabric_physical = sum(value[1] for value in live.values() if value[1] is not None)
    cuda_requested = sum(value[0] for value in live.values() if value[1] is None)
    accounted = fabric_physical + cuda_requested
    return {"scope": "instrumented HybridEP ExtendedMemoryAllocator only",
            "requested_live_bytes": requested, "accounted_live_bytes": accounted,
            "fabric_physical_live_bytes": fabric_physical,
            "cuda_malloc_requested_live_bytes": cuda_requested,
            "physical_live_bytes": accounted if not cuda_requested else None,
            "fabric_alignment_live_bytes": accounted - requested,
            "peak_accounted_bytes": peak, "cuda_malloc_physical_rounding_known": False,
            "unmatched_events": issues, "complete_external_memory_coverage": False}


def summarize_snapshot(snapshot, inventory, device=None):
    """Partition active allocated bytes without double counting aliased storage."""
    totals, pools = defaultdict(int), defaultdict(lambda: defaultdict(int))
    categories, hints = defaultdict(int), defaultdict(int)
    for segment in snapshot.get("segments", []):
        if device is not None and segment.get("device") != device:
            continue
        pool = str(segment.get("segment_pool_id", "unspecified"))
        totals["reserved_bytes"] += segment["total_size"]
        for block in segment.get("blocks", []):
            state, size = block["state"], block["size"]
            totals[state] += size
            pools[pool][state] += size
            if state != "active_allocated":
                continue
            start, end = block["address"], block["address"] + size
            requested_end = start + block.get("requested_size", size)
            if not start <= requested_end <= end:
                raise ValueError("Snapshot requested size exceeds its allocator block")
            overlapping = [entry for entry in inventory if
                entry.get("device_index", segment.get("device")) == segment.get("device")
                and entry["address"] < end and entry["address"] + entry["size_bytes"] > start]
            cuts = {start, end, requested_end}
            for entry in overlapping:
                cuts.update((max(start, entry["address"]), min(end, entry["address"] + entry["size_bytes"])))
            cuts = sorted(cuts)
            for left, right in zip(cuts, cuts[1:]):
                owners = {c for entry in overlapping
                          if entry["address"] <= left and entry["address"] + entry["size_bytes"] >= right
                          for c in entry["categories"]}
                key = next(iter(owners)) if len(owners) == 1 else "shared:" + "+".join(sorted(owners)) if owners else "unattributed"
                if not owners and left >= requested_end:
                    key = "allocator_block_slack"
                categories[key] += right - left
                if key == "unattributed":
                    stack = " ".join(f.get("filename", "") + " " + f.get("name", "") for f in block.get("frames", [])).lower()
                    functions = [f.get("name", "").lower() for f in block.get("frames", [])]
                    # These are evidence hints, not confirmed tensor ownership.
                    hint = "unknown"
                    if "get_dummy_wgrad" in functions:
                        hint = "te_dummy_weight_gradient_candidate"
                    elif any(any(token in name for token in (
                            "cublas_workspace", "gemm_setup_workspace", "setworkspaceforhandle"))
                            for name in functions):
                        hint = "gemm_workspace_candidate"
                    elif any(s in stack for s in ("quantiz", "mxfp8", "nvfp4", "cast_transpose")):
                        hint = "quantization_candidate"
                    elif any(s in stack for s in ("deep_ep", "fused_a2a", "token_dispatch", "nccl")):
                        hint = "communication_or_routing_candidate"
                    elif any(s in stack for s in ("attention", "softmax", "cross_entropy", "linear")):
                        hint = "activation_or_kernel_storage_candidate"
                    hints[hint] += right - left
    return {"totals_bytes": dict(totals), "exact_storage_categories_bytes": dict(categories),
            "unattributed_stack_hints_bytes": dict(hints),
            "pools_by_state_bytes": dict(pools),
            "pool_interpretation": "Non-default pool IDs may be graph or custom pools; not classified from ID alone",
            "attributed_plus_unattributed_equals_active": sum(categories.values()) == totals["active_allocated"]}


def storage_inventory(models, optimizer):
    """Inspect known ownership fields, retaining only integer metadata."""
    import torch
    found = {}

    def add(tensor, category, name):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return
        try:
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
        except (RuntimeError, NotImplementedError):
            return
        if not key[2]:
            return
        entry = found.setdefault(key, {"device_index": key[0], "address": key[1], "size_bytes": key[2], "categories": set(), "examples": []})
        entry["categories"].add(category)
        if len(entry["examples"]) < 3:
            entry["examples"].append(name)

    cache_fields = ("_fp8_workspaces", "_weight_workspace", "_rowwise_data", "_columnwise_data",
                    "_rowwise_scale_inv", "_columnwise_scale_inv", "_scale_inv", "_data", "_transpose")
    def cache(value, name, depth=0):
        if depth > 4:
            return
        add(value, "quantization_cache", name)
        if isinstance(value, dict):
            for key, child in value.items():
                cache(child, f"{name}.{key}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                cache(child, f"{name}.{index}", depth + 1)
        else:
            for key in cache_fields[2:]:
                child = getattr(value, key, None)
                if child is not None and child is not value:
                    cache(child, f"{name}.{key}", depth + 1)

    for chunk, model in enumerate(models or []):
        for name, parameter in model.named_parameters():
            add(parameter, "parameters", f"{chunk}.{name}")
            add(getattr(parameter, "main_grad", None), "model_gradient_buffers", name + ".main_grad")
        for name, buffer in model.named_buffers():
            add(buffer, "registered_model_buffers", name)
        for name, module in model.named_modules():
            for key in cache_fields[:2]:
                value = getattr(module, key, None)
                if value is not None:
                    cache(value, name + "." + key)
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer]) if optimizer is not None else []
    for index, opt in enumerate(optimizers):
        for group in getattr(opt, "shard_fp32_from_float16_groups", []):
            for param in group:
                add(param, "master_weights", f"optimizer{index}.master")
                add(param.grad, "optimizer_gradients", f"optimizer{index}.master.grad")
        inner = getattr(opt, "optimizer", None)
        for state in getattr(inner, "state", {}).values():
            for name, value in state.items():
                add(value, "adam_first_moment" if name == "exp_avg" else "adam_second_moment" if name == "exp_avg_sq" else "optimizer_other", name)
    return [{**entry, "categories": sorted(entry["categories"])} for entry in found.values()]


class AllocationRecorder:
    """Bounded observation hooks; no tensors retained and no explicit GPU sync."""

    def __init__(self, output, *, rank, detailed_ranks=(0, 1), history_entries=100000):
        self.output = Path(output) / f"rank{rank}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.rank, self.detailed = rank, rank in detailed_ranks
        self.history_entries, self.step = history_entries, 0
        self.models, self.optimizer, self.sequence = [], None, 0
        self.restore = []
        self.enabled = False

    def start(self):
        import torch
        if self.detailed:
            torch.cuda.memory._record_memory_history(max_entries=self.history_entries)
        self.enabled = True
        self.record("before_model_setup", snapshot=self.detailed)

    def record(self, phase, *, snapshot=False):
        import torch
        if not self.enabled or torch.cuda.is_current_stream_capturing():
            return
        self.sequence += 1
        stats = torch.cuda.memory_stats()
        row = {"rank": self.rank, "step": self.step, "phase": phase, "sequence": self.sequence,
               "explicit_synchronize": False, "allocator_backend": torch.cuda.get_allocator_backend(),
               "counters": {k: int(stats.get(k, 0)) for k in (
                   "allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current",
                   "reserved_bytes.all.peak", "active_bytes.all.current", "inactive_split_bytes.all.current")}}
        try:
            row["device_wide_used_bytes"] = torch.cuda.device_memory_used(torch.cuda.current_device())
        except Exception as exc:
            row["device_wide_used_error"] = type(exc).__name__
        row["device_wide_scope"] = "whole device, not exact process/NCCL attribution"
        native_dir = os.environ.get("TITANSIM_DEEPEP_MEMORY_LOG_DIR")
        if native_dir:
            native = Path(native_dir) / f"rank{self.rank}-pid{os.getpid()}.jsonl"
            row["deepep_log"] = str(native.relative_to(self.output.parent))
            row["deepep_log_bytes"] = native.stat().st_size if native.exists() else 0
        if self.detailed and snapshot:
            prefix = f"{self.sequence:03d}-step{self.step}-{phase}"
            inventory = storage_inventory(self.models, self.optimizer)
            # CPU serialization work can perturb overlap; a counters-only control
            # run is required. Never insert snapshot operations into CUDA capture.
            data = torch.cuda.memory._snapshot()
            with (self.output / f"{prefix}.pickle").open("wb") as handle:
                pickle.dump(data, handle)
            (self.output / f"{prefix}-inventory.json").write_text(json.dumps(inventory) + "\n")
            row["snapshot"] = prefix + ".pickle"
            row["inventory"] = prefix + "-inventory.json"
        with (self.output / "phases.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def wrap(self, cls, method, label):
        original = getattr(cls, method)
        recorder = self
        @wraps(original)
        def observed(obj, *args, **kwargs):
            active = recorder.step <= 2
            if active:
                recorder.record("before_" + label, snapshot=recorder.step in (1, 2))
            result = original(obj, *args, **kwargs)
            if active:
                recorder.record("after_" + label, snapshot=recorder.step in (1, 2))
            return result
        setattr(cls, method, observed)
        self.restore.append((cls, method, original))

    def install_megatron_hooks(self):
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from megatron.core.models.gpt.gpt_model import GPTModel
        self.wrap(DistributedOptimizer, "zero_grad", "optimizer_zero_grad")
        self.wrap(DistributedOptimizer, "_copy_model_grads_to_main_grads", "fp32_gradient_copy")
        self.wrap(DistributedOptimizer, "step_with_ready_grads", "adam_update")
        self.wrap(GPTModel, "forward", "model_forward")

    def close(self):
        import torch
        if self.enabled:
            self.record("finished", snapshot=self.detailed)
            if self.detailed:
                torch.cuda.memory._record_memory_history(enabled=None)
        for cls, method, original in reversed(self.restore):
            setattr(cls, method, original)
        self.enabled = False


def make_bridge_callback(recorder):
    from megatron.bridge.training.callbacks import Callback
    class MemoryCallback(Callback):
        def on_train_start(self, context):
            recorder.models, recorder.optimizer = context.model, context.optimizer
            recorder.record("after_model_optimizer_setup", snapshot=True)

        def on_train_step_start(self, context):
            recorder.step = context.state.train_state.step + 1
            recorder.record("step_start", snapshot=recorder.step <= 2)

        def on_train_step_end(self, context):
            recorder.record("step_end", snapshot=recorder.step in (1, 2, 3, 4, 6))
    return MemoryCallback()
