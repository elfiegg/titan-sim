"""Tensor-dispatch traces of logical storage and operations, never GPU predictions.

The same observer accepts fake tensors for projections and real tensors for
observations. Logical storage bytes exclude allocator blocks and kernel scratch.
"""
from __future__ import annotations

import contextlib
import math
import time
from collections import Counter
from typing import Any

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.flop_counter import flop_registry
from torch.utils.module_tracker import ModuleTracker


def _local_tensor(tensor):
    from torch.distributed.tensor import DTensor
    # Parameter DTensors are wrappers: storage and calibration use the local
    # rank's tensor. Calling to_local here would itself introduce dispatch.
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _tensors(value):
    return [_local_tensor(item) for item in tree_flatten(value)[0] if isinstance(item, torch.Tensor)]


class TensorCapture(TorchDispatchMode):
    """Record dispatch, unique StorageImpl lifetimes, aliases and autograd saves.

    Enter inside FakeTensorMode so this mode observes its user-visible results.
    Storage expiration is checked at each operator boundary; release timestamps
    therefore have operator-boundary precision. No tensor or storage is retained
    by the observer. Event byte traffic is a logical tensor payload estimate,
    not measured DRAM traffic. Unknown FLOP formulas remain explicitly reported.
    """

    def __init__(self, *, source: str = "cpu_projection", rank: int = 0, modeled_device: str | None = None):
        super().__init__()
        if source not in {"cpu_projection", "gpu_observation"}:
            raise ValueError("source must be cpu_projection or gpu_observation")
        self.source, self.rank = source, rank
        self.modeled_device = modeled_device
        self.stream = "compute"
        self.events: list[dict[str, Any]] = []
        self.storages: dict[str, dict[str, Any]] = {}
        self.saved_tensors: list[dict[str, Any]] = []
        self.unknown_flop_ops: Counter = Counter()
        self._live: dict[int, tuple[str, StorageWeakRef]] = {}
        self._phase = "setup"
        self._tracker = ModuleTracker()
        self._stack = contextlib.ExitStack()
        self._last_by_stream: dict[str, str] = {}
        self._writers: dict[str, str] = {}
        self._readers: dict[str, set[str]] = {}
        self._host_time_provider = None
        self._event_listener = None

    def __enter__(self):
        self._stack.enter_context(self._tracker)
        self._stack.enter_context(torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack))
        return super().__enter__()

    def __exit__(self, *exc):
        result = super().__exit__(*exc)
        self._stack.close()
        self.sweep()
        return result

    @contextlib.contextmanager
    def phase(self, name: str):
        previous, self._phase = self._phase, name
        try:
            yield self
        finally:
            self.sweep()
            self._phase = previous

    def _event(self, kind: str, name: str, **fields):
        defer_notify = fields.pop('_defer_notify', False)
        event_id = f"r{self.rank}:e{len(self.events)}"
        event = dict(event_id=event_id, kind=kind, rank=self.rank, stream=self.stream,
                     name=name, deps=[],
                     flops=0.0, bytes_read=0, bytes_write=0, dtype="float32", size_bytes=0,
                     category=self._phase, reads=[], writes=[], phase=self._phase,
                     module_scope=sorted(self._tracker.parents), device=self.modeled_device or "cpu")
        event.update(fields)
        if self._host_time_provider is not None:
            event['host_time_us'] = self._host_time_provider()
        deps = set(event["deps"])
        stream = event["stream"]
        if stream in self._last_by_stream:
            deps.add(self._last_by_stream[stream])
        accesses = set(event["reads"]) | set(event["writes"])
        if kind == "release":
            accesses.add(event["storage_id"])
        for sid in accesses:
            if sid in self._writers:
                deps.add(self._writers[sid])
            if sid in event["writes"] or kind == "release":
                deps.update(self._readers.get(sid, ()))
        event["deps"] = sorted(deps)
        self.events.append(event)
        self._last_by_stream[stream] = event_id
        for sid in event["reads"]:
            self._readers.setdefault(sid, set()).add(event_id)
        for sid in event["writes"]:
            self._writers[sid] = event_id
            self._readers[sid] = set()
        if kind == "allocate":
            self._writers[event["storage_id"]] = event_id
        if not defer_notify:
            self._notify_event(event)
        return event

    def _notify_event(self, event):
        if self._event_listener is not None:
            self._event_listener(event)

    def register_tensor(self, tensor, *, category: str | None = None, name: str | None = None, modeled_device: str | None = None):
        tensor = _local_tensor(tensor)
        if tensor.layout != torch.strided:
            raise NotImplementedError(f"Storage accounting requires strided tensors: {tensor.layout}")
        from torch._subclasses.fake_tensor import FakeTensor
        if self.source == "cpu_projection" and tensor.is_cuda and not isinstance(tensor, FakeTensor):
            raise ValueError("Real CUDA tensors cannot supply a CPU projection")
        storage = tensor.untyped_storage()
        key = storage._cdata
        if key in self._live and self._live[key][1].expired():
            self._release(key)
        reallocation_of = None
        if key in self._live and int(storage.nbytes()) != self.storages[self._live[key][0]]["size_bytes"]:
            reallocation_of = self._live[key][0]
            self._release(key)
        if key not in self._live:
            sid = f"r{self.rank}:s{len(self.storages)}"
            self._live[key] = sid, StorageWeakRef(storage)
            entry = dict(storage_id=sid, size_bytes=int(storage.nbytes()),
                         dtype=str(tensor.dtype).removeprefix("torch."), device=modeled_device or (self.modeled_device if isinstance(tensor, FakeTensor) else None) or str(tensor.device),
                         capture_device=str(tensor.device),
                         category=category or self._phase, name=name,
                         allocation_event=len(self.events), release_event=None, reallocation_of=reallocation_of)
            self.storages[sid] = entry
            self._event("allocate", name or "tensor_storage", storage_id=sid,
                        size_bytes=entry["size_bytes"], dtype=entry["dtype"], category=entry["category"], device=entry["device"])
        sid = self._live[key][0]
        entry = self.storages[sid]
        if category is not None:
            entry["category"] = category
            self.events[entry["allocation_event"]]["category"] = category
        if name is not None:
            entry["name"] = name
        if modeled_device is not None:
            entry["device"] = modeled_device
            self.events[entry["allocation_event"]]["device"] = modeled_device
        if category is not None or modeled_device is not None:
            self._notify_event(self.events[entry["allocation_event"]])
        return sid

    def register_model(self, model):
        for name, tensor in model.named_parameters():
            self.register_tensor(tensor, category="parameters", name=name)
            if tensor.grad is not None:
                self.register_tensor(tensor.grad, category="gradients", name=name + ".grad")
        for name, tensor in model.named_buffers():
            self.register_tensor(tensor, category="buffers", name=name)

    def register_optimizer(self, optimizer):
        for index, state in enumerate(optimizer.state.values()):
            for name, value in state.items():
                for tensor in _tensors(value):
                    self.register_tensor(tensor, category="optimizer", name=f"state.{index}.{name}",
                                         modeled_device="cpu" if name == "step" and tensor.device.type == "cpu" else None)

    def _release(self, key):
        sid, _ = self._live.pop(key)
        entry = self.storages[sid]
        entry["release_event"] = len(self.events)
        self._event("release", entry["name"] or "tensor_storage", storage_id=sid,
                    size_bytes=entry["size_bytes"], dtype=entry["dtype"], category=entry["category"], device=entry["device"])

    def sweep(self):
        for key, (sid, ref) in list(self._live.items()):
            if ref.expired():
                self._release(key)
            else:
                # FSDP frees backing memory with storage.resize_(0), outside
                # tensor dispatch, while views of the StorageImpl remain alive.
                storage = torch.UntypedStorage._new_with_weak_ptr(ref.cdata)
                if int(storage.nbytes()) != self.storages[sid]["size_bytes"]:
                    self._release(key)

    def _pack(self, tensor):
        sid = self.register_tensor(tensor)
        self.saved_tensors.append(dict(storage_id=sid, action="save", event_index=len(self.events),
                                       phase=self._phase, shape=list(tensor.shape),
                                       module_scope=sorted(self._tracker.parents)))
        # Saving an output tensor itself can form a grad_fn reference cycle.
        # A detached alias preserves storage lifetime without extending the graph.
        with _disable_current_modes():
            return tensor.detach()

    def _unpack(self, tensor):
        self.saved_tensors.append(dict(storage_id=self.register_tensor(tensor), action="unpack",
                                       event_index=len(self.events), phase=self._phase))
        return tensor

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.name in {"aten::resize_", "aten::resize_as_"}:
            raise NotImplementedError("Tensor resize_ dispatch is not modeled; FSDP direct storage resize is supported")
        self.sweep()
        inputs = _tensors((args, kwargs))
        reads = list(dict.fromkeys(self.register_tensor(tensor) for tensor in inputs))
        start = time.perf_counter_ns()
        output = func(*args, **kwargs)
        capture_wall_us = (time.perf_counter_ns() - start) / 1000
        outputs = _tensors(output)
        output_ids = list(dict.fromkeys(self.register_tensor(tensor) for tensor in outputs))
        mutated = []
        mutated_tensors = []
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is not None and argument.alias_info.is_write:
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                mutated_tensors.extend(_tensors(value))
                mutated.extend(self.register_tensor(tensor) for tensor in _tensors(value))
        # Views create aliases, but cause no data write or DRAM transfer themselves.
        is_alias = bool(outputs) and all(sid in reads for sid in output_ids) and not mutated
        writes = list(dict.fromkeys(mutated + ([] if is_alias else output_ids)))
        name = str(func)
        metadata_only = func._schema.name in {
            "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::new_empty",
            "aten::new_empty_strided", "prim::device", "aten::sym_size", "aten::sym_stride",
            "aten::sym_numel", "aten::size", "aten::stride", "aten::numel", "aten::is_contiguous",
        }
        scalar_read = func._schema.name == "aten::_local_scalar_dense"
        scalar_on_cuda = scalar_read and any(self.storages[sid]["device"].startswith("cuda") for sid in reads)
        if scalar_read and not scalar_on_cuda:
            metadata_only = True
        formula = flop_registry.get(func._overloadpacket)
        flops = 0
        if formula is not None:
            flops = formula(*tree_map(_local_tensor, args), **tree_map(_local_tensor, kwargs),
                            out_val=tree_map(_local_tensor, output))
        elif not is_alias and not metadata_only:
            self.unknown_flop_ops[str(func)] += 1
        kind = "collective" if any(word in name for word in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")) else "compute"
        if "wait_tensor" in name:
            kind = "wait"
        event = self._event(kind, name, _defer_notify=True, reads=reads, writes=writes, flops=float(flops),
                           stream="__host_control__" if scalar_read and not scalar_on_cuda else self.stream,
                           bytes_read=0 if is_alias or metadata_only else self._payload_bytes(inputs),
                           bytes_write=0 if is_alias or metadata_only else self._payload_bytes(outputs + mutated_tensors),
                           dtype=str((outputs or inputs)[0].dtype).removeprefix("torch.") if outputs or inputs else "float32",
                           capture_wall_us=capture_wall_us, aliases=output_ids if is_alias else [],
                           flop_formula="pytorch_registry" if formula else "unavailable")
        event["traffic_basis"] = "unique_contiguous_regions_or_strided_enclosing_regions"
        event["metadata_only"] = metadata_only or is_alias
        if metadata_only or is_alias:
            event["flops"] = 0.0
            event["execution_kind"] = "metadata"
        event["signature"] = self._signature(func, args, kwargs)
        event["input_tensors"] = [self._tensor_metadata(t) for t in inputs]
        event["output_tensors"] = [self._tensor_metadata(t) for t in outputs]
        event["shapes"] = [list(t.shape) for t in inputs]
        if inputs or outputs:
            sid = (output_ids or reads)[0]
            event["device"] = self.storages[sid]["device"]
        devices = {self.storages[sid]["device"].split(":")[0] for sid in reads + output_ids}
        blocking_copy = False
        if "cpu" in devices and "cuda" in devices and func._schema.name in {"aten::_to_copy", "aten::copy_"}:
            event["traffic_domain"] = "host_device"
            event["device"] = next(self.storages[sid]["device"] for sid in reads + output_ids
                                   if self.storages[sid]["device"].startswith("cuda"))
            event["host_device_transfer_bytes"] = max(event["bytes_read"], event["bytes_write"])
            non_blocking = next(
                args[index] if index < len(args) else kwargs.get(argument.name, argument.default_value)
                for index, argument in enumerate(func._schema.arguments)
                if argument.name == "non_blocking"
            )
            blocking_copy = not non_blocking
        if scalar_on_cuda:
            event["traffic_domain"] = "host_device"
            event["host_device_transfer_bytes"] = event["bytes_read"]
        self._notify_event(event)
        if blocking_copy or scalar_on_cuda:
            self._event("host_sync", "blocking_host_device_copy" if blocking_copy else "cuda_scalar_read", stream="__host_control__",
                        deps=[event["event_id"]], sync_scope="event", device="cpu")
        return output

    def _signature(self, func, args, kwargs):
        unsupported = set()

        def value(item):
            if isinstance(item, torch.Tensor):
                item = _local_tensor(item)
                sid = self.register_tensor(item)
                return {"tensor": {"shape": list(item.shape), "stride": list(item.stride()),
                                   "dtype": str(item.dtype), "layout": str(item.layout),
                                   "storage_offset": item.storage_offset(),
                                   "device_type": self.storages[sid]["device"].split(":")[0]}}
            if item is None or isinstance(item, (str, bool, int)):
                return item
            if isinstance(item, float):
                return item if math.isfinite(item) else {"float": str(item)}
            if isinstance(item, complex):
                return {"complex": [value(item.real), value(item.imag)]}
            if isinstance(item, torch.device):
                return {"device_type": self.modeled_device.split(":")[0] if self.modeled_device else item.type}
            if isinstance(item, (torch.dtype, torch.layout, torch.memory_format)):
                return str(item)
            if isinstance(item, (tuple, list)):
                return [value(element) for element in item]
            if isinstance(item, dict) and all(isinstance(key, str) for key in item):
                return {key: value(element) for key, element in sorted(item.items())}
            type_name = f"{type(item).__module__}.{type(item).__qualname__}"
            unsupported.add(type_name)
            return {"unsupported_type": type_name}

        arguments = {}
        for index, argument in enumerate(func._schema.arguments):
            item = args[index] if index < len(args) else kwargs.get(argument.name, argument.default_value)
            arguments[argument.name] = value(item)
        return dict(version=1, status="unsupported" if unsupported else "complete",
                    arguments=arguments, unsupported_types=sorted(unsupported))

    def _payload_bytes(self, tensors):
        intervals = {}
        for tensor in tensors:
            if tensor.numel() == 0:
                continue
            sid = self.register_tensor(tensor)
            item_bytes = tensor.element_size()
            start = tensor.storage_offset() * item_bytes
            # Exact for contiguous views. Strided views use the enclosing byte
            # interval, a conservative upper bound that includes stride gaps.
            span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
            intervals.setdefault(sid, []).append((start, start + span * item_bytes))
        total = 0
        for regions in intervals.values():
            cursor_start, cursor_end = sorted(regions)[0]
            for start, end in sorted(regions)[1:]:
                if start > cursor_end:
                    total += cursor_end - cursor_start
                    cursor_start, cursor_end = start, end
                else:
                    cursor_end = max(cursor_end, end)
            total += cursor_end - cursor_start
        return total

    @staticmethod
    def _tensor_metadata(tensor):
        return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                    dtype=str(tensor.dtype).removeprefix("torch."), layout=str(tensor.layout),
                    device=str(tensor.device), storage_offset=tensor.storage_offset())

    def document(self, *, metadata=None):
        self.sweep()
        return dict(schema_version=1, source=self.source, metadata=metadata or {},
                    events=self.events, storages=list(self.storages.values()),
                    saved_tensors=self.saved_tensors, unknown_flop_ops=dict(self.unknown_flop_ops),
                    limitations=["Logical storage only: excludes CUDA allocator reservations, workspaces and communication buffers hidden from dispatch.",
                                 "Releases are observed at operator boundaries; tensor aliases share one StorageImpl.",
                                 "Host dispatch durations are diagnostic and are not predicted GPU durations.",
                                 "FLOPs are partial when unknown_flop_ops is nonempty; traffic is a union of accessed byte intervals (strided gaps conservatively included), not DRAM measurement.",
                                 "Cross-stream ordering is represented only when an explicit runtime shim records waits and stream selection."])
