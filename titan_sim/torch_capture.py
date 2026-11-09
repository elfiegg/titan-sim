"""Optional PyTorch dispatch observer. No kernels or model forwards are replaced.

This is an operator/storage trace, not a CUDA kernel trace. Import this module
only inside a compatible PyTorch environment; the simulator never imports it.
"""
from contextlib import contextmanager
import hashlib
import json

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.module_tracker import ModuleTracker

from .core import Node, Storage
from .workload import Workload


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class TorchCapture(TorchDispatchMode):
    """Observe a rank without retaining tensors or extending their lifetimes.

    Storage identity includes an allocation generation: allocator address reuse
    and resize_(0), used by FSDP, must not merge unrelated allocations. Releases
    are sampled at operator boundaries. Dependencies deliberately serialize host
    dispatch; they do not claim to recover CUDA stream/event ordering.
    """

    # Record HOP boundaries and call the original implementation. Their nested
    # workspaces/compiled kernels remain opaque and are explicitly reported.
    supports_higher_order_operators = True

    def __init__(self, *, rank=0, world_size=1, device='cpu', provenance=None):
        super().__init__()
        self.rank, self.world_size = rank, world_size
        self.device = torch.device(device)
        self.provenance = provenance or {}
        self.provenance_key = digest(self.provenance)
        self.nodes, self.storage = [], []
        self.storage_metadata, self.state = {}, {}
        self._live = {}
        self._phase, self.step = 'training_step', 0
        self._tracker = ModuleTracker()
        self.gaps = {
            'Operator dispatch is not a selected CUDA kernel or compiled-region trace.',
            'Dispatch/module instrumentation can change compiler behavior; compare against an uninstrumented run.',
            'Host dispatch is serialized; CUDA streams, waits, overlap and contention are not reconstructed.',
            'Logical tensor storage excludes allocator reservation, external buffers and kernel workspaces.',
        }
        if world_size > 1:
            self.gaps.add('Rank-local trace: communication participants, rendezvous and all-rank schedule need reconciliation.')

    def __enter__(self):
        self._tracker.__enter__()
        try:
            return super().__enter__()
        except BaseException:
            self._tracker.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self._tracker.__exit__(*exc)
            self.sweep()

    @contextmanager
    def phase(self, name):
        previous, self._phase = self._phase, name
        try:
            yield
        finally:
            self._phase = previous

    def _node(self, name, **attrs):
        node = Node(len(self.nodes), name, self.rank,
                    deps=[self.nodes[-1].id] if self.nodes else [],
                    attrs=dict(phase=self._phase, step=self.step, **attrs))
        self.nodes.append(node)
        return node

    def marker(self, name):
        return self._node(name, zero_cost_semantic=True).id

    def sweep(self):
        released = []
        for key, (record, ref) in list(self._live.items()):
            backing = None if ref.expired() else torch.UntypedStorage._new_with_weak_ptr(ref.cdata)
            if backing is None or backing.nbytes() != record.size:
                released.append(record)
                del self._live[key]
        if released:
            boundary = self.marker('storage_release_boundary')
            for record in released:
                record.consumers.append(boundary)
                self.storage_metadata[record.id]['release_boundary'] = boundary

    @staticmethod
    def _local(tensor):
        from torch.distributed.tensor import DTensor
        return tensor._local_tensor if isinstance(tensor, DTensor) else tensor

    def register_tensor(self, tensor, *, producer=None, category=None, name=None):
        tensor = self._local(tensor)
        if tensor.device != self.device:
            # CPU scalars/offload storage must not be charged to GPU memory.
            if tensor.device.type != 'cpu':
                self.gaps.add('A rank used another device; only the declared device has storage accounting.')
            return None
        if tensor.layout != torch.strided:
            self.gaps.add('Storage accounting unsupported for layout '+str(tensor.layout))
            return None
        try:
            backing = tensor.untyped_storage()
        except (RuntimeError, NotImplementedError) as error:
            self.gaps.add('Native tensor storage is unavailable ('+type(error).__name__+'); its allocation is unverified.')
            return None
        key = backing._cdata
        if key in self._live:
            record, ref = self._live[key]
            if ref.expired() or record.size != backing.nbytes():
                self.sweep()
        if key not in self._live:
            sid = f'r{self.rank}:s{len(self.storage)}'
            record = Storage(sid, self.rank, int(backing.nbytes()), producer,
                             category=category or 'tensor')
            self.storage.append(record)
            self._live[key] = record, StorageWeakRef(backing)
            self.storage_metadata[sid] = {'device': str(tensor.device), 'names': []}
        record = self._live[key][0]
        if category is not None:
            record.category = category
        if name is not None and name not in self.storage_metadata[record.id]['names']:
            self.storage_metadata[record.id]['names'].append(name)
        return record.id

    def describe(self, value, *, node=None):
        if isinstance(value, torch.Tensor):
            tensor = self._local(value)
            sid = self.register_tensor(value, producer=node)
            result = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                          device=str(tensor.device), layout=str(tensor.layout),
                          storage_id=sid)
            if tensor.layout == torch.strided:
                result.update(stride=list(tensor.stride()), storage_offset=tensor.storage_offset())
            if value is not tensor:
                result['distributed'] = dict(global_shape=list(value.shape),
                    placements=[str(p) for p in value.placements],
                    mesh_names=list(value.device_mesh.mesh_dim_names or ()),
                    mesh=value.device_mesh.mesh.tolist())
            if sid is not None and node is not None:
                record = self._live[tensor.untyped_storage()._cdata][0]
                if not record.consumers or record.consumers[-1] != node:
                    record.consumers.append(node)
            return result
        if isinstance(value, (tuple, list)):
            return [self.describe(v, node=node) for v in value]
        if isinstance(value, dict):
            return {str(k): self.describe(v, node=node) for k, v in value.items()}
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            # JSON cannot represent infinities used by some attention masks.
            return value if float('-inf') < value < float('inf') else {'float': str(value)}
        if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return str(value)
        self.gaps.add('Opaque operator arguments require a semantic adapter: '+type(value).__qualname__)
        return {'opaque_type': type(value).__module__+'.'+type(value).__qualname__}

    def register_model(self, model, prefix='model'):
        for name, tensor in model.named_parameters():
            self.register_tensor(tensor, category='parameters', name=prefix+'.'+name)
            self.state[prefix+'.'+name] = self.describe(tensor)
            if tensor.grad is not None:
                self.register_tensor(tensor.grad, category='gradients', name=prefix+'.'+name+'.grad')
        for name, tensor in model.named_buffers():
            self.register_tensor(tensor, category='buffers', name=prefix+'.'+name)
            self.state[prefix+'.'+name] = self.describe(tensor)

    def register_optimizer(self, optimizer):
        def visit(value, path):
            if isinstance(value, torch.Tensor):
                self.register_tensor(value, category='optimizer', name=path)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, path+'.'+str(key))
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    visit(item, path+'.'+str(index))
        for index, state in enumerate(optimizer.state.values()):
            visit(state, f'optimizer.{index}')

    @staticmethod
    def _signature(value):
        # Allocation occurrence IDs must not make identical calibration cases
        # different. Shapes, strides, offsets, precision and scalars do matter.
        if isinstance(value, dict):
            return {k: TorchCapture._signature(v) for k, v in value.items() if k != 'storage_id'}
        if isinstance(value, list):
            return [TorchCapture._signature(v) for v in value]
        return value

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.sweep()
        name = str(func)
        node = self._node(name, operator=name, module_scope=sorted(self._tracker.parents),
                          in_backward=self._tracker.is_bw, status='issued')
        node.attrs['arguments'] = self.describe({'args': args, 'kwargs': kwargs or {}}, node=node.id)
        if 'c10d' in name or 'collective' in name:
            node.attrs['semantics'] = 'unresolved_communication'
            self.gaps.add('Communication dispatch observed, but transport/group/wait semantics are not lowered.')
            try:
                from .communication_cases import describe_communication
                communication = describe_communication(func, args, kwargs, self.describe)
                if communication is not None:
                    node.attrs['communication'] = communication
            except Exception as error:
                # Observation failure must not replace the native collective or
                # prevent this rank from entering it while peers are waiting.
                node.attrs['communication_error'] = type(error).__name__+': '+str(error)
                self.gaps.add('Native communication metadata could not be resolved for '+name)
        elif not name.startswith(('aten.', 'prims.')):
            self.gaps.add('Custom/higher-order operator needs implementation matching: '+name)
        try:
            output = func(*args, **(kwargs or {}))
        except Exception as error:
            node.attrs['status'] = 'failed'
            node.attrs['error_type'] = type(error).__qualname__
            self.gaps.add('An operator failed during capture: '+name)
            raise
        node.attrs['outputs'] = self.describe(output, node=node.id)
        node.attrs['status'] = 'returned'
        signature = dict(operator=name, arguments=self._signature(node.attrs['arguments']),
                         outputs=self._signature(node.attrs['outputs']), provenance_sha256=self.provenance_key)
        node.attrs['cost_key'] = 'torch-dispatch:'+digest(signature)
        return output

    def finish(self, *, hardware, manifest=None):
        self.sweep()
        boundary = self.marker('capture_end')
        for record, _ in self._live.values():
            record.consumers.append(boundary)
        details = dict(manifest or {})
        details.update(ranks=[self.rank], world_size=self.world_size,
                       provenance=self.provenance, storage_metadata=self.storage_metadata,
                       model_state=self.state,
                       unresolved=sorted(self.gaps | set(details.get('unresolved', []))))
        return Workload(self.nodes, self.storage, details, hardware)
