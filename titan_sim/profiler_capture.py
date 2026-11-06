"""Native execution-trace capture, compatible with compiled PyTorch execution.

Raw execution and Chrome traces are retained. The Workload projection preserves
leaf operations and tensor references, but deliberately declares incomplete
stream, communication and allocation semantics. It is not a ready timing model.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch

from .core import Node, Storage
from .torch_capture import TorchCapture, digest
from .workload import Workload


def _tensor_ref(value):
    return (isinstance(value, list) and len(value) == 6
            and all(type(v) is int for v in value[:5])
            and isinstance(value[5], str) and value[5].split(':')[0] in ('cpu', 'cuda', 'meta', 'xpu', 'mps'))


def _references(value):
    if _tensor_ref(value):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _signature(value):
    if _tensor_ref(value):
        return {'tensor': value[2:]}
    if isinstance(value, list):
        return [_signature(v) for v in value]
    if isinstance(value, dict):
        return {k: _signature(v) for k, v in value.items()}
    return value


def project_trace(trace, *, rank, step, device, offset, provenance_key):
    """Conservative projection; preserve raw hierarchy rather than double-count it."""
    if trace.get('schema') != '1.1.1-chakra.0.0.4':
        raise ValueError('unsupported PyTorch execution-trace schema: '+str(trace.get('schema')))
    raw = {n['id']: n for n in trace['nodes']}
    parents = {n['ctrl_deps'] for n in raw.values()}
    leaves = [n for n in raw.values() if n['id'] not in parents
              and not n['name'].startswith('[pytorch|')]
    attrs = lambda n: {a['name']: a['value'] for a in n['attrs']}
    leaves.sort(key=lambda n: (attrs(n).get('rf_id', 0), n['id']))
    nodes, storages = [], {}
    for original in leaves:
        metadata = attrs(original)
        ancestors, parent, seen = [], original['ctrl_deps'], set()
        while parent in raw and parent not in seen:
            seen.add(parent)
            ancestors.append(raw[parent]['name'])
            parent = raw[parent]['ctrl_deps']
        phase = next((n.removeprefix('titan_sim::') for n in ancestors if n.startswith('titan_sim::')), 'training_step')
        signature = dict(name=original['name'], schema=metadata.get('op_schema'),
                         inputs=_signature(original['inputs']), outputs=_signature(original['outputs']),
                         kernel_backend=metadata.get('kernel_backend'), provenance=provenance_key)
        node = Node(offset+len(nodes), original['name'], rank,
                    deps=[offset+len(nodes)-1] if offset+len(nodes) else [],
                    attrs=dict(operator=original['name'], phase=phase, step=step,
                               in_backward=any('Backward' in n or 'backward' in n for n in ancestors),
                               cost_key='pytorch-execution-trace:'+digest(signature),
                               execution_trace_id=original['id'], execution_trace_attrs=metadata,
                               ancestors=ancestors, inputs=original['inputs'], outputs=original['outputs']))
        nodes.append(node)
        for side in ('inputs', 'outputs'):
            for ref in _references(original[side]['values']):
                _, storage_id, storage_offset, numel, itemsize, tensor_device = ref
                if tensor_device != str(device):
                    continue
                sid = f'r{rank}:step{step}:et-storage{storage_id}'
                # This is an observed contiguous span, NOT actual backing size.
                # Strided views and unobserved allocations remain explicit gaps.
                extent = (storage_offset+numel)*itemsize
                if sid not in storages:
                    storages[sid] = Storage(sid, rank, extent, offset if side == 'inputs' else node.id,
                                           category='observed_tensor_extent')
                storages[sid].size = max(storages[sid].size, extent)
                storages[sid].consumers.append(node.id)
    if nodes:
        # ET does not establish frees. Retain observed allocations to step end.
        for storage in storages.values():
            if storage.consumers[-1] != nodes[-1].id:
                storage.consumers.append(nodes[-1].id)
    return nodes, list(storages.values())


class ProfilerCapture:
    """One native profiler session per captured step; no Python dispatch mode."""

    def __init__(self, *, trace_dir, rank=0, world_size=1, device='cpu', provenance=None):
        self.root = Path(trace_dir)/f'rank{rank}'
        self.root.mkdir(parents=True, exist_ok=False)
        self.rank, self.world_size, self.device = rank, world_size, torch.device(device)
        self.provenance = provenance or {}
        self.provenance_key = digest(self.provenance)
        self.step = 0
        self.nodes, self.storage, self.artifacts = [], [], []
        self.state_observer = TorchCapture(rank=rank, world_size=world_size, device=device)
        self.gaps = {
            'Execution-trace leaf projection serializes host issue order; CUDA stream/event dependencies need lowering.',
            'Tensor extents are not backing allocations; frees, strided spans, unused state and hidden workspaces need reconciliation.',
            'Profiler instrumentation and shape recording can perturb timing and tensor lifetimes.',
            'Operator signatures still require native kernel/compiled-region matching and held-out validation.',
            'Initialization, checkpointing and work outside selected train_step calls are excluded.',
        }
        if world_size > 1:
            self.gaps.add('Rank-local traces require all-rank communication/group/rendezvous reconciliation.')

    def register_model(self, model, prefix='model'):
        self.state_observer.register_model(model, prefix)

    def register_optimizer(self, optimizer):
        self.state_observer.register_optimizer(optimizer)

    @contextmanager
    def phase(self, name):
        with torch.profiler.record_function('titan_sim::'+name):
            yield

    def __enter__(self):
        self.et_path = self.root/f'step{self.step}.et.json'
        self.chrome_path = self.root/f'step{self.step}.chrome.json'
        observer = torch.profiler.ExecutionTraceObserver().register_callback(str(self.et_path))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(activities=activities, record_shapes=True,
                                             profile_memory=True, execution_trace_observer=observer)
        try:
            self.profiler.__enter__()
        except BaseException:
            observer.cleanup()
            raise
        return self

    def __exit__(self, *exc):
        self.profiler.__exit__(*exc)
        self.profiler.export_chrome_trace(str(self.chrome_path))
        trace = json.loads(self.et_path.read_text())
        nodes, storage = project_trace(trace, rank=self.rank, step=self.step, device=self.device,
                                      offset=len(self.nodes), provenance_key=self.provenance_key)
        self.nodes.extend(nodes)
        self.storage.extend(storage)
        for path in (self.et_path, self.chrome_path):
            self.artifacts.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def finish(self, *, hardware, manifest=None):
        details = dict(manifest or {})
        details.update(ranks=[self.rank], world_size=self.world_size, provenance=self.provenance,
                       native_trace_artifacts=self.artifacts, model_state=self.state_observer.state,
                       state_storage_metadata=self.state_observer.storage_metadata,
                       observed_state_storage=[dict(id=s.id, size_bytes=s.size, category=s.category)
                                               for s in self.state_observer.storage],
                       observer='pytorch_execution_trace_and_profiler',
                       unresolved=sorted(self.gaps | set(details.get('unresolved', []))))
        return Workload(self.nodes, self.storage, details, hardware)
