"""Native Kineto capture with explicitly optional execution-trace observation.

ET mode retains leaf operations and tensor references. Chrome-only mode avoids
ExecutionTraceObserver and binds individual GPU activities to retained evidence,
without claiming reusable operator signatures or tensor storage coverage. Both
modes declare incomplete stream, communication and allocation semantics.
"""
from contextlib import contextmanager
import hashlib
import json
import math
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


def project_chrome_trace(trace, *, rank, step, offset, artifact_sha256,
                         provenance_key, hardware_id=None, stream_tails=None):
    """Bind GPU occurrences without inferring missing ET shapes or identities.

    Exact keys include artifact/index, so a symbol cannot silently reuse a cost
    across different captured occurrences. This is observed timing, not an
    executable-region implementation match or an allocation reconstruction.
    """
    from .native_runner import GPU_CATEGORIES, native_cost_key
    tails = {} if stream_tails is None else stream_tails
    events = []
    for index, event in enumerate(trace.get('traceEvents', [])):
        if event.get('ph') != 'X' or event.get('cat') not in GPU_CATEGORIES:
            continue
        if any(isinstance(event.get(k), bool) or not isinstance(event.get(k), (int, float))
               or not math.isfinite(event[k]) or event[k] < 0 for k in ('ts', 'dur')):
            raise ValueError('invalid Chrome GPU event timestamp/duration')
        events.append((event['ts'], index, event))
    nodes = []
    for _, index, event in sorted(events, key=lambda item: (item[0], item[1])):
        args = event.get('args', {})
        implementation = dict(name=event['name'], category=event['cat'],
            launch={k: args[k] for k in ('grid', 'block', 'registers per thread', 'shared memory', 'bytes') if k in args})
        context = dict(hardware_id=hardware_id, occurrence_only=True,
                       artifact_sha256=artifact_sha256, event_index=index,
                       rank=rank, step=step, provenance_sha256=provenance_key)
        binding = dict(kind='chrome_gpu_event', artifact_sha256=artifact_sha256,
                       event_index=index, event_name=event['name'], event_category=event['cat'],
                       implementation=implementation, context=context)
        stream = args.get('stream')
        stream_key = (rank, str(args.get('device', event.get('pid'))), str(stream))
        dependencies = [tails[stream_key]] if stream is not None and stream_key in tails else []
        node = Node(offset+len(nodes), event['name'], rank, deps=dependencies,
            attrs=dict(operator=event['name'], step=step, stream=stream,
                       resources=[f'gpu:{rank}:stream:{stream}'] if stream is not None else [f'gpu:{rank}'],
                       cost_key=native_cost_key(implementation, context), native_measurement=binding,
                       measurement_scope='occurrence_only'))
        nodes.append(node)
        if stream is not None:
            tails[stream_key] = node.id
    return nodes


class ProfilerCapture:
    """One native profiler session per captured step; no Python dispatch mode."""

    def __init__(self, *, trace_dir, rank=0, world_size=1, device='cpu', provenance=None, execution_trace=True):
        if type(execution_trace) is not bool:
            raise ValueError('execution_trace must be a boolean')
        self.execution_trace = execution_trace
        self.stream_tails = {}
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
        if not execution_trace:
            self.gaps = {
                'Execution trace disabled explicitly; GPU costs are occurrence-only, without reusable operator/shape/scalar matching.',
                'Chrome-only capture does not reconstruct backing allocations, tensor lifetimes, aliases or physical memory coverage.',
                'Cross-stream dependencies, CUDA event waits and stream contention require explicit evidence.',
                'Profiler instrumentation and shape recording can perturb timing and tensor lifetimes.',
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
        observer = None
        if self.execution_trace:
            observer = torch.profiler.ExecutionTraceObserver().register_callback(str(self.et_path))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        options = dict(activities=activities, record_shapes=True, profile_memory=True)
        if observer is not None:
            options['execution_trace_observer'] = observer
        self.profiler = torch.profiler.profile(**options)
        try:
            self.profiler.__enter__()
        except BaseException:
            if observer is not None:
                observer.cleanup()
            raise
        return self

    def __exit__(self, *exc):
        self.profiler.__exit__(*exc)
        self.profiler.export_chrome_trace(str(self.chrome_path))
        if self.execution_trace:
            trace = json.loads(self.et_path.read_text())
            nodes, storage = project_trace(trace, rank=self.rank, step=self.step, device=self.device,
                                          offset=len(self.nodes), provenance_key=self.provenance_key)
            self.nodes.extend(nodes)
            self.storage.extend(storage)
            paths = (self.et_path, self.chrome_path)
        else:
            raw = self.chrome_path.read_bytes()
            nodes = project_chrome_trace(json.loads(raw), rank=self.rank, step=self.step,
                offset=len(self.nodes), artifact_sha256=hashlib.sha256(raw).hexdigest(),
                provenance_key=self.provenance_key, stream_tails=self.stream_tails)
            self.nodes.extend(nodes)
            if any(n.attrs['stream'] is None for n in nodes):
                self.gaps.add('GPU activities without stream identity cannot establish execution dependencies.')
            paths = (self.chrome_path,)
        for path in paths:
            self.artifacts.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       step=self.step))

    def finish(self, *, hardware, manifest=None):
        if not self.execution_trace:
            from .native_runner import native_cost_key
            for node in self.nodes:
                binding = node.attrs['native_measurement']
                binding['context']['hardware_id'] = hardware['id']
                node.attrs['cost_key'] = native_cost_key(binding['implementation'], binding['context'])
        details = dict(manifest or {})
        details['memory'] = dict(details.get('memory', {}), storage_evidence='unverified')
        details.update(ranks=[self.rank], world_size=self.world_size, provenance=self.provenance,
                       native_trace_artifacts=self.artifacts, model_state=self.state_observer.state,
                       state_storage_metadata=self.state_observer.storage_metadata,
                       observed_state_storage=[dict(id=s.id, size_bytes=s.size, category=s.category)
                                               for s in self.state_observer.storage],
                       execution_trace=self.execution_trace,
                       observer='pytorch_execution_trace_and_profiler' if self.execution_trace else 'pytorch_profiler_chrome_only',
                       unresolved=sorted(self.gaps | set(details.get('unresolved', []))))
        return Workload(self.nodes, self.storage, details, hardware)
