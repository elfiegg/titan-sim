"""Read original native state and CUDA allocator history without tensor retention.

The optional history recorder is for a dedicated observation process only. It is
left enabled: there is no reliable API for restoring somebody else's recorder.
The caller owns process lifetime. No synchronization, allocator reset, compiler
setting, optimizer initialization or tensor allocation is introduced here.
"""
from contextlib import contextmanager
from copy import deepcopy
import inspect

from .identity import digest

SCHEMA = 'titan_sim.native_allocation_capture.v1'


def inspect_state(*, models=(), optimizers=(), torch_module=None):
    """Inventory actual backing stores, retaining addresses only in local evidence.

    A backing store may have many aliases/categories. CPU state is recorded too,
    but is never charged to CUDA. No state is manufactured for lazy optimizers.
    """
    if torch_module is None:
        import torch as torch_module
    rows, identities, issues, containers = [], {}, [], set()

    def visit(value, category, path):
        if torch_module.is_tensor(value):
            try:
                local = value.to_local() if callable(getattr(value, 'to_local', None)) else value
                if local.device.type == 'meta':
                    raise ValueError('meta tensor has no physical backing allocation')
                storage = local.untyped_storage()
                identity = (str(local.device), int(storage._cdata))
                if identity not in identities:
                    entry = dict(id='state:' + str(len(rows)), device=str(local.device),
                                 address=int(storage.data_ptr()), bytes=int(storage.nbytes()),
                                 aliases=[], categories=[])
                    rows.append(entry)
                    identities[identity] = entry
                entry = identities[identity]
                entry['aliases'].append(path)
                if category not in entry['categories']:
                    entry['categories'].append(category)
            except Exception as error:
                issues.append(category + ': ' + type(error).__name__ + ': ' + str(error))
        elif isinstance(value, (dict, list, tuple)):
            if id(value) in containers:
                issues.append('cyclic state container: ' + '/'.join(path))
                return
            containers.add(id(value))
            children = value.values() if isinstance(value, dict) else value
            for index, child in enumerate(children):
                visit(child, category, path + [str(index)])
            containers.remove(id(value))
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            issues.append(category + ': unsupported native state object ' + type(value).__name__)

    for index, model in enumerate(models):
        for name, parameter in model.named_parameters():
            visit(parameter, 'parameters', ['model', str(index), name])
            visit(parameter.grad, 'gradients', ['model', str(index), name, 'grad'])
        for name, buffer in model.named_buffers():
            visit(buffer, 'buffers', ['model', str(index), name])
    for index, optimizer in enumerate(optimizers):
        if not hasattr(optimizer, 'state'):
            issues.append('optimizer.state unavailable: ' + str(index))
        else:
            visit(optimizer.state, 'optimizer_state', ['optimizer', str(index)])
    return dict(allocations=rows, unresolved=issues,
                unique_storage_bytes=sum(row['bytes'] for row in rows),
                scope='exact byte facts at this sample; persistence is not established')


class NativeAllocationCapture:
    """Capture actual snapshots at caller-selected native boundaries.

    ``region(node_id)`` wraps the original call and records the corresponding
    workload node, without claiming that host return means CUDA completion.
    Only allocator ``free_completed`` events establish actual reuse eligibility.
    An imported snapshot history with unknown recorder settings remains partial.
    """
    def __init__(self, *, device_index, physical_device, rank=0, provenance=None,
                 torch_module=None):
        if type(device_index) is not int or device_index < 0 or type(rank) is not int or rank < 0:
            raise ValueError('device_index and rank must be nonnegative integers')
        if not isinstance(physical_device, str) or not physical_device:
            raise ValueError('physical_device must be explicit')
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device_index, self.physical_device, self.rank = device_index, physical_device, rank
        self.provenance = deepcopy(provenance or {})
        self.samples, self.issues = [], []
        self.history = dict(started_by_capture=False, max_entries=None,
                            reason='existing global history settings are unknown')
        self._active_region = False

    def start_history(self, *, max_entries, dedicated_process=False):
        """Explicitly enable all allocator events in an isolated observation process.

        Known existing history is rejected. ``dedicated_process`` attests a
        fresh process because native APIs cannot query all recorder settings.
        Call before any sample. The training call itself remains untouched.
        """
        if dedicated_process is not True:
            raise ValueError('history ownership requires a dedicated observation process')
        if type(max_entries) is not int or max_entries < 2:
            raise ValueError('max_entries must be an integer of at least two')
        if self.samples or self.history['started_by_capture']:
            raise ValueError('start history exactly once, before sampling')
        function = getattr(self.torch.cuda.memory, '_record_memory_history', None)
        if not callable(function):
            raise RuntimeError('native allocator history API unavailable')
        snapshot = getattr(self.torch.cuda.memory, '_snapshot', None)
        if not callable(snapshot):
            raise RuntimeError('native allocator snapshot API unavailable')
        prior = snapshot()
        if not isinstance(prior, dict) or 'device_traces' not in prior:
            raise RuntimeError('cannot inspect existing allocator history')
        if any(prior['device_traces']):
            raise RuntimeError('existing allocator history detected; use a fresh process')
        kwargs = dict(enabled='all', context=None, stacks='python', max_entries=max_entries)
        # Older native APIs have no skip_actions; their default records all
        # actions. We never infer support for a new argument from **kwargs.
        parameters = inspect.signature(function).parameters
        if 'skip_actions' in parameters:
            kwargs['skip_actions'] = []
        function(**kwargs)
        self.history = dict(started_by_capture=True, max_entries=max_entries,
                            arguments=kwargs, restoration='left enabled until dedicated process exits')

    @contextmanager
    def record_history(self, *, max_entries, dedicated_process=False):
        """Record explicit entry/exit snapshots; never synchronize native work."""
        self.start_history(max_entries=max_entries, dedicated_process=dedicated_process)
        self.sample('history:entry')
        try:
            yield self
        finally:
            self.sample('history:exit')

    def sample(self, label, *, models=(), optimizers=(), node_id=None, edge=None, phase='schedule'):
        if not isinstance(label, str) or not label or not isinstance(phase, str) or not phase:
            raise ValueError('sample label and phase must be nonempty')
        if node_id is not None and (type(node_id) is not int or node_id < 0):
            raise ValueError('node_id must be a nonnegative integer or null')
        if edge not in (None, 'before', 'after'):
            raise ValueError('edge must be before, after or null')
        state = inspect_state(models=models, optimizers=optimizers, torch_module=self.torch)
        function = getattr(self.torch.cuda.memory, '_snapshot', None)
        raw, issue = None, None
        try:
            if not callable(function):
                raise RuntimeError('native _snapshot API unavailable')
            raw = function()
            if not isinstance(raw, dict) or not isinstance(raw.get('segments'), list):
                raise ValueError('unsupported native allocator snapshot format')
        except Exception as error:
            issue = type(error).__name__ + ': ' + str(error)
            self.issues.append(issue)
        sample = dict(label=label, node_id=node_id, edge=edge, phase=phase,
                      state=state, snapshot=raw, snapshot_error=issue)
        self.samples.append(deepcopy(sample))
        return sample

    @contextmanager
    def region(self, node_id, *, models=(), optimizers=(), phase='schedule'):
        if self._active_region:
            raise ValueError('nested allocation capture regions are ambiguous')
        self._active_region = True
        try:
            self.sample(str(node_id) + ':before', node_id=node_id, edge='before',
                        models=models, optimizers=optimizers, phase=phase)
            try:
                yield
            except BaseException:
                self.issues.append('native region failed: ' + str(node_id))
                raise
            finally:
                self.sample(str(node_id) + ':after', node_id=node_id, edge='after',
                            models=models, optimizers=optimizers, phase=phase)
        finally:
            self._active_region = False

    def report(self):
        result = deepcopy(dict(schema=SCHEMA, rank=self.rank, device_index=self.device_index,
            physical_device=self.physical_device, provenance=self.provenance,
            runtime=dict(torch=str(self.torch.__version__), cuda=getattr(self.torch.version, 'cuda', None)),
            history=self.history, samples=self.samples, unresolved=self.issues))
        result['sha256'] = digest(result)
        return result
