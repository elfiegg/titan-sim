"""Observe one native CUDA process without changing its recipe or allocator policy.

Allocator counters and sampled device totals are separate evidence domains. This
module deliberately cannot produce a simulator memory-fit verdict.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import time


_SCHEMA = 'titan_sim.memory_observation.v1'
_RESET_OBSERVERS = set()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _signature(function):
    if not callable(function):
        return None
    try:
        return str(inspect.signature(function))
    except (TypeError, ValueError):
        return 'signature unavailable'


class MemoryObserver:
    """Synchronized boundary samples plus allocator high-water marks.

    The caller supplies physical placement and immutable recipe/runtime provenance.
    CUDA peak counters are reset on entry and at each non-nested phase, so use a
    dedicated observation run. No collectives or background sampling are added.
    Failures propagate; evidence from a failed body remains available via report().
    """
    def __init__(self, *, device=None, rank=0, world_size=1, provenance=None,
                 physical_device=None, snapshot=False, observe_peak_resets=False,
                 torch_module=None):
        if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
            raise ValueError('rank and world_size must describe a valid process rank')
        if physical_device is not None and (not isinstance(physical_device, str) or not physical_device):
            raise ValueError('physical_device must be a nonempty globally unique device identifier')
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError('provenance must be a JSON object')
        if type(snapshot) is not bool:
            raise ValueError('snapshot must be a boolean')
        if type(observe_peak_resets) is not bool:
            raise ValueError('observe_peak_resets must be a boolean')
        if torch_module is None:
            import torch as torch_module
        self.torch, self.cuda = torch_module, torch_module.cuda
        if not self.cuda.is_available():
            raise ValueError('MemoryObserver requires a real available CUDA device')
        self.device = torch_module.device('cuda', self.cuda.current_device()) if device is None else torch_module.device(device)
        if self.device.type != 'cuda':
            raise ValueError('MemoryObserver only observes CUDA devices')
        if self.device.index is None:
            self.device = torch_module.device('cuda', self.cuda.current_device())
        self.rank, self.world_size = rank, world_size
        self.provenance = deepcopy(provenance or {})
        self.provenance_sha256 = _digest(self.provenance)
        self.physical_device, self.want_snapshot = physical_device, snapshot
        self.samples, self.phases, self.errors = [], [], []
        self.observe_peak_resets = observe_peak_resets
        self.peak_counter_resets, self._reset_patches = [], []
        self._inside_reset_hook = False
        self._original_reset = None
        self.active, self.finished, self.current_phase = False, False, None
        self.status, self.started = 'not_started', None
        memory = getattr(self.cuda, 'memory', self.cuda)
        self.snapshot_function = getattr(self.cuda, 'memory_snapshot', None)
        self.capabilities = {
            'memory_snapshot': _signature(self.snapshot_function),
            'record_memory_history': _signature(getattr(memory, 'record_memory_history', None)),
            'private_record_memory_history_available': callable(getattr(memory, '_record_memory_history', None)),
            'history_enabled': False,
            'history_reason': 'History is not enabled: no stable public history ownership/restoration contract is assumed.',
        }
        backend = getattr(memory, 'get_allocator_backend', None)
        self.runtime = {'torch_version': str(torch_module.__version__),
                        'cuda_version': torch_module.version.cuda,
                        'allocator_backend': backend() if callable(backend) else 'unknown'}

    def __enter__(self):
        if self.active or self.finished:
            raise ValueError('MemoryObserver instances are single-use')
        if id(self.cuda) in _RESET_OBSERVERS:
            raise ValueError('another MemoryObserver owns this process CUDA reset API')
        self._original_reset = self.cuda.reset_peak_memory_stats
        try:
            if self.observe_peak_resets:
                self._install_reset_observation()
            self.cuda.synchronize(self.device)
            self._reset_owned_counters()
            self.started, self.active, self.status = time.monotonic(), True, 'running'
            self.sample('baseline')
        except BaseException as error:
            self.status, self.active, self.finished = 'failed', False, True
            self.errors.append('baseline sample: '+type(error).__name__+': '+str(error))
            self._restore_reset_observation()
            raise
        return self

    def _install_reset_observation(self):
        _RESET_OBSERVERS.add(id(self.cuda))
        owners = [self.cuda]
        memory = getattr(self.cuda, 'memory', None)
        if memory is not None and memory is not self.cuda:
            owners.append(memory)
        for owner in owners:
            original = getattr(owner, 'reset_peak_memory_stats', None)
            if not callable(original):
                continue

            def make_wrapper(original):
                def observed_reset(*args, **kwargs):
                    if self._inside_reset_hook:
                        return original(*args, **kwargs)
                    self._inside_reset_hook = True
                    try:
                        if self.active:
                            try:
                                requested = args[0] if args else kwargs.get('device')
                                if requested is None:
                                    index = self.cuda.current_device()
                                elif type(requested) is int:
                                    index = requested
                                else:
                                    target = self.torch.device(requested)
                                    index = (self.cuda.current_device() if target.index is None
                                             else target.index) if target.type == 'cuda' else None
                                if index == self.device.index:
                                    self.peak_counter_resets.append({
                                        'phase': self.current_phase,
                                        'elapsed_seconds': time.monotonic() - self.started,
                                        'peak_allocated_bytes': int(self.cuda.max_memory_allocated(self.device)),
                                        'peak_reserved_bytes': int(self.cuda.max_memory_reserved(self.device))})
                            except Exception as error:
                                self.errors.append('pre-reset counter observation: '+type(error).__name__+': '+str(error))
                        return original(*args, **kwargs)
                    finally:
                        self._inside_reset_hook = False

                return observed_reset

            observed_reset = make_wrapper(original)
            # Keep aliases separate: each wrapper delegates the exact callable
            # originally exposed on its namespace, with unchanged arguments.
            existed = 'reset_peak_memory_stats' in vars(owner)
            self._reset_patches.append((owner, original, observed_reset, existed))
            setattr(owner, 'reset_peak_memory_stats', observed_reset)

    def _reset_owned_counters(self):
        previous = self._inside_reset_hook
        self._inside_reset_hook = True
        try:
            return self._original_reset(self.device)
        finally:
            self._inside_reset_hook = previous

    def _restore_reset_observation(self):
        for owner, original, wrapper, existed in reversed(self._reset_patches):
            try:
                if getattr(owner, 'reset_peak_memory_stats') is not wrapper:
                    self.errors.append('reset API changed during observation; foreign replacement was not overwritten')
                elif existed:
                    setattr(owner, 'reset_peak_memory_stats', original)
                else:
                    delattr(owner, 'reset_peak_memory_stats')
            except Exception as error:
                self.errors.append('restore reset API: '+type(error).__name__+': '+str(error))
        self._reset_patches.clear()
        if self.observe_peak_resets:
            _RESET_OBSERVERS.discard(id(self.cuda))

    def set_provenance(self, provenance):
        """Bind resolved recipe/source identities after native construction."""
        if not isinstance(provenance, dict):
            raise ValueError('provenance must be a JSON object')
        copied = deepcopy(provenance)
        digest = _digest(copied)
        self.provenance, self.provenance_sha256 = copied, digest

    def __exit__(self, error_type, error, traceback):
        self.status = 'failed' if error is not None else 'complete'
        try:
            self.sample('final')
        except Exception as sample_error:
            self.status = 'failed'
            self.errors.append('final sample: '+type(sample_error).__name__+': '+str(sample_error))
            if error is None:
                raise
        finally:
            self.active, self.finished = False, True
            self._restore_reset_observation()
        if error is not None:
            self.errors.append(type(error).__name__+': '+str(error))
        return False

    def sample(self, label, *, engine=None):
        """Capture allocated, reserved and free/total; never add the first two."""
        if not self.active:
            raise ValueError('sample requires an active observation context')
        if not isinstance(label, str) or not label:
            raise ValueError('sample label must be a nonempty string')
        self.cuda.synchronize(self.device)
        allocated = int(self.cuda.memory_allocated(self.device))
        reserved = int(self.cuda.memory_reserved(self.device))
        free, total = map(int, self.cuda.mem_get_info(self.device))
        record = {'label': label, 'phase': self.current_phase,
                  'elapsed_seconds': time.monotonic() - self.started,
                  'process_allocator': {'allocated_bytes': allocated, 'reserved_bytes': reserved,
                      'slack_bytes': reserved - allocated,
                      'peak_allocated_bytes': int(self.cuda.max_memory_allocated(self.device)),
                      'peak_reserved_bytes': int(self.cuda.max_memory_reserved(self.device))},
                  'device_global_sample': {'free_bytes': free, 'total_bytes': total,
                      'used_bytes': total - free,
                      'used_minus_process_reserved_bytes': total - free - reserved}}
        if engine is not None:
            record['state_inventory'] = self._state_inventory(engine)
        if self.want_snapshot:
            record['allocator_snapshot'] = self._snapshot()
        self.samples.append(record)
        return deepcopy(record)

    @contextmanager
    def phase(self, name, *, engine=None):
        """Measure one caller-defined native phase, retaining its allocator peak."""
        if not self.active or self.current_phase is not None:
            raise ValueError('phases require an active context and cannot nest')
        if not isinstance(name, str) or not name:
            raise ValueError('phase name must be a nonempty string')
        self.sample(name+':before', engine=engine)
        self._reset_owned_counters()
        self.current_phase = name
        first = len(self.samples)
        first_reset = len(self.peak_counter_resets)
        status = 'complete'
        try:
            yield self
        except BaseException:
            status = 'failed'
            raise
        finally:
            try:
                end = self.sample(name+':after', engine=engine)
                counters = [end['process_allocator']] + self.peak_counter_resets[first_reset:]
                self.phases.append({'name': name, 'status': status, 'first_sample': first,
                                    'last_sample': len(self.samples)-1,
                                    'observed_native_reset_count': len(self.peak_counter_resets)-first_reset,
                                    'process_peak_allocated_bytes': max(c['peak_allocated_bytes'] for c in counters),
                                    'process_peak_reserved_bytes': max(c['peak_reserved_bytes'] for c in counters)})
            except Exception as error:
                self.errors.append('phase final sample: '+type(error).__name__+': '+str(error))
                if status == 'complete':
                    raise
            finally:
                self.current_phase = None

    def _snapshot(self):
        if not callable(self.snapshot_function):
            return {'status': 'unsupported', 'reason': 'public memory_snapshot is unavailable'}
        try:
            # Public API returns segments in older runtimes; newer versions may
            # return a mapping containing segments and optional device traces.
            raw = self.snapshot_function()
            segments = raw.get('segments', []) if isinstance(raw, dict) else raw
            selected = [s for s in segments if s.get('device') == self.device.index]
            states = {}
            for segment in selected:
                for block in segment.get('blocks', []):
                    state = block.get('state', 'unknown')
                    states[state] = states.get(state, 0) + int(block.get('size', 0))
            return {'status': 'observed', 'scope': 'process allocator at sample boundary',
                    'segment_count': len(selected),
                    'segment_bytes': sum(int(s['total_size']) for s in selected),
                    'block_state_bytes': states, 'history_enabled': False}
        except Exception as error:
            return {'status': 'unavailable', 'reason': type(error).__name__+': '+str(error)}

    def _state_inventory(self, engine):
        """Deduplicate live local backing storage, including nested optimizer state."""
        categories = dict.fromkeys(('parameters', 'gradients', 'buffers', 'optimizer_state'), 0)
        seen, issues, excluded = set(), [], 0

        def visit(value, category):
            nonlocal excluded
            if isinstance(value, dict):
                for item in value.values():
                    visit(item, category)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, category)
            elif self.torch.is_tensor(value):
                try:
                    local = value.to_local() if callable(getattr(value, 'to_local', None)) else value
                    if local.device != self.device:
                        excluded += 1
                        return
                    storage = local.untyped_storage()
                    # The pointer is used only for in-sample alias identity and
                    # never serialized or persisted across allocator reuse.
                    key = (str(local.device), int(storage.data_ptr()), int(storage.nbytes()))
                    if key not in seen:
                        seen.add(key)
                        categories[category] += int(storage.nbytes())
                except Exception as error:
                    issues.append(category+': '+type(error).__name__+': '+str(error))

        models = getattr(engine, 'model_parts', None)
        if models is None:
            issues.append('engine.model_parts unavailable')
        for model in models or ():
            for parameter in model.parameters():
                visit(parameter, 'parameters')
                visit(parameter.grad, 'gradients')
            for buffer in model.buffers():
                visit(buffer, 'buffers')
        optim = getattr(engine, 'optim', None)
        optimizers = getattr(optim, 'optimizers', None)
        if optimizers is None:
            issues.append('engine.optim.optimizers unavailable')
        for optimizer in optimizers or ():
            if hasattr(optimizer, 'state'):
                visit(optimizer.state, 'optimizer_state')
            else:
                issues.append('optimizer.state unavailable')
        return {'category_bytes': categories, 'unique_storage_bytes': sum(categories.values()),
                'other_device_tensor_references': excluded, 'unresolved': issues,
                'scope': 'live visible state only; aliases assigned to first category; not additive to allocator totals'}

    def report(self):
        allocator = [sample['process_allocator'] for sample in self.samples] + self.peak_counter_resets
        global_samples = [sample['device_global_sample'] for sample in self.samples]
        peak_allocated = max((s['peak_allocated_bytes'] for s in allocator), default=0)
        peak_reserved = max((s['peak_reserved_bytes'] for s in allocator), default=0)
        phases = {p['name'] for p in self.phases if p['status'] == 'complete'}
        return deepcopy({'schema': _SCHEMA, 'status': self.status, 'rank': self.rank,
            'world_size': self.world_size, 'logical_device': str(self.device),
            'physical_device': self.physical_device, 'provenance': self.provenance,
            'provenance_sha256': self.provenance_sha256, 'runtime': self.runtime,
            'capabilities': self.capabilities, 'samples': self.samples, 'phases': self.phases,
            'peak_reset_observation': {'enabled': self.observe_peak_resets,
                'scope': 'calls through wrapped public CUDA reset API aliases only; cached references and private/C++ resets may bypass observation',
                'pre_reset_counters': self.peak_counter_resets},
            'process_allocator_peak': {'allocated_bytes': peak_allocated, 'reserved_bytes': peak_reserved},
            'device_sampled_used_peak_bytes': max((s['used_bytes'] for s in global_samples), default=0),
            'observed_configuration_bound': {'scope': 'this process and exact observed execution window only',
                'process_allocator_peak_lower_bytes': peak_reserved,
                'physical_peak_upper_bytes': None, 'supports_unseen_configuration_fit': False},
            'lifecycle': {name: 'observed_phase' if name in phases else 'unknown'
                          for name in ('initialization', 'checkpoint_loading')},
            'unresolved': ['Global device usage is sampled, not a continuous peak.',
                'External/library allocations and other processes are not attributed; global used minus reserved is not an external-allocation estimate.',
                'Initialization/checkpoint phases are covered only when the caller brackets their full execution.',
                ('Unobserved cached/private/concurrent peak-counter resets can truncate peaks.' if self.observe_peak_resets else
                 'Allocator peak-counter resets or concurrent observers inside the window invalidate counter ownership.')] + self.errors})


def merge_observations(reports, *, expected_ranks=None):
    """Audit rank completeness and placement; never sum asynchronous rank peaks."""
    reports = list(reports)
    if not reports:
        raise ValueError('at least one observation is required')
    first = reports[0]
    world = first['world_size']
    expected = list(range(world)) if expected_ranks is None else list(expected_ranks)
    if (any(type(rank) is not int or not 0 <= rank < world for rank in expected)
            or len(set(expected)) != len(expected) or not expected):
        raise ValueError('expected_ranks must be distinct ranks in the observed world')
    by_rank, devices, unresolved = {}, {}, []
    for report in reports:
        if report.get('schema') != _SCHEMA or report['world_size'] != world:
            raise ValueError('observations must have the same schema and world size')
        if report['provenance_sha256'] != _digest(report['provenance']):
            raise ValueError('observation provenance digest mismatch')
    config_comparison = None
    if any(report.get('provenance_sha256') != first['provenance_sha256'] for report in reports):
        def without_config(provenance):
            return {key: value for key, value in provenance.items()
                    if key not in ('recipe_config', 'resolved_config_sha256')}
        if all(without_config(report['provenance']) == without_config(first['provenance']) for report in reports):
            from .prepare import _function_repr_config_match
            config_comparison = _function_repr_config_match([
                {'ranks': [report['rank']], 'provenance': report['provenance']} for report in reports])
        if config_comparison is None:
            raise ValueError('cannot merge different recipe/runtime observations')
        unresolved.append('Recipe configurations differ only in function repr addresses; callable identity remains unverified.')
    for report in reports:
        if ((report['provenance_sha256'] != first['provenance_sha256'] and config_comparison is None)
                or report['runtime'] != first['runtime']):
            raise ValueError('cannot merge different recipe/runtime observations')
        rank = report['rank']
        if type(rank) is not int or rank not in expected or rank in by_rank:
            raise ValueError('unexpected or duplicate rank observation')
        by_rank[rank] = report
        if report['status'] != 'complete':
            unresolved.append('rank '+str(rank)+' observation did not complete')
        device = report['physical_device']
        if device is None:
            unresolved.append('rank '+str(rank)+' physical placement is unknown')
            device = 'unplaced-rank:'+str(rank)
        entry = devices.setdefault(device, {'ranks': [], 'observed_process_peak_lower_bytes': 0,
                                            'sampled_global_used_peak_bytes': 0,
                                            'physical_peak_upper_bytes': None})
        entry['ranks'].append(rank)
        entry['observed_process_peak_lower_bytes'] = max(entry['observed_process_peak_lower_bytes'],
                                                        report['process_allocator_peak']['reserved_bytes'])
        entry['sampled_global_used_peak_bytes'] = max(entry['sampled_global_used_peak_bytes'],
                                                     report['device_sampled_used_peak_bytes'])
    missing = sorted(set(expected) - by_rank.keys())
    if missing:
        unresolved.append('missing ranks: '+', '.join(map(str, missing)))
    if len(expected) != world:
        unresolved.append('expected_ranks covers only part of the declared world')
    return {'schema': _SCHEMA+'.aggregate', 'status': 'incomplete' if unresolved else 'complete',
            'expected_ranks': expected, 'observed_ranks': sorted(by_rank), 'missing_ranks': missing,
            'provenance_sha256': first['provenance_sha256'] if config_comparison is None else None,
            'rank_provenance_sha256': {str(rank): report['provenance_sha256'] for rank, report in by_rank.items()},
            'recipe_config_comparison': config_comparison, 'per_device': devices,
            'memory_verdict': 'inconclusive', 'supports_unseen_configuration_fit': False,
            'unresolved': unresolved + ['Per-process peaks are not concurrent device-wide peaks; collocated rank peaks are maxed, never summed.',
                                      'Physical upper bounds remain unknown; these observations do not close simulation coverage.']}
