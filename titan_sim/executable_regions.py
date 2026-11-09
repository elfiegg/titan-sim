"""Live native callable retention; serialization never invents executable handles."""
from contextlib import contextmanager
from copy import deepcopy
import functools
import hashlib
import inspect
import uuid

from .core import Node
from .native_runner import native_cost_key
from .workload import Workload

_REGISTRIES = {}


def _unsupported(contract):
    if isinstance(contract, dict):
        return contract.get('replay_unsupported', False) or any(_unsupported(v) for v in contract.values())
    return isinstance(contract, list) and any(_unsupported(v) for v in contract)


def _tensor_versions(value):
    import torch
    if isinstance(value, torch.Tensor):
        try:
            return value._version
        except RuntimeError:
            return None  # Inference tensors have no usable mutation counter.
    if isinstance(value, (tuple, list)):
        return [_tensor_versions(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _tensor_versions(v) for k, v in value.items()}
    return None


def _dispatch_context():
    import torch
    keys = (torch._C._dispatch_tls_local_include_set(), torch._C._dispatch_tls_local_exclude_set())
    return dict(include=str(keys[0]), exclude=str(keys[1])), keys


class ArgumentRetention:
    """Bound optional live storage retention by exact callable names and call count."""
    def __init__(self, enabled=False, names=None, max_occurrences=None):
        if names is not None and (not isinstance(names, list) or not all(isinstance(n, str) for n in names)):
            raise ValueError('retain_native_names must be a list of exact callable names')
        if max_occurrences is not None and (type(max_occurrences) is not int or max_occurrences < 1):
            raise ValueError('retain_native_max_occurrences must be positive')
        self.enabled, self.names, self.limit, self.counts = enabled, names, max_occurrences, {}

    def __call__(self, name):
        count = self.counts.get(name, 0)
        self.counts[name] = count + 1
        return bool(self.enabled and (self.names is None or name in self.names) and
                    (self.limit is None or count < self.limit))


def _contract(value, aliases=None):
    """Argument shape/layout and storage alias contract without tensor serialization."""
    aliases = {} if aliases is None else aliases
    import torch
    if isinstance(value, torch.Tensor):
        result = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                      layout=str(value.layout), requires_grad=value.requires_grad)
        if value.layout == torch.strided:
            storage = value.untyped_storage()
            key = (str(value.device), storage._cdata)
            result.update(stride=list(value.stride()), storage_offset=value.storage_offset(),
                          alias=aliases.setdefault(key, len(aliases)), storage_bytes=storage.nbytes())
        return result
    if isinstance(value, (tuple, list)):
        return [_contract(item, aliases) for item in value]
    if isinstance(value, dict):
        return {str(k): _contract(v, aliases) for k, v in value.items()}
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return dict(immutable_type=type(value).__name__, value=str(value))
    if value is None or type(value) in (str, int, bool, float):
        return value
    return dict(opaque_type=type(value).__module__+'.'+type(value).__qualname__, replay_unsupported=True)


def _runtime_identity():
    import torch
    from .operator_cases import execution_context
    return dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                execution_context=execution_context(torch),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                grad_enabled=torch.is_grad_enabled(),
                inference_mode=torch.is_inference_mode_enabled(),
                autocast_cpu=torch.is_autocast_enabled('cpu'),
                autocast_cpu_dtype=str(torch.get_autocast_dtype('cpu')),
                autocast_cuda=torch.is_autocast_enabled('cuda'),
                autocast_cuda_dtype=str(torch.get_autocast_dtype('cuda')))


def _active(recorder):
    if recorder is None:
        return False
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    return recorder in _get_current_dispatch_mode_stack()


class LiveRegionRegistry:
    """Retain original callable objects and arguments in this process only.

    Registration does not establish safe state replay. A caller must provide
    an explicit state contract and reset callback (or declare read-only state).
    """
    def __init__(self):
        self.token = uuid.uuid4().hex
        self.entries = []
        _REGISTRIES[self.token] = self

    def close(self):
        _REGISTRIES.pop(self.token, None)
        self.entries.clear()

    def register(self, run, *, args=(), kwargs=None, implementation, context,
                 source_workload_sha256, source_node_ids, state_contract,
                 before_sample=None, replay_safe=False):
        if not callable(run) or not implementation or not context:
            raise ValueError('native callable and nonempty implementation/context required')
        if not isinstance(source_workload_sha256, str) or len(source_workload_sha256) != 64 or not source_node_ids:
            raise ValueError('source workload fingerprint and occurrence IDs required')
        if not isinstance(state_contract, dict) or not state_contract:
            raise ValueError('explicit state contract required')
        if replay_safe and before_sample is None and state_contract.get('mutation') != 'read_only':
            raise ValueError('mutable replay requires a before_sample reset callback')
        if before_sample is not None and not callable(before_sample):
            raise ValueError('before_sample must be callable')
        if not isinstance(context.get('hardware_id'), str) or not context['hardware_id']:
            raise ValueError('context.hardware_id is required for native callable replay')
        kwargs = {} if kwargs is None else kwargs
        contract = _contract(dict(args=args, kwargs=kwargs))
        if replay_safe and _unsupported(contract):
            raise ValueError('opaque native arguments have no supported replay contract')
        identity = deepcopy(implementation)
        identity.update(handle_scope='live_process', registry_token=self.token,
                        handle_index=len(self.entries), argument_contract=contract,
                        state_contract=deepcopy(state_contract))
        ctx = deepcopy(context)
        ctx['runtime_identity'] = _runtime_identity()
        identity['identity_basis'] = 'original_live_handle_and_trusted_caller_descriptor'
        ctx.update(source_workload_sha256=source_workload_sha256, source_node_ids=list(source_node_ids))
        key = native_cost_key(identity, ctx)
        entry = dict(cost_key=key, implementation=identity, context=ctx, run=functools.partial(run, *args, **kwargs),
                     native_handle=run, arguments=(args, kwargs), before_sample=before_sample,
                     tensor_versions=_tensor_versions(dict(args=args, kwargs=kwargs)),
                     replay_safe=replay_safe, stream_semantics='current_stream', supports_cuda_graph=False)
        self.entries.append(entry)
        return key

    def register_observed(self, record, *, native_callable_index, source_workload_sha256,
                          context, state_contract, occurrence=0, before_sample=None, replay_safe=False):
        """Bind one actual observed native call without replacing its handle."""
        members = record.get('source_occurrences', [])
        retained = record.get('source_arguments', [])
        contracts = record.get('source_argument_contracts', [])
        runtimes = record.get('source_runtime_identities', [])
        if not 0 <= occurrence < len(members) or not members[occurrence]:
            raise ValueError('observed native call has no source occurrence')
        if occurrence >= len(retained) or retained[occurrence] is None or occurrence >= len(contracts):
            raise ValueError('observed native call arguments were not retained')
        if occurrence >= len(runtimes) or runtimes[occurrence] != _runtime_identity():
            raise ValueError('observed native runtime/math context differs from replay')
        args, kwargs = retained[occurrence]
        if _contract(dict(args=args, kwargs=kwargs)) != contracts[occurrence]:
            raise ValueError('observed native argument contract changed before registration')
        versions = record.get('source_tensor_versions', [])
        if (before_sample is None and state_contract.get('mutation') == 'read_only' and
                occurrence < len(versions) and
                _tensor_versions(dict(args=args, kwargs=kwargs)) != versions[occurrence]):
            raise ValueError('observed native tensor state changed before registration; reset required')
        implementation = dict(name=record['name'], native_callable_index=native_callable_index,
            reference_source_sha256=record.get('reference_source_sha256'),
            compiled_correspondence=record.get('compiled_correspondence', 'unverified'),
            observed_occurrence=occurrence, boundary_kind=record.get('boundary_kind', 'compiled_callable'))
        custom = record.get('boundary_kind') == 'custom_dispatch'
        if custom:
            if replay_safe and state_contract.get('communication') != 'none':
                raise ValueError('custom dispatch replay requires explicit communication=none state contract')
            if record.get('status') != 'returned':
                raise ValueError('custom dispatch did not return successfully')
            if record.get('observation_error') or 'input_output_contract' not in record:
                raise ValueError('custom dispatch input/output observation is incomplete')
            keys = record.get('source_dispatch_keys', [])
            contexts = record.get('source_dispatch_contexts', [])
            if occurrence >= len(keys) or occurrence >= len(contexts):
                raise ValueError('custom dispatch context was not retained')
            if _unsupported(record.get('input_output_contract', {})):
                raise ValueError('opaque native input/output state has no supported replay contract')
            schema = getattr(record['native_handle'], '_schema', None)
            if schema is None or str(schema) != record.get('operator_schema'):
                raise ValueError('custom native overload schema changed')
            if replay_safe and any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments):
                if before_sample is None or state_contract.get('mutation') == 'read_only':
                    raise ValueError('mutable custom schema requires explicit mutation contract and reset')
            implementation.update(operator_schema=record['operator_schema'], dispatch_context=contexts[occurrence])
        key = self.register(record['native_handle'], args=args, kwargs=kwargs,
            implementation=implementation, context=context,
            source_workload_sha256=source_workload_sha256, source_node_ids=members[occurrence],
            state_contract=state_contract, before_sample=before_sample, replay_safe=replay_safe)
        if custom:
            import torch
            native = self.entries[-1]['run']
            include, exclude = keys[occurrence]
            def redispatch():
                # __torch_dispatch__ observes below-autograd execution. Calling
                # the overload from Python without these keys can add autograd
                # setup or choose another implementation.
                with torch._C._ForceDispatchKeyGuard(include, exclude):
                    return native()
            self.entries[-1]['run'] = redispatch
        return key

    def calibration_workload(self, *, hardware):
        selected = [entry for entry in self.entries if entry['replay_safe']]
        if not selected:
            raise ValueError('no retained callable has an explicit safe replay contract')
        if any(entry['context']['hardware_id'] != hardware.get('id') for entry in selected):
            raise ValueError('native callable hardware differs from calibration workload')
        nodes = [Node(i, 'retained_native_callable', 0, attrs=dict(cost_key=e['cost_key'],
                 native_measurement=dict(kind='live_native_callable', implementation=e['implementation'], context=e['context'])))
                 for i, e in enumerate(selected)]
        return Workload(nodes, [], dict(world_size=1, ranks=[0], registry_token=self.token,
            scope='isolated live native callable measurement', unresolved=[
                'Live callable retention does not verify full-training compiled correspondence.',
                'Retained arguments extend storage lifetimes; this workload is not memory evidence.']), hardware)


def region_provider(workload, config):
    token = config.get('registry_token', workload.manifest.get('registry_token'))
    registry = _REGISTRIES.get(token)
    if registry is None:
        raise ValueError('live callable registry unavailable; execute in its owning process')
    requested = {node.attrs['cost_key']: node.attrs['native_measurement'] for node in workload.nodes}
    regions = []
    for entry in registry.entries:
        if entry['cost_key'] not in requested:
            continue
        binding = requested[entry['cost_key']]
        if entry['context']['hardware_id'] != workload.hardware.get('id'):
            raise ValueError('native callable hardware identity mismatch')
        if entry['context']['runtime_identity'] != _runtime_identity():
            raise ValueError('native callable runtime/math context changed')
        if not entry['replay_safe'] or any(binding[k] != entry[k] for k in ('implementation', 'context')):
            raise ValueError('live callable binding or state replay contract mismatch')
        if _contract(dict(args=entry['arguments'][0], kwargs=entry['arguments'][1])) != entry['implementation']['argument_contract']:
            raise ValueError('retained native arguments changed shape/layout/alias contract')
        if (entry['before_sample'] is None and entry['implementation']['state_contract'].get('mutation') == 'read_only' and
                _tensor_versions(dict(args=entry['arguments'][0], kwargs=entry['arguments'][1])) != entry['tensor_versions']):
            raise ValueError('retained native tensor state changed; reset required')
        regions.append({key: entry[key] for key in ('cost_key', 'implementation', 'context', 'run',
                        'before_sample', 'stream_semantics', 'supports_cuda_graph')})
    if len(regions) != len(requested):
        raise ValueError('requested live callable handle is unavailable')
    return dict(regions=regions, provenance=dict(registry_token=token, kind='retained_original_native_callable'))


@contextmanager
def observe_native_compile(records, *, retain_arguments=False, recorder_getter=lambda: None, boundary_only=False,
                           retention=None):
    """Observe the pinned TorchTitan local-compile hook without replacing its backend.

    Handles and optional arguments stay live. Unknown state prevents automatic
    replay. A verified FX-to-generated-kernel correspondence is never inferred.
    """
    from torchtitan.distributed.local_compile import LocalCompileConfig
    original = LocalCompileConfig._compile
    retention = retention or ArgumentRetention(retain_arguments)
    def compile_observed(reference, **options):
        native = original(reference, **options)
        record = dict(native_handle=native, reference=reference, compile_options=options,
            name=reference.__module__+'.'+reference.__qualname__, calls=0,
            compiled_correspondence='unverified', state_contract={'mutation': 'unknown'},
            source_occurrences=[], boundary_kind='compiled_callable')
        try:
            source = inspect.getsource(reference)
            record['reference_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        except (OSError, TypeError):
            record['reference_source_sha256'] = None
        record_index = len(records)
        records.append(record)
        @functools.wraps(native)
        def observed(*args, **kwargs):
            record['calls'] += 1
            recorder = recorder_getter()
            boundary, begin = None, None
            input_contract, runtime_identity, versions = None, None, None
            try:
                if 'argument_contract' not in record:
                    record['argument_contract'] = _contract(dict(args=args, kwargs=kwargs))
                if (boundary_only and recorder is not None) or _active(recorder):
                    input_contract = _contract(dict(args=args, kwargs=kwargs))
                    runtime_identity = _runtime_identity()
                    versions = _tensor_versions(dict(args=args, kwargs=kwargs))
                    begin = len(recorder.nodes)
                    boundary = recorder._node('native_callable:'+record['name'],
                        operator='native_callable:'+record['name'], native_callable_boundary=True,
                        native_callable_index=record_index, status='issued',
                        compiled_correspondence='unverified', state_contract={'mutation':'unknown'})
                    boundary.attrs['arguments'] = recorder.describe(dict(args=args, kwargs=kwargs), node=boundary.id)
                    recorder.gaps.add('Native callable boundary observed; generated-kernel correspondence and workspace coverage unverified.')
                    if boundary_only:
                        recorder.gaps.add('Boundary-only observation does not capture internal dispatch, communication or intermediate storage.')
            except Exception as error:
                record['observation_error'] = type(error).__name__+': '+str(error)
            try:
                output = native(*args, **kwargs)
            except BaseException:
                if boundary is not None:
                    boundary.attrs['status'] = 'failed'
                raise
            try:
                if 'input_output_contract' not in record:
                    record['input_output_contract'] = _contract(dict(args=args, kwargs=kwargs, output=output))
                if boundary is not None:
                    boundary.attrs.update(status='returned', outputs=recorder.describe(output, node=boundary.id))
                    members = [n.id for n in recorder.nodes[begin:]]
                    boundary.attrs['native_region_member_node_ids'] = members
                    record['source_occurrences'].append(members)
                    record.setdefault('source_argument_contracts', []).append(input_contract)
                    record.setdefault('source_runtime_identities', []).append(runtime_identity)
                    record.setdefault('source_tensor_versions', []).append(versions)
                    record.setdefault('source_arguments', []).append((args, kwargs) if retention(record['name']) else None)
            except Exception as error:
                record['observation_error'] = type(error).__name__+': '+str(error)
            return output
        return observed
    LocalCompileConfig._compile = staticmethod(compile_observed)
    try:
        yield
    finally:
        LocalCompileConfig._compile = staticmethod(original)
