"""Real local CUDA execution with explicit virtual-rank communication substitution.

This is a discovery experiment, not distributed training or a numerical result.
The original trainer and recipe execute; supported communication boundaries use
finite local representatives while target group membership remains visible.
"""
from contextlib import ExitStack, contextmanager, redirect_stdout
import inspect
import os
import sys

# Live handles are deliberately not serialized. Their retention is opt-in.
LIVE_CAPTURES = {}


@contextmanager
def virtual_groups(*, rank, world_size, records):
    import torch.distributed as dist
    import torch.testing._internal.distributed.fake_pg  # Registers native fake backend.
    from .torchtitan import _replace
    if dist.is_initialized():
        raise ValueError('local CUDA discovery requires a fresh process without a process group')
    old_environment = {key: os.environ.get(key) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')}
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK='0', LOCAL_WORLD_SIZE='1')
    try:
        with ExitStack() as stack:
            for name in ('init_process_group', 'new_group'):
                original = getattr(dist, name)
                def substituted(*args, _original=original, _name=name, **kwargs):
                    bound = inspect.signature(_original).bind_partial(*args, **kwargs)
                    values = bound.arguments
                    requested_backend = values.get('backend')
                    inherited_backend = next((g['target_backend'] for g in records if g['api'] == 'init_process_group'), None)
                    target_backend = str(requested_backend) if requested_backend is not None else inherited_backend
                    if target_backend and 'cuda:' in target_backend:
                        target_backend = next(part.split(':', 1)[1] for part in target_backend.split(',') if part.startswith('cuda:'))
                    if _name == 'init_process_group':
                        configured_world = values.get('world_size', -1)
                        configured_rank = values.get('rank', -1)
                        if configured_world not in (-1, world_size) or configured_rank not in (-1, rank):
                            raise ValueError('native recipe process group differs from requested target rank/world')
                        values.update(backend='fake', store=dist.HashStore(), rank=rank, world_size=world_size)
                        values.pop('init_method', None)
                        values.pop('pg_options', None)
                        values.pop('device_id', None)
                    else:
                        values['backend'] = 'fake'
                        values.pop('pg_options', None)
                        values.pop('device_id', None)
                    group = _original(*bound.args, **bound.kwargs)
                    actual = dist.group.WORLD if _name == 'init_process_group' else group
                    participants = (dist.get_process_group_ranks(actual)
                                    if actual != dist.GroupMember.NON_GROUP_MEMBER else list(values.get('ranks', [])))
                    records.append(dict(api=_name, requested_backend=str(requested_backend),
                        execution_backend='fake', target_backend=target_backend, participants=participants,
                        group_name=(str(dist.distributed_c10d._get_process_group_name(actual))
                                    if actual != dist.GroupMember.NON_GROUP_MEMBER else None)))
                    return group
                stack.enter_context(_replace(dist, name, substituted))
            yield
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def communication_mode(records, *, native_records=None, retain_arguments=False, recorder_getter=lambda: None):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from .communication_cases import describe_communication
    from .executable_regions import _contract, _runtime_identity

    class LocalCommunication(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            if not ('c10d' in name or 'collective' in name):
                if native_records is not None and not name.startswith(('aten.', 'prims.')):
                    recorder = recorder_getter()
                    record = dict(native_handle=func, name=name, operator_schema=str(getattr(func, '_schema', '')),
                        compiled_correspondence='unverified', state_contract={'mutation': 'unknown'},
                        source_occurrences=[[recorder.nodes[-1].id]] if recorder is not None and recorder.nodes else [])
                    try:
                        record['argument_contract'] = _contract(dict(args=args, kwargs=kwargs))
                    except Exception as error:
                        record['observation_error'] = type(error).__name__+': '+str(error)
                    if record['source_occurrences'] and 'argument_contract' in record:
                        record['source_argument_contracts'] = [record['argument_contract']]
                        record['source_runtime_identities'] = [_runtime_identity()]
                    if retain_arguments:
                        record['arguments'] = (args, kwargs)
                        record['source_arguments'] = [(args, kwargs)] if record['source_occurrences'] else []
                    native_records.append(record)
                return func(*args, **kwargs)
            if name == '_c10d_functional.wait_tensor.default':
                records.append(dict(operator=name, operation='wait', substitution='already_completed_local_representative'))
                return func(*args, **kwargs)
            if name == '_c10d_functional._wrap_tensor_autograd.default':
                records.append(dict(operator=name, operation='autograd_wrapper', substitution='native_local_wrapper_no_transport'))
                return func(*args, **kwargs)
            if name == 'c10d.barrier.default':
                records.append(dict(operator=name, operation='barrier', substitution='virtual_group_no_transport'))
                return func(*args, **kwargs)
            metadata = describe_communication(func, args, kwargs, _contract)
            if metadata is None:
                raise NotImplementedError('local CUDA communication substitution unsupported: '+name)
            values = {field.name: value for field, value in zip(func._schema.arguments, args)}
            values.update(kwargs)
            operation = metadata['operation']
            source = values.get('input', values.get('input_tensor'))
            if source is None:
                source = values['tensors'][0]
                if len(values['tensors']) != 1:
                    raise NotImplementedError('local substitution supports one tensor per allreduce')
            if not source.is_contiguous():
                raise NotImplementedError('local substitution requires contiguous collective inputs')
            if source.is_floating_point() and not torch.isfinite(source).all().item():
                raise ValueError('communication source is not finite')
            representative = source.clone()
            output = func(*args, **kwargs)  # Native fake backend supplies genuine Work handles.
            functional = name.startswith('_c10d_functional.')
            target = output if functional else output[0]
            if isinstance(target, (tuple, list)):
                target = target[0]
            group_size = metadata['group_size']
            if operation == 'all_gather_into_tensor':
                target.view(-1).copy_(representative.reshape(-1).repeat(group_size))
                policy = 'repeat_local_shard_for_each_target_participant'
            elif operation == 'reduce_scatter_tensor':
                participants = metadata['participants']
                local_index = participants.index(torch.distributed.get_rank())
                shard = representative.reshape(group_size, -1)[local_index]
                target.view(-1).copy_(shard)
                policy = 'local_rank_slice_without_cross_rank_reduction'
            elif operation == 'all_reduce':
                target.copy_(representative)
                policy = 'local_value_without_cross_rank_reduction'
            else:
                raise NotImplementedError('unsupported representative collective '+operation)
            if target.is_floating_point() and not torch.isfinite(target).all().item():
                raise ValueError('communication representative output is not finite')
            recorder = recorder_getter()
            occurrence = (recorder.nodes[-1] if recorder is not None and recorder.nodes and
                          recorder.nodes[-1].attrs.get('operator') == name and
                          recorder.nodes[-1].attrs.get('status') == 'issued' else None)
            metadata.update(source_node_id=occurrence.id if occurrence is not None else None,
                            source_step=occurrence.attrs.get('step') if occurrence is not None else None,
                            substitution=policy, representative_outputs_finite=True,
                            transport_executed=False, numerical_training_equivalence=False)
            records.append(metadata)
            return output
    return LocalCommunication()


def capture(config):
    import torch
    from .torchtitan import _build_trainer, capture_trainer
    from .torch_capture import TorchCapture
    from .executable_regions import observe_native_compile
    from .operator_cases import execution_context
    if config.get('execution') != 'local_cuda':
        raise ValueError('local discovery requires explicit execution=local_cuda')
    rank, world = config.get('target_rank'), config.get('target_world_size')
    if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('target_rank and positive target_world_size are required')
    argv = config.get('recipe_args')
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv) or '--print-config' in argv:
        raise ValueError('recipe_args must be an executable native argument list')
    if not isinstance(config.get('hardware'), dict):
        raise ValueError('hardware must describe the target')
    if config.get('observer', 'dispatch') != 'dispatch':
        raise ValueError('local CUDA discovery currently requires observer=dispatch')
    if not torch.cuda.is_available():
        raise ValueError('local CUDA discovery requires actual CUDA; no fake tensors are used')
    handoff = None
    if config.get('runtime_manifest'):
        from .runtime import check_runtime
        handoff = check_runtime(config['runtime_manifest'], config.get('capability_report'), config.get('expected_runtime'))
        if handoff['status'] != 'ready':
            raise ValueError('prepare-runtime handoff is blocked: '+'; '.join(handoff['unresolved']))
    groups, communication, handles = [], [], []
    provenance = dict(execution='local_cuda', torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        recipe_args=argv, runtime=config.get('runtime_provenance', {}), target_rank=rank, target_world_size=world,
        numerical_training_equivalence=False, compiled_correspondence='unverified')
    if handoff is not None:
        provenance['runtime_handoff'] = handoff
    trainer, workload = None, None
    active_recorder = [None]
    def recorder_callback(recorder):
        active_recorder[0] = recorder
    with redirect_stdout(sys.stderr), virtual_groups(rank=rank, world_size=world, records=groups):
        try:
            with communication_mode(communication, native_records=handles,
                    retain_arguments=config.get('retain_executable_arguments', False),
                    recorder_getter=lambda: active_recorder[0]), observe_native_compile(handles,
                    retain_arguments=config.get('retain_executable_arguments', False),
                    recorder_getter=lambda: active_recorder[0]):
                trainer = _build_trainer(argv, provenance)
                provenance.update(operator_execution_context=execution_context(torch),
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision())
                workload = capture_trainer(trainer, hardware=config['hardware'],
                    steps=config.get('steps', 1), warmup_steps=config.get('warmup_steps', 0),
                    provenance=provenance, observer='dispatch', recorder_callback=recorder_callback)
        except Exception as error:
            recorder = TorchCapture(rank=rank, world_size=world, device='cuda:0', provenance=provenance)
            recorder.gaps.add('Local CUDA discovery failed; no complete native training trace.')
            workload = recorder.finish(hardware=config['hardware'], manifest=dict(capture_status='blocked',
                error_type=type(error).__name__, error=str(error)))
        finally:
            if trainer is not None:
                try:
                    trainer.close()
                except Exception as error:
                    workload.manifest['capture_status'] = 'blocked'
                    workload.manifest['unresolved'].append('Trainer cleanup failed: '+str(error))
    node_by_id = {node.id: node for node in workload.nodes}
    for record in communication:
        matches = [group for group in groups if group['group_name'] == record.get('group_name') and
                   group['participants'] == record.get('participants')]
        target_backends = {group['target_backend'] for group in matches}
        if len(target_backends) == 1:
            record.update(target_backend=next(iter(target_backends)), execution_backend='fake',
                          evidence='local_cuda_substitution_and_target_group')
            node = node_by_id.get(record.get('source_node_id'))
            if node is not None and 'communication' in node.attrs:
                node.attrs['communication'].update(target_backend=record['target_backend'], execution_backend='fake',
                    evidence=record['evidence'], substitution=record['substitution'],
                    representative_outputs_finite=True, transport_executed=False,
                    source_node_id=node.id, source_step=node.attrs.get('step'))
    workload.manifest.update(execution='local_cuda', virtual_process_groups=groups,
        substituted_communications=communication, compiled_correspondence='unverified',
        retained_native_boundaries=[{key: value for key, value in record.items()
            if key not in ('native_handle', 'reference', 'arguments', 'source_arguments')} for record in handles],
        local_cuda_memory_scope='discovery execution only; substituted communication and retained handles change lifetimes')
    workload.manifest['unresolved'] += [
        'Communication outputs are finite local representatives, not distributed numerical results.',
        'Compiled correspondence to uninstrumented target training is unverified.',
        'Local discovery cannot establish all-rank schedule, overlap or target physical memory fit.']
    # Keep a process-local access point; serialized artifacts contain metadata only.
    if config.get('retain_executable_arguments', False):
        import uuid
        token = uuid.uuid4().hex
        LIVE_CAPTURES[token] = handles
        workload.manifest['live_capture_token'] = token
        workload.manifest['unresolved'].append('Retained executable arguments extend tensor lifetimes during discovery.')
    from .identity import workload_fingerprint
    source_fingerprint = workload_fingerprint(workload)
    for record in handles:
        record['source_workload_sha256'] = source_fingerprint
    return workload
