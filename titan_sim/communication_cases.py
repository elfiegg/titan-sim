"""Identify native collective cases and measure their isolated completion.

Cases come from real dispatch and resolved process-group membership. Replaying a
case uses the same native overload on an existing group of the same size. It
does not predict a larger group, recover a distributed schedule, or model overlap.
"""
from copy import deepcopy
import hashlib
import json
import os
import ast
import math

from .core import Node
from .native_runner import native_cost_key
from .workload import Workload


_OPERATIONS = {
    'c10d._allgather_base_.default': 'all_gather_into_tensor',
    'c10d._reduce_scatter_base_.default': 'reduce_scatter_tensor',
    'c10d.allreduce_.default': 'all_reduce',
    '_c10d_functional.all_gather_into_tensor.default': 'all_gather_into_tensor',
    '_c10d_functional.reduce_scatter_tensor.default': 'reduce_scatter_tensor',
    '_c10d_functional.all_reduce.default': 'all_reduce',
}
_DTYPES = {'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4,
           'torch.float64': 8, 'torch.int8': 1, 'torch.uint8': 1,
           'torch.int32': 4, 'torch.int64': 8}


def execution_context(torch):
    """Record NCCL settings that cannot be inferred from an operator schema."""
    try:
        version = torch.cuda.nccl.version()
        version = list(version) if isinstance(version, tuple) else version
    except (AttributeError, RuntimeError):
        version = None
    return dict(nccl_version=version, torch_git_version=torch.version.git_version,
        environment={key: os.environ.get(key) for key in (
            'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_COLLNET_ENABLE', 'NCCL_NVLS_ENABLE',
            'NCCL_NET', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_IB_DISABLE',
            'NCCL_IB_HCA', 'NCCL_SOCKET_IFNAME', 'NCCL_CUMEM_ENABLE',
            'NCCL_CUMEM_HOST_ENABLE', 'TORCH_NCCL_BLOCKING_WAIT',
            'TORCH_NCCL_ASYNC_ERROR_HANDLING', 'CUDA_DEVICE_MAX_CONNECTIONS')})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def describe_communication(func, args, kwargs, describe):
    """Read an existing group before dispatch; never create a group or wait."""
    name = str(func)
    if name not in _OPERATIONS:
        return None
    import torch
    dist = torch.distributed
    values = {a.name: value for a, value in zip(func._schema.arguments, args)}
    values.update(kwargs or {})
    for argument in func._schema.arguments:
        if argument.name not in values and argument.has_default_value():
            values[argument.name] = argument.default_value
    group = values.get('process_group')
    if group is None:
        group = values['group_name']
        if isinstance(group, str):
            group = dist.distributed_c10d._resolve_process_group(group)
    if isinstance(group, torch.ScriptObject):
        group = dist.ProcessGroup.unbox(group)
    participants = dist.get_process_group_ranks(group)
    if 'group_size' in values and values['group_size'] != len(participants):
        raise ValueError('native functional group_size disagrees with resolved membership')
    if values.get('sparse_indices') is not None:
        raise ValueError('sparse-index collective replay is unsupported')
    extras = {}
    known = {'input', 'input_tensor', 'output_tensor', 'tensors', 'process_group',
             'group_name', 'group_size', 'reduce_op', 'sparse_indices', 'async_op', 'timeout'}
    for argument in func._schema.arguments:
        if argument.name not in known:
            value = values[argument.name]
            if (not argument.has_default_value() or value != argument.default_value
                    or not (value is None or type(value) in (bool, int, float, str))):
                raise ValueError('unsupported nondefault native collective argument: '+argument.name)
            extras[argument.name] = value
    reduction = values.get('reduce_op')
    if isinstance(reduction, torch.ScriptObject):
        reduction = dist.ReduceOp.unbox(reduction)
    reduction_factor = None
    if reduction is not None and not isinstance(reduction, str):
        operation = getattr(reduction, 'op', reduction)
        if operation == dist.ReduceOp.PREMUL_SUM:
            reduction_factor = reduction.factor
            if (type(reduction_factor) not in (int, float) or not math.isfinite(reduction_factor)):
                raise ValueError('PREMUL_SUM requires a captured finite host scalar factor; tensor factors unsupported')
        reduction = next((key.lower() for key in ('SUM', 'AVG', 'PRODUCT', 'MIN', 'MAX', 'PREMUL_SUM')
                          if operation == getattr(dist.ReduceOp, key)), None)
        if reduction is None:
            raise ValueError('unsupported native reduction operator')
    tensors = {key: describe(values[key]) for key in
               ('input', 'input_tensor', 'output_tensor', 'tensors') if key in values}
    return dict(operator=name, operator_schema=str(func._schema), operation=_OPERATIONS[name],
                participants=participants, group_size=len(participants),
                backend=str(dist.get_backend(group)),
                group_name=str(dist.distributed_c10d._get_process_group_name(group)),
                reduce_op=reduction, reduction_factor=reduction_factor,
                async_op=values.get('async_op'), tensors=tensors,
                timeout=values.get('timeout', -1), extra_arguments=extras,
                runtime_context=execution_context(torch),
                evidence='native_dispatch_and_resolved_process_group')


def _tensor_signature(value):
    if isinstance(value, list):
        return [_tensor_signature(item) for item in value]
    if not isinstance(value, dict):
        raise ValueError('collective tensor metadata is missing')
    if 'distributed' in value or value.get('dtype') not in _DTYPES:
        raise ValueError('collective replay requires supported local tensor dtype')
    if value.get('layout') != 'torch.strided' or not str(value.get('device', '')).startswith('cuda'):
        raise ValueError('collective replay requires CUDA strided tensors')
    shape, stride = value.get('shape'), value.get('stride')
    if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('collective replay requires nonempty static shapes')
    expected, size = [], 1
    for dim in reversed(shape):
        expected.insert(0, size)
        size *= dim
    if (stride != expected or type(value.get('storage_offset')) is not int
            or value['storage_offset'] < 0):
        raise ValueError('collective replay requires contiguous tensors and nonnegative offsets')
    return {key: deepcopy(value[key]) for key in ('shape', 'stride', 'dtype', 'layout', 'storage_offset') if key in value}


def _validate_tensors(record, tensors):
    functional = record['operator'].startswith('_c10d_functional.')
    operation = record['operation']
    expected = {'input'} if functional else {'tensors'} if operation == 'all_reduce' else {'input_tensor', 'output_tensor'}
    if set(tensors) != expected:
        raise ValueError('collective tensor argument keys differ from native schema')
    def count(value):
        total = 1
        for dim in value['shape']:
            total *= dim
        return total
    source = tensors['input'] if functional else tensors['tensors'][0] if operation == 'all_reduce' else tensors['input_tensor']
    if functional and operation != 'all_reduce' and not source['shape']:
        raise ValueError('functional gather/scatter requires a non-scalar input')
    group_size = record['group_size']
    input_elements = count(source)
    output_elements = input_elements
    if operation == 'all_gather_into_tensor':
        output_elements *= group_size
    elif operation == 'reduce_scatter_tensor':
        if input_elements % group_size or (functional and source['shape'][0] % group_size):
            raise ValueError('reduce-scatter input is not divisible by group size')
        output_elements //= group_size
    if 'output_tensor' in tensors:
        target = tensors['output_tensor']
        if target['dtype'] != source['dtype'] or count(target) != output_elements:
            raise ValueError('collective output dtype/size does not match native input and group')
    return dict(input_bytes=input_elements * _DTYPES[source['dtype']],
                output_bytes=output_elements * _DTYPES[source['dtype']])


def _extra_arguments(record, node):
    """Preserve defaults even for captures made before extra-argument recording."""
    schema = record['operator_schema']
    if '(' not in schema:
        return deepcopy(record.get('extra_arguments', {}))  # Synthetic unit fixture.
    declaration = schema.split(' -> ', 1)[0]
    body = declaration[declaration.index('(')+1:declaration.rfind(')')]
    fields, begin, depth = [], 0, 0
    for index, char in enumerate(body):
        if char in '([':
            depth += 1
        elif char in ')]':
            depth -= 1
        elif char == ',' and depth == 0:
            fields.append(body[begin:index].strip())
            begin = index+1
    fields.append(body[begin:].strip())
    known = {'input', 'input_tensor', 'output_tensor', 'tensors', 'process_group',
             'group_name', 'group_size', 'reduce_op', 'sparse_indices', 'async_op', 'timeout'}
    captured = node.attrs.get('arguments', {})
    args, kwargs = captured.get('args', []), captured.get('kwargs', {})
    extras = {}
    for index, field in enumerate(fields):
        declaration, equal, default_text = field.partition('=')
        name = declaration.rsplit(' ', 1)[-1]
        if name in known:
            continue
        if 'arguments' not in node.attrs and 'extra_arguments' not in record:
            raise ValueError('native collective argument evidence is missing')
        if not equal:
            raise ValueError('unsupported required native collective argument: '+name)
        try:
            default = ast.literal_eval(default_text)
        except (ValueError, SyntaxError):
            raise ValueError('unsupported native collective default: '+name) from None
        value = kwargs.get(name, args[index] if index < len(args) else default)
        if value != default or not (value is None or type(value) in (bool, int, float, str)):
            raise ValueError('unsupported nondefault native collective argument: '+name)
        extras[name] = value
    retained = record.get('extra_arguments')
    if retained is not None and retained != extras:
        raise ValueError('native collective extra arguments disagree with capture')
    return extras


def _alias_contract(record, tensors, rank):
    raw = record['tensors']
    if not {'input_tensor', 'output_tensor'} <= raw.keys():
        if any(item.get('storage_offset', 0) for value in tensors.values()
               for item in (value if isinstance(value, list) else [value])):
            raise ValueError('unaliased collective tensor offsets must be zero')
        return {'kind': 'separate_allocations'}
    source, target = raw['input_tensor'], raw['output_tensor']
    if not source.get('storage_id') or not target.get('storage_id'):
        raise ValueError('collective alias determination requires storage identities')
    if source['storage_id'] != target['storage_id']:
        if source['storage_offset'] or target['storage_offset']:
            raise ValueError('unaliased collective tensor offsets must be zero')
        return {'kind': 'separate_allocations'}
    if record['operation'] == 'all_gather_into_tensor':
        base, view = 'output_tensor', 'input_tensor'
    elif record['operation'] == 'reduce_scatter_tensor':
        base, view = 'input_tensor', 'output_tensor'
    else:
        raise ValueError('unsupported collective alias pattern')
    elements = 1
    for size in raw[view]['shape']:
        elements *= size
    group_rank = record['participants'].index(rank)
    if raw[base]['storage_offset'] != 0 or raw[view]['storage_offset'] != group_rank * elements:
        raise ValueError('collective alias is not the owning group rank slice')
    # The stored signature is group-rank relative. Literal observed offsets and
    # storage IDs remain in source_nodes; replay evaluates this contract per rank.
    tensors[view]['storage_offset'] = 0
    return dict(kind='own_group_rank_slice', base_tensor=base, view_tensor=view,
                slice_elements=elements, offset_rule='group_rank * slice_elements')


def _peer_evidence(sources):
    """Reject mixed runs when capture IDs exist; report their absence honestly."""
    def identities(source):
        provenance = source.manifest.get('provenance', {})
        runtime = provenance.get('runtime', {})
        return {key: value for key, value in {
            'manifest_capture_run_id': source.manifest.get('capture_run_id'),
            'capture_run_id': provenance.get('capture_run_id'),
            'slurm_job_id': runtime.get('slurm_job_id'),
        }.items() if value is not None}
    expected = identities(sources[0])
    primary = sources[0].manifest.get('provenance', {})
    for source in sources[1:]:
        if identities(source) != expected:
            raise ValueError('peer capture run identity differs or is missing')
        provenance = source.manifest.get('provenance', {})
        for field in ('torch_version', 'cuda_version', 'torchtitan_python_source_sha256', 'loader', 'trainer'):
            if provenance.get(field) != primary.get(field):
                raise ValueError('peer capture source/runtime differs: '+field)
        for field in ('torchtitan_revision', 'container_sha256'):
            if provenance.get('runtime', {}).get(field) != primary.get('runtime', {}).get(field):
                raise ValueError('peer capture source/runtime differs: '+field)
    return dict(capture_identity=expected,
                identity_basis='recorded capture/run identifiers' if expected else
                    'no run identifier recorded; caller asserts same capture run')


def prepare_communication_cases(workload, peer_workloads=()):
    """Extract cases; rank-slice aliases require matching evidence from all peers."""
    sources = [workload, *peer_workloads]
    peer_evidence = _peer_evidence(sources)
    cases, skipped = {}, []
    for source in sources:
        if source.hardware != workload.hardware:
            raise ValueError('peer source hardware differs from primary capture')
        for node in source.nodes:
            if node.attrs.get('zero_cost_semantic'):
                continue
            record = node.attrs.get('communication')
            if record is None:
                if 'c10d' in node.name or 'collective' in node.name:
                    skipped.append(dict(rank=node.rank, node_id=node.id, operator=node.name,
                        reason=node.attrs.get('communication_error', 'no supported resolved process-group evidence')))
                continue
            try:
                if node.attrs.get('status') != 'returned':
                    raise ValueError('native collective did not return successfully')
                if record.get('operator') not in _OPERATIONS or record.get('evidence') != 'native_dispatch_and_resolved_process_group':
                    raise ValueError('unsupported collective implementation evidence')
                if record.get('operation') != _OPERATIONS[record['operator']]:
                    raise ValueError('native collective operation does not match its overload')
                participants = record.get('participants')
                if (not isinstance(participants, list) or not participants or
                        any(type(r) is not int or r < 0 for r in participants) or
                        len(set(participants)) != len(participants) or node.rank not in participants or
                        record.get('group_size') != len(participants)):
                    raise ValueError('collective membership is invalid')
                if record.get('backend') != 'nccl':
                    raise ValueError('GPU collective replay currently requires NCCL')
                if record.get('reduce_op') not in (None, 'sum', 'avg', 'product', 'min', 'max', 'premul_sum'):
                    raise ValueError('unsupported reduction')
                factor = record.get('reduction_factor')
                if record.get('reduce_op') == 'premul_sum':
                    if (not record['operator'].startswith('c10d.') or type(factor) not in (int, float)
                            or not math.isfinite(factor)):
                        raise ValueError('PREMUL_SUM requires exact finite native host scalar factor')
                elif factor is not None:
                    raise ValueError('unexpected factor for native reduction')
                tensors = {key: _tensor_signature(value) for key, value in record['tensors'].items()}
                if 'tensors' in tensors and len(tensors['tensors']) != 1:
                    raise ValueError('tensor-list/coalesced collective replay requires a dedicated contract')
                volumes = _validate_tensors(record, tensors)
                alias = _alias_contract(record, tensors, node.rank)
                case = {key: deepcopy(record[key]) for key in
                        ('operator', 'operator_schema', 'operation', 'participants', 'group_size',
                         'backend', 'group_name', 'reduce_op', 'async_op', 'timeout', 'runtime_context')}
                case.update(tensors=tensors, logical_payload=volumes, alias=alias, reduction_factor=factor,
                            peer_evidence=peer_evidence,
                            extra_arguments=_extra_arguments(record, node))
                # Group by the complete native invocation, then retain all distinct
                # per-rank source provenance. Rank-local resolved recipes can differ.
                identity = _digest(case)
                if identity not in cases:
                    cases[identity] = dict(case, source_nodes=[], source_provenance_by_rank={})
                candidate = cases[identity]
                rank_key = str(node.rank)
                provenance = deepcopy(source.manifest.get('provenance', {}))
                if (rank_key in candidate['source_provenance_by_rank'] and
                        candidate['source_provenance_by_rank'][rank_key] != provenance):
                    raise ValueError('conflicting source provenance for one collective rank')
                candidate['source_provenance_by_rank'][rank_key] = provenance
                candidate['source_nodes'].append(dict(rank=node.rank, node_id=node.id,
                    cost_key=node.attrs.get('cost_key'), tensor_storage=deepcopy(record['tensors'])))
            except (ValueError, KeyError, TypeError) as error:
                skipped.append(dict(rank=node.rank, node_id=node.id, operator=node.name, reason=str(error)))
    eligible = []
    for candidate in cases.values():
        peers = candidate['source_provenance_by_rank']
        if (candidate['alias']['kind'] == 'own_group_rank_slice' and
                {int(rank) for rank in peers} != set(candidate['participants'])):
            for occurrence in candidate['source_nodes']:
                skipped.append(dict(rank=occurrence['rank'], node_id=occurrence['node_id'],
                    operator=candidate['operator'], reason='rank-slice collective alias requires matching evidence from every participant'))
            continue
        candidate['source_provenance'] = deepcopy(peers[str(min(map(int, peers)))])
        candidate['case_id'] = _digest({key: value for key, value in candidate.items() if key != 'source_nodes'})
        eligible.append(candidate)
    return dict(schema='native-communication-cases', version=1,
                cases=eligible, skipped=skipped,
                scope='isolated same-size group completion; no training overlap or larger-group extrapolation')


def communication_case_workload(workload, case_ids=None, *, topology, peer_workloads=()):
    """Build an all-rank measurement workload for one selected group size."""
    bundle = prepare_communication_cases(workload, peer_workloads=peer_workloads)
    if not isinstance(topology, dict) or not topology:
        raise ValueError('explicit measurement topology is required for collective cost identity')
    wanted = None if case_ids is None else set(case_ids)
    cases = [case for case in bundle['cases'] if wanted is None or case['case_id'] in wanted]
    if wanted is not None and wanted != {case['case_id'] for case in cases}:
        raise ValueError('unknown collective case requested')
    sizes = {case['group_size'] for case in cases}
    if len(sizes) != 1:
        raise ValueError('select nonempty collective cases with one native group size')
    world = sizes.pop()
    nodes = []
    for case in cases:
        implementation = dict(kind='native_collective_overload', operator=case['operator'],
                              operator_schema=case['operator_schema'])
        context = dict(case={key: value for key, value in case.items() if key != 'source_nodes'},
                       hardware_id=workload.hardware['id'], cuda_graph=False,
                       measurement_topology=deepcopy(topology),
                       completion='native work wait on timing stream',
                       input_values='ones; inputs reset before every sample',
                       rank_mapping='measurement rank i corresponds to source participants[i]')
        key = native_cost_key(implementation, context)
        for rank in range(world):
            nodes.append(Node(len(nodes), case['operator'], rank,
                attrs=dict(cost_key=key, native_measurement=dict(kind='executable_region',
                           implementation=implementation, context=context))))
    return Workload(nodes, [], dict(ranks=list(range(world)), world_size=world,
        communication_cases=bundle, scope='isolated native collective calibration',
        unresolved=['Isolated collective calibration does not establish distributed training overlap or physical memory fit.']),
        deepcopy(workload.hardware))


def _allocate_tensors(torch, case, rank, device):
    """Materialize the verified group-relative layout, including shared storage."""
    def allocate(desc):
        if isinstance(desc, list):
            return [allocate(item) for item in desc]
        dtype = getattr(torch, desc['dtype'].removeprefix('torch.'), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError('unknown collective dtype')
        return torch.ones(desc['shape'], dtype=dtype, device=device)
    alias = case.get('alias', {'kind': 'separate_allocations'})
    if alias['kind'] == 'separate_allocations':
        return {key: allocate(desc) for key, desc in case['tensors'].items()}
    if alias['kind'] != 'own_group_rank_slice' or not 0 <= rank < case['group_size']:
        raise ValueError('invalid group-relative collective alias contract')
    base_key, view_key = alias['base_tensor'], alias['view_tensor']
    base = allocate(case['tensors'][base_key])
    desc = case['tensors'][view_key]
    view = base.as_strided(desc['shape'], desc['stride'], rank * alias['slice_elements'])
    return {base_key: base, view_key: view}


def region_provider(workload, config):
    """Run original native collective overloads in the caller's existing group."""
    import torch
    dist = torch.distributed
    if not dist.is_initialized():
        raise ValueError('collective calibration requires an initialized process group')
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    group_name = dist.distributed_c10d._get_process_group_name(group)
    regions, active = [], {}
    for node in workload.nodes:
        if node.rank != rank:
            continue
        binding = node.attrs['native_measurement']
        case = binding['context']['case']
        if config.get('topology') != binding['context']['measurement_topology']:
            raise ValueError('measurement topology differs from the collective cost identity')
        if case['group_size'] != world or case['backend'] != str(dist.get_backend(group)):
            raise ValueError('measurement group does not match identified native collective')
        if case['case_id'] != _digest({key: value for key, value in case.items() if key != 'case_id'}):
            raise ValueError('native collective case digest mismatch')
        provenance = case['source_provenance']
        if case.get('runtime_context') != execution_context(torch):
            raise ValueError('NCCL runtime/settings differ from native capture')
        for key, actual in (('torch_version', str(torch.__version__)), ('cuda_version', torch.version.cuda)):
            if any(peer.get(key) != actual for peer in case.get('source_provenance_by_rank', {'0': provenance}).values()):
                raise ValueError('collective runtime mismatch: '+key)
        namespace, operation, overload = case['operator'].split('.')
        function = getattr(getattr(getattr(torch.ops, namespace), operation), overload)
        if str(function._schema) != case['operator_schema']:
            raise ValueError('native collective schema differs from captured implementation')

        reduction = (dist._make_nccl_premul_sum(case['reduction_factor'])
                     if case['reduce_op'] == 'premul_sum' else
                     dist.ReduceOp(getattr(dist.ReduceOp, case['reduce_op'].upper())) if case['reduce_op'] else None)
        kwargs = dict(case.get('extra_arguments', {}))
        if namespace == 'c10d':
            if type(case['async_op']) is not bool:
                raise ValueError('native asynchronous collective mode is missing')
            kwargs.update(process_group=group.boxed(), async_op=case['async_op'], timeout=case['timeout'])
            if operation == 'allreduce_':
                kwargs['sparse_indices'] = None
            if reduction is not None:
                kwargs['reduce_op'] = reduction.boxed()
        else:
            kwargs['group_name'] = group_name
            if operation != 'all_reduce':
                kwargs['group_size'] = world
            if case['reduce_op'] is not None:
                kwargs['reduce_op'] = case['reduce_op']

        def run(_function=function, _kwargs=kwargs, _namespace=namespace, _case_id=case['case_id']):
            if active.get('case_id') != _case_id:
                raise ValueError('collective case must be prepared before timing')
            result = _function(**dict(_kwargs, **active['tensors']))
            if _namespace == 'c10d':
                result[-1].wait()
            else:
                torch.ops._c10d_functional.wait_tensor.default(result)

        def reset(_case=case):
            # Only one case owns memory. Share preparation errors before any
            # peer starts its timed collective; otherwise a one-rank OOM could
            # strand the other ranks in an unmatched communication call.
            error = None
            try:
                if active.get('case_id') != _case['case_id']:
                    active.clear()
                    tensors = _allocate_tensors(torch, _case, rank, torch.cuda.current_device())
                    active.update(case_id=_case['case_id'], tensors=tensors)
                seen = set()
                # Fill the shared base, not just a view. This resets both the
                # in-place input slice and all destination elements outside it.
                values = active['tensors']
                alias = _case.get('alias', {})
                if alias.get('kind') == 'own_group_rank_slice':
                    values = {alias['base_tensor']: values[alias['base_tensor']]}
                for value in values.values():
                    for tensor in value if isinstance(value, list) else [value]:
                        identity = tensor.untyped_storage()._cdata
                        if identity not in seen:
                            tensor.fill_(1)
                            seen.add(identity)
            except Exception as exception:
                error = type(exception).__name__+': '+str(exception)
            errors = [None] * world
            dist.all_gather_object(errors, error)
            if any(errors):
                active.clear()
                raise ValueError('collective input preparation failed on rank(s): '+str(errors))

        regions.append(dict(cost_key=node.attrs['cost_key'], implementation=binding['implementation'],
            context=binding['context'], run=run, before_sample=reset, stream_semantics='current_stream',
            supports_cuda_graph=False, round_id=case['case_id'],
            round_context=dict(participants=list(range(world)), case_id=case['case_id'],
                               source_participants=case['participants'])))
    return dict(regions=regions, provenance=dict(provider='native collective overload replay',
        group_backend=str(dist.get_backend(group)), world_size=world,
        topology=config.get('topology', 'caller-owned placement; see retained allocation evidence'),
        limitations=['Same-size isolated collective; no topology extrapolation or overlap model.']))
