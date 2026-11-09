"""Conservative, CPU-only reconciliation of native rank dispatch artifacts.

``recover_schedule(workloads)`` returns a new Workload. Recovery evidence and
rejections are in ``manifest.schedule_recovery``; every remaining limitation is
also a manifest gap. No recipe, trainer, CUDA operation or recorded code runs.
"""
from collections import defaultdict, deque
from copy import deepcopy

from .communication_cases import (
    _OPERATIONS, _alias_contract, _peer_evidence, _tensor_signature,
    _validate_tensors,
)
from .core import Node, validate
from .identity import digest, workload_fingerprint
from .prepare import _validate_rendezvous, prepare
from .workload import Workload, read_workload


_WAIT = '_c10d_functional.wait_tensor.default'


def _storages(value):
    if isinstance(value, dict):
        if isinstance(value.get('storage_id'), str):
            return {value['storage_id']}
        return set().union(*(_storages(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_storages(item) for item in value)) if value else set()
    return set()


def _ordering(nodes):
    """Topological positions are used only with explicit reachability checks."""
    dependencies = {node.id: tuple(node.deps) for node in nodes}
    children, remaining = defaultdict(list), {}
    for node in nodes:
        remaining[node.id] = len(node.deps)
        for dep in node.deps:
            children[dep].append(node.id)
    ready = deque(key for key, count in remaining.items() if not count)
    order = {}
    while ready:
        key = ready.popleft()
        order[key] = len(order)
        for child in children[key]:
            remaining[child] -= 1
            if not remaining[child]:
                ready.append(child)

    def before(first, second):
        if order[first] >= order[second]:
            return False
        pending, seen = list(dependencies[second]), set()
        while pending:
            current = pending.pop()
            if current == first:
                return True
            if current in seen or order[current] < order[first]:
                continue
            seen.add(current)
            pending.extend(dependencies[current])
        return False
    return order, before


def _descriptor(node):
    record = node.attrs['communication']
    participants = record.get('participants')
    if (not isinstance(participants, list) or not participants or
            any(type(rank) is not int or rank < 0 for rank in participants) or
            len(set(participants)) != len(participants) or node.rank not in participants or
            record.get('group_size') != len(participants)):
        raise ValueError('invalid resolved process-group membership')
    if not isinstance(record.get('group_name'), str) or not record['group_name']:
        raise ValueError('missing resolved process-group identity')
    if node.attrs.get('status') != 'returned':
        raise ValueError('native dispatch did not return successfully')
    operator = record.get('operator')
    if operator not in _OPERATIONS or _OPERATIONS[operator] != record.get('operation'):
        raise ValueError('unsupported or inconsistent native collective operator')
    if node.name != operator or node.attrs.get('operator', operator) != operator:
        raise ValueError('collective descriptor contradicts captured operator')
    if record['operation'] in ('send', 'recv'):
        raise ValueError('point-to-point schedule recovery is unsupported')
    if record.get('evidence') not in ('native_dispatch_and_resolved_process_group',
                                      'local_cuda_substitution_and_target_group'):
        raise ValueError('native dispatch/group evidence is missing')
    backend = record.get('target_backend', record.get('backend'))
    if backend != 'nccl':
        raise ValueError('only NCCL target collective semantics are supported')
    tensors = {key: _tensor_signature(value) for key, value in record['tensors'].items()}
    if 'tensors' in tensors and len(tensors['tensors']) != 1:
        raise ValueError('coalesced tensor-list collectives are unsupported')
    volumes = _validate_tensors(record, tensors)
    alias = _alias_contract(record, tensors, node.rank)
    signature = {key: deepcopy(record.get(key)) for key in (
        'operator', 'operator_schema', 'operation', 'participants', 'group_size',
        'group_name', 'reduce_op', 'reduction_factor', 'async_op', 'timeout',
        'runtime_context', 'split_sizes', 'extra_arguments')}
    signature.update(backend=backend, tensors=tensors, alias=alias, logical_payload=volumes)
    if not signature['operator_schema'] or not isinstance(signature['runtime_context'], dict):
        raise ValueError('native schema or runtime context is missing')
    return signature


def _native(record, source):
    return (record.get('evidence') == 'native_dispatch_and_resolved_process_group' and
            record.get('backend') == 'nccl' and
            record.get('execution_backend', 'nccl') == 'nccl' and
            record.get('transport_executed') is not False and
            not record.get('substitution') and
            source.manifest.get('execution', source.manifest.get('provenance', {}).get('execution'))
            not in ('fake', 'local_cuda'))


def _functional_waits(node, nodes, before, source_workload):
    """Require exact native output identity and a unique, ordered wait lineage."""
    output = node.attrs.get('outputs')
    if not isinstance(output, dict) or len(_storages(output)) != 1:
        raise ValueError('functional collective output storage identity is missing')
    storage = output['storage_id']
    signature = _tensor_signature(output)
    record = node.attrs['communication']
    source = _tensor_signature(record['tensors']['input'])
    expected = list(source['shape'])
    if record['operation'] == 'all_gather_into_tensor':
        expected[0] *= record['group_size']
    elif record['operation'] == 'reduce_scatter_tensor':
        expected[0] //= record['group_size']
    if signature['shape'] != expected or signature['dtype'] != source['dtype']:
        raise ValueError('functional output shape/dtype contradicts native collective')
    allocations = {item.id: item for item in source_workload.storage if item.rank == node.rank}
    inputs = _storages(record['tensors'])
    if not inputs or any(identity not in allocations for identity in inputs | {storage}):
        raise ValueError('functional collective references missing backing storage')
    volumes = _validate_tensors(record, {'input': source})
    output_allocation = allocations[storage]
    if (storage in inputs or output_allocation.producer != node.attrs['source_node_id'] or
            signature['storage_offset'] != 0 or output_allocation.size != volumes['output_bytes'] or
            any(allocations[identity].size < volumes['input_bytes'] for identity in inputs)):
        raise ValueError('functional collective backing storage extent or producer is inconsistent')
    if any(other is not node and other.rank == node.rank and other.attrs.get('communication') and
           storage in _storages(other.attrs.get('outputs')) for other in nodes):
        raise ValueError('functional output storage generation has multiple collective producers')
    waits = []
    for candidate in nodes:
        if candidate.rank != node.rank or candidate.name != _WAIT:
            continue
        args = candidate.attrs.get('arguments', {})
        positional = args.get('args', [])
        value = args.get('kwargs', {}).get('tensor', positional[0] if positional else None)
        if _storages(value) != {storage}:
            continue
        if not before(node.id, candidate.id):
            raise ValueError('functional wait lacks explicit dependency order after issue')
        if (candidate.attrs.get('status') != 'returned' or
                _tensor_signature(value) != signature or
                _storages(candidate.attrs.get('outputs')) != {storage}):
            raise ValueError('functional wait is failed or has contradictory tensor identity')
        waits.append(candidate)
    if not waits:
        raise ValueError('no native functional wait matches the collective output storage')
    first = next((wait for wait in waits if all(
        other is wait or before(wait.id, other.id) for other in waits)), None)
    if first is None:
        raise ValueError('functional wait ordering is ambiguous')
    # Do not infer synchronization through aliases, reused storage or an output
    # consumer. Such nodes require a dedicated semantic observer.
    for candidate in nodes:
        if candidate.rank != node.rank or candidate is node or candidate in waits:
            continue
        if storage not in (_storages(candidate.attrs.get('arguments')) |
                           _storages(candidate.attrs.get('outputs'))):
            continue
        if before(node.id, candidate.id) and not before(first.id, candidate.id):
            raise ValueError('collective output has an unordered or pre-wait use')
    return waits


def recover_schedule(workloads):
    """Recover supported rendezvous and functional waits from original captures.

    Inputs are Workload objects or JSON paths, one original dispatch capture per
    rank. Existing native host plans should use ``plan_lowering.lower_plan``.
    Unsupported/missing evidence returns gaps, not invented peer executions.
    Malformed artifacts, conflicting provenance and dependency cycles raise.
    """
    sources = [deepcopy(value) if isinstance(value, Workload) else read_workload(value)
               for value in workloads]
    if not sources:
        raise ValueError('schedule recovery requires at least one rank artifact')
    peer_evidence = _peer_evidence(sources)
    # Disable Chrome projection: output identity and original host dispatch
    # dependencies must remain intact for this particular recovery pass.
    result = prepare(sources, {'chrome_traces': []})
    source_by_rank = {rank: source for source in sources for rank in source.manifest['ranks']}
    fingerprints = {rank: workload_fingerprint(source) for rank, source in source_by_rank.items()}
    report = dict(version=1, source_workload_sha256_by_rank={str(k): v for k, v in fingerprints.items()},
                  peer_evidence=peer_evidence, rendezvous=[], rejected=[],
                  scope='native dispatch rendezvous and exact functional wait lineage; no stream recovery')
    result.manifest['schedule_recovery'] = report
    gaps = set(result.manifest.get('unresolved', []))

    def reject(reason, nodes=()):
        refs = [dict(rank=node.rank, id=node.attrs.get('source_node_id', node.id)) for node in nodes]
        report['rejected'].append(dict(reason=reason, source_nodes=refs))
        gaps.add('Schedule recovery: ' + reason + '.')

    complete = set(result.manifest['ranks']) == set(range(result.manifest['world_size']))
    if not complete:
        reject('missing rank artifacts; global schedule recovery requires every declared rank')
    identity = any(peer_evidence['capture_identity'].get(key) for key in
                   ('manifest_capture_run_id', 'capture_run_id'))
    if not identity:
        reject('same-run identity is missing; independent rank traces cannot establish a native global schedule')
    eligible = all(source.manifest.get('observer') == 'dispatch' and
                   source.manifest.get('capture_status') == 'complete' for source in sources)
    if not eligible:
        reject('recovery requires complete original native dispatch artifacts')

    original_nodes = list(result.nodes)
    # prepare preserves input storage order when no storage assertions or Chrome
    # lowering are requested. Keep an explicit source-to-merged allocation map.
    allocation_map = {(original.rank, original.id): merged for original, merged in zip(
        (item for source in sources for item in source.storage), result.storage)}
    order, before = _ordering(original_nodes)
    groups = defaultdict(list)
    signatures = {}
    invalid_groups = set()
    for node in original_nodes:
        record = node.attrs.get('communication')
        if not isinstance(record, dict):
            if node.attrs.get('semantics') == 'unresolved_communication' and node.name != _WAIT:
                reject('communication lacks a supported native descriptor', [node])
            continue
        key = digest([record.get('group_name'), record.get('participants')])
        groups[key].append(node)
        try:
            signatures[node.id] = _descriptor(node)
        except (ValueError, TypeError, KeyError) as error:
            invalid_groups.add(key)
            reject(str(error), [node])

    for key, members in groups.items():
        if key in invalid_groups:
            continue
        participants = members[0].attrs['communication']['participants']
        per_rank = {rank: sorted((node for node in members if node.rank == rank),
                                key=lambda node: order[node.id]) for rank in participants}
        if len({len(sequence) for sequence in per_rank.values()}) != 1 or not all(per_rank.values()):
            reject('missing peer collective occurrences', members)
            continue
        if any(not before(left.id, right.id) for sequence in per_rank.values()
               for left, right in zip(sequence, sequence[1:])):
            reject('ambiguous collective issue order within a process group', members)
            continue
        for sequence, peers in enumerate(zip(*(per_rank[rank] for rank in participants))):
            if (len({digest(signatures[node.id]) for node in peers}) != 1 or
                    len({digest(node.attrs.get('step')) for node in peers}) != 1):
                reject('peer collective signatures or steps disagree; uneven shards are unsupported', peers)
                continue
            signature = signatures[peers[0].id]
            rendezvous = 'recovered-native:' + digest(dict(group=key, sequence=sequence,
                signature=signature, sources=report['source_workload_sha256_by_rank']))
            occurrence = dict(rendezvous=rendezvous, sequence=sequence, participants=participants,
                signature=signature, source_nodes=[dict(rank=node.rank, id=node.attrs['source_node_id'])
                                                  for node in peers], status='candidate')
            report['rendezvous'].append(occurrence)
            if not all(_native(node.attrs['communication'], source_by_rank[node.rank]) for node in peers):
                reject('fake or substituted transport cannot establish native completion', peers)
                continue
            if not complete or not identity or not eligible:
                continue
            if not signature['operator'].startswith('_c10d_functional.'):
                reject('native c10d Work completion needs an observed Work wait; host return is insufficient', peers)
                continue
            try:
                waits = {node.id: _functional_waits(node, original_nodes, before,
                                                   source_by_rank[node.rank]) for node in peers}
            except (ValueError, TypeError, KeyError) as error:
                reject(str(error), peers)
                continue
            # Commit an occurrence only after all ranks pass; never lower half
            # of a rendezvous or manufacture a missing peer.
            occurrence.update(status='recovered', completion_nodes=[], waits=[])
            for node in peers:
                source_id = node.attrs['source_node_id']
                completion_id = max((item.id for item in result.nodes), default=-1) + 1
                attrs = dict(communication=deepcopy(node.attrs['communication']),
                    communication_completion='device', rendezvous=rendezvous,
                    participants=deepcopy(participants), group=signature['group_name'],
                    operation=signature['operation'], payload_bytes=signature['logical_payload']['input_bytes'],
                    source_node_id=source_id, source_rank=node.rank,
                    source_workload_sha256=fingerprints[node.rank], source_issue_node_id=node.id,
                    cost_key='native-communication-completion:' + digest(dict(signature=signature,
                        source_workload_sha256=fingerprints[node.rank], source_node_id=source_id)),
                    completion_evidence='native functional wait_tensor storage lineage',
                    completion_measured=False)
                completion = Node(completion_id, node.name + ':device_completion', node.rank,
                                  kind='collective', deps=[node.id], attrs=attrs)
                result.nodes.append(completion)
                occurrence['completion_nodes'].append(completion_id)
                node.attrs['source_cost_key'] = node.attrs.pop('cost_key', None)
                node.attrs.pop('native_measurement', None)
                node.attrs.update(zero_cost_semantic=True, resources=[],
                                  semantics='native_collective_issue', completion_node_id=completion_id)
                for wait in waits[node.id]:
                    wait.deps = sorted(set(wait.deps + [completion_id]))
                    wait.attrs['source_cost_key'] = wait.attrs.pop('cost_key', None)
                    wait.attrs.pop('native_measurement', None)
                    wait.attrs.update(zero_cost_semantic=True, resources=[],
                                      semantics='native_functional_wait', completion_node_id=completion_id)
                    occurrence['waits'].append(dict(rank=wait.rank,
                        source_node_id=wait.attrs['source_node_id'], node_id=wait.id,
                        completion_node_id=completion_id))
                for allocation in result.storage:
                    # Output storage is allocated at dispatch, potentially long
                    # before peers rendezvous. Retain its original producer.
                    if node.id in allocation.consumers:
                        allocation.consumers = sorted({completion_id if ref == node.id else ref
                                                       for ref in allocation.consumers})
                for storage_id in (_storages(node.attrs['communication']['tensors']) |
                                   _storages(node.attrs['outputs'])):
                    allocation = allocation_map[(node.rank, storage_id)]
                    allocation.consumers = sorted(set(allocation.consumers + [completion_id]))
    recovered = [record for record in report['rendezvous'] if record['status'] == 'recovered']
    if recovered:
        gaps.add('Recovered collective completions require exact native device-completion calibration; dispatch costs are not reused.')
        gaps.add('Native functional waits establish device dependency, not measured CPU blocking time; CUDA stream/event ordering remains unresolved.')
    report['recovered_count'] = len(recovered)
    report['candidate_count'] = len(report['rendezvous']) - len(recovered)
    result.manifest['unresolved'] = sorted(gaps)
    validate(result.nodes, result.storage, result.manifest)
    _validate_rendezvous(result.nodes)
    return result
