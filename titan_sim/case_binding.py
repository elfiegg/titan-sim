"""Verify native measurement cases and bind them to immutable source occurrences.

Binding supplies implementation correspondence, not missing stream semantics or
physical allocation coverage. Communication completion costs require an explicit
completion node in an already reconciled native plan; dispatch issue alone is
insufficient. No PyTorch import or executable replay occurs here.
"""
from copy import deepcopy

from .core import validate
from .identity import digest, workload_fingerprint
from .native_runner import _descriptor
from .prepare import prepare


def _live_bindings(sources, measurement):
    """Require an observed callable boundary; a manually invented handle is insufficient."""
    if measurement.manifest.get('scope') != 'isolated live native callable measurement':
        raise ValueError('invalid live native measurement scope')
    if measurement.manifest.get('ranks') != [0] or measurement.manifest.get('world_size') != 1:
        raise ValueError('live native local measurement requires one GPU')
    for node in measurement.nodes:
        binding = _descriptor(node)
        implementation, context = binding['implementation'], binding['context']
        if (binding.get('kind') != 'live_native_callable' or
                implementation.get('registry_token') != measurement.manifest.get('registry_token') or
                context.get('hardware_id') != measurement.hardware['id']):
            raise ValueError('live native measurement handle/context mismatch')
        matches = [source for source in sources
                   if workload_fingerprint(source) == context.get('source_workload_sha256')]
        if len(matches) != 1:
            raise ValueError('live native source workload fingerprint mismatch')
        source = matches[0]
        inventory = source.manifest.get('retained_native_boundaries', [])
        index = implementation.get('native_callable_index')
        if type(index) is not int or not 0 <= index < len(inventory):
            raise ValueError('live native handle has no observed source boundary')
        record = inventory[index]
        runtime_identities = record.get('source_runtime_identities', [])
        if not runtime_identities or not runtime_identities[0] or context.get('runtime_identity') != runtime_identities[0]:
            raise ValueError('live native runtime/math context differs from observed boundary')
        source_contracts = record.get('source_argument_contracts', [])
        argument_contract = source_contracts[0] if source_contracts else record.get('argument_contract')
        if (implementation.get('name') != record.get('name') or
                implementation.get('reference_source_sha256') != record.get('reference_source_sha256') or
                implementation.get('argument_contract') != argument_contract):
            raise ValueError('live native callable/arguments differ from observed boundary')
        occurrences = record.get('source_occurrences', [])
        members = context.get('source_node_ids')
        # First-call argument metadata cannot justify later specializations.
        if not occurrences or members != occurrences[0]:
            raise ValueError('live native binding requires the first observed argument specialization')
        nodes = {n.id: n for n in source.nodes}
        if not members or len(set(members)) != len(members) or any(i not in nodes for i in members):
            raise ValueError('invalid native region occurrence membership')
        selected = [nodes[i] for i in members]
        if len({n.rank for n in selected}) != 1:
            raise ValueError('local native region cannot span ranks')
        boundary = selected[0]
        if not boundary.attrs.get('native_callable_boundary') or boundary.attrs.get('native_callable_index') != index:
            raise ValueError('native region does not start at its observed callable boundary')
        if boundary.attrs.get('status') != 'returned' or boundary.attrs.get('native_region_member_node_ids') != members:
            raise ValueError('native callable did not complete with the declared members')
        for member in selected:
            if member.kind != 'compute' or 'communication' in member.attrs or 'c10d' in member.name:
                raise ValueError('single-GPU native region contains distributed communication')
        yield selected, node.attrs['cost_key'], binding


def _contract_region(workload, members, key, binding):
    by_id = {node.id: node for node in workload.nodes}
    selected = [by_id[member] for member in members]
    owned = set(members)
    children = {member: set() for member in members}
    for node in selected:
        for parent in node.deps:
            if parent in owned:
                children[parent].add(node.id)
    terminals = {member for member in members if not children[member]}
    for node in workload.nodes:
        if node.id not in owned and (set(node.deps) & owned) - terminals:
            raise ValueError('native region has an externally visible intermediate dependency')
    root = selected[0]
    placement = workload.manifest.get('rank_placement', {})
    resources = set()
    for node in selected:
        device = placement.get(str(node.rank), placement.get(node.rank, node.rank))
        resources.update(node.attrs.get('resources', ['gpu:'+str(device)]))
    root.deps = sorted({parent for node in selected for parent in node.deps if parent not in owned})
    root.attrs.update(cost_key=key, native_measurement=deepcopy(binding),
                      contracted_node_ids=list(members),
                      resources=sorted(resources))
    workload.nodes = [node for node in workload.nodes if node.id not in owned or node is root]
    for node in workload.nodes:
        if node is not root:
            node.deps = sorted({root.id if parent in owned else parent for parent in node.deps})
    for storage in workload.storage:
        if storage.producer in owned:
            storage.producer = root.id
        storage.consumers = sorted({root.id if user in owned else user for user in storage.consumers})
    # Collapsing internal events retains their buffers for the whole region.
    # That is a conservative projection, never an unavoidable live-byte lower bound.
    workload.manifest.setdefault('memory', {})['storage_evidence'] = 'unverified'
    validate(workload.nodes, workload.storage, workload.manifest)
    return root


def _compute_bindings(sources, measurement):
    from .operator_cases import operator_case_workload
    inventory = measurement.manifest['operator_case_catalog']
    fingerprint = inventory.get('source_workload_sha256')
    matches = [source for source in sources if workload_fingerprint(source) == fingerprint]
    if len(matches) != 1:
        raise ValueError('operator source workload fingerprint is missing, stale or ambiguous')
    source = matches[0]
    ids = [node.attrs['operator_case']['case_id'] for node in measurement.nodes]
    expected = operator_case_workload(source, ids)
    if workload_fingerprint(expected) != workload_fingerprint(measurement):
        raise ValueError('operator measurement workload differs from verified source cases')
    for node in measurement.nodes:
        case = node.attrs['operator_case']
        if any(not value['device'].startswith('cuda') for value in case['arguments']['args']):
            raise ValueError('native GPU cost binding requires captured CUDA operands')
        binding = _descriptor(node)
        for occurrence in case['occurrences']:
            yield occurrence, node.attrs['cost_key'], binding, None


def _communication_bindings(sources, measurement):
    from .communication_cases import communication_case_workload
    inventory = measurement.manifest['communication_cases']
    recorded = inventory.get('source_workload_sha256_by_rank')
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError('communication source workload fingerprints are missing')
    selected = []
    actual = {}
    for source in sources:
        relevant = set(map(str, source.manifest['ranks'])) & set(recorded)
        if relevant:
            selected.append(source)
            for rank in source.manifest['ranks']:
                actual[str(rank)] = workload_fingerprint(source)
    if actual != recorded:
        raise ValueError('communication source workload fingerprint mismatch')
    if not selected:
        raise ValueError('communication source workloads are missing')
    keys = {}
    for node in measurement.nodes:
        binding = _descriptor(node)
        keys[binding['context']['case']['case_id']] = (node.attrs['cost_key'], binding)
    topologies = [binding['context']['measurement_topology'] for _, binding in keys.values()]
    if not topologies or any(value != topologies[0] for value in topologies):
        raise ValueError('communication measurement topology is inconsistent')
    expected = communication_case_workload(selected[0], list(keys), topology=topologies[0],
                                           peer_workloads=selected[1:])
    # Case identities exclude occurrence ordering. Compare executable nodes by
    # rank/key rather than requiring a particular input-file ordering.
    def descriptors(workload):
        return sorted((node.rank, node.attrs['cost_key'], node.name, node.kind,
                       tuple(node.deps), digest(_descriptor(node))) for node in workload.nodes)
    if descriptors(measurement) != descriptors(expected) or measurement.storage:
        raise ValueError('communication measurement differs from verified native cases')
    original_cases = {case['case_id']: case for case in expected.manifest['communication_cases']['cases']}
    supplied_cases = {case['case_id']: case for case in inventory['cases']}
    for case_id, (key, binding) in keys.items():
        expected_case = original_cases[case_id]
        def ordered(case):
            value = deepcopy(case)
            value['source_nodes'] = sorted(value['source_nodes'], key=lambda n: (n['rank'], n['node_id']))
            return value
        if case_id not in supplied_cases or ordered(supplied_cases[case_id]) != ordered(expected_case):
            raise ValueError('communication occurrence inventory differs from source')
        for occurrence in expected_case['source_nodes']:
            yield occurrence, key, binding, expected_case


def bind_cases(sources, measurements, config=None):
    """Return a derived prepared workload with audited native cost-key bindings.

    ``config`` accepts only target communication topology and a global useful
    token count. Topology must exactly match the measured placement declaration.
    Communication nodes must already carry kind=collective, a rendezvous,
    participants and communication_completion='device'. Unresolved issue/wait
    relationships are reported as unbound rather than guessed from node order.
    """
    sources, measurements = list(sources), list(measurements)
    config = deepcopy(config or {})
    if set(config) - {'communication_topology', 'useful_tokens'}:
        raise ValueError('unknown case binding configuration')
    if not sources or not measurements:
        raise ValueError('source and measurement workloads are required')
    for source in sources:
        if source.manifest.get('capture_status', 'complete') != 'complete':
            raise ValueError('case binding requires complete source capture')
        if source.manifest.get('memory', {}).get('evidence_binding'):
            raise ValueError('bind native cases before attaching memory evidence')
    options = {'chrome_traces': []}
    if 'useful_tokens' in config:
        options['useful_tokens'] = config['useful_tokens']
    result = prepare(sources, options)
    original = {(node.rank, node.id): node for source in sources for node in source.nodes}
    prepared = {(node.rank, node.attrs['source_node_id']): node for node in result.nodes}
    fingerprints = {str(rank): workload_fingerprint(source)
                    for source in sources for rank in source.manifest['ranks']}
    bound, skipped, claimed = [], [], set()
    gaps = set(result.manifest.get('unresolved', []))
    for measurement in measurements:
        validate(measurement.nodes, measurement.storage, measurement.manifest)
        if measurement.hardware != result.hardware:
            raise ValueError('measurement hardware differs from target source')
        if measurement.manifest.get('scope') == 'isolated live native callable measurement':
            for members, key, binding in _live_bindings(sources, measurement):
                identities = [(member.rank, member.id) for member in members]
                if any(identity in claimed for identity in identities):
                    raise ValueError('source region overlaps an already bound occurrence')
                claimed.update(identities)
                root = _contract_region(result, [prepared[identity].id for identity in identities], key, binding)
                audit = dict(rank=members[0].rank, source_node_id=members[0].id,
                             source_node_ids=[member.id for member in members], node_id=root.id,
                             source_workload_sha256=binding['context']['source_workload_sha256'],
                             measurement_workload_sha256=workload_fingerprint(measurement),
                             source_cost_key=members[0].attrs.get('cost_key'), cost_key=key)
                root.attrs['measurement_binding'] = deepcopy(audit)
                bound.append(audit)
                gaps.add('Live native region costs use the retained state/reset contract; distributed equivalence and physical allocations remain unverified.')
            continue
        if 'operator_case_catalog' in measurement.manifest:
            bindings = _compute_bindings(sources, measurement)
        elif 'communication_cases' in measurement.manifest:
            bindings = _communication_bindings(sources, measurement)
        else:
            raise ValueError('unsupported measurement workload; native case inventory is required')
        measurement_sha = workload_fingerprint(measurement)
        for occurrence, key, binding, communication in bindings:
            identity = occurrence['rank'], occurrence['node_id']
            if identity not in original:
                raise ValueError('measurement references an absent source occurrence')
            if identity in claimed:
                raise ValueError('source occurrence already bound by another measurement')
            claimed.add(identity)
            source_node, node = original[identity], prepared[identity]
            audit = dict(rank=identity[0], source_node_id=identity[1], node_id=node.id,
                         source_workload_sha256=fingerprints[str(identity[0])],
                         measurement_workload_sha256=measurement_sha,
                         source_cost_key=source_node.attrs.get('cost_key'), cost_key=key)
            if communication is not None:
                topology = binding['context']['measurement_topology']
                if config.get('communication_topology') != topology:
                    skipped.append(dict(audit, reason='target communication topology is not bound to measured placement'))
                    continue
                event_kind_matches = (node.kind in ('send', 'recv') if communication['operation'] == 'p2p'
                                      else node.kind == 'collective')
                if (not event_kind_matches or not node.attrs.get('rendezvous') or
                        node.attrs.get('participants') != communication['participants'] or
                        node.attrs.get('communication_completion') != 'device'):
                    skipped.append(dict(audit, reason='source communication has no verified collective device-completion endpoint'))
                    continue
                gaps.add('Collective costs describe isolated completion; concurrent interference remains unqualified.')
            else:
                gaps.add('Bound native operator costs use synthetic inputs and isolated execution; compiled correspondence and interference remain separate.')
            node.attrs['cost_key'] = key
            node.attrs['native_measurement'] = deepcopy(binding)
            node.attrs['measurement_binding'] = deepcopy(audit)
            bound.append(audit)
    unbound = [dict(rank=node.rank, source_node_id=node.attrs['source_node_id'], node_id=node.id,
                    name=node.name) for node in result.nodes
               if 'measurement_binding' not in node.attrs and not node.attrs.get('zero_cost_semantic')]
    if unbound:
        gaps.add('Native case binding is partial: some execution occurrences have no verified measurement binding.')
    report = dict(schema='native-case-binding', version=1, source_workload_sha256_by_rank=fingerprints,
                  bound=bound, skipped=skipped, unbound=unbound,
                  scope='implementation correspondence; execution dependencies and physical memory gaps are retained')
    result.manifest['case_binding'] = report
    result.manifest['unresolved'] = sorted(gaps)
    validate(result.nodes, result.storage, result.manifest)
    return result
