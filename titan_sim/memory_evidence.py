"""Target-bound memory contracts and empirical native observations.

No report or inventory is a predictive upper bound. Only explicitly referenced
producer contracts can contribute allocation bounds or completeness assertions.
"""
from copy import deepcopy
from .identity import digest as canonical_digest, workload_fingerprint as workload_digest

from .memory import ALLOCATION_CATEGORIES, COVERAGE_CATEGORIES, LIFECYCLE_PHASES
from .workload import Workload

SCHEMA = 'titan_sim.memory_evidence.v1'
CONTEXT_KEYS = ('source', 'target', 'runtime', 'placement', 'phases', 'implementation')


def _context(value, workload):
    if not isinstance(value, dict) or set(value) != set(CONTEXT_KEYS):
        raise ValueError('applicability requires source, target, runtime, placement, phases and implementation')
    if any(not value[key] for key in CONTEXT_KEYS):
        raise ValueError('applicability fields must be explicit and nonempty')
    if value['placement'] != workload.manifest.get('rank_placement'):
        raise ValueError('applicability placement mismatch')
    phases = value['phases']
    if not isinstance(phases, list) or any(not isinstance(p, str) or not p for p in phases) or len(set(phases)) != len(phases):
        raise ValueError('applicability phases must be distinct nonempty names')
    required = {'schedule'}
    if workload.manifest.get('memory', {}).get('scope', 'recipe') == 'recipe':
        required.update(LIFECYCLE_PHASES)
    if not required <= set(phases):
        raise ValueError('applicability omits required lifecycle/schedule phases')
    canonical_digest(value)


def build_evidence_bundle(workload, *, applicability, observations=(), allocations=(),
                          coverage=(), producer_contracts=(), evidence_refs=(), phase_bounds=()):
    """Build a JSON-only bundle; checksums detect stale data, not truthfulness."""
    _context(applicability, workload)
    result = deepcopy(dict(schema=SCHEMA, workload_sha256=workload_digest(workload),
        applicability=applicability, observations=list(observations), allocations=list(allocations),
        coverage=list(coverage), producer_contracts=list(producer_contracts), evidence_refs=list(evidence_refs),
        phase_bounds=list(phase_bounds)))
    result['sha256'] = canonical_digest(result)
    _validate_bundle(workload, result, applicability)
    return result


def _bytes(value):
    return value is None or (type(value) is int and value >= 0)


def _validate_bundle(workload, bundle, applicability):
    if bundle.get('schema') != SCHEMA:
        raise ValueError('unsupported memory evidence schema')
    if bundle.get('sha256') != canonical_digest({k: v for k, v in bundle.items() if k != 'sha256'}):
        raise ValueError('memory evidence checksum mismatch')
    if bundle.get('workload_sha256') != workload_digest(workload):
        raise ValueError('memory evidence workload hash mismatch')
    _context(applicability, workload)
    if bundle['applicability'] != applicability:
        raise ValueError('memory evidence applicability mismatch')
    refs = {}
    for ref in bundle['evidence_refs']:
        if not isinstance(ref, dict) or not ref.get('id') or ref['id'] in refs:
            raise ValueError('evidence references require unique IDs')
        if not isinstance(ref.get('sha256'), str) or len(ref['sha256']) != 64 or any(c not in '0123456789abcdef' for c in ref['sha256']):
            raise ValueError('evidence reference requires SHA256')
        refs[ref['id']] = ref
    contracts = {}
    for contract in bundle['producer_contracts']:
        if not isinstance(contract, dict) or not contract.get('id') or contract['id'] in contracts:
            raise ValueError('producer contracts require unique IDs')
        if not contract.get('producer') or not contract.get('statement') or contract.get('evidence_ref') not in refs:
            raise ValueError('producer contract requires producer, statement and evidence reference')
        if contract.get('applicability_sha256') != canonical_digest(applicability):
            raise ValueError('producer contract applicability mismatch')
        contracts[contract['id']] = contract
    devices = set(applicability['placement'].values())
    devices.update(workload.hardware.get('devices', {}))
    phases = set(applicability['phases'])
    by_node = {node.id: node for node in workload.nodes}
    storages = {storage.id: storage for storage in workload.storage}
    generations, claimed_storage, claimed_roots = set(), set(), {}
    existing_allocation_ids = {row.get('id') for row in workload.manifest.get('memory', {}).get('allocations', [])}
    prior_aliases = workload.manifest.get('memory', {}).get('storage_aliases', {})
    def storage_root(sid):
        seen = set()
        while sid in prior_aliases:
            if sid in seen:
                raise ValueError('storage alias cycle')
            seen.add(sid)
            sid = prior_aliases[sid]
        return sid
    for record in bundle['allocations']:
        if record.get('kind') not in ('exact', 'conditional') or record.get('contract_id') not in contracts:
            raise ValueError('allocation bound requires explicit producer contract')
        if record.get('evidence_ref') not in refs:
            raise ValueError('allocation bound requires evidence reference')
        identity = record.get('allocation_id'), record.get('generation')
        if not isinstance(identity[0], str) or not identity[0] or type(identity[1]) is not int or identity[1] < 0 or identity in generations:
            raise ValueError('allocation generations must be distinct nonnegative identities')
        generations.add(identity)
        if identity[0] in existing_allocation_ids or ('evidence:' + identity[0] + ':' + str(identity[1])) in existing_allocation_ids:
            raise ValueError('allocation identity conflicts with existing memory allocation; do not count it twice')
        if record.get('device') not in devices or record.get('phase') not in phases:
            raise ValueError('allocation device or phase outside applicability')
        if record.get('category') not in ALLOCATION_CATEGORIES:
            raise ValueError('unknown allocation category')
        if 'lower_bytes' not in record or 'upper_bytes' not in record or not _bytes(record['lower_bytes']) or not _bytes(record['upper_bytes']):
            raise ValueError('allocation bounds require nonnegative bytes or explicit null')
        low, high = record['lower_bytes'], record['upper_bytes']
        if low is not None and high is not None and high < low:
            raise ValueError('allocation upper bound below lower bound')
        if record['kind'] == 'exact' and (low is None or low != high):
            raise ValueError('exact allocation needs equal known bounds')
        if record['kind'] == 'conditional' and (not isinstance(record.get('conditions'), list) or not record['conditions'] or any(not isinstance(item, str) or not item for item in record['conditions'])):
            raise ValueError('conditional allocation requires explicit conditions')
        if type(record.get('persistent')) is not bool or 'creation_node_id' not in record or not isinstance(record.get('completion_node_ids'), list):
            raise ValueError('allocation requires persistence and creation/completion node IDs')
        events = record['completion_node_ids'] + ([] if record['creation_node_id'] is None else [record['creation_node_id']])
        if any(type(event) is not int or event not in by_node for event in events):
            raise ValueError('allocation references unknown creation/completion node')
        if record['creation_node_id'] is not None and applicability['placement'][str(by_node[record['creation_node_id']].rank)] != record['device']:
            raise ValueError('allocation creation node physical placement mismatch')
        ids = record.get('storage_ids', [])
        if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(sid not in storages or sid in claimed_storage for sid in ids):
            raise ValueError('storage IDs must identify one distinct allocation generation')
        claimed_storage.update(ids)
        for sid in ids:
            root = storage_root(sid)
            if root in claimed_roots and claimed_roots[root] != identity:
                raise ValueError('existing storage aliases cannot identify separate allocation generations')
            claimed_roots[root] = identity
        if ids and (low != high or low is None or any(storages[sid].size != low or applicability['placement'][str(storages[sid].rank)] != record['device'] for sid in ids)):
            raise ValueError('storage aliases require identical exact sizes and physical placement')
        if record['phase'] != 'schedule':
            raise ValueError('allocation lifetimes currently require schedule phase; retain lifecycle measurements as observations')
    for row in bundle['phase_bounds']:
        if (row.get('device') not in devices or row.get('phase') not in LIFECYCLE_PHASES or
                row.get('contract_id') not in contracts or row.get('evidence_ref') not in refs or
                row.get('scope') != 'whole_device' or row.get('kind') != 'exact'):
            raise ValueError('lifecycle bound requires whole-device exact producer contract/evidence')
        if row.get('lower_bytes') is None or row.get('upper_bytes') is None or not _bytes(row['lower_bytes']) or not _bytes(row['upper_bytes']) or row['upper_bytes'] < row['lower_bytes']:
            raise ValueError('lifecycle bounds require finite ordered bytes')
    for row in bundle['coverage']:
        if (row.get('device') not in devices or row.get('phase') not in phases or
                row.get('category') not in COVERAGE_CATEGORIES or type(row.get('complete')) is not bool or
                row.get('contract_id') not in contracts or row.get('evidence_ref') not in refs):
            raise ValueError('coverage requires applicable device/phase/category and producer contract/evidence')
        if row.get('complete') and not row.get('statement'):
            raise ValueError('complete coverage requires a scoped completeness statement')
    for observation in bundle['observations']:
        if observation.get('kind') != 'observation' or observation.get('physical_peak_upper_bytes') is not None:
            raise ValueError('observations cannot assert physical upper bounds')


def bind_memory_evidence(workload, bundle, *, applicability):
    """Return a copy retaining all original gaps, with a revalidated binding."""
    if 'evidence_binding' in workload.manifest.get('memory', {}):
        raise ValueError('workload already has memory evidence; bind the original workload')
    _validate_bundle(workload, bundle, applicability)
    result = deepcopy(workload)
    memory_existed = 'memory' in result.manifest
    memory = result.manifest.setdefault('memory', {})
    memory['evidence_binding'] = deepcopy(dict(bundle=bundle, applicability=applicability,
                                              memory_existed=memory_existed))
    return result


def _base_workload(nodes, storage, hardware, manifest):
    base = Workload(deepcopy(nodes), deepcopy(storage), deepcopy(manifest), deepcopy(hardware))
    binding = base.manifest['memory'].pop('evidence_binding')
    if not binding['memory_existed']:
        del base.manifest['memory']
    _validate_bundle(base, binding['bundle'], binding['applicability'])
    return base, binding['bundle']


def assess_evidence_memory(nodes, storage, timing, hardware, manifest):
    """Realize event lifetimes then delegate envelopes to the existing calculator.

    Per-device calls preserve global gaps while allowing localized coverage. No
    measured maxima enter these predictive envelopes.
    """
    from .memory import assess_memory
    base, bundle = _base_workload(nodes, storage, hardware, manifest)
    original = deepcopy(base)
    context = bundle['applicability']
    phases = context['phases']
    memory = base.manifest.setdefault('memory', {})
    memory.setdefault('unresolved', [])
    aliases = memory.setdefault('storage_aliases', {})
    extras = memory.setdefault('allocations', [])
    by_storage = {item.id: item for item in base.storage}
    exact_storage, conditions = set(), []
    local_issues = {device: [] for device in set(context['placement'].values()) | set(hardware.get('devices', {}))}
    finish = max((value[1] for value in timing.values()), default=0)
    def event_time(event, side):
        value = timing.get(str(event), timing.get(event))
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('missing allocation event timing')
        return value[side]
    for row in bundle['allocations']:
        if row['kind'] == 'conditional':
            conditions.extend(row['conditions'])
        ids = row.get('storage_ids', [])
        first = row['creation_node_id']
        start = 0 if first is None else event_time(first, 0)
        ends = row['completion_node_ids']
        unknown_end = not row['persistent'] and not ends
        end = finish if row['persistent'] or unknown_end else max(event_time(event, 1) for event in ends)
        if first is not None:
            end = max(end, event_time(first, 1))
        if unknown_end:
            local_issues[row['device']].append('allocation completion remains unknown: ' + row['allocation_id'])
            # Unknown retention is an open upper bound, never evidence that
            # these bytes coexist with every later allocation. Keep a separate
            # zero-lower unbounded marker and attest only creation completion.
            extras.append(dict(id='evidence-lifetime:' + row['allocation_id'] + ':' + str(row['generation']),
                device=row['device'], category='external_allocations', lower_bytes=0, upper_bytes=None))
            end = start if first is None else event_time(first, 1)
        if ids:
            root = ids[0]
            seen_aliases = set()
            while root in aliases:
                if root in seen_aliases:
                    raise ValueError('storage alias cycle')
                seen_aliases.add(root)
                root = aliases[root]
            for sid in ids:
                if sid == root:
                    continue
                if sid in aliases and aliases[sid] != root:
                    raise ValueError('evidence alias conflicts with existing storage alias')
                aliases[sid] = root
            for sid in ids:
                item = by_storage[sid]
                original_users = item.consumers + ([] if item.producer is None else [item.producer])
                original_start = 0 if item.producer is None else event_time(item.producer, 0)
                original_end = finish if item.persistent or not original_users else max(event_time(event, 1) for event in original_users)
                # Retain previous users and lifetime assertions. A narrower
                # attested interval supplies a lower bound through the alias
                # below but cannot promote a longer projected lifetime.
                if row['kind'] == 'exact' and not unknown_end and start <= original_start and end >= original_end:
                    exact_storage.add(sid)
                item.consumers = sorted(set(original_users + ends))
                if first is None or (item.producer is not None and start < original_start):
                    item.producer = first
                item.persistent |= row['persistent']
        if row['lower_bytes'] is None:
            local_issues[row['device']].append('allocation lower bound remains unknown: ' + row['allocation_id'])
            continue
        extra = dict(id='evidence:' + row['allocation_id'] + ':' + str(row['generation']),
                     device=row['device'], category=row['category'] if row['category'] != 'storage' else 'external_allocations',
                     lower_bytes=row['lower_bytes'], upper_bytes=row['upper_bytes'], start_seconds=start)
        if not row['persistent']:
            extra['end_seconds'] = end
            extra['end_inclusive'] = any(event_time(event, 0) == end for event in ends)
        if ids:
            extra['storage_id'] = ids[0]
        extras.append(extra)
    for row in bundle['phase_bounds']:
        memory.setdefault('lifecycle_peaks', []).append({key: row[key] for key in ('device', 'phase', 'lower_bytes', 'upper_bytes')})
    devices = set(context['placement'].values()) | set(hardware.get('devices', {}))
    required = COVERAGE_CATEGORIES if memory.get('scope', 'recipe') == 'recipe' else ALLOCATION_CATEGORIES
    audit, per_device, issues, assumptions = {}, {}, [], []
    # Allocation categories require a statement for every requested phase. The
    # legacy lifecycle flags additionally require their own named phase.
    for device in sorted(devices):
        local = deepcopy(base.manifest)
        local_memory = local['memory']
        local_memory['unresolved'].extend(local_issues[device])
        coverage = {}
        audit[device] = {}
        for category in required:
            needed = [category] if category in LIFECYCLE_PHASES else phases
            checks = {}
            for phase in needed:
                matches = [row for row in bundle['coverage'] if (row['device'], row['phase'], row['category']) == (device, phase, category)]
                checks[phase] = bool(matches) and all(row['complete'] for row in matches)
            coverage[category] = all(checks.values())
            audit[device][category] = checks
        local_memory['coverage'] = coverage
        selected_storage = [item for item in base.storage if context['placement'][str(item.rank)] == device]
        # A contract may promote backing evidence only for explicitly bound
        # storage, never through the category-complete boolean alone.
        if all(item.id in exact_storage for item in selected_storage) and coverage.get('storage'):
            local_memory['storage_evidence'] = 'exact'
        result = assess_memory(base.nodes, base.storage, timing, base.hardware, local)
        per_device[device] = result['per_device'][device]
        issues.extend('device ' + device + ': ' + issue for issue in result['memory_unresolved'])
        assumptions.extend(result['memory_assumptions'])
    compatibility = {category: all(all(audit[device][category].values()) for device in devices) for category in required}
    for item in per_device.values():
        lower, upper, budget, capacity = (item[k] for k in ('peak_lower_bytes', 'peak_upper_bytes', 'budget_bytes', 'capacity_bytes'))
        item['budget_outcome'] = 'overflow' if lower > budget else 'within_budget' if upper is not None and upper <= budget else 'unknown'
        item['physical_capacity_outcome'] = 'oom_lower_bound' if lower > capacity else 'within_capacity' if upper is not None and upper <= capacity else 'unknown'
    not_fit = any(item['budget_outcome'] == 'overflow' for item in per_device.values())
    fits = bool(per_device) and all(item['budget_outcome'] == 'within_budget' for item in per_device.values())
    if conditions:
        # Reassess only unconditional evidence. Conditional bytes cannot prove
        # unconditional fit, overlap, budget overflow or physical-capacity OOM.
        unconditional_bundle = deepcopy(bundle)
        unconditional_bundle['allocations'] = [row for row in bundle['allocations'] if row['kind'] != 'conditional']
        unconditional_bundle['sha256'] = canonical_digest({key: value for key, value in unconditional_bundle.items() if key != 'sha256'})
        bound = bind_memory_evidence(original, unconditional_bundle, applicability=context)
        result = assess_evidence_memory(bound.nodes, bound.storage, timing, bound.hardware, bound.manifest)
        for device in {row['device'] for row in bundle['allocations'] if row['kind'] == 'conditional'}:
            item = result['per_device'][device]
            item['peak_upper_bytes'] = item['modeled_peak_upper_bytes'] = item['scheduled_peak_upper_bytes'] = None
            item['peak_upper_time_seconds'] = None
            item['peak_upper_categories_bytes'] = {}
            item['budget_outcome'] = 'overflow' if item['peak_lower_bytes'] > item['budget_bytes'] else 'unknown'
            item['physical_capacity_outcome'] = 'oom_lower_bound' if item['peak_lower_bytes'] > item['capacity_bytes'] else 'unknown'
            result['memory_unresolved'].append('device ' + device + ': allocation bound conditions are unverified')
        result.update(memory_verdict='inconclusive', memory_bound_kind='conditional',
            conditional_memory_verdict='estimated_not_fit' if not_fit else 'estimated_fit' if fits else 'inconclusive',
            conditional_assumptions=conditions, conditional_per_device=per_device,
            evidence_sha256=bundle['sha256'])
        return result
    return dict(memory_verdict='estimated_not_fit' if not_fit else 'estimated_fit' if fits else 'inconclusive',
                per_device=per_device, memory_unresolved=list(dict.fromkeys(issues)),
                memory_assumptions=list(dict.fromkeys(assumptions)), memory_scope=memory.get('scope', 'recipe'),
                storage_evidence='per_device_evidence_binding', coverage_audit=audit,
                compatibility_coverage=compatibility, conditional_assumptions=conditions,
                memory_bound_kind='producer_contract', conditional_memory_verdict=None,
                evidence_sha256=bundle['sha256'], observations=deepcopy(bundle['observations']))


def observation_from_report(report):
    """Adapt a MemoryObserver report without changing its evidentiary meaning."""
    if report.get('schema') != 'titan_sim.memory_observation.v1':
        raise ValueError('expected a MemoryObserver report')
    if report.get('provenance_sha256') != canonical_digest(report.get('provenance')):
        raise ValueError('observation provenance checksum mismatch')
    return dict(kind='observation', method='MemoryObserver', report_sha256=canonical_digest(report),
                rank=report.get('rank'), device=report.get('physical_device'),
                status=report.get('status'), runtime=deepcopy(report.get('runtime')),
                provenance=deepcopy(report.get('provenance')), phases=deepcopy(report.get('phases', [])),
                samples=deepcopy(report.get('samples', [])),
                process_allocator_peak=deepcopy(report.get('process_allocator_peak')),
                physical_peak_upper_bytes=None, supports_unseen_configuration_fit=False,
                unresolved=deepcopy(report.get('unresolved', [])))


def inspect_native_state(*, models, optimizers, torch_module=None):
    """Snapshot real local native backing storage, including lazy optimizer state.

    Caller supplies original native objects. No state is initialized or changed;
    DTensor is inspected locally. IDs are valid within this sample only.
    """
    if torch_module is None:
        import torch as torch_module
    allocations, by_identity, unresolved = [], {}, []
    def visit(value, category, path):
        if isinstance(value, dict):
            for index, child in enumerate(value.values()):
                visit(child, category, path + [str(index)])
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, category, path + [str(index)])
        elif torch_module.is_tensor(value):
            try:
                local = value.to_local() if callable(getattr(value, 'to_local', None)) else value
                if local.device.type == 'meta':
                    raise ValueError('meta tensor has no native allocation')
                backing = local.untyped_storage()
                identity = (str(local.device), int(backing._cdata))
                if identity not in by_identity:
                    entry = dict(id='sample-storage:' + str(len(allocations)), device=str(local.device),
                                 bytes=int(backing.nbytes()), categories=[], aliases=[])
                    allocations.append(entry)
                    by_identity[identity] = entry
                entry = by_identity[identity]
                if category not in entry['categories']:
                    entry['categories'].append(category)
                entry['aliases'].append(path)
            except Exception as error:
                unresolved.append(category + ': ' + type(error).__name__ + ': ' + str(error))
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            unresolved.append(category + ': unsupported native state object ' + type(value).__name__)
    for index, model in enumerate(models):
        for name, parameter in model.named_parameters():
            visit(parameter, 'parameters', ['model', str(index), name])
            visit(parameter.grad, 'gradients', ['model', str(index), name, 'grad'])
        for name, buffer in model.named_buffers():
            visit(buffer, 'buffers', ['model', str(index), name])
    for index, optimizer in enumerate(optimizers):
        if not hasattr(optimizer, 'state'):
            unresolved.append('optimizer.state unavailable: ' + str(index))
        else:
            visit(optimizer.state, 'optimizer_state', ['optimizer', str(index)])
    return dict(kind='observation', method='native_state_inventory', allocations=allocations,
                unique_storage_bytes=sum(entry['bytes'] for entry in allocations),
                physical_peak_upper_bytes=None, supports_unseen_configuration_fit=False,
                scope='sample only; IDs are not allocation generations; not additive to allocator totals',
                unresolved=unresolved)
