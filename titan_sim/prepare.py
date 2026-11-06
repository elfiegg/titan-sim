"""Evidence-driven, CPU-only preparation of rank-local native workloads.

``prepare(workloads, config=None)`` accepts Workload objects or JSON paths.
Configuration is JSON: ``chrome_traces`` contains {rank, step, path} entries;
paths must match SHA256 artifacts registered by capture. GPU events are matched
through Chrome External id and CPU Record function id to execution-trace rf_id.
Only unambiguous matches are lowered. Same-stream event order is retained;
cross-stream synchronization cannot be reconstructed by timestamp comparison.

Optional explicit assertions use original node references {rank, id}:
* dependencies: [{node: ref, deps: [ref, ...]}] replaces incoming dependencies.
* regions: [{nodes: [ref, ...], implementation: {...}, context: {...}, name: str}]
  contracts an exact, externally verified execution region to one cost event.
* communications: [{kind: 'collective' | 'send_recv', nodes: [ref, ...],
  group: str, sequence: int, operation: str, participants: [rank, ...],
  payload_bytes: int}]. For send_recv the first node sends, the second receives.
* allocations: [{rank: int, id: str, source_storage_ids: [str, ...], size: int,
  producer: ref | null, consumers: [ref, ...], persistent: bool, category: str}]
  replaces named observed storage spans with a verified backing allocation.
* resolutions: [{gap: exact original string, reason: str,
  scope: 'integration_assertion', evidence: {path: str, sha256: str}}] records a
  caller's specific gap resolution after checking the referenced evidence file.
  These are labeled integration assertions, never empirical verification.

Assertions require config.evidence={source: nonempty str, sha256: 64 hex chars}.
This records the caller's evidence identity, not independent proof of fidelity.
Unknown gaps are preserved. The function never imports PyTorch or executes code
recorded in a workload. Input objects are not modified.
"""
from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .core import Node, Storage, validate
from .memory import classify_storage_evidence
from .workload import Workload, read_workload


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def native_cost_key(implementation, context):
    """Exact implementation/context identity shared with native measurement."""
    if not isinstance(implementation, dict) or not implementation or not isinstance(context, dict) or not context:
        raise ValueError('native implementation and context must be nonempty objects')
    return 'native-runtime:' + _digest(dict(implementation=implementation, context=context))


def _nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + ' must be a nonnegative integer')
    return value


def _evidence(config, required=False):
    if required or any(config.get(k) for k in ('dependencies', 'regions', 'communications', 'allocations')):
        evidence = config.get('evidence', {})
        checksum = evidence.get('sha256', '')
        if (not evidence.get('source') or not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in '0123456789abcdefABCDEF' for c in checksum)):
            raise ValueError('explicit preparation assertions require evidence source and sha256')


def _chrome(workload, entries, maps, gaps):
    """Lower only directly correlated GPU events, retaining unmatched nodes."""
    nodes = {n.id: n for n in workload.nodes}
    lowered, replacement, streams = [], {}, defaultdict(list)
    direct = {}
    verified_direct = set()
    for node in nodes.values():
        binding = node.attrs.get('native_measurement', {})
        if binding.get('kind') == 'chrome_gpu_event':
            identity = (node.rank, node.attrs.get('step'), binding.get('artifact_sha256'), binding.get('event_index'))
            if identity in direct:
                raise ValueError('same GPU event bound to multiple captured nodes')
            direct[identity] = node
    next_id = max(nodes, default=-1) + 1
    used = set()
    for entry in entries:
        rank, step = entry['rank'], entry['step']
        if (rank, step) in used:
            raise ValueError('duplicate Chrome trace for rank/step')
        used.add((rank, step))
        path = Path(entry['path'])
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        matching_artifacts = [a for m in workload.manifest['source_manifests'] if rank in m['ranks']
                              for a in m.get('native_trace_artifacts', []) if a['sha256'] == checksum]
        if not matching_artifacts:
            raise ValueError('Chrome trace is not a captured SHA256 artifact')
        if not any(a.get('step') == step or Path(a['path']).name == f'step{step}.chrome.json'
                   for a in matching_artifacts):
            raise ValueError('Chrome artifact does not match the declared capture step')
        events = json.loads(data).get('traceEvents', [])
        candidates = defaultdict(list)
        for node in nodes.values():
            if node.rank == rank and node.attrs.get('step') == step:
                rf_id = node.attrs.get('execution_trace_attrs', {}).get('rf_id')
                if rf_id is not None:
                    candidates[str(rf_id)].append(node)
        external = defaultdict(set)
        for event in events:
            args = event.get('args', {})
            if event.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                if 'External id' in args and 'Record function id' in args:
                    external[str(args['External id'])].add(str(args['Record function id']))
        for index, event in enumerate(events):
            if event.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset') or event.get('ph') != 'X':
                continue
            args = event.get('args', {})
            existing = direct.get((rank, step, checksum, index))
            if existing is not None:
                binding = existing.attrs['native_measurement']
                implementation = dict(name=event['name'], category=event['cat'],
                    launch={k: args[k] for k in ('grid', 'block', 'registers per thread',
                                               'shared memory', 'bytes') if k in args})
                if (binding.get('event_name') != event['name'] or binding.get('event_category') != event['cat']
                        or binding.get('implementation') != implementation):
                    raise ValueError('captured GPU binding does not match retained activity')
                context = binding.get('context', {})
                if context.get('hardware_id') != workload.hardware['id']:
                    raise ValueError('captured GPU binding hardware mismatch')
                if not context.get('operator_cost_key'):
                    occurrence = dict(occurrence_only=True, artifact_sha256=checksum, event_index=index,
                                      rank=rank, step=step)
                    if any(context.get(key) != value for key, value in occurrence.items()):
                        raise ValueError('GPU activity without an operator signature requires an exact occurrence binding')
                    gaps.add('Occurrence-only GPU costs cannot be reused across shapes, scalars, recipes or new executions.')
                if existing.attrs.get('cost_key') != native_cost_key(implementation, context):
                    raise ValueError('captured GPU binding cost key mismatch')
                if not all(type(event.get(k)) in (int, float) and math.isfinite(event[k])
                           and event[k] >= 0 for k in ('ts', 'dur')):
                    raise ValueError('invalid Chrome GPU event timestamps')
                if 'stream' not in args:
                    gaps.add('GPU events without explicit stream identity remain unsupported.')
                else:
                    if existing.attrs.get('stream') != args['stream']:
                        raise ValueError('captured GPU binding stream mismatch')
                    streams[(rank, str(args['stream']))].append((step, event['ts'], index, existing))
                verified_direct.add(existing.id)
                continue
            matches = [n for rf in external.get(str(args.get('External id')), ()) for n in candidates.get(rf, ())]
            if len(matches) != 1:
                gaps.add('GPU events without an unambiguous execution-trace correlation remain unsupported.')
                continue
            original = matches[0]
            if not all(isinstance(event.get(k), (int, float)) and math.isfinite(event[k])
                       and event[k] >= 0 for k in ('ts', 'dur')):
                raise ValueError('invalid Chrome GPU event timestamps')
            if 'stream' not in args:
                gaps.add('GPU events without explicit stream identity remain unsupported.')
                continue
            implementation = dict(name=event['name'], category=event['cat'],
                                  launch={k: args[k] for k in ('grid', 'block', 'registers per thread',
                                      'shared memory', 'bytes') if k in args})
            # ET cost identity includes shapes, scalar values and source/runtime
            # provenance; a kernel symbol alone is not an exact measurement case.
            context = dict(operator_cost_key=original.attrs.get('cost_key'),
                           hardware_id=workload.hardware['id'])
            if not context['operator_cost_key']:
                gaps.add('GPU correlation without an operator signature cannot establish an exact cost key.')
                continue
            measurement = dict(kind='chrome_gpu_event', artifact_sha256=checksum,
                               event_index=index, event_name=event['name'], event_category=event['cat'],
                               implementation=implementation, context=context)
            attrs = dict(cost_key=native_cost_key(implementation, context), native_measurement=measurement,
                         step=step, stream=args['stream'], resources=[f'gpu:{rank}:stream:{args["stream"]}'],
                         source_node_ids=[original.attrs['source_node_id']],
                         execution_trace_id=original.attrs.get('execution_trace_id'))
            node = Node(next_id, event['name'], rank, attrs=attrs)
            next_id += 1
            lowered.append(node)
            replacement.setdefault(original.id, []).append(node.id)
            streams[(rank, str(args['stream']))].append((step, event['ts'], index, node))
    if entries and set(n.id for n in direct.values()) != verified_direct:
        gaps.add('Some captured GPU bindings were not verified against the selected preparation artifacts.')
    if not lowered and not verified_direct:
        return
    # ET's preliminary host-issue chains are not CUDA dependencies. Discard them
    # only in known profiler projections; keep the original limitation visible.
    for node in nodes.values():
        if node.attrs.get('execution_trace_id') is not None:
            node.deps = []
        else:
            node.deps = sorted({new for old in node.deps for new in replacement.get(old, [old])})
    for events in streams.values():
        events.sort(key=lambda item: (item[0], item[1], item[2]))
        for previous, current in zip(events, events[1:]):
            current[3].deps = sorted(set(current[3].deps + [previous[3].id]))
    for storage in workload.storage:
        if storage.producer in replacement:
            # A projected span is conservatively live before all its launches.
            # This still cannot assert an actual allocation/lifetime.
            storage.producer = None
        storage.consumers = sorted({new for old in storage.consumers for new in replacement.get(old, [old])})
    for key, old in list(maps.items()):
        if old in replacement:
            # Multiple launches have no unique endpoint without evidence.
            maps[key] = replacement[old]
    workload.nodes = [n for n in nodes.values() if n.id not in replacement] + lowered
    gaps.add('Cross-stream dependencies, CUDA event waits and stream contention require explicit evidence.')
    workload.manifest['preparation']['chrome_gpu_events'] = len(lowered) + len(verified_direct)
    workload.manifest['preparation']['verified_captured_gpu_events'] = len(verified_direct)
    workload.manifest['preparation']['lowered_operator_occurrences'] = len(replacement)


def prepare(workloads, config=None):
    """Combine rank-local workloads and apply only explicitly evidenced lowering."""
    config = deepcopy(config or {})
    _evidence(config)
    inputs = [deepcopy(w) if isinstance(w, Workload) else read_workload(w) for w in workloads]
    if not inputs:
        raise ValueError('prepare requires at least one workload')
    hardware = inputs[0].hardware
    nodes, storage, maps, storage_maps, ranks, manifests = [], [], {}, {}, set(), []
    gaps, artifacts = set(), []
    world_sizes = set()
    for workload in inputs:
        validate(workload.nodes, workload.storage, workload.manifest)
        if workload.hardware != hardware:
            raise ValueError('rank workloads have incompatible hardware descriptions')
        local_ranks = set(workload.manifest['ranks'])
        if ranks & local_ranks:
            raise ValueError('duplicate rank workload; combine steps in one capture per rank')
        ranks.update(local_ranks)
        if 'world_size' in workload.manifest:
            world_sizes.add(workload.manifest['world_size'])
        manifests.append(deepcopy(workload.manifest))
        gaps.update(workload.manifest.get('unresolved', []))
        artifacts.extend(workload.manifest.get('native_trace_artifacts', []))
        mapping = {n.id: len(nodes) + i for i, n in enumerate(workload.nodes)}
        for node in workload.nodes:
            old = node.id
            node.id = mapping[old]
            node.deps = [mapping[d] for d in node.deps]
            node.attrs['source_node_id'] = old
            maps[(node.rank, old)] = node.id
            nodes.append(node)
        for allocation in workload.storage:
            old = allocation.id
            # JSON tuple encoding prevents separator/embedded-rank collisions.
            allocation.id = 'prepared:' + json.dumps([allocation.rank, old], separators=(',', ':'))
            allocation.producer = mapping[allocation.producer] if allocation.producer is not None else None
            allocation.consumers = sorted(set(mapping[c] for c in allocation.consumers))
            storage_maps[(allocation.rank, old)] = allocation.id
            storage.append(allocation)
    if len(world_sizes) > 1:
        raise ValueError('rank workloads disagree on world_size')
    expected = next(iter(world_sizes), len(ranks))
    if type(expected) is not int or expected < 1:
        raise ValueError('invalid world_size')
    if ranks != set(range(expected)):
        gaps.add('All-rank preparation is incomplete: expected ranks ' + str(list(range(expected))) + '.')
    # Recipe/runtime identity must agree when present. Capture-side fields such
    # as device rank or artifact paths need not be identical.
    for key in ('recipe_args', 'resolved_config_sha256', 'torchtitan_python_source_sha256', 'torch_version', 'cuda_version'):
        values = [_digest(m.get('provenance', {}).get(key)) for m in manifests]
        if len(set(values)) > 1:
            raise ValueError('rank workloads have mismatched provenance: ' + key)
    statuses = [m.get('capture_status', 'complete') for m in manifests]
    if any(status != 'complete' for status in statuses):
        gaps.add('At least one source capture is blocked or incomplete.')
    manifest = dict(ranks=sorted(ranks), world_size=expected, source_manifests=manifests,
                    native_trace_artifacts=artifacts, unresolved=[],
                    capture_status='blocked' if 'blocked' in statuses else ('incomplete' if 'incomplete' in statuses else 'complete'),
                    preparation=dict(version=1, evidence=config.get('evidence'), input_count=len(inputs)),
                    provenance=deepcopy(manifests[0].get('provenance', {})))
    if 'useful_tokens' in config:
        manifest['useful_tokens'] = _nonnegative(config['useful_tokens'], 'useful_tokens')
    elif len(inputs) == 1 and 'useful_tokens' in manifests[0]:
        manifest['useful_tokens'] = manifests[0]['useful_tokens']
    else:
        manifest['useful_tokens'] = 0
        if any(m.get('useful_tokens') for m in manifests):
            gaps.add('Global useful token count must be supplied; rank counts may duplicate data-parallel or tensor-parallel work.')
    result = Workload(nodes, storage, manifest, deepcopy(hardware))
    traces = config.get('chrome_traces')
    if traces is None:
        traces = []
        for source in manifests:
            if len(source['ranks']) != 1:
                continue
            for artifact in source.get('native_trace_artifacts', []):
                match = re.fullmatch(r'step(\d+)\.chrome\.json', Path(artifact['path']).name)
                if match:
                    traces.append(dict(rank=source['ranks'][0], step=int(match[1]), path=artifact['path']))
    _chrome(result, traces, maps, gaps)
    for key in ('memory', 'rank_placement'):
        if key in config:
            manifest[key] = deepcopy(config[key])
        elif len(inputs) == 1 and key in manifests[0]:
            manifest[key] = deepcopy(manifests[0][key])
    unverified_storage = [m['ranks'] for m in manifests
                          if classify_storage_evidence(m) == 'unverified']
    if unverified_storage:
        memory = manifest.setdefault('memory', {})
        if memory.get('storage_evidence') == 'exact':
            _evidence(config, required=True)
            manifest['preparation']['storage_evidence_promotion'] = dict(
                source_ranks=unverified_storage, previous='unverified', declared='exact',
                evidence=deepcopy(config['evidence']), scope='integration_assertion', empirical_verification=False)
        else:
            memory['storage_evidence'] = 'unverified'

    def ref(value):
        key = (value['rank'], value['id'])
        if key not in maps:
            raise ValueError('unknown source node reference: ' + str(key))
        mapped = maps[key]
        if isinstance(mapped, list):
            if len(mapped) != 1:
                raise ValueError('source node has multiple GPU launches; explicit assertion needs a unique event')
            return mapped[0]
        return mapped

    by_id = {n.id: n for n in result.nodes}
    asserted = set()
    for assertion in config.get('dependencies', []):
        node = ref(assertion['node'])
        if node in asserted:
            raise ValueError('duplicate dependency assertion')
        asserted.add(node)
        by_id[node].deps = [ref(d) for d in assertion['deps']]
    claimed = set()
    for region in config.get('regions', []):
        members = [ref(r) for r in region['nodes']]
        if not members or len(set(members)) != len(members) or claimed.intersection(members):
            raise ValueError('empty, duplicate or overlapping execution region')
        claimed.update(members)
        selected = [by_id[i] for i in members]
        if len({n.rank for n in selected}) != 1 or any(n.kind != 'compute' for n in selected):
            raise ValueError('execution region must contain same-rank compute nodes')
        key = native_cost_key(region['implementation'], region['context'])
        root = min(members)
        attrs = dict(cost_key=key, implementation=region['implementation'], context=region['context'],
                     native_measurement=dict(kind='executable_region', implementation=region['implementation'],
                                             context=region['context']),
                     preparation_evidence=config['evidence'], source_node_ids=[n.attrs['source_node_id'] for n in selected
                         if 'source_node_id' in n.attrs], resources=sorted({r for n in selected for r in n.attrs.get('resources', [f'gpu:{n.rank}'])}))
        by_id[root] = Node(root, region.get('name', selected[0].name), selected[0].rank,
                           deps=sorted({d for n in selected for d in n.deps if d not in members}), attrs=attrs)
        for member in members:
            if member != root:
                del by_id[member]
        for node in by_id.values():
            node.deps = sorted({root if d in members else d for d in node.deps})
        for allocation in result.storage:
            if allocation.producer in members:
                allocation.producer = root
            allocation.consumers = sorted({root if d in members else d for d in allocation.consumers})
        for old, mapped in list(maps.items()):
            if not isinstance(mapped, list) and mapped in members:
                maps[old] = root
            elif isinstance(mapped, list):
                maps[old] = list(dict.fromkeys(root if d in members else d for d in mapped))
    communication_nodes, communication_keys = set(), set()
    for comm in config.get('communications', []):
        ids = [ref(r) for r in comm['nodes']]
        if not ids or len(set(ids)) != len(ids) or communication_nodes.intersection(ids):
            raise ValueError('duplicate/empty communication occurrence')
        communication_nodes.update(ids)
        selected = [by_id[i] for i in ids]
        participants = comm['participants']
        if (not isinstance(participants, list) or len(set(participants)) != len(participants)
                or sorted(n.rank for n in selected) != sorted(participants)):
            raise ValueError('communication participant mismatch')
        size = _nonnegative(comm['payload_bytes'], 'payload_bytes')
        sequence = _nonnegative(comm['sequence'], 'sequence')
        if not isinstance(comm['group'], str) or not comm['group'] or not isinstance(comm['operation'], str) or not comm['operation']:
            raise ValueError('communication requires group and operation identity')
        identity = dict(group=comm['group'], sequence=sequence, participants=participants)
        rendezvous = 'prepared-comm:' + _digest(identity)
        if rendezvous in communication_keys:
            raise ValueError('duplicate communication group/sequence')
        communication_keys.add(rendezvous)
        if comm['kind'] == 'collective':
            kinds = ['collective'] * len(ids)
        elif comm['kind'] == 'send_recv' and len(ids) == 2:
            kinds = ['send', 'recv']
        else:
            raise ValueError('unsupported communication kind or endpoint count')
        for index, (node, kind) in enumerate(zip(selected, kinds)):
            for field, expected_value in (('payload_bytes', size), ('participants', participants),
                                          ('operation', comm['operation'])):
                if field in node.attrs and node.attrs[field] != expected_value:
                    raise ValueError('communication evidence contradicts captured ' + field)
            node.kind = kind
            node.attrs.update(rendezvous=rendezvous, participants=participants, payload_bytes=size,
                              group=comm['group'], sequence=sequence, operation=comm['operation'],
                              preparation_evidence=config['evidence'])
            if kind != 'collective':
                node.attrs['peer'] = selected[1-index].rank
            communication_context = dict(group=comm['group'], operation=comm['operation'],
                participants=participants, payload_bytes=size, kind=kind, rank=node.rank,
                peer=node.attrs.get('peer'))
            node.attrs['communication_signature'] = communication_context
            binding = node.attrs.get('native_measurement')
            if binding:
                binding = deepcopy(binding)
                node.attrs['native_measurement'] = binding
                binding['context']['communication'] = communication_context
                node.attrs['cost_key'] = native_cost_key(binding['implementation'], binding['context'])
                if 'context' in node.attrs:
                    node.attrs['context'] = deepcopy(binding['context'])
            elif 'cost_key' in node.attrs:
                node.attrs['cost_key'] = 'prepared-communication:' + _digest(dict(
                    source_cost_key=node.attrs['cost_key'], communication=communication_context))
    replaced, storage_replacements = set(), {}
    for allocation in config.get('allocations', []):
        rank = allocation['rank']
        if rank not in ranks or not allocation['source_storage_ids']:
            raise ValueError('allocation assertion needs known rank and source storage')
        sources = [storage_maps.get((rank, sid)) for sid in allocation['source_storage_ids']]
        if None in sources or len(set(sources)) != len(sources) or replaced.intersection(sources):
            raise ValueError('unknown, duplicate or overlapping allocation source')
        replaced.update(sources)
        producer = None if allocation['producer'] is None else ref(allocation['producer'])
        consumers = [ref(r) for r in allocation['consumers']]
        if any(by_id[n].rank != rank for n in consumers + ([] if producer is None else [producer])):
            raise ValueError('backing allocation users must belong to its rank')
        target_id = 'allocation:' + json.dumps([rank, allocation['id']], separators=(',', ':'))
        storage_replacements.update({sid: target_id for sid in sources})
        result.storage.append(Storage(target_id,
            rank, _nonnegative(allocation['size'], 'allocation size'), producer, sorted(set(consumers)),
            allocation.get('persistent', False), allocation.get('category', 'verified_backing_allocation')))
    result.storage = [s for s in result.storage if s.id not in replaced]
    _remap_memory_storage(manifest.get('memory', {}), result.storage, storage_maps, storage_replacements)
    result.nodes = sorted(by_id.values(), key=lambda n: n.id)
    # A resolution is an explicit, individually scoped producer attestation.
    # Hash verification binds its evidence without claiming empirical fidelity.
    resolutions = []
    for resolution in config.get('resolutions', []):
        gap = resolution['gap']
        if gap not in gaps:
            raise ValueError('gap resolution must name one existing unresolved limitation')
        if gap.startswith('All-rank preparation is incomplete:') or gap == 'At least one source capture is blocked or incomplete.':
            raise ValueError('missing ranks or incomplete captures require new capture evidence, not an attestation')
        if resolution.get('scope') != 'integration_assertion' or not isinstance(resolution.get('reason'), str) or not resolution['reason'].strip():
            raise ValueError('gap resolution requires integration_assertion scope and a reason')
        evidence = resolution['evidence']
        actual = hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest()
        if actual != evidence.get('sha256'):
            raise ValueError('gap resolution evidence SHA256 mismatch')
        resolutions.append(dict(gap=gap, reason=resolution['reason'], scope=resolution['scope'],
                                evidence=deepcopy(evidence), validation='artifact_checksum_only',
                                empirical_verification=False))
        gaps.remove(gap)
    manifest['preparation']['resolved_gaps'] = resolutions
    manifest['preparation']['coverage_basis'] = 'integration_assertions' if resolutions else 'observed_evidence_and_remaining_gaps'
    manifest['unresolved'] = sorted(gaps)
    manifest['preparation'].update(dependency_assertions=len(asserted), regions=len(config.get('regions', [])),
                                   communications=len(config.get('communications', [])),
                                   allocation_assertions=len(config.get('allocations', [])))
    validate(result.nodes, result.storage, result.manifest)
    _validate_rendezvous(result.nodes)
    return result


def _remap_memory_storage(memory, storage, storage_maps, replacements):
    """Carry physical-memory alias references through both namespace changes."""
    available = {s.id for s in storage}
    def resolve(value):
        if isinstance(value, dict):
            mapped = storage_maps.get((value['rank'], value['id']))
        elif value in available or value in replacements:
            mapped = value
        else:
            choices = [mapped for (_, original), mapped in storage_maps.items() if original == value]
            if len(choices) != 1:
                raise ValueError('ambiguous or unknown memory storage reference: ' + str(value))
            mapped = choices[0]
        mapped = replacements.get(mapped, mapped)
        if mapped not in available:
            raise ValueError('unknown memory storage reference: ' + str(value))
        return mapped
    if 'storage_aliases' in memory:
        aliases = {}
        for source, target in memory['storage_aliases'].items():
            source, target = resolve(source), resolve(target)
            if source != target:
                if source in aliases and aliases[source] != target:
                    raise ValueError('contradictory backing allocation aliases')
                aliases[source] = target
        memory['storage_aliases'] = aliases
    for allocation in memory.get('allocations', []):
        if 'storage_id' in allocation:
            allocation['storage_id'] = resolve(allocation['storage_id'])


def _validate_rendezvous(nodes):
    groups = defaultdict(list)
    for node in nodes:
        if node.kind in ('collective', 'send', 'recv'):
            groups[node.attrs.get('rendezvous', 'node:' + str(node.id))].append(node)
    owners = {n.id: key for key, members in groups.items() for n in members}
    for key, members in groups.items():
        if all(n.kind == 'collective' for n in members):
            participants = members[0].attrs.get('participants', [])
            if sorted(n.rank for n in members) != sorted(participants) or any(n.attrs.get('participants') != participants for n in members):
                raise ValueError('collective participant mismatch')
            if len({(n.attrs.get('operation'), n.attrs.get('payload_bytes')) for n in members}) != 1:
                raise ValueError('collective signature mismatch')
        elif sorted(n.kind for n in members) == ['recv', 'send']:
            if members[0].attrs.get('payload_bytes') != members[1].attrs.get('payload_bytes') or any(
                n.attrs.get('peer') != members[1-i].rank for i, n in enumerate(members)):
                raise ValueError('point-to-point signature mismatch')
        else:
            raise ValueError('unmatched communication rendezvous')
        if any(owners.get(d) == key for n in members for d in n.deps):
            raise ValueError('dependency within communication rendezvous')
    # Individual node DAGs may become cyclic only after rendezvous contraction.
    owner = {n.id: ('comm', owners[n.id]) if n.id in owners else ('node', n.id) for n in nodes}
    parents = {key: set() for key in owner.values()}
    children = defaultdict(set)
    for node in nodes:
        for dep in node.deps:
            parents[owner[node.id]].add(owner[dep])
            children[owner[dep]].add(owner[node.id])
    ready = deque(key for key in parents if not parents[key])
    count = 0
    while ready:
        key = ready.popleft()
        count += 1
        for child in children[key]:
            parents[child].remove(key)
            if not parents[child]:
                ready.append(child)
    if count != len(parents):
        raise ValueError('rendezvous contraction creates a dependency cycle')
