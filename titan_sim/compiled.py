"""Lower validated serial compiled-region plans into ordinary runtime events.

A compiler's actual GPU allocation plan is an input, never inferred from FX.
Declared plans remain declared. Fused kernel traffic is authoritative: logical
operator traffic is deliberately not added back to the fused boundary traffic.
"""
import hashlib
import json
from .events import Event
from .quantization import _integer


def artifact_digest(artifact: dict) -> str:
    return hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def export_cpu_graph(module=None, inputs=None) -> dict:
    """Obtain real torch.export FX graph metadata; this is NOT GPU compiler IR."""
    import torch
    if module is None:
        class Example(torch.nn.Module):
            def forward(self, x, weight):
                return torch.relu(x @ weight)
        module = Example()
        inputs = (torch.zeros(2, 4), torch.zeros(4, 3))
    if inputs is None:
        raise ValueError('inputs are required for a custom module')
    exported = torch.export.export(module, inputs)
    nodes = []
    for node in exported.graph_module.graph.nodes:
        if node.op != 'call_function':
            continue
        value = node.meta.get('val')
        nodes.append({'name': node.name, 'target': str(node.target),
                      'shape': list(value.shape) if isinstance(value, torch.Tensor) else None,
                      'dtype': str(value.dtype) if isinstance(value, torch.Tensor) else None})
    return {'kind': 'torch_export_fx', 'torch_version': torch.__version__, 'nodes': nodes,
            'graph_code': exported.graph_module.code,
            'provenance': 'actual torch.export invocation on CPU; graph metadata only, no GPU allocation or fusion plan'}


def lower_compiled_manifest(manifest: dict, *, artifact: dict | None = None) -> dict:
    """Validate and lower an explicit serial kernel/physical-storage plan.

    External storages must already exist when the resulting events are simulated.
    Their allocation and lifetime belong to the caller (e.g. quantized training).
    Logical buffers can alias or reuse physical slots, but overlapping live
    ranges require explicit alias ownership. Graph pools are retained explicitly.
    """
    allowed = {'schema_version', 'id', 'rank', 'stream', 'provenance', 'plan_source',
               'artifact_sha256', 'calibration_context', 'device', 'retained_buffers', 'external_storages', 'slots', 'buffers', 'kernels', 'graph', 'deps'}
    if not isinstance(manifest, dict) or set(manifest) - allowed or manifest.get('schema_version') != 1:
        raise ValueError('unsupported compiled manifest schema or field')
    prefix = manifest.get('id')
    if not isinstance(prefix, str) or not prefix or not manifest.get('provenance'):
        raise ValueError('compiled id and provenance are required')
    source = manifest.get('plan_source')
    if source not in ('declared', 'compiler_artifact'):
        raise ValueError('plan_source must be declared or compiler_artifact')
    digest = manifest.get('artifact_sha256')
    if digest is not None or artifact is not None or source == 'compiler_artifact':
        if artifact is None or digest != artifact_digest(artifact):
            raise ValueError('compiled artifact mismatch: an exact artifact_sha256 is required')
        if source == 'compiler_artifact' and artifact.get('kind') not in ('gpu_compiler_allocation_plan', 'inductor_wrapper_allocation_plan'):
            raise ValueError('FX metadata is not a GPU compiler allocation plan; use a declared plan')
        if source == 'compiler_artifact':
            # The hashed compiler artifact must bind the actual memory/work plan,
            # not merely an unrelated graph or compiler version string.
            bound = {key: manifest.get(key, {}) for key in ('external_storages', 'slots', 'buffers', 'kernels', 'graph')}
            bound.update({key: manifest[key] for key in ('device', 'retained_buffers') if key in manifest})
            if artifact.get('plan') != bound:
                raise ValueError('compiler artifact allocation plan mismatch')
    context = manifest.get('calibration_context', {})
    if not isinstance(context, dict):
        raise ValueError('calibration_context must be an object')
    plan_digest = artifact_digest(manifest)
    rank = _integer(manifest.get('rank', 0), 'rank', 0)
    stream = manifest.get('stream', 'default')
    device = manifest.get('device', 'cuda')
    if device not in ('cpu', 'cuda'):
        raise ValueError('compiled device must be cpu or cuda')
    external = manifest.get('external_storages', {})
    if not isinstance(external, dict) or any(not isinstance(k, str) or not k for k in external):
        raise ValueError('external_storages must map storage IDs to sizes')
    for sid, size in external.items():
        _integer(size, f'external storage {sid} size', 0)
    kernels = manifest.get('kernels')
    if not isinstance(kernels, list) or not kernels:
        raise ValueError('kernels must be a nonempty ordered list')
    kernel_fields = {'id', 'name', 'dtype', 'flops', 'bytes_read', 'bytes_write', 'reads', 'writes', 'logical_ops', 'provenance'}
    kernel_ids, ops = {}, []
    for i, kernel in enumerate(kernels):
        if not isinstance(kernel, dict) or set(kernel) != kernel_fields:
            raise ValueError('kernel must explicitly declare id/name/dtype/flops/traffic/reads/writes/logical_ops/provenance')
        kid = kernel['id']
        if not isinstance(kid, str) or not kid or kid in kernel_ids:
            raise ValueError('kernel IDs must be unique strings')
        kernel_ids[kid] = i
        logical = kernel['logical_ops']
        if not isinstance(logical, list) or not logical or any(not isinstance(x, str) or not x for x in logical):
            raise ValueError('logical_ops must be a nonempty list of graph node identifiers')
        ops.extend(logical)
        if not kernel['provenance']:
            raise ValueError('kernel requires arithmetic/traffic provenance')
    if len(set(ops)) != len(ops):
        raise ValueError('logical graph node appears in multiple fused kernels')
    if artifact is not None and artifact.get('kind') == 'torch_export_fx':
        if set(ops) != {node['name'] for node in artifact['nodes']}:
            raise ValueError('compiled plan logical_ops do not match artifact graph nodes')
    slots = {}
    for slot in manifest.get('slots', []):
        if not isinstance(slot, dict) or set(slot) != {'id', 'size_bytes', 'category'}:
            raise ValueError('slot requires id, size_bytes, category')
        sid = slot['id']
        if not isinstance(sid, str) or not sid or sid in slots:
            raise ValueError('slot IDs must be unique strings')
        _integer(slot['size_bytes'], 'slot size_bytes')
        if not isinstance(slot['category'], str) or not slot['category']:
            raise ValueError('slot category must be a nonempty string')
        slots[sid] = slot
    buffers = {}
    buffer_fields = {'id', 'slot', 'size_bytes', 'offset_bytes', 'first_kernel', 'last_kernel'}
    for raw in manifest.get('buffers', []):
        if not isinstance(raw, dict) or set(raw) - (buffer_fields | {'alias_of'}) or not buffer_fields <= set(raw):
            raise ValueError('buffer requires explicit slot, offset, size and kernel lifetime')
        bid = raw['id']
        if not isinstance(bid, str) or not bid or bid in buffers or bid in external:
            raise ValueError('buffer IDs must be unique and distinct from external storages')
        if raw['slot'] not in slots or raw['first_kernel'] not in kernel_ids or raw['last_kernel'] not in kernel_ids:
            raise ValueError('buffer references unknown slot or kernel')
        size = _integer(raw['size_bytes'], 'buffer size_bytes')
        offset = _integer(raw['offset_bytes'], 'buffer offset_bytes', 0)
        if size + offset > slots[raw['slot']]['size_bytes']:
            raise ValueError('buffer exceeds physical slot')
        first, last = kernel_ids[raw['first_kernel']], kernel_ids[raw['last_kernel']]
        if first > last:
            raise ValueError('buffer lifetime is reversed')
        buffers[bid] = {**raw, 'first': first, 'last': last}
    retained = manifest.get('retained_buffers', [])
    if not isinstance(retained, list) or any(bid not in buffers for bid in retained) or len(set(retained)) != len(retained):
        raise ValueError('retained_buffers must name unique internal buffers')
    retained_slots = {buffers[bid]['slot'] for bid in retained}
    # Returned views retain their byte ranges until region completion. Extend
    # alias owners too, before validating overlap with later slot reuse.
    for retained_id in retained:
        current, seen = retained_id, set()
        while current is not None:
            if current in seen or current not in buffers:
                raise ValueError('retained buffer has invalid alias ownership')
            seen.add(current)
            buffers[current]['last'] = len(kernels) - 1
            buffers[current]['last_kernel'] = kernels[-1]['id']
            current = buffers[current].get('alias_of')
    def root_alias(bid):
        seen = set()
        current = bid
        while buffers[current].get('alias_of') is not None:
            if current in seen:
                raise ValueError('buffer alias cycle')
            seen.add(current)
            parent = buffers[current]['alias_of']
            if parent not in buffers:
                raise ValueError('alias references unknown buffer')
            child, target = buffers[current], buffers[parent]
            if (child['slot'] != target['slot'] or child['offset_bytes'] < target['offset_bytes']
                    or child['offset_bytes'] + child['size_bytes'] > target['offset_bytes'] + target['size_bytes']
                    or child['first'] < target['first'] or child['last'] > target['last']):
                raise ValueError('alias must fit within owner storage and lifetime')
            current = parent
        return current
    roots = {bid: root_alias(bid) for bid in buffers}
    values = list(buffers.values())
    for index, a in enumerate(values):
        for b in values[index + 1:]:
            same_slot = a['slot'] == b['slot']
            live_overlap = max(a['first'], b['first']) <= min(a['last'], b['last'])
            byte_overlap = max(a['offset_bytes'], b['offset_bytes']) < min(a['offset_bytes'] + a['size_bytes'], b['offset_bytes'] + b['size_bytes'])
            if same_slot and live_overlap and byte_overlap and roots[a['id']] != roots[b['id']]:
                raise ValueError('overlapping live storage reuse requires explicit alias_of')
    used = set()
    for i, kernel in enumerate(kernels):
        for access in ('reads', 'writes'):
            ids = kernel[access]
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
                raise ValueError('kernel reads/writes must be unique storage or buffer IDs')
            for bid in ids:
                if bid in external:
                    continue
                if bid not in buffers:
                    raise ValueError(f'kernel references unknown buffer {bid!r}')
                if not buffers[bid]['first'] <= i <= buffers[bid]['last']:
                    raise ValueError('kernel accesses buffer outside its lifetime')
                used.add(bid)
    if used != set(buffers):
        raise ValueError('all declared buffers must have an explicit kernel use')
    if set(slots) != {b['slot'] for b in buffers.values()}:
        raise ValueError('all physical slots must back a declared buffer')
    graph = manifest.get('graph', {'mode': 'none'})
    if not isinstance(graph, dict) or graph.get('mode') not in ('none', 'capture_replay'):
        raise ValueError('unsupported graph mode')
    replay = graph['mode'] == 'capture_replay'
    if set(graph) != ({'mode', 'replays', 'private_pool', 'release_pool'} if replay else {'mode'}):
        raise ValueError('graph capture requires explicit replay count, private_pool, release_pool')
    repeats, pool_sid = 1, None
    if replay:
        repeats += _integer(graph['replays'], 'replays', 0)
        if retained and graph['release_pool']:
            raise ValueError('cannot release graph pool while output buffers are retained')
        if not isinstance(graph['release_pool'], bool):
            raise ValueError('release_pool must be boolean')
        pool = graph['private_pool']
        if not isinstance(pool, dict) or set(pool) != {'id', 'size_bytes', 'slot_offsets'}:
            raise ValueError('private_pool requires id, size_bytes and slot_offsets')
        if not isinstance(pool['id'], str) or not pool['id']:
            raise ValueError('private pool id must be nonempty')
        pool_size = _integer(pool['size_bytes'], 'private pool size_bytes')
        offsets = pool['slot_offsets']
        if not isinstance(offsets, dict) or set(offsets) != set(slots):
            raise ValueError('private pool slot offsets must cover every physical slot')
        ranges = []
        for sid, offset in offsets.items():
            _integer(offset, 'private pool slot offset', 0)
            end = offset + slots[sid]['size_bytes']
            if end > pool_size:
                raise ValueError('slot exceeds private pool')
            if any(max(offset, start) < min(end, stop) for start, stop in ranges):
                raise ValueError('distinct private-pool slots overlap')
            ranges.append((offset, end))
        pool_sid = f'{prefix}:pool:{pool["id"]}'
    physical = {bid: pool_sid if replay else f'{prefix}:slot:{b["slot"]}' for bid, b in buffers.items()}
    if set(physical.values()) & set(external):
        raise ValueError('internal physical storage ID collides with an external storage')
    events = []
    def emit(kind, suffix, **kw):
        raw = {'event_id': f'{prefix}:{suffix}', 'kind': kind, 'rank': rank, 'stream': stream,
               'metadata': {'provenance': manifest['provenance'], 'plan_source': source}, **kw}
        raw['metadata'] = {**raw['metadata'], 'device': device}
        if not events:
            raw['deps'] = manifest.get('deps', [])
        Event.from_dict(raw)
        events.append(raw)
    if replay:
        emit('allocate', 'pool:allocate', storage_id=pool_sid, size_bytes=pool_size,
             category='graph_private_pool',
             metadata={'provenance': manifest['provenance'], 'graph_pool': pool_sid})
    # Slots are physical allocations whose lifespan covers every assigned logical
    # buffer. Reuse within a slot never creates another physical allocation.
    firsts = {sid: min(b['first'] for b in buffers.values() if b['slot'] == sid) for sid in slots}
    lasts = {sid: max(b['last'] for b in buffers.values() if b['slot'] == sid) for sid in slots}
    for iteration in range(repeats):
        phase = 'capture' if replay and iteration == 0 else f'replay{iteration}' if replay else 'execute'
        for i, kernel in enumerate(kernels):
            if not replay:
                for sid, slot in slots.items():
                    if firsts[sid] == i:
                        emit('allocate', f'slot:{sid}:allocate', storage_id=f'{prefix}:slot:{sid}',
                             size_bytes=slot['size_bytes'], category=slot['category'])
            resolve = lambda ids: sorted({physical.get(bid, bid) for bid in ids})
            emit('compute', f'{phase}:{kernel["id"]}', name=kernel['name'], dtype=kernel['dtype'],
                 flops=kernel['flops'], bytes_read=kernel['bytes_read'], bytes_write=kernel['bytes_write'],
                 read_storage_ids=resolve(kernel['reads']), write_storage_ids=resolve(kernel['writes']),
                 metadata={'provenance': kernel['provenance'], 'plan_source': source, 'graph_phase': phase,
                           'logical_ops': kernel['logical_ops'], 'artifact_sha256': digest,
                           'calibration_context': context,
                           'signature': {'kind': 'compiled_kernel', 'kernel': kernel,
                                         'artifact_sha256': digest, 'plan_sha256': plan_digest,
                                         'context': context},
                           'traffic_basis': 'declared fused kernel HBM traffic; no unfused intermediate traffic added'})
            if not replay:
                for sid in slots:
                    if lasts[sid] == i and sid not in retained_slots:
                        emit('release', f'slot:{sid}:release', storage_id=f'{prefix}:slot:{sid}')
    if replay and graph['release_pool']:
        emit('release', 'pool:release', storage_id=pool_sid)
    return {'schema_version': 1, 'source': 'compiled_region_manifest',
            'metadata': {'plan_source': source, 'provenance': manifest['provenance'],
                         'artifact_sha256': digest, 'calibration_context': context,
                         **{k: context[k] for k in ('torch_version', 'torchtitan_commit') if k in context}, 'graph': graph, 'external_storages': external,
                         'device': device, 'retained_storage_ids': sorted({physical[bid] for bid in retained}),
                         'buffer_to_storage': physical, 'logical_buffer_lifetimes': list(buffers.values())},
            'events': events,
            'limitations': ['Serial kernel order only; host/compiler latency and launch overhead are not inferred.',
                            'Capture executes the declared work once; replay repeats it without additional pool allocations.',
                            'FX/export graphs establish graph provenance, not GPU fusion or allocator equivalence.',
                            'Physical slots/private pool include declared reservation; general allocator caching is not modeled.',
                            'release_pool drops logical pool ownership; allocator replay can retain graph-pool cached segments.']}
