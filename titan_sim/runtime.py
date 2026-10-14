"""CPU discrete-event scheduler with asynchronous host submission and storage lifetimes.

Each rank has one compute lane and one communication lane. Their overlap is an
optimistic scenario; contention, collectives rendezvous and allocator caching are
not inferred from this event log.
"""
from collections import defaultdict
from .events import Event
from .costs import estimate_cost, validate_hardware


def logical_memory_events(events: list[dict]) -> list[dict]:
    """Replay reference-owned storage in program order, without a GPU clock.

    These samples represent tensor storage ownership observed by dispatch. They
    exclude caching allocator reserve, kernel workspaces and deferred GPU uses.
    Event indices, not estimated timestamps, are the horizontal coordinate.
    """
    live = {}
    allocated = defaultdict(int)
    categories = defaultdict(lambda: defaultdict(int))
    samples = []
    for index, raw in enumerate(events):
        event = raw if isinstance(raw, Event) else Event.from_dict(raw)
        key = (event.rank, event.storage_id)
        delta = 0
        category = event.category
        if event.kind == 'allocate':
            if key in live:
                storage = live[key]
                if event.size_bytes not in (0, storage['size_bytes']):
                    raise ValueError(f'{event.event_id}: inconsistent alias size')
                storage['refs'] += 1
            else:
                live[key] = {'size_bytes': event.size_bytes, 'refs': 1, 'category': category,
                             'device': event.metadata.get('device', 'cuda')}
                delta = event.size_bytes if not str(live[key]['device']).startswith(('cpu', 'meta')) else 0
        elif event.kind == 'release':
            if key not in live:
                raise ValueError(f'{event.event_id}: release of unknown storage')
            storage = live[key]
            storage['refs'] -= 1
            category = storage['category']
            if not storage['refs']:
                delta = -storage['size_bytes'] if not str(storage['device']).startswith(('cpu', 'meta')) else 0
                del live[key]
        else:
            continue
        allocated[event.rank] += delta
        categories[event.rank][category] += delta
        samples.append({'event_index': index, 'event_id': event.event_id,
                        'rank': event.rank, 'action': event.kind,
                        'storage_id': event.storage_id, 'delta_bytes': delta,
                        'allocated_bytes': allocated[event.rank], 'category': category,
                        'phase': event.metadata.get('phase', event.category),
                        'by_category': dict(categories[event.rank])})
    return samples


def simulate(events: list[dict], hardware: dict, timing='roofline', calibrations=None) -> dict:
    validate_hardware(hardware)
    if timing not in ('roofline', 'calibrated', 'hybrid'):
        raise ValueError(f'unknown timing mode: {timing}')
    parsed = [e if isinstance(e, Event) else Event.from_dict(e) for e in events]
    explicit_host_schedule = bool(events) and all(isinstance(e, dict) and 'host_time_us' in e for e in events)
    logical_memory = logical_memory_events(parsed)
    logical_peaks = defaultdict(int)
    for sample in logical_memory:
        logical_peaks[sample['rank']] = max(logical_peaks[sample['rank']], sample['allocated_bytes'])
    by_id = {e.event_id: e for e in parsed}
    if len(by_id) != len(parsed):
        raise ValueError('duplicate event_id')
    position = {e.event_id: i for i, e in enumerate(parsed)}
    deps = {e.event_id: set(e.deps) for e in parsed}
    stream_last, barrier, host_floor = {}, {}, defaultdict(float)
    previous_rank = defaultdict(list)
    submit_floor, submit_barrier = {}, {}
    for e in parsed:
        unknown = deps[e.event_id] - by_id.keys()
        if unknown:
            raise ValueError(f'{e.event_id}: missing dependencies {sorted(unknown)}')
        host_floor[e.rank] = max(host_floor[e.rank], e.host_time_us)
        submit_floor[e.event_id] = host_floor[e.rank]
        submit_barrier[e.event_id] = barrier.get(e.rank)
        if e.rank in barrier:
            deps[e.event_id].add(barrier[e.rank])
        stream = (e.rank, e.stream)
        if stream in stream_last:
            deps[e.event_id].add(stream_last[stream])
        if e.kind == 'host_sync':
            scope = e.metadata.get('sync_scope', 'device')
            if scope not in ('device', 'stream', 'event'):
                raise ValueError(f'{e.event_id}: unknown host sync_scope {scope!r}')
            # Device sync drains prior rank work; stream/event sync waits only
            # for its own stream order and explicitly recorded dependencies.
            if scope == 'device':
                deps[e.event_id].update(previous_rank[e.rank])
            barrier[e.rank] = e.event_id
        stream_last[stream] = e.event_id
        previous_rank[e.rank].append(e.event_id)

    # Determine physical-storage generations and Python-side reference ownership
    # in submission order. A released generation remains resident until use ends.
    live = {}
    generations = []
    for e in parsed:
        key = (e.rank, e.storage_id)
        if e.kind == 'allocate':
            if key in live:
                allocation = live[key]
                if e.size_bytes not in (0, allocation['size_bytes']):
                    raise ValueError(f'{e.event_id}: alias allocation size differs from backing storage')
                allocation['refs'] += 1
            else:
                allocation = {'rank': e.rank, 'storage_id': e.storage_id,
                              'generation': len(generations), 'size_bytes': e.size_bytes,
                              'category': e.category, 'device': e.metadata.get('device', 'cuda'),
                              'allocate': e.event_id,
                              'release': None, 'uses': [], 'refs': 1}
                generations.append(allocation)
                live[key] = allocation
        elif e.kind == 'release':
            if key not in live:
                raise ValueError(f'{e.event_id}: release of unknown storage {e.storage_id}')
            allocation = live[key]
            allocation['refs'] -= 1
            if not allocation['refs']:
                allocation['release'] = e.event_id
                del live[key]
        for storage in set(e.read_storage_ids + e.write_storage_ids):
            use_key = (e.rank, storage)
            if use_key not in live:
                raise ValueError(f'{e.event_id}: use of unallocated/released storage {storage}')
            allocation = live[use_key]
            allocation['uses'].append(e.event_id)
            deps[e.event_id].add(allocation['allocate'])

    children = defaultdict(list)
    pending = {}
    for event_id, parents in deps.items():
        pending[event_id] = len(parents)
        for parent in parents:
            children[parent].append(event_id)
    ready = [position[event_id] for event_id, count in pending.items() if not count]
    lane_end = defaultdict(float)
    timeline, completed = [], {}
    def eligible_start(index):
        candidate = parsed[index]
        prior = submit_barrier[candidate.event_id]
        submit = max(submit_floor[candidate.event_id], completed[prior]['end_us'] if prior else 0.0)
        start = max([submit] + [completed[d]['end_us'] for d in deps[candidate.event_id]])
        lane = 'compute' if candidate.kind == 'compute' else 'communication' if candidate.kind == 'collective' else None
        if lane:
            start = max(start, lane_end[(candidate.rank, lane)])
        return start, index

    while ready:
        # Select the earliest eligible launch; a stream waiting on communication
        # must not reserve an idle compute lane ahead of other ready streams.
        index = min(ready, key=eligible_start)
        ready.remove(index)
        e = parsed[index]
        prior_barrier = submit_barrier[e.event_id]
        submit_us = max(submit_floor[e.event_id], completed[prior_barrier]['end_us'] if prior_barrier else 0.0)
        start_us = max([submit_us] + [completed[d]['end_us'] for d in deps[e.event_id]])
        cost = estimate_cost(e, hardware, timing, calibrations)
        lane = 'compute' if e.kind == 'compute' else 'communication' if e.kind == 'collective' else None
        if cost['source'] in ('host_unmodeled', 'meta_unmodeled'):
            lane = None
        if lane:
            start_us = max(start_us, lane_end[(e.rank, lane)])
        end_us = start_us + cost['duration_us']
        if lane:
            lane_end[(e.rank, lane)] = end_us
        record = {'event_id': e.event_id, 'kind': e.kind, 'name': e.name,
                  'rank': e.rank, 'stream': e.stream, 'submit_us': submit_us,
                  'start_us': start_us, 'end_us': end_us, 'lane': lane,
                  'deps': sorted(deps[e.event_id]), 'flops': e.flops,
                  'bytes_read': e.bytes_read, 'bytes_write': e.bytes_write,
                  'dtype': e.dtype, **cost}
        if e.kind == 'host_sync':
            record['sync_scope'] = e.metadata.get('sync_scope', 'device')
        timeline.append(record)
        completed[e.event_id] = record
        for child in children[e.event_id]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(position[child])
    if len(timeline) != len(parsed):
        raise ValueError(f'dependency cycle involving {sorted(k for k, n in pending.items() if n)}')

    changes = []
    lifetimes = []
    for alloc in generations:
        if str(alloc['device']).startswith(('cpu', 'meta')):
            continue
        start = completed[alloc['allocate']]['submit_us']
        final_release = alloc['release']
        end = None
        if final_release:
            end = max([completed[final_release]['submit_us'], start] +
                      [completed[use]['end_us'] for use in alloc['uses']])
        detail = {k: alloc[k] for k in ('rank', 'storage_id', 'generation', 'size_bytes', 'category')}
        lifetimes.append({**detail, 'allocate_us': start, 'reclaim_us': end,
                          'remaining_refs': alloc['refs']})
        # All allocations at a timestamp precede reclamations: conservative peak
        # at ties, including zero-duration lifetimes.
        changes.append((start, 0, alloc['generation'], alloc['size_bytes'], detail))
        if end is not None:
            changes.append((end, 1, alloc['generation'], -alloc['size_bytes'], detail))
    allocated, peaks = defaultdict(int), defaultdict(int)
    categories = defaultdict(lambda: defaultdict(int))
    peak_categories, memory = {}, []
    for time_us, reclaim, generation, delta, detail in sorted(changes):
        rank = detail['rank']
        allocated[rank] += delta
        categories[rank][detail['category']] += delta
        if allocated[rank] > peaks[rank]:
            peaks[rank] = allocated[rank]
            peak_categories[rank] = dict(categories[rank])
        memory.append({'time_us': time_us, 'rank': rank, 'allocated_bytes': allocated[rank],
                       'delta_bytes': delta, 'action': 'reclaim' if reclaim else 'allocate',
                       'storage_id': detail['storage_id'], 'generation': generation,
                       'category': detail['category'], 'by_category': dict(categories[rank])})
    work = [r for r in timeline if r['kind'] in ('compute', 'collective') and r['source'] not in ('host_unmodeled', 'meta_unmodeled', 'metadata_only')]
    fallback = [r for r in work if r['source'] == 'roofline_fallback']
    calibrated = [r for r in work if r['source'] == 'calibrated']
    explicit = [r for r in work if r['source'] == 'explicit']
    total_flops = sum(r['flops'] for r in work)
    total_work_us = sum(r['duration_us'] for r in work)
    ranks = sorted({e.rank for e in parsed})
    per_rank = {str(rank): {'runtime_us': max((r['end_us'] for r in timeline if r['rank'] == rank), default=0.0),
                            'host_time_us': max((r['end_us'] if r['kind'] == 'host_sync' else r['submit_us']
                                                 for r in timeline if r['rank'] == rank), default=0.0),
                            'peak_allocated_bytes': peaks[rank],
                            'final_allocated_bytes': allocated[rank],
                            'peak_by_category': peak_categories.get(rank, {}),
                            'logical_peak_allocated_bytes': logical_peaks[rank],
                            'capacity_exceeded': (peaks[rank] > hardware['memory_capacity_bytes']) if explicit_host_schedule else None,
                            'scheduled_capacity_exceeded': peaks[rank] > hardware['memory_capacity_bytes']}
                for rank in ranks}
    summary = {'runtime_us': max((r['end_us'] for r in timeline), default=0.0),
               'peak_allocated_bytes': max(peaks.values(), default=0),
               'per_rank': per_rank,
               'capacity_exceeded': any(r['scheduled_capacity_exceeded'] for r in per_rank.values()) if explicit_host_schedule else None,
               'scheduled_capacity_exceeded': any(r['scheduled_capacity_exceeded'] for r in per_rank.values()),
               'logical_peak_allocated_bytes': max(logical_peaks.values(), default=0),
               'memory_confidence': 'explicit_host_schedule_scenario' if explicit_host_schedule else 'unsupported_without_explicit_host_schedule',
               'logical_memory_model': 'program-order reference-owned GPU tensor storage; excludes CPU/meta storage, allocator reserve, kernel workspaces and deferred GPU usage',
               'coverage': {'work_events': len(work), 'calibrated_events': len(calibrated),
                            'explicit_events': len(explicit),
                            'fallback_events': len(fallback),
                            'fallback_event_fraction': len(fallback) / len(work) if work else 0.0,
                            'fallback_flop_fraction': sum(r['flops'] for r in fallback) / total_flops if total_flops else 0.0,
                            'fallback_duration_fraction': sum(r['duration_us'] for r in fallback) / total_work_us if total_work_us else 0.0},
               'timing': timing, 'hardware': hardware.get('name', 'unnamed'),
               'scheduling_model': 'one serialized compute lane and one communication lane per rank',
               'overlap_model': 'ideal compute/communication overlap scenario; contention is not modeled',
               'memory_model': 'physical live storage with alias references and deferred reclamation; no caching/reserved model; conservative allocation-before-free ties',
               'limitations': ['CPU/meta-device arithmetic/control durations are unmodeled and contribute no device time.',
                               'Collective participant rendezvous must be supplied as explicit dependencies.',
                               'Host launch overhead is zero unless host_time_us encodes it.',
                               'Analytical durations are lower bounds, not predicted measured runtimes.',
                               'No same-stream allocator reuse before physical reclamation.']}
    return {'timeline': timeline, 'memory': memory, 'logical_memory': logical_memory,
            'storage_lifetimes': lifetimes, 'summary': summary}
