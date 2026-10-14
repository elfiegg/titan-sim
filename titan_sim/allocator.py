"""Approximate capacity/fragmentation replay, not a CUDA allocator implementation.

The runtime supplies async-safe physical storage lifetimes. Cached segments belong
exclusively to their graph pool. Driver state, allocator stream arenas and hidden
workspaces cannot be recovered from a tensor event log.
"""
from collections import defaultdict
from .events import Event, nonnegative


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _round(size, alignment):
    return ((size + alignment - 1) // alignment) * alignment


def replay_allocator(result, hardware, config=None, events=None):
    """Replay ``storage_lifetimes``; stop each rank at its first failed request.

    Config: policy ideal|caching, alignment_bytes (512), segment_bytes (2 MiB),
    bin_sizes_bytes (optional ascending rounding bins), split/coalesce (true),
    release_cached_on_oom (true), external_memory_bytes (scalar or rank mapping).
    Graph pool segments are retained even when inactive. The conservative
    allocation-before-free timestamp ordering matches runtime memory samples.
    """
    config = dict(config or {})
    allowed = {'policy', 'alignment_bytes', 'segment_bytes', 'bin_sizes_bytes',
               'split', 'coalesce', 'release_cached_on_oom', 'external_memory_bytes'}
    if config.keys() - allowed:
        raise ValueError(f'unknown allocator options: {sorted(config.keys() - allowed)}')
    policy = config.get('policy', 'caching')
    if policy not in ('ideal', 'caching'):
        raise ValueError('allocator policy must be ideal or caching')
    alignment = _integer(config.get('alignment_bytes', 512), 'alignment_bytes', 1)
    segment_size = _integer(config.get('segment_bytes', 2 * 1024**2), 'segment_bytes', 1)
    bins = config.get('bin_sizes_bytes', [])
    if not isinstance(bins, list) or any(_integer(x, 'bin size', 1) % alignment for x in bins):
        raise ValueError('bin_sizes_bytes must contain alignment multiples')
    if bins != sorted(set(bins)):
        raise ValueError('bin_sizes_bytes must be strictly increasing')
    for key in ('split', 'coalesce', 'release_cached_on_oom'):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f'{key} must be boolean')
    capacity = _integer(hardware['memory_capacity_bytes'], 'memory_capacity_bytes', 1)
    external_config = config.get('external_memory_bytes', 0)
    if isinstance(external_config, dict):
        normalized_ranks = set()
        for rank, value in external_config.items():
            if isinstance(rank, bool) or not ((isinstance(rank, int) and rank >= 0) or
                                              (isinstance(rank, str) and rank.isdecimal() and str(int(rank)) == rank)):
                raise ValueError('external_memory_bytes rank keys must be nonnegative integers')
            if int(rank) in normalized_ranks:
                raise ValueError('duplicate external_memory_bytes rank')
            normalized_ranks.add(int(rank))
            _integer(value, 'external_memory_bytes')
    else:
        _integer(external_config, 'external_memory_bytes')
    # Recover allocation metadata from the same generation traversal as runtime.
    generation_metadata, live, generation = {}, {}, 0
    for raw in events or []:
        event = raw if isinstance(raw, Event) else Event.from_dict(raw)
        key = (event.rank, event.storage_id)
        if event.kind == 'allocate':
            if key in live:
                live[key][1] += 1
            else:
                generation_metadata[generation] = event.metadata
                live[key] = [generation, 1]
                generation += 1
        elif event.kind == 'release' and key in live:
            live[key][1] -= 1
            if not live[key][1]:
                del live[key]
    lifetimes = result.get('storage_lifetimes')
    if lifetimes is None:
        raise ValueError('allocator replay requires storage_lifetimes')
    changes, known = [], set()
    for life in lifetimes:
        rank = _integer(life['rank'], 'rank')
        _integer(life['generation'], 'generation')
        key = (rank, life['generation'])
        if key in known:
            raise ValueError('duplicate storage lifetime generation')
        known.add(key)
        size = _integer(life['size_bytes'], 'size_bytes')
        start = nonnegative(life['allocate_us'], 'allocate_us')
        end = life.get('reclaim_us')
        if end is not None and nonnegative(end, 'reclaim_us') < start:
            raise ValueError('reclaim_us precedes allocate_us')
        metadata = {**generation_metadata.get(life['generation'], {}), **life.get('metadata', {})}
        pool = str(life.get('allocator_pool', metadata.get('graph_pool', metadata.get('allocator_pool', 'default'))))
        external = life.get('external_memory', metadata.get('external_memory', False))
        if not isinstance(external, bool):
            raise ValueError('external_memory allocation metadata must be boolean')
        record = {**life, 'pool': pool, 'external_memory': external, 'size_bytes': size}
        changes.append((start, 0, life['generation'], record))
        if end is not None:
            changes.append((end, 1, life['generation'], record))
    states = {}
    ranks = {life['rank'] for life in lifetimes} | {int(r) for r in result.get('summary', {}).get('per_rank', {})}
    if isinstance(external_config, dict):
        ranks |= {int(r) for r in external_config}
    elif external_config and not ranks:
        ranks.add(0)
    def state(rank):
        if rank not in states:
            ext = external_config.get(str(rank), external_config.get(rank, 0)) if isinstance(external_config, dict) else external_config
            states[rank] = {'segments': [], 'active': {}, 'external': ext, 'failed': False,
                            'peaks': defaultdict(int), 'released_cached_bytes': 0}
        return states[rank]
    def counters(st):
        allocated = sum(x['block_bytes'] for x in st['active'].values() if not x['external_memory'])
        requested = sum(x['size_bytes'] for x in st['active'].values() if not x['external_memory'])
        reserved = sum(x['size'] for x in st['segments']) if policy == 'caching' else allocated
        inactive = reserved - allocated
        split = sum(b['size'] for seg in st['segments'] if any(b['key'] is not None for b in seg['blocks'])
                    for b in seg['blocks'] if b['key'] is None)
        return {'requested_bytes': requested, 'allocated_bytes': allocated,
                'reserved_bytes': reserved, 'inactive_bytes': inactive,
                'inactive_split_bytes': split, 'external_bytes': st['external'],
                'device_used_bytes': reserved + st['external']}
    samples, allocations, oom = [], [], []
    def sample(time, rank, action, life=None, **extra):
        st = state(rank)
        count = counters(st)
        for k, value in count.items():
            st['peaks'][k] = max(st['peaks'][k], value)
        samples.append({'time_us': time, 'rank': rank, 'action': action,
                        'generation': life['generation'] if life else None,
                        'storage_id': life['storage_id'] if life else None, **count, **extra})
    for rank in sorted(ranks):
        st = state(rank)
        sample(0, rank, 'initial')
        if st['external'] > capacity:
            st['failed'] = True
            oom.append({'rank': rank, 'time_us': 0, 'reason': 'external_memory_exceeds_capacity',
                        'requested_bytes': st['external'], 'capacity_bytes': capacity})
    for time, freeing, _, life in sorted(changes, key=lambda x: x[:3]):
        rank, key = life['rank'], (life['rank'], life['generation'])
        st = state(rank)
        if st['failed']:
            continue
        size = life['size_bytes']
        if freeing:
            allocation = st['active'].pop(key)
            if life['external_memory']:
                st['external'] -= size
            elif policy == 'caching':
                for seg in st['segments']:
                    for block in seg['blocks']:
                        if block['key'] == key:
                            block['key'] = None
                    if config.get('coalesce', True):
                        merged = []
                        for block in seg['blocks']:
                            if merged and merged[-1]['key'] is None and block['key'] is None:
                                merged[-1]['size'] += block['size']
                            else:
                                merged.append(block)
                        seg['blocks'] = merged
            sample(time, rank, 'reclaim', life)
            continue
        rounded = size if policy == 'ideal' else _round(size, alignment)
        if policy == 'caching' and rounded:
            rounded = next((b for b in bins if b >= rounded), rounded)
        chosen = None
        before = counters(st)
        needed = size if life['external_memory'] else rounded
        if not life['external_memory'] and policy == 'caching' and rounded:
            candidates = [(block['size'], seg['id'], i, seg, block)
                          for seg in st['segments'] if seg['pool'] == life['pool']
                          for i, block in enumerate(seg['blocks'])
                          if block['key'] is None and block['size'] >= rounded]
            if candidates:
                _, _, block_index, seg, block = min(candidates, key=lambda x: x[:3])
                chosen = (seg, block_index, block)
                needed = 0
            else:
                needed = _round(max(segment_size, rounded), alignment)
        if before['device_used_bytes'] + needed > capacity and config.get('release_cached_on_oom', True):
            retained = []
            for seg in st['segments']:
                # Private graph memory stays pinned until graph destruction,
                # which this input schema does not currently model.
                if seg['pool'] == 'default' and all(b['key'] is None for b in seg['blocks']):
                    st['released_cached_bytes'] += seg['size']
                else:
                    retained.append(seg)
            st['segments'] = retained
            before = counters(st)
        if before['device_used_bytes'] + needed > capacity:
            st['failed'] = True
            free_blocks = [b['size'] for seg in st['segments'] if seg['pool'] == life['pool']
                           for b in seg['blocks'] if b['key'] is None]
            failure = {'rank': rank, 'time_us': time, 'storage_id': life['storage_id'],
                       'generation': life['generation'], 'pool': life['pool'],
                       'requested_bytes': size, 'rounded_request_bytes': rounded,
                       'new_segment_bytes': needed, 'capacity_bytes': capacity,
                       'largest_free_block_bytes': max(free_blocks, default=0),
                       'total_pool_free_bytes': sum(free_blocks), **{'before_' + k: v for k, v in before.items()},
                       'reason': 'request_cannot_fit_capacity_or_fragmentation'}
            oom.append(failure)
            sample(time, rank, 'oom', life, failed_request_bytes=size)
            continue
        block_bytes = rounded
        if life['external_memory']:
            st['external'] += size
            block_bytes = 0
        elif policy == 'caching' and rounded:
            if chosen is None:
                seg = {'id': life['generation'], 'pool': life['pool'], 'size': needed,
                       'blocks': [{'size': needed, 'key': None}]}
                st['segments'].append(seg)
                chosen = (seg, 0, seg['blocks'][0])
            seg, index, block = chosen
            block_bytes = rounded if config.get('split', True) else block['size']
            replacement = [{'size': block_bytes, 'key': key}]
            if block['size'] > block_bytes:
                replacement.append({'size': block['size'] - block_bytes, 'key': None})
            seg['blocks'][index:index + 1] = replacement
        entry = {**life, 'block_bytes': block_bytes, 'reused_block': chosen is not None and needed == 0}
        st['active'][key] = entry
        allocations.append(entry)
        sample(time, rank, 'allocate', life)
    per_rank = {str(rank): {'oom': st['failed'], 'final': counters(st),
                           'peaks': dict(st['peaks']), 'released_cached_bytes': st['released_cached_bytes']}
                for rank, st in states.items()}
    limitations = ['Approximate best-fit segment allocator; not exact CUDA allocator or driver behavior.',
                   'Allocation-before-free ties are conservative; same-stream early reuse is not modeled.',
                   'Unknown workspaces, library/driver allocations and communication buffers are excluded unless explicitly recorded.',
                   'Graph private pools remain pinned; graph destruction and allocator stream arenas are not modeled.',
                   'Replay stops a rank at its first OOM; the original runtime timeline is a counterfactual after that point.']
    return {'samples': samples, 'allocations': allocations, 'oom': oom,
            'summary': {'policy': policy, 'capacity_exceeded': bool(oom), 'per_rank': per_rank,
                        'capacity_status': 'oom' if oom else 'fits_recorded_allocations_only',
                        'full_target_fit_proven': False,
                        'peak_reserved_bytes': max((s['peaks'].get('reserved_bytes', 0) for s in states.values()), default=0),
                        'peak_allocated_bytes': max((s['peaks'].get('allocated_bytes', 0) for s in states.values()), default=0),
                        'coverage': {'physical_lifetimes': len(lifetimes),
                                     'allocation_metadata_available': bool(events) or all('metadata' in x for x in lifetimes),
                                     'external_memory_supplied': 'external_memory_bytes' in config,
                                     'unclassified_lifetimes': sum(x.get('category', 'unclassified') == 'unclassified' for x in lifetimes),
                                     'hidden_allocations': 'unmodeled_unless_recorded'},
                        'limitations': limitations, 'config': config}}
