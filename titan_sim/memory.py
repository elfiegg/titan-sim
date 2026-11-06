"""Physical-device memory envelopes from declared placement and live intervals.

This module never infers missing allocation sizes or allocator reservations.
``memory.storage_evidence='unverified'`` preserves projected model peaks but
excludes them from credible lower bounds and leaves physical upper bounds unknown.
The default 'exact' is a producer assertion about backing allocations/lifetimes.
``manifest.rank_placement`` maps ranks to physical device IDs. Hardware may use
``devices={id: {capacity_bytes, reserve_fraction?}}`` or a uniform
``capacity_bytes``. ``manifest.memory.coverage`` explicitly attests each category
in COVERAGE_CATEGORIES; a missing category leaves the physical upper bound open.

Additional ``manifest.memory.allocations`` have an id, rank or device, category,
and either bytes or lower_bytes/upper_bytes (null upper means unknown). Optional
start_seconds/end_seconds default to the complete simulated interval. Allocator
slack is reserved minus allocated; adding reserved memory to storage is invalid.
``storage_aliases`` maps a Storage ID to another Storage ID for the same backing
allocation. An additional allocation may reference ``storage_id`` instead of
being counted again. Aliased intervals are unioned, including across ranks.

The default ``memory.scope='recipe'`` also requires initialization and
checkpoint_loading coverage. ``memory.lifecycle_peaks`` contains device-wide
phase/lower_bytes/upper_bytes bounds; these are maxed against the scheduled
peak, never added to it. A rank may identify a lifecycle device only when it is
not collocated. Explicit ``scope='schedule'`` limits the coverage claim.

Bounds describe declared evidence, not runtime validation. Zero-duration
operations have conservative boundary envelopes when timing cannot order them.
"""
from collections import defaultdict
import math


ALLOCATION_CATEGORIES = (
    'storage', 'external_allocations', 'workspaces', 'communication_buffers',
    'graph_pools', 'allocator_slack',
)
LIFECYCLE_PHASES = ('initialization', 'checkpoint_loading')
COVERAGE_CATEGORIES = ALLOCATION_CATEGORIES + LIFECYCLE_PHASES


def _bytes(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(label + ' must be a nonnegative integer number of bytes')
    return value


def _seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(label + ' must be finite and nonnegative')
    return value


def _fraction(value):
    _seconds(value, 'reserve_fraction')
    if value >= 1:
        raise ValueError('reserve_fraction must be less than one')
    return value


def _identifier(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == '':
        raise ValueError(label + ' must be a nonempty string or integer')
    return str(value)


def _merge(intervals):
    """Union lifetimes of aliases; adjacent intervals share one allocation."""
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _peak(records, bound):
    events = defaultdict(lambda: [[], [], [], []])
    for record in records:
        size = record[bound]
        if size is None:
            return None, None, {}
        for start, end in _merge(record['intervals']):
            if start == end:
                events[start][2].append((size, record['category']))
            else:
                events[start][1].append((size, record['category']))
                events[end][0].append((size, record['category']))
                if end in record.get('closed_ends', set()):
                    events[end][3].append((size, record['category']))
    live = defaultdict(int)
    peak, when, categories = 0, 0, {}

    def observe(values, at):
        nonlocal peak, when, categories
        total = sum(values.values())
        if total > peak:
            peak, when = total, at
            categories = {key: value for key, value in values.items() if value}

    for at, (ends, starts, instantaneous, closed_ends) in sorted(events.items()):
        # Ordinary intervals are half open: a completed user's allocation can be
        # reused by an operation beginning at the same timestamp.
        before = dict(live)
        for size, category in ends:
            live[category] -= size
        for size, category in starts:
            live[category] += size
        observe(live, at)
        if bound == 'upper' and closed_ends:
            values = defaultdict(int, live)
            for size, category in closed_ends:
                values[category] += size
            observe(values, at)
        if instantaneous:
            if bound == 'lower':
                # Timing alone cannot order zero-duration events at a boundary.
                for size, category in instantaneous:
                    observe({category: size}, at)
            else:
                # Safe upper envelope includes both sides of that boundary.
                values = defaultdict(int, before)
                for size, category in starts + instantaneous:
                    values[category] += size
                observe(values, at)
    return peak, when, categories


def classify_storage_evidence(manifest):
    """Honor explicit producer assertions, otherwise recognize native projections.

    Older captures lack the classification field. Source manifests retain that
    provenance through preparation, so legacy native projections cannot silently
    become exact backing-allocation evidence. Explicit promotions are trusted by
    this calculator; preparation is responsible for retaining their proof.
    """
    memory = manifest.get('memory', {})
    if not isinstance(memory, dict):
        raise ValueError('manifest.memory must be an object')
    if 'storage_evidence' in memory:
        result = memory['storage_evidence']
        if result not in ('exact', 'unverified'):
            raise ValueError('memory.storage_evidence must be exact or unverified')
        return result
    if manifest.get('observer') in ('pytorch_execution_trace_and_profiler', 'pytorch_profiler_chrome_only'):
        return 'unverified'
    sources = manifest.get('source_manifests', [])
    if not isinstance(sources, list) or any(not isinstance(source, dict) for source in sources):
        raise ValueError('source_manifests must be a list of objects')
    if any(classify_storage_evidence(source) == 'unverified' for source in sources):
        return 'unverified'
    return 'exact'


def assess_memory(nodes, storage, timing, hardware, manifest):
    """Return physical per-device bounds and a conservative fit verdict.

    ``timing`` is the simulator's node-ID -> (start, finish) mapping; integer or
    JSON string keys are accepted. Missing placement uses separate logical
    rank devices for reporting, but cannot establish physical fit. Capacity
    headroom is a budget reduction, distinct from explicit allocator slack.
    """
    memory = manifest.get('memory', {})
    if not isinstance(memory, dict):
        raise ValueError('manifest.memory must be an object')
    ranks = manifest.get('ranks', [])
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError('ranks must be distinct nonnegative integers')
    rank_ids = {str(rank) for rank in ranks}
    issues, assumptions = [], []
    storage_evidence = classify_storage_evidence(manifest)
    if storage_evidence == 'unverified':
        issues.append('storage sizes, backing identities or lifetimes are unverified; projected storage is not a credible lower bound')
    placement = manifest.get('rank_placement')
    placement_known = placement is not None
    if placement is None:
        placement = {str(rank): ('unplaced-rank:' + str(rank) if hardware.get('devices') is not None else str(rank)) for rank in ranks}
        issues.append('physical rank placement is not declared')
        assumptions.append('Each rank is reported on a separate provisional device.')
    if not isinstance(placement, dict):
        raise ValueError('rank_placement must be an object')
    normalized = {}
    for rank, device in placement.items():
        key = _identifier(rank, 'rank')
        if key in normalized:
            raise ValueError('duplicate normalized rank placement')
        normalized[key] = _identifier(device, 'device')
    if set(normalized) != rank_ids:
        raise ValueError('rank_placement must map every declared rank exactly once')
    placement = normalized
    default_reserve = _fraction(hardware.get('reserve_fraction', 0.1))
    devices = hardware.get('devices')
    if devices is None:
        capacity = _bytes(hardware.get('capacity_bytes'), 'capacity_bytes')
        devices = {device: {'capacity_bytes': capacity} for device in placement.values()}
    if not isinstance(devices, dict):
        raise ValueError('hardware.devices must be an object')
    device_specs = {}
    for key, spec in devices.items():
        device = _identifier(key, 'device')
        if device in device_specs or not isinstance(spec, dict):
            raise ValueError('invalid or duplicate device specification')
        capacity = _bytes(spec.get('capacity_bytes'), 'capacity_bytes')
        reserve = _fraction(spec.get('reserve_fraction', default_reserve))
        device_specs[device] = (capacity, capacity * (1 - reserve))
    if not placement_known:
        for device in placement.values():
            device_specs.setdefault(device, (None, None))
    if set(placement.values()) - set(device_specs):
        raise ValueError('rank placement references an unknown hardware device')
    coverage = memory.get('coverage', {})
    if not isinstance(coverage, dict) or any(key not in COVERAGE_CATEGORIES or type(value) is not bool for key, value in coverage.items()):
        raise ValueError('memory.coverage must contain known categories with boolean values')
    scope = memory.get('scope', 'recipe')
    if scope not in ('recipe', 'schedule'):
        raise ValueError('memory.scope must be recipe or schedule')
    required = COVERAGE_CATEGORIES if scope == 'recipe' else ALLOCATION_CATEGORIES
    for category in required:
        if not coverage.get(category, False):
            issues.append('memory coverage not established: ' + category)
    if manifest.get('unresolved'):
        issues.append('workload has unresolved capture or execution evidence')
    unresolved = memory.get('unresolved', [])
    if not isinstance(unresolved, list) or any(not isinstance(item, str) for item in unresolved):
        raise ValueError('memory.unresolved must be a list of strings')
    issues.extend(unresolved)

    intervals = {}
    for node in nodes:
        if node.rank not in ranks or node.id in intervals:
            raise ValueError('invalid node rank or duplicate node id')
        value = timing.get(str(node.id), timing.get(node.id))
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('missing or malformed node timing: ' + str(node.id))
        start, end = (_seconds(item, 'node timing') for item in value)
        if end < start:
            raise ValueError('node finish precedes start')
        intervals[node.id] = (start, end)
    finish = max((end for _, end in intervals.values()), default=0)
    storages = {}
    for item in storage:
        if not isinstance(item.id, str) or not item.id or item.id in storages:
            raise ValueError('storage ids must be unique nonempty strings')
        if item.rank not in ranks:
            raise ValueError('storage has unknown rank')
        _bytes(item.size, 'storage.size')
        if type(item.persistent) is not bool or not isinstance(item.category, str) or not item.category:
            raise ValueError('invalid storage persistence or category')
        storages[item.id] = item
    aliases = memory.get('storage_aliases', {})
    if not isinstance(aliases, dict) or any(key not in storages or value not in storages for key, value in aliases.items()):
        raise ValueError('storage_aliases must reference existing storage ids')

    def root(sid):
        seen = set()
        while sid in aliases:
            if sid in seen:
                raise ValueError('storage alias cycle')
            seen.add(sid)
            sid = aliases[sid]
        return sid

    records = {}
    for sid, item in storages.items():
        users = set(item.consumers)
        if item.producer is not None:
            users.add(item.producer)
        if users - intervals.keys():
            raise ValueError('unknown storage producer or consumer')
        start = intervals[item.producer][0] if item.producer is not None else 0
        end = finish if item.persistent or not users else max(intervals[user][1] for user in users)
        if any(intervals[user][0] < start for user in item.consumers):
            raise ValueError('storage consumer starts before allocation')
        identity = root(sid)
        device = placement[str(item.rank)]
        record = records.setdefault(identity, dict(device=device, lower=item.size, upper=item.size,
                                                   category=storages[identity].category, intervals=[], closed_ends=set(), storage=True))
        if (record['device'], record['lower']) != (device, item.size):
            raise ValueError('storage aliases must have identical size and physical device')
        record['intervals'].append((start, end))
        if item.persistent or not users or any(intervals[user] == (end, end) for user in users):
            record['closed_ends'].add(end)

    extras = memory.get('allocations', [])
    if not isinstance(extras, list):
        raise ValueError('memory.allocations must be a list')
    seen = set(storages)
    credible_aliases = {}
    for allocation in extras:
        if not isinstance(allocation, dict):
            raise ValueError('memory allocation must be an object')
        aid = allocation.get('id')
        if not isinstance(aid, str) or not aid or aid in seen:
            raise ValueError('allocation id must be a unique nonempty string')
        seen.add(aid)
        category = allocation.get('category')
        if category not in ALLOCATION_CATEGORIES[1:]:
            raise ValueError('unknown allocation category; allocator reservations must be expressed as allocator_slack (reserved minus allocated)')
        if ('rank' in allocation) == ('device' in allocation):
            raise ValueError('allocation needs exactly one of rank or device')
        if 'rank' in allocation:
            rank = _identifier(allocation['rank'], 'rank')
            if rank not in placement:
                raise ValueError('allocation references unknown rank')
            device = placement[rank]
        else:
            device = _identifier(allocation['device'], 'device')
        if device not in device_specs:
            raise ValueError('allocation references unknown device')
        if 'bytes' in allocation:
            if 'lower_bytes' in allocation or 'upper_bytes' in allocation:
                raise ValueError('use bytes or lower_bytes/upper_bytes, not both')
            lower = upper = _bytes(allocation['bytes'], 'allocation.bytes')
        else:
            lower = _bytes(allocation.get('lower_bytes'), 'allocation.lower_bytes')
            if 'upper_bytes' not in allocation:
                raise ValueError('allocation.upper_bytes is required (null means unknown)')
            upper = allocation['upper_bytes']
            if upper is not None:
                _bytes(upper, 'allocation.upper_bytes')
                if upper < lower:
                    raise ValueError('allocation upper bound is below lower bound')
        start = _seconds(allocation.get('start_seconds', 0), 'allocation.start_seconds')
        end = _seconds(allocation.get('end_seconds', finish), 'allocation.end_seconds')
        if end < start:
            raise ValueError('allocation end precedes start')
        if 'storage_id' in allocation:
            sid = allocation['storage_id']
            if sid not in storages:
                raise ValueError('allocation aliases unknown storage')
            record = records[root(sid)]
            if (record['device'], record['lower'], record['upper']) != (device, lower, upper):
                raise ValueError('aliased external allocation must match storage size and device')
            record['intervals'].append((start, end))
            if storage_evidence == 'unverified':
                # Only the separately attested interval supplies a lower bound;
                # do not inherit the projection's conservative lifetime.
                proof = credible_aliases.setdefault(root(sid), dict(
                    device=device, lower=lower, upper=0, category=category,
                    intervals=[], closed_ends=set()))
                proof['intervals'].append((start, end))
            if 'end_seconds' not in allocation:
                record['closed_ends'].add(end)
        else:
            records[aid] = dict(device=device, lower=lower, upper=upper,
                                category=category, intervals=[(start, end)],
                                closed_ends={end} if 'end_seconds' not in allocation else set())

    lifecycle = defaultdict(list)
    lifecycle_peaks = memory.get('lifecycle_peaks', [])
    if not isinstance(lifecycle_peaks, list):
        raise ValueError('memory.lifecycle_peaks must be a list')
    for entry in lifecycle_peaks:
        if not isinstance(entry, dict) or entry.get('phase') not in LIFECYCLE_PHASES:
            raise ValueError('lifecycle peak needs an initialization or checkpoint_loading phase')
        if ('rank' in entry) == ('device' in entry):
            raise ValueError('lifecycle peak needs exactly one of rank or device')
        if 'rank' in entry:
            rank = _identifier(entry['rank'], 'rank')
            if rank not in placement:
                raise ValueError('lifecycle peak references unknown rank')
            device = placement[rank]
            if sum(value == device for value in placement.values()) != 1:
                raise ValueError('collocated lifecycle peaks must give a device-wide bound, not a rank peak')
        else:
            device = _identifier(entry['device'], 'device')
        if device not in device_specs:
            raise ValueError('lifecycle peak references unknown device')
        lower = _bytes(entry.get('lower_bytes'), 'lifecycle.lower_bytes')
        if 'upper_bytes' not in entry:
            raise ValueError('lifecycle.upper_bytes is required (null means unknown)')
        upper = entry['upper_bytes']
        if upper is not None:
            _bytes(upper, 'lifecycle.upper_bytes')
            if upper < lower:
                raise ValueError('lifecycle upper bound is below lower bound')
        lifecycle[device].append((entry['phase'], lower, upper))

    global_incomplete = bool(issues)
    per_device = {}
    for device, (capacity, budget) in device_specs.items():
        selected = [record for record in records.values() if record['device'] == device]
        modeled_lower, _, _ = _peak(selected, 'lower')
        credible = [dict(record, lower=0) if storage_evidence == 'unverified' and record.get('storage')
                    else record for record in selected]
        credible.extend(record for record in credible_aliases.values() if record['device'] == device)
        lower, lower_time, lower_categories = _peak(credible, 'lower')
        upper, upper_time, upper_categories = _peak(selected, 'upper')
        scheduled_lower, scheduled_upper = lower, upper
        peak_phase = 'schedule'
        for phase, phase_lower, phase_upper in lifecycle[device]:
            modeled_lower = max(modeled_lower, phase_lower)
            if phase_lower > lower:
                lower, lower_time, lower_categories = phase_lower, None, {phase: phase_lower}
                peak_phase = phase
            if upper is not None:
                if phase_upper is None:
                    upper, upper_time, upper_categories = None, None, {}
                elif phase_upper > upper:
                    upper, upper_time, upper_categories = phase_upper, None, {phase: phase_upper}
        if upper is None:
            issues.append('unbounded allocation on device ' + device)
        per_device[device] = dict(
            ranks=[rank for rank in ranks if placement[str(rank)] == device],
            capacity_bytes=capacity, budget_bytes=budget, placement_verified=placement_known,
            peak_lower_bytes=lower, modeled_peak_lower_bytes=modeled_lower, modeled_peak_upper_bytes=upper,
            peak_upper_bytes=upper if not global_incomplete else None,
            scheduled_peak_lower_bytes=scheduled_lower, scheduled_peak_upper_bytes=scheduled_upper,
            peak_lower_phase=peak_phase,
            peak_lower_time_seconds=lower_time, peak_upper_time_seconds=upper_time,
            peak_lower_categories_bytes=lower_categories,
            peak_upper_categories_bytes=upper_categories,
        )
    # Unknown evidence affects every device unless explicitly localized by an
    # allocation with an unknown upper bound. Never expose a finite fit bound
    # while global coverage is incomplete.
    global_issues = [issue for issue in issues if not issue.startswith('unbounded allocation on device ')]
    if global_issues:
        for item in per_device.values():
            item['peak_upper_bytes'] = None
    not_fit = placement_known and any(item['peak_lower_bytes'] > item['budget_bytes'] for item in per_device.values())
    fits = bool(per_device) and all(item['peak_upper_bytes'] is not None and item['peak_upper_bytes'] <= item['budget_bytes'] for item in per_device.values())
    return dict(memory_verdict='estimated_not_fit' if not_fit else 'estimated_fit' if fits else 'inconclusive',
                per_device=per_device, memory_unresolved=list(dict.fromkeys(issues)),
                memory_assumptions=assumptions,
                memory_scope=scope, storage_evidence=storage_evidence)
