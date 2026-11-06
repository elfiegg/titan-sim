"""Experimental, framework-independent discrete-event and storage simulator."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import heapq
import hashlib
import json
import math


@dataclass
class Node:
    id: int
    name: str
    rank: int
    kind: str = 'compute'
    deps: list[int] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


@dataclass
class Storage:
    id: str
    rank: int
    size: int
    producer: int | None = None
    consumers: list[int] = field(default_factory=list)
    persistent: bool = False
    category: str = 'tensor'


def validate(nodes, storage, manifest):
    by_id = {n.id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('duplicate node id')
    ranks = set(manifest['ranks'])
    remaining, successors = {}, defaultdict(list)
    for n in nodes:
        if n.kind not in ['compute', 'send', 'recv', 'collective']:
            raise ValueError('unsupported event kind: '+str(n.kind))
        if 'cost_key' in n.attrs and (not isinstance(n.attrs['cost_key'], str) or not n.attrs['cost_key']):
            raise ValueError('cost_key must be a nonempty string')
        if n.rank not in ranks:
            raise ValueError('unknown rank')
        if len(set(n.deps)) != len(n.deps):
            raise ValueError('duplicate dependency')
        remaining[n.id] = len(n.deps)
        for dep in n.deps:
            if dep not in by_id:
                raise ValueError('missing dependency: ' + str(dep))
            successors[dep].append(n.id)
        for key in ['flops', 'bytes', 'payload_bytes', 'legacy_seconds']:
            if key in n.attrs and (not math.isfinite(n.attrs[key]) or n.attrs[key] < 0):
                raise ValueError('invalid cost quantity')
    ready = deque(i for i, d in remaining.items() if d == 0)
    visited = 0
    while ready:
        visited += 1
        for child in successors[ready.popleft()]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        raise ValueError('dependency cycle')
    seen = set()
    for s in storage:
        if s.id in seen:
            raise ValueError('storage ids must identify unique allocations, not tensor views')
        seen.add(s.id)
        if s.size < 0 or s.rank not in ranks:
            raise ValueError('invalid allocation')
        if s.producer is not None and s.producer not in by_id:
            raise ValueError('unknown allocation producer')
        if any(c not in by_id for c in s.consumers):
            raise ValueError('unknown storage consumer')
    return {'nodes': len(nodes), 'storages': len(storage), 'ranks': len(ranks)}


class CostModel:
    """Calibration is exact-key lookup; no unlabelled fallback or inferred measurements."""
    def __init__(self, hardware, calibration=None):
        self.hw = hardware
        self.calibration = calibration or {}
        self.coverage = Counter()

    def seconds(self, n):
        a, h = n.attrs, self.hw
        key = a.get('cost_key')
        if key and key in self.calibration:
            rec = self.calibration[key]
            if rec.get('hardware_id') != h['id'] or not rec.get('evidence_sha256'):
                raise ValueError('calibration hardware/evidence mismatch: ' + key)
            if 'samples_seconds' not in rec or hashlib.sha256(json.dumps(rec['samples_seconds'],separators=(',',':')).encode()).hexdigest() != rec['evidence_sha256']:
                raise ValueError('calibration samples missing or checksum mismatch')
            t = float(rec['seconds'])
            source = 'measured_exact_key'
        elif n.kind in ['send', 'recv', 'collective']:
            b = a.get('payload_bytes', 0)
            bw = h['interconnect_bytes_per_second'] if n.kind == 'collective' else h['pp_bytes_per_second']
            t = h['comm_latency_seconds'] + b / bw
            source = 'analytical_network_assumption'
        elif 'legacy_seconds' in a:
            t = a['legacy_seconds'] * h.get('cost_multiplier', 1.0)
            source = 'historical_analytical_proxy'
        elif 'flops' in a or 'bytes' in a:
            t = max(a.get('flops', 0) / h['effective_flops_per_second'],
                    a.get('bytes', 0) / h['effective_hbm_bytes_per_second'])
            t += h.get('launch_seconds', 0)
            source = 'analytical_roofline_assumption'
        elif a.get('zero_cost_semantic'):
            t, source = 0., 'semantic_zero'
        else:
            raise ValueError('no cost model for ' + n.name)
        if not math.isfinite(t) or t < 0:
            raise ValueError('invalid duration')
        self.coverage[source] += 1
        return t


def simulate(nodes, storage, manifest, hardware, calibration=None, strict=False):
    validation = validate(nodes, storage, manifest)
    if strict and manifest.get('unresolved'):
        raise ValueError('incomplete workload: ' + '; '.join(manifest['unresolved']))
    model = CostModel(hardware, calibration)
    by_id = {n.id: n for n in nodes}
    # A paired transfer / collective becomes one rendezvous event. All local
    # participants must be ready before any of them can complete.
    groups = defaultdict(list)
    for n in nodes:
        key = n.attrs.get('rendezvous', 'node:' + str(n.id))
        groups[key].append(n)
    owner = {n.id: key for key, ns in groups.items() for n in ns}
    durations, resources, deps = {}, {}, {}
    for key, ns in groups.items():
        comm = [n for n in ns if n.kind in ['send', 'recv', 'collective']]
        if comm:
            if len(comm) != len(ns):
                raise ValueError('mixed communication rendezvous')
            if comm[0].kind == 'collective':
                expected = comm[0].attrs['participants']
                if sorted(n.rank for n in ns) != sorted(expected):
                    raise ValueError('collective participant mismatch')
                if any(n.attrs['participants'] != expected for n in ns):
                    raise ValueError('collective group mismatch')
            else:
                if sorted(n.kind for n in ns) != ['recv', 'send']:
                    raise ValueError('unmatched point-to-point transfer')
                if ns[0].attrs['payload_bytes'] != ns[1].attrs['payload_bytes']:
                    raise ValueError('point-to-point size mismatch')
                if ns[0].attrs.get('peer') != ns[1].rank or ns[1].attrs.get('peer') != ns[0].rank:
                    raise ValueError('point-to-point peer mismatch')
        durations[key] = max(model.seconds(n) for n in ns)
        resources[key] = set()
        for n in ns:
            resources[key].update(n.attrs.get('resources', ['gpu:' + str(n.rank)]))
        deps[key] = {owner[d] for n in ns for d in n.deps}
        if key in deps[key]:
            raise ValueError('dependency within communication rendezvous')
    if strict and any(k not in ['measured_exact_key', 'semantic_zero'] for k in model.coverage):
        raise ValueError('uncalibrated operation costs')
    successors = defaultdict(set)
    remaining = {k: len(v) for k, v in deps.items()}
    for key, parents in deps.items():
        for p in parents:
            successors[p].add(key)
    priorities = {k: min(n.id for n in v) for k, v in groups.items()}
    ready = [(priorities[k], k) for k, v in remaining.items() if not v]
    heapq.heapify(ready)
    running, held, done, timing = [], set(), set(), {}
    now, serial = 0., 0
    # Allocations start when their producer is issued, and survive until ALL
    # registered users finish. Tensor aliases share one Storage record.
    alloc_at = defaultdict(list)
    consumed_at = defaultdict(list)
    users, live = {}, set()
    usage = defaultdict(int)
    peak = defaultdict(int)
    category = defaultdict(Counter)
    peak_categories, peak_times = {}, {}
    allocations = {s.id: s for s in storage}
    def allocate(s):
        if s.id in live:
            return
        live.add(s.id)
        usage[s.rank] += s.size
        category[s.rank][s.category] += s.size
        if usage[s.rank] > peak[s.rank]:
            peak[s.rank] = usage[s.rank]
            peak_categories[s.rank] = dict(category[s.rank])
            peak_times[s.rank] = now
    for s in storage:
        # The producer is also a user, so even an unused output survives its op.
        u = {owner[c] for c in s.consumers}
        if s.producer is not None:
            u.add(owner[s.producer])
            alloc_at[owner[s.producer]].append(s)
        else:
            allocate(s)
        users[s.id] = u
        for k in u:
            consumed_at[k].append(s.id)
    while ready or running:
        waiting = []
        while ready:
            priority, key = heapq.heappop(ready)
            if held & resources[key]:
                waiting.append((priority, key))
                continue
            held.update(resources[key])
            for s in alloc_at[key]:
                allocate(s)
            finish = now + durations[key]
            timing[key] = (now, finish)
            heapq.heappush(running, (finish, serial, key))
            serial += 1
        for item in waiting:
            heapq.heappush(ready, item)
        if not running:
            raise ValueError('scheduler deadlock')
        now = running[0][0]
        completed = []
        while running and running[0][0] == now:
            _, _, key = heapq.heappop(running)
            held.difference_update(resources[key])
            done.add(key)
            completed.append(key)
        for key in completed:
            for sid in consumed_at[key]:
                users[sid].discard(key)
                s = allocations[sid]
                if not users[sid] and not s.persistent and sid in live:
                    live.remove(sid)
                    usage[s.rank] -= s.size
                    category[s.rank][s.category] -= s.size
            for child in successors[key]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    heapq.heappush(ready, (priorities[child], child))
    if len(done) != len(groups):
        raise ValueError('rendezvous contraction created a cycle / unresolved dependency')
    total_tokens = manifest.get('useful_tokens', 0)
    budget = hardware['capacity_bytes'] * (1 - hardware.get('reserve_fraction', 0.1))
    worst = max(peak, key=peak.get) if peak else None
    incomplete = bool(manifest.get('unresolved'))
    return {
        'validation': validation, 'event_count': len(groups), 'seconds': now,
        'useful_tokens_per_second': total_tokens / now if now else None,
        'cost_coverage': dict(model.coverage),
        'memory_verdict': 'inconclusive' if incomplete else ('estimated_fit' if max(peak.values(), default=0) <= budget else 'estimated_not_fit'),
        'status': 'exploratory_uncalibrated' if incomplete or any(k not in ['measured_exact_key','semantic_zero'] for k in model.coverage) else 'calibrated_prediction',
        'worst_rank': worst, 'modeled_peak_bytes': dict(peak),
        'peak_time_seconds': peak_times, 'peak_categories_bytes': peak_categories,
        'budget_bytes': budget, 'unresolved': manifest.get('unresolved', []),
        'timing': {str(n.id): timing[owner[n.id]] for n in nodes},
    }
