"""Derive allocation generations and conditional envelopes from native evidence.

Allocator history establishes recorded allocation/reuse facts, not a bound on a
future invocation. Every forecast produced here remains conditional. Missing
physical domains are explicit even when the observed allocator history closes.
"""
from copy import deepcopy

from .allocation_capture import SCHEMA
from .identity import digest
from .memory import COVERAGE_CATEGORIES
from .memory_evidence import build_evidence_bundle


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + ' must be a nonnegative integer')
    return value


def _blocks(snapshot, device):
    blocks, reserved = {}, 0
    for segment in snapshot['segments']:
        if segment.get('device') != device:
            continue
        address = _integer(segment['address'], 'segment address')
        total = _integer(segment['total_size'], 'segment size')
        reserved += total
        offset = address
        for block in segment['blocks']:
            size = _integer(block['size'], 'block size')
            start = _integer(block.get('address', offset), 'block address')
            if start != offset or start + size > address + total:
                raise ValueError('noncontiguous allocator blocks')
            offset += size
            state = block['state']
            # Newer native allocators name the same stream-pending state
            # active_pending_free. Reconciliation below still requires an
            # actual free_requested and forbids reuse before free_completed.
            if state == 'active_pending_free':
                state = 'active_awaiting_free'
            if state not in ('active_allocated', 'active_awaiting_free', 'inactive'):
                raise ValueError('unknown allocator block state')
            if state != 'inactive':
                if start in blocks:
                    raise ValueError('overlapping allocator block addresses')
                blocks[start] = dict(bytes=size, state=state,
                    requested_bytes=_integer(block.get('requested_size', size), 'requested size'),
                    pool_id=list(segment.get('segment_pool_id', (0, 0))))
        if offset != address + total:
            raise ValueError('allocator segment is not fully partitioned')
    return blocks, reserved


def _trace(snapshot, device):
    traces = snapshot.get('device_traces')
    if not isinstance(traces, list):
        raise ValueError('native allocator history unavailable')
    # A CUDA device with no allocator activity can have no trace slot yet.
    return traces[device] if len(traces) > device else []


def analyze_allocation_capture(capture):
    """Audit raw native history and derive reuse generations and state aliases.

    Endpoint reconciliation checks allocation bytes and pending asynchronous
    frees. Prefix preservation plus the owned bounded recorder detects ring
    truncation. External CUDA allocators and compiler-private workspaces are
    outside this domain and never receive complete physical coverage.
    """
    if capture.get('schema') != SCHEMA:
        raise ValueError('unsupported native allocation capture schema')
    if capture.get('sha256') != digest({k: v for k, v in capture.items() if k != 'sha256'}):
        raise ValueError('native allocation capture checksum mismatch')
    device = _integer(capture['device_index'], 'device index')
    samples = capture['samples']
    issues = list(capture.get('unresolved', []))
    history = capture.get('history', {})
    owned = history.get('started_by_capture') is True
    limit = history.get('max_entries')
    if owned and (type(limit) is not int or limit < 2):
        raise ValueError('owned history requires a positive bounded recorder')
    if not owned:
        issues.append('allocator history settings are unknown; omitted transient events cannot be excluded')
    if len(samples) < 2:
        issues.append('at least two native allocator snapshots are required')
    rows, live, identities, generations, states, boundaries = [], {}, {}, {}, [], []
    previous = None
    active_bytes = peak = 0
    reserved_bytes = None

    def create(address, size, index, *, baseline=False, awaiting=False, pool_id=None):
        nonlocal active_bytes, peak
        _integer(address, 'allocation address')
        _integer(size, 'allocation size')
        if address in live:
            raise ValueError('allocation address reused before free_completed')
        if address not in identities:
            identities[address] = 'allocator:' + str(len(identities))
        generation = generations.get(address, -1) + 1
        generations[address] = generation
        row = dict(allocation_id=identities[address], generation=generation, bytes=size,
                   creation_sample=None if baseline else index, creation_node_id=None,
                   free_requested_sample=0 if awaiting else None, free_completed_sample=None,
                   completion_node_ids=[], retained_at_capture_end=True,
                   state_aliases=[], pool_id=pool_id)
        if not baseline:
            row['creation_node_id'] = region_node(index)
        rows.append(row)
        live[address] = row
        active_bytes += size
        peak = max(peak, active_bytes)

    def region_node(index):
        current = samples[index]
        prior = samples[index - 1] if index else {}
        if (current.get('edge') == 'after' and prior.get('edge') == 'before' and
                current.get('node_id') == prior.get('node_id') and
                current.get('phase') == prior.get('phase') == 'schedule'):
            return current.get('node_id')
        return None

    for index, sample in enumerate(samples):
        state = deepcopy(sample.get('state', {}))
        states.append(dict(sample_index=index, label=sample.get('label'), phase=sample.get('phase'),
                           **state))
        issues.extend('state sample ' + str(index) + ': ' + text for text in state.get('unresolved', []))
        try:
            snapshot = sample.get('snapshot')
            if not isinstance(snapshot, dict):
                raise ValueError('native allocator snapshot unavailable')
            blocks, reserved = _blocks(snapshot, device)
            trace = _trace(snapshot, device)
            if owned and len(trace) >= limit:
                raise ValueError('allocator history reached bounded capacity; possible ring truncation')
            if index == 0:
                reserved_bytes = reserved
                for address, block in blocks.items():
                    create(address, block['bytes'], index, baseline=True,
                           awaiting=block['state'] == 'active_awaiting_free', pool_id=block['pool_id'])
            else:
                if previous is None or len(trace) < len(previous) or trace[:len(previous)] != previous:
                    raise ValueError('allocator history prefix lost or changed')
                new = trace[len(previous):]
                if not any(event.get('action') == 'snapshot' for event in new):
                    raise ValueError('allocator history lacks a new snapshot boundary')
                for event in new:
                    action = event['action']
                    if action == 'alloc':
                        create(event['addr'], event['size'], index, pool_id=event.get('pool_id'))
                    elif action in ('free_requested', 'free_completed'):
                        address = event['addr']
                        if address not in live or live[address]['bytes'] != event['size']:
                            raise ValueError('free does not match a live allocation generation')
                        row = live[address]
                        if action == 'free_requested':
                            if row['free_requested_sample'] is not None:
                                raise ValueError('duplicate free_requested for one allocation generation')
                            row['free_requested_sample'] = index
                        else:
                            if row['free_requested_sample'] is None:
                                raise ValueError('free_completed lacks free_requested history')
                            row['free_completed_sample'] = index
                            node = region_node(index)
                            row['completion_node_ids'] = [] if node is None else [node]
                            row['retained_at_capture_end'] = False
                            active_bytes -= row['bytes']
                            del live[address]
                    elif action == 'oom':
                        issues.append('native allocator recorded OOM; execution is incomplete')
                    elif action in ('segment_alloc', 'segment_map', 'segment_free', 'segment_unmap'):
                        # Expandable segments reserve a large virtual arena,
                        # then map physical pages on demand. Only mapped bytes
                        # enter the physical reservation, not the arena size.
                        size = _integer(event['size'], 'segment event size')
                        reserved_bytes += size if action in ('segment_alloc', 'segment_map') else -size
                        if reserved_bytes < 0:
                            raise ValueError('segment history released unrecorded reserved bytes')
                    elif action not in ('snapshot', 'annotate'):
                        raise ValueError('unsupported allocator action: ' + str(action))
            if reserved_bytes != reserved:
                raise ValueError('segment history and endpoint reserved bytes disagree')
            if set(live) != set(blocks):
                raise ValueError('allocator history and endpoint live blocks disagree')
            for address, row in live.items():
                block = blocks[address]
                if row['bytes'] != block['bytes']:
                    raise ValueError('allocator history and endpoint block sizes disagree')
                if ((row['free_requested_sample'] is not None) != (block['state'] == 'active_awaiting_free')):
                    raise ValueError('allocator asynchronous free state disagrees with snapshot')
                row['pool_id'] = block['pool_id']
            for entry in state.get('allocations', []):
                if entry.get('device') != 'cuda:' + str(device) or entry.get('bytes') == 0:
                    continue
                row = live.get(entry['address'])
                if row is None or entry['bytes'] > row['bytes']:
                    issues.append('state storage is outside the recorded allocator: sample ' + str(index) + ':' + entry['id'])
                    continue
                row['state_aliases'].append(dict(sample_index=index, state_id=entry['id'],
                    bytes=entry['bytes'], aliases=entry['aliases'], categories=entry['categories']))
            boundaries.append(dict(sample_index=index, phase=sample.get('phase'),
                active_bytes=sum(block['bytes'] for block in blocks.values()),
                reserved_bytes=reserved, allocator_slack_bytes=reserved - sum(block['bytes'] for block in blocks.values())))
            previous = trace
        except (KeyError, TypeError, ValueError) as error:
            issues.append('allocator sample ' + str(index) + ': ' + str(error))
            previous = None
    complete = bool(samples) and len(samples) >= 2 and owned and not issues
    phases = sorted({sample.get('phase', 'schedule') for sample in samples})
    physical_gaps = [
        'recorded allocator events do not bound future control flow or lazy state creation',
        'external CUDA allocations and library/communication workspaces are unbounded',
        'allocator reservation and graph/private-pool growth have no predictive upper bound',
        'initialization and checkpoint-loading physical peaks require separate evidence',
        'host region boundaries do not establish CUDA stream completion',
    ]
    result = dict(schema='titan_sim.native_allocation_analysis.v1', capture_sha256=capture['sha256'],
        device=capture['physical_device'], rank=capture['rank'], allocations=rows,
        state_samples=states, allocator_boundaries=boundaries,
        observed_allocator_active_peak_bytes=peak if complete else None,
        recorded_allocator_coverage=[dict(device=capture['physical_device'], phase=phase,
            domain='process native caching allocator', complete=complete) for phase in phases],
        reconstruction_complete=complete, physical_peak_upper_bytes=None,
        supports_unseen_configuration_fit=False, unresolved=list(dict.fromkeys(issues + physical_gaps)))
    result['sha256'] = digest(result)
    return result


def produce_allocation_evidence(workload, capture, *, applicability, storage_bindings=None):
    """Produce binder-ready conditional records from audited native events.

    ``storage_bindings`` maps existing workload storage IDs to
    ``{sample_index, state_id}`` from the actual native inventory. It supplies
    identity correspondence only: sizes, aliases and generations are checked
    against native evidence. Missing correspondence produces observations only,
    preventing double counting of projected storage and allocator blocks.
    """
    analysis = analyze_allocation_capture(capture)
    device = capture['physical_device']
    if capture.get('provenance', {}).get('applicability') != applicability:
        raise ValueError('native capture applicability mismatch; record the exact source/target/runtime context')
    if applicability['placement'].get(str(capture['rank'])) != device:
        raise ValueError('native capture physical placement mismatch')
    evidence_id = 'native-capture:' + capture['sha256']
    contract_id = 'native-allocator-replay:' + capture['sha256']
    allocations, bindings = [], storage_bindings or {}
    projected = {item.id: item for item in workload.storage
                 if applicability['placement'].get(str(item.rank)) == device}
    if set(bindings) - set(projected):
        raise ValueError('storage binding references unknown or other-device storage')
    mapped = {}
    for sid, binding in bindings.items():
        matches = [(row, alias) for row in analysis['allocations'] for alias in row['state_aliases']
                   if alias['sample_index'] == binding.get('sample_index') and alias['state_id'] == binding.get('state_id')]
        if len(matches) != 1 or matches[0][1]['bytes'] != projected[sid].size:
            raise ValueError('storage binding must match one native generation with exact backing bytes')
        row, alias = matches[0]
        mapped.setdefault((row['allocation_id'], row['generation']), []).append(sid)
    issues = list(analysis['unresolved'])
    can_forecast = analysis['reconstruction_complete'] and set(bindings) == set(projected)
    if set(bindings) != set(projected):
        issues.append('projected storage lacks native identity correspondence; no allocations added')
    if workload.manifest.get('memory', {}).get('allocations'):
        can_forecast = False
        issues.append('existing extra allocations lack native identity correspondence; no allocations added')
    if any(sample.get('phase') != 'schedule' for sample in capture['samples']):
        can_forecast = False
        issues.append('multi-phase allocation-to-schedule lowering is unsupported; retained as observations')
    # The binder's storage-alias contracts require equal size bounds over the
    # entire interval. Native region boundaries only bound transient lifetimes;
    # they do not prove coexistence throughout those widened intervals. A store
    # present at entry and never freed is the supported exception under replay.
    if any(mapped.get((row['allocation_id'], row['generation'])) and
           not (row['creation_sample'] is None and row['retained_at_capture_end'])
           for row in analysis['allocations']):
        can_forecast = False
        issues.append('projected transient storage requires finer lifetime correspondence; no allocations added')
    if can_forecast:
        by_node = {node.id: node for node in workload.nodes}
        for row in analysis['allocations']:
            ids = mapped.get((row['allocation_id'], row['generation']), [])
            size = projected[ids[0]].size if ids else row['bytes']
            if any(projected[sid].size != size for sid in ids):
                raise ValueError('native aliases have incompatible backing sizes')
            events = row['completion_node_ids'] + ([] if row['creation_node_id'] is None else [row['creation_node_id']])
            if any(event not in by_node or by_node[event].rank != capture['rank'] for event in events):
                raise ValueError('native region references an unknown or other-rank workload node')
            result = dict(allocation_id=row['allocation_id'], generation=row['generation'], device=device,
                phase='schedule', category='storage', kind='conditional',
                lower_bytes=size if row['creation_sample'] is None and row['retained_at_capture_end'] else 0,
                upper_bytes=size,
                creation_node_id=row['creation_node_id'], completion_node_ids=row['completion_node_ids'],
                persistent=row['retained_at_capture_end'], storage_ids=ids,
                contract_id=contract_id, evidence_ref=evidence_id,
                conditions=['the target repeats the captured native allocation sizes, order, control flow and state creation',
                            'native region identity and completion ordering match the simulated workload',
                            'allocations retained at capture end remain resident for the modeled schedule'])
            allocations.append(result)
            if ids and row['bytes'] > size:
                slack = deepcopy(result)
                slack.update(allocation_id=row['allocation_id'] + ':rounding', category='allocator_slack',
                             storage_ids=[], lower_bytes=row['bytes'] - size, upper_bytes=row['bytes'] - size)
                allocations.append(slack)
    coverage = [dict(device=device, phase=phase, category=category, complete=False,
        contract_id=contract_id, evidence_ref=evidence_id,
        statement='native observed allocator/state domain does not establish predictive physical coverage')
        for phase in applicability['phases'] for category in COVERAGE_CATEGORIES]
    observation = dict(kind='observation', method='native_allocator_history',
        capture_sha256=capture['sha256'], analysis=analysis, physical_peak_upper_bytes=None,
        supports_unseen_configuration_fit=False, unresolved=issues)
    return build_evidence_bundle(workload, applicability=applicability, allocations=allocations,
        observations=[observation], coverage=coverage,
        evidence_refs=[dict(id=evidence_id, sha256=capture['sha256'])],
        producer_contracts=[dict(id=contract_id, producer='titan_sim.allocation_contracts',
            statement='native sizes/generations and free_completed events; future reuse is conditional',
            evidence_ref=evidence_id, applicability_sha256=digest(applicability))])
