"""Strict participant rendezvous and explicit lowering to the event scheduler.

The input order is each rank's submission order, never an inferred global clock.
A rendezvous is one network operation with one completion token on each rank.
"""
from collections import defaultdict
from dataclasses import asdict
from .events import Event

_COLLECTIVES = {'all_reduce', 'all_gather', 'reduce_scatter', 'all_to_all', 'barrier'}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _event_dict(event):
    data = asdict(event)
    if not data['name']:
        del data['name']
    return data


def lower_distributed_events(events, groups):
    """Match explicit group/sequence participants, rejecting incomplete traces.

    Participant metadata ``distributed`` contains group, sequence, op and
    payload_bytes. P2P send/recv also require peer and tag. All-to-all split
    vectors are measured in BYTES in the group's explicitly supplied order.
    Payload means local input bytes (full input for reduce-scatter). The original
    event ID becomes a completion wait, preserving all downstream dependencies.
    """
    groups = {name: tuple(ranks) for name, ranks in groups.items()}
    for name, ranks in groups.items():
        if not isinstance(name, str) or not name or not ranks or len(set(ranks)) != len(ranks):
            raise ValueError('groups require unique nonempty IDs and unique nonempty rank lists')
        for rank in ranks:
            _integer(rank, 'group rank')
    parsed = [e if isinstance(e, Event) else Event.from_dict(e) for e in events]
    ids = [e.event_id for e in parsed]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate event_id')
    if any(x.startswith('__dist_') for x in ids):
        raise ValueError('event IDs beginning __dist_ are reserved')
    buckets, member_key, previous = {}, {}, {}
    for event in parsed:
        spec = event.metadata.get('distributed')
        if spec is None:
            continue
        if event.kind != 'collective' or not isinstance(spec, dict):
            raise ValueError('distributed participants must be collective events with object metadata')
        group = spec.get('group')
        if group not in groups or event.rank not in groups[group]:
            raise ValueError(f'{event.event_id}: unknown group or rank outside group')
        seq = _integer(spec.get('sequence'), 'sequence')
        payload = _integer(spec.get('payload_bytes'), 'payload_bytes')
        op = spec.get('op')
        if op in ('send', 'recv'):
            peer = _integer(spec.get('peer'), 'peer')
            tag = _integer(spec.get('tag'), 'tag')
            if peer not in groups[group] or peer == event.rank:
                raise ValueError('P2P peer must be a different member of the group')
            src, dst = (event.rank, peer) if op == 'send' else (peer, event.rank)
            key = ('p2p', group, src, dst, tag, seq)
        elif op in _COLLECTIVES:
            key = ('collective', group, seq)
            order_key = (group, event.rank)
            if seq <= previous.get(order_key, -1):
                raise ValueError(f'{event.event_id}: collective sequence must increase on every rank')
            previous[order_key] = seq
        else:
            raise ValueError(f'unsupported distributed operation: {op!r}')
        member_key[event.event_id] = key
        buckets.setdefault(key, []).append(event)
    transfers, arrival_ids = {}, {}
    for index, (key, members) in enumerate(buckets.items()):
        group = key[1]
        specs = [e.metadata['distributed'] for e in members]
        op = specs[0]['op']
        ranks = groups[group] if key[0] == 'collective' else (key[2], key[3])
        if len(members) != len(ranks) or {e.rank for e in members} != set(ranks):
            raise ValueError(f'{key}: missing or duplicate participants; expected {list(ranks)}, got {[e.rank for e in members]}')
        if len({e.dtype for e in members}) != 1:
            raise ValueError(f'{key}: mismatched dtype')
        if key[0] == 'p2p':
            if {s['op'] for s in specs} != {'send', 'recv'}:
                raise ValueError(f'{key}: missing matching send/recv')
            op = 'p2p'
        elif len({s['op'] for s in specs}) != 1:
            raise ValueError(f'{key}: mismatched collective kind')
        payloads = [s['payload_bytes'] for s in specs]
        if op != 'all_to_all' and len(set(payloads)) != 1:
            raise ValueError(f'{key}: mismatched payload size')
        if op == 'barrier' and any(payloads):
            raise ValueError('barrier payload_bytes must be zero')
        reduction = None
        if op in ('all_reduce', 'reduce_scatter'):
            reductions = [s.get('reduce_op', 'sum') for s in specs]
            if any(not isinstance(value, str) or value not in {'sum', 'min', 'max', 'product', 'avg'}
                   for value in reductions):
                raise ValueError(f'{key}: unsupported reduction operator')
            if len(set(reductions)) != 1:
                raise ValueError(f'{key}: mismatched reduction operator')
            reduction = reductions[0]
        if op == 'reduce_scatter' and payloads[0] % len(ranks):
            raise ValueError('reduce_scatter payload must divide evenly over the group')
        metadata = {'collective': op, 'participants': list(ranks),
                    'participant_ranks': list(ranks), 'group': group,
                    'sequence': specs[0]['sequence'], 'distributed_lowered': True}
        if reduction is not None:
            metadata['reduce_op'] = reduction
        if op == 'all_to_all':
            by_rank = {e.rank: e.metadata['distributed'] for e in members}
            inputs, outputs = [], []
            for rank in ranks:
                spec = by_rank[rank]
                pair = []
                for field in ('input_splits', 'output_splits'):
                    splits = spec.get(field)
                    if not isinstance(splits, (list, tuple)) or len(splits) != len(ranks):
                        raise ValueError(f'all_to_all {field} must provide one byte count per group rank')
                    pair.append([_integer(x, field) for x in splits])
                if sum(pair[0]) != spec['payload_bytes']:
                    raise ValueError('all_to_all input split sum does not match payload_bytes')
                inputs.append(pair[0]); outputs.append(pair[1])
            if any(inputs[i][j] != outputs[j][i] for i in range(len(ranks)) for j in range(len(ranks))):
                raise ValueError('all_to_all send/receive splits mismatch')
            metadata.update(input_splits=inputs, output_splits=outputs)
        message_bytes = max(payloads) * (len(ranks) if op == 'all_gather' else 1)
        metadata['message_bytes'] = message_bytes
        if op == 'p2p':
            metadata.update(src=key[2], dst=key[3], tag=key[4])
        arrivals = [f'__dist_arrival_{index}_{e.rank}' for e in members]
        arrival_ids.update(zip((e.event_id for e in members), arrivals))
        transfer = {'event_id': f'__dist_transfer_{index}', 'kind': 'collective',
                    'rank': members[0].rank, 'stream': f'__dist_network_{index}',
                    'name': op, 'dtype': members[0].dtype, 'deps': arrivals,
                    'bytes_read': message_bytes, 'metadata': metadata}
        durations = {(e.duration_us, e.metadata.get('duration_provenance')) for e in members}
        if any(e.duration_us is not None for e in members):
            if len(durations) != 1:
                raise ValueError('participant explicit durations/provenance must match')
            transfer['duration_us'], metadata['duration_provenance'] = durations.pop()
        transfers[key] = transfer
    # A group has an explicit collective sequence, independent of rank/input merge order.
    by_group = defaultdict(list)
    for key, transfer in transfers.items():
        if key[0] == 'collective':
            by_group[key[1]].append((key[2], transfer))
    for entries in by_group.values():
        ordered = [t for _, t in sorted(entries)]
        for earlier, later in zip(ordered, ordered[1:]):
            later['deps'].append(earlier['event_id'])
    result, emitted = [], set()
    for event in parsed:
        if event.event_id not in member_key:
            result.append(_event_dict(event)); continue
        key = member_key[event.event_id]
        arrival = _event_dict(event)
        arrival.update(event_id=arrival_ids[event.event_id], kind='wait', duration_us=None,
                       flops=0, bytes_read=0, bytes_write=0,
                       read_storage_ids=(), write_storage_ids=())
        arrival['metadata'] = {'distributed_arrival': True}
        result.append(arrival)
        transfer = transfers[key]
        if key not in emitted:
            result.append(transfer)
            emitted.add(key)
        completion = _event_dict(event)
        completion.update(kind='wait', deps=[transfer['event_id']], duration_us=None,
                          flops=0, bytes_read=0, bytes_write=0)
        completion['metadata'] = {k: v for k, v in completion['metadata'].items() if k != 'distributed'}
        completion['metadata']['distributed_completion'] = True
        result.append(completion)
    return result


class SimulatedWorld:
    """Cooperative CPU ProcessGroups backed by a shared virtual execution clock.

    Actual ``torch.distributed`` functions dispatch into ProcessGroup subclasses.
    Tensors MUST be meta/FakeTensor: this backend transports metadata, not values.
    Submit asynchronous calls from each simulated rank, then wait. Waiting before
    every peer submits raises a diagnostic instead of pretending to complete.
    ``backend`` can be the same LiveExecution used by fake device compute/waits.
    """
    def __init__(self, hardware=None, *, groups=None, world_size=2, backend=None):
        from .live import LiveExecution
        if groups is None:
            if _integer(world_size, 'world_size') == 0:
                raise ValueError('world_size must be positive')
            groups = {'world': list(range(world_size))}
        self.groups = {name: list(ranks) for name, ranks in groups.items()}
        if not self.groups:
            raise ValueError('at least one process group is required')
        lower_distributed_events([], self.groups)
        self.backend = backend if backend is not None else LiveExecution(hardware)
        self._pending = {}
        self._works = []
        self._counter = 0
        self._transaction = 0
        self._pgs = {}

    def process_group(self, rank, group='world'):
        if group not in self.groups or rank not in self.groups[group]:
            raise ValueError('unknown group or rank outside group')
        key = (group, rank)
        if key not in self._pgs:
            pg_class, _ = _torch_classes()
            self._pgs[key] = pg_class(self, group, rank)
        return self._pgs[key]

    def _enqueue(self, group, rank, spec, tensors, *, deps=(), reads=(), writes=()):
        _, work_class = _torch_classes()
        event_id = f'pg:{self._counter}:r{rank}'
        self._counter += 1
        event = {'event_id': event_id, 'kind': 'collective', 'rank': rank,
                 'stream': f'pg:{group}' if spec['op'] not in ('send', 'recv') else f'pg:p2p:{event_id}',
                 'dtype': str(tensors[0].dtype).removeprefix('torch.') if tensors else 'float32',
                 'deps': list(deps), 'read_storage_ids': list(reads), 'write_storage_ids': list(writes),
                 'metadata': {'distributed': dict(spec, group=group)}}
        work = work_class(self, event_id, tensors)
        self._works.append(work)
        # Pin backing storage before peers arrive: host ownership may be dropped
        # while this asynchronous operation is still awaiting its rendezvous.
        storage_ids = sorted(set(reads) | set(writes))
        for storage_id in storage_ids:
            references = 0
            for prior in self.backend.events:
                if prior.get('rank', 0) == rank and prior.get('storage_id') == storage_id:
                    references += 1 if prior['kind'] == 'allocate' else -1 if prior['kind'] == 'release' else 0
            if references <= 0:
                raise ValueError(f'cannot pin unknown or released communication storage {storage_id}')
        pins = [{'event_id': f'pin:{event_id}:{index}', 'kind': 'allocate', 'rank': rank,
                 'stream': f'pin:{event_id}', 'storage_id': storage_id, 'size_bytes': 0}
                for index, storage_id in enumerate(storage_ids)]
        if pins:
            self.backend.submit(pins)
        work.storage_pins = storage_ids
        if spec['op'] in ('send', 'recv'):
            src, dst = (rank, spec['peer']) if spec['op'] == 'send' else (spec['peer'], rank)
            key = (group, 'p2p', src, dst, spec['tag'], spec['sequence'])
            count = 2
        else:
            key = (group, 'collective', spec['sequence'])
            count = len(self.groups[group])
        pending = self._pending.setdefault(key, [])
        pending.append((event, work))
        if len(pending) == count:
            try:
                lowered = lower_distributed_events([e for e, _ in pending], self.groups)
                prefix = f'pg_tx{self._transaction}:'
                names = {e['event_id']: prefix + e['event_id'] for e in lowered if e['event_id'].startswith('__dist_')}
                for item in lowered:
                    item['event_id'] = names.get(item['event_id'], item['event_id'])
                    item['deps'] = [names.get(d, d) for d in item['deps']]
                    if item['stream'].startswith('__dist_'):
                        item['stream'] = prefix + item['stream']
                    item['host_time_us'] = self.backend.host_time_us
                for participant, pending_work in pending:
                    for index, storage_id in enumerate(pending_work.storage_pins):
                        lowered.append({'event_id': f'unpin:{pending_work.event_id}:{index}',
                                        'kind': 'release', 'rank': participant['rank'],
                                        'stream': f'pin:{pending_work.event_id}', 'storage_id': storage_id,
                                        'deps': [pending_work.event_id], 'host_time_us': self.backend.host_time_us})
                self.backend.submit(lowered)
                self._transaction += 1
                for _, item in pending:
                    item.submitted = True
            except Exception as error:
                aborted_pins = [{'event_id': f'abort-unpin:{item.event_id}:{index}',
                                 'kind': 'release', 'rank': participant['rank'],
                                 'stream': f'pin:{item.event_id}', 'storage_id': storage_id}
                                for participant, item in pending
                                for index, storage_id in enumerate(item.storage_pins)]
                if aborted_pins:
                    self.backend.submit(aborted_pins)
                for _, item in pending:
                    item.error = error
                    item.future.set_exception(error)
                raise
            finally:
                del self._pending[key]
        return work

    def assert_complete(self):
        if self._pending:
            raise RuntimeError(f'missing distributed participants: {list(self._pending)}')

    def _complete_futures(self):
        for work in self._works:
            if work.submitted and not work.future.done() and self.backend.query(work.event_id):
                work.future.set_result(work.result_tensors)
                work.tensors = []
                work.future.work = None
        self._works = [work for work in self._works if not work.future.done()]


_CLASSES = None


def _torch_classes():
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import torch
    import torch.distributed as dist
    from torch._subclasses.fake_tensor import FakeTensor

    def check(tensors):
        if len(tensors) != 1:
            raise NotImplementedError('simulated ProcessGroup supports one tensor per call')
        for tensor in tensors:
            if tensor.device.type != 'meta' and not isinstance(tensor, FakeTensor):
                raise ValueError('simulated ProcessGroup requires meta or FakeTensor; numeric transport is unsupported')

    class SimulatedFuture(torch.futures.Future):
        def __init__(self, work):
            super().__init__()
            self.work = work

        def wait(self):
            if self.work is not None:
                self.work.wait()
            return super().wait()

    class SimulatedWork(dist.Work):
        def __init__(self, world, event_id, tensors):
            super().__init__()
            self.world, self.event_id, self.tensors = world, event_id, tensors
            self.result_tensors = tensors
            self.submitted, self.error = False, None
            self.future = SimulatedFuture(self)

        def wait(self, timeout=None):
            if timeout is not None:
                raise NotImplementedError('per-call Work timeout is not modeled; registered backend uses init_process_group timeout')
            if hasattr(self.world, '_resolve'):
                self.world._resolve(self, blocking=True)
            if self.error is not None:
                raise self.error
            if not self.submitted:
                raise RuntimeError('missing distributed participants: submit every rank asynchronously before waiting')
            self.world.backend.wait(self.event_id)
            self.world._complete_futures()
            return True

        def is_completed(self):
            if hasattr(self.world, '_resolve'):
                self.world._resolve(self, blocking=False)
            if self.error is not None:
                raise self.error
            return self.submitted and self.world.backend.query(self.event_id)

        def get_future(self):
            self.world._complete_futures()
            return self.future

        def synchronize(self):
            return self.wait()

    class SimulatedProcessGroup(dist.ProcessGroup):
        def __init__(self, world, group, global_rank):
            self.world, self.group_id, self.global_rank = world, group, global_rank
            self.ranks = list(world.groups[group])
            super().__init__(self.ranks.index(global_rank), len(self.ranks))
            self.sequence = 0
            self.p2p_sequences = defaultdict(int)
            self.dependencies = ()
            self.storage_reads, self.storage_writes = (), ()

        def set_dependencies(self, *event_ids):
            """Bind explicit producer completion tokens from the shared backend."""
            if any(not isinstance(x, str) or not x for x in event_ids):
                raise ValueError('dependencies must be nonempty event IDs')
            self.dependencies = tuple(event_ids)

        def set_storage_accesses(self, *, reads=(), writes=()):
            """Bind capture storage IDs; caller owns tensor-to-storage correspondence.

            Accesses persist until replaced or cleared with no arguments. These
            explicit bindings keep storage live until collective completion.
            """
            if any(not isinstance(x, str) or not x for x in (*reads, *writes)):
                raise ValueError('storage accesses must be nonempty storage IDs')
            self.storage_reads, self.storage_writes = tuple(reads), tuple(writes)

        def _set_group_name(self, name):
            self._sim_group_name = name

        @property
        def group_name(self):
            if not hasattr(self, '_sim_group_name'):
                raise RuntimeError('simulated ProcessGroup must be registered for functional lookup')
            return self._sim_group_name

        def getBackendName(self):
            return 'titan_sim'

        def _collective(self, op, tensors, *, result_tensors=None, **extra):
            check(tensors) if tensors else None
            spec = {'op': op, 'sequence': self.sequence,
                    'payload_bytes': tensors[0].numel() * tensors[0].element_size() if tensors else 0, **extra}
            self.sequence += 1
            work = self.world._enqueue(self.group_id, self.global_rank, spec, tensors, deps=self.dependencies,
                                       reads=self.storage_reads, writes=self.storage_writes)
            work.result_tensors = tensors if result_tensors is None else result_tensors
            return work

        def allreduce(self, tensors, opts=dist.AllreduceOptions()):
            if opts is not None and opts.reduceOp != dist.ReduceOp.SUM:
                raise NotImplementedError('simulated allreduce currently supports SUM only')
            return self._collective('all_reduce', tensors, reduce_op='sum')

        def allgather(self, outputs, inputs, opts=None):
            check(inputs)
            if len(outputs) != 1 or len(outputs[0]) != len(self.ranks):
                raise ValueError('allgather output list must match group size')
            for output in outputs[0]:
                check([output])
                if output.numel() != inputs[0].numel() or output.dtype != inputs[0].dtype:
                    raise ValueError('allgather input/output metadata mismatch')
            return self._collective('all_gather', inputs, result_tensors=outputs[0])

        def _allgather_base(self, output, input, opts=None):
            check([output]); check([input])
            if output.numel() != input.numel() * len(self.ranks) or output.dtype != input.dtype:
                raise ValueError('allgather input/output metadata mismatch')
            return self._collective('all_gather', [input], result_tensors=[output])

        def _reduce_scatter_base(self, output, input, opts=None):
            check([output]); check([input])
            if opts is not None and opts.reduceOp != dist.ReduceOp.SUM:
                raise NotImplementedError('simulated reduce_scatter supports SUM only')
            if input.numel() != output.numel() * len(self.ranks) or output.dtype != input.dtype:
                raise ValueError('reduce_scatter input/output metadata mismatch')
            return self._collective('reduce_scatter', [input], result_tensors=[output])

        def alltoall_base(self, output, input, output_splits, input_splits, opts=None):
            check([output]); check([input])
            if input.dtype != output.dtype or input.dim() < 1 or output.dim() < 1:
                raise ValueError('all_to_all requires matching dtypes and nonscalar tensors')
            n = len(self.ranks)
            def split_bytes(tensor, splits):
                if not splits:
                    if tensor.shape[0] % n:
                        raise ValueError('equal all_to_all split requires divisible dimension zero')
                    splits = [tensor.shape[0] // n] * n
                if len(splits) != n or sum(splits) != tensor.shape[0]:
                    raise ValueError('all_to_all splits must cover dimension zero')
                row_bytes = tensor.element_size()
                for dimension in tensor.shape[1:]:
                    row_bytes *= dimension
                return [_integer(x, 'split') * row_bytes for x in splits]
            return self._collective('all_to_all', [input], result_tensors=[output],
                                    input_splits=split_bytes(input, input_splits),
                                    output_splits=split_bytes(output, output_splits))

        # PyTorch 2.14 renamed these ProcessGroup entry points.
        all_gather_single = _allgather_base
        reduce_scatter_single = _reduce_scatter_base
        all_to_all_single = alltoall_base

        def barrier(self, opts=dist.BarrierOptions()):
            return self._collective('barrier', [])

        def _p2p(self, op, tensors, peer, tag):
            check(tensors)
            if not 0 <= peer < len(self.ranks):
                raise ValueError('P2P peer is outside group')
            key = (op, peer, tag)
            sequence = self.p2p_sequences[key]
            self.p2p_sequences[key] += 1
            return self.world._enqueue(self.group_id, self.global_rank,
                {'op': op, 'sequence': sequence, 'peer': self.ranks[peer], 'tag': tag,
                 'payload_bytes': tensors[0].numel() * tensors[0].element_size()}, tensors, deps=self.dependencies,
                reads=self.storage_reads, writes=self.storage_writes)

        def send(self, tensors, dst, tag):
            return self._p2p('send', tensors, dst, tag)

        def recv(self, tensors, src, tag):
            return self._p2p('recv', tensors, src, tag)

    def unsupported(self, *args, **kwargs):
        raise NotImplementedError('this ProcessGroup operation is not modeled by titan_sim')

    # Base-class meta kernels can otherwise return a silent FakeWork for these APIs.
    for name in ('broadcast', 'reduce', 'gather', 'scatter', 'recv_anysource',
                 'reduce_scatter', 'allreduce_coalesced', 'allgather_coalesced',
                 'allgather_into_tensor_coalesced', 'reduce_scatter_tensor_coalesced',
                 'all_gather_single_coalesced', 'reduce_scatter_single_coalesced',
                 'alltoall', 'all_to_all', 'monitored_barrier', 'gather_single',
                 'gather_into_tensor', '_start_coalescing', '_end_coalescing'):
        setattr(SimulatedProcessGroup, name, unsupported)

    _CLASSES = SimulatedProcessGroup, SimulatedWork
    return _CLASSES


class _StoreWorld(SimulatedWorld):
    """Store rendezvous for independent CPU processes; collective-only scope."""
    def __init__(self, store, hardware, ranks, rank, timeout):
        import hashlib
        import json
        super().__init__(hardware, groups={'world': ranks})
        self.store, self.rank, self.timeout = store, rank, timeout
        self._store_pending = {}
        self._hardware_key = hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest()

    def _enqueue(self, group, rank, spec, tensors, *, deps=(), reads=(), writes=()):
        import json
        if spec['op'] in ('send', 'recv'):
            raise NotImplementedError('registered Store backend currently supports collectives only; use cooperative world for P2P')
        if deps or reads or writes:
            raise NotImplementedError('registered Store backend cannot yet exchange captured compute/storage graphs')
        _, work_class = _torch_classes()
        sequence = spec['sequence']
        event_id = f'store:{sequence}:r{rank}'
        work = work_class(self, event_id, tensors)
        self._works.append(work)
        self._store_pending[sequence] = work
        event = {'event_id': event_id, 'kind': 'collective', 'rank': rank,
                 'stream': 'registered_collectives',
                 'host_time_us': self.backend.host_time_us,
                 'dtype': str(tensors[0].dtype).removeprefix('torch.') if tensors else 'float32',
                 'metadata': {'distributed': dict(spec, group=group), 'hardware_key': self._hardware_key}}
        self.store.set(f'operation/{sequence}/{rank}', json.dumps(event))
        return work

    def _resolve(self, work, *, blocking):
        import json
        if work.submitted or work.error is not None:
            return
        # Resolve earlier calls first: a later wait must preserve collective ordering.
        target = int(work.event_id.split(':')[1])
        for sequence in sorted(self._store_pending):
            if sequence > target:
                break
            item = self._store_pending[sequence]
            if item.submitted:
                continue
            keys = [f'operation/{sequence}/{rank}' for rank in self.groups['world']]
            if not blocking and not self.store.check(keys):
                return
            try:
                self.store.wait(keys, self.timeout)
                events = [json.loads(self.store.get(key).decode()) for key in keys]
                if any(event['metadata']['hardware_key'] != self._hardware_key for event in events):
                    raise ValueError('registered participants use different hardware scenarios')
                lowered = lower_distributed_events(events, self.groups)
                names = {e['event_id']: f'store_tx{sequence}:' + e['event_id']
                         for e in lowered if e['event_id'].startswith('__dist_')}
                # Preserve actual enqueue clocks. Waiting later must not delay
                # already issued communication or change another rank's timeline.
                for event in lowered:
                    event['event_id'] = names.get(event['event_id'], event['event_id'])
                    event['deps'] = [names.get(dep, dep) for dep in event.get('deps', ())]
                    if event['stream'].startswith('__dist_'):
                        event['stream'] = f'store_tx{sequence}:' + event['stream']
                    event.setdefault('host_time_us', events[0]['host_time_us'])
                self.backend.submit_enqueued(lowered)
                item.submitted = True
            except Exception as error:
                item.error = RuntimeError(f'registered collective {sequence} rendezvous failed: {error}')
                item.future.set_exception(item.error)
                raise item.error from error

    def assert_complete(self):
        for work in self._store_pending.values():
            self._resolve(work, blocking=False)
            if not work.submitted:
                raise RuntimeError('missing registered distributed participants')


def register_simulated_backend(hardware, *, name='titan_sim'):
    """Register a metadata-only backend for real ``init_process_group`` workers.

    Each process must call this with the same hardware before initializing. The
    PyTorch Store carries small JSON control records only. Supported eager
    collectives use meta/FakeTensor payloads and rank-local virtual host clocks.
    Functional collectives require the explicit functional_collectives() mode.
    DeviceMesh/FSDP, P2P and captured compute/storage exchange are outside this
    registered backend's supported scope.
    """
    import torch.distributed as dist
    from copy import deepcopy
    from .costs import validate_hardware
    hardware = deepcopy(validate_hardware(hardware))
    def create(options, backend_options):
        ranks = list(options.global_ranks_in_group) or list(range(options.group_size))
        rank = ranks[options.group_rank]
        world = _StoreWorld(options.store, hardware, ranks, rank, options.timeout)
        return world.process_group(rank)
    dist.Backend.register_backend(name, create, extended_api=True, devices=['cpu', 'meta'])
    return name


def functional_collectives():
    """Explicit dispatch mode for modeled functional collectives and wait_tensor.

    Enter inside FakeTensorMode (or use meta tensors). This intercepts the actual
    c10d functional operators before metadata dispatch or native Work wrappers
    can bypass modeled completion.
    All returned functional tensors must be waited inside this scope. Registered
    simulated groups are required; other families/operations fail explicitly.
    """
    import torch
    import torch.distributed as dist
    from torch.utils._python_dispatch import TorchDispatchMode

    class FunctionalCollectiveMode(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.pending = {}
            self.completed = {}

        def _wait_tensor(self, tensor):
            from torch.distributed._functional_collectives import AsyncCollectiveTensor
            if isinstance(tensor, AsyncCollectiveTensor):
                tensor = tensor.elem
            key = tensor.untyped_storage()._cdata
            entry = self.pending.get(key)
            if entry is None:
                if key in self.completed:
                    return tensor
                raise RuntimeError('functional wait_tensor has no matching simulated collective in this scope')
            entry[1].wait()
            self.completed[key] = entry[0]
            del self.pending[key]
            return tensor

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            namespace = func._schema.name.split('::')[0]
            if namespace not in ('_c10d_functional', '_c10d_functional_autograd'):
                from torch.distributed._functional_collectives import AsyncCollectiveTensor
                from torch.utils._pytree import tree_map_only
                args = tree_map_only(AsyncCollectiveTensor, self._wait_tensor, args)
                kwargs = tree_map_only(AsyncCollectiveTensor, self._wait_tensor, kwargs)
                return func(*args, **kwargs)
            name = func._schema.name.split('::')[1]
            if kwargs:
                values = dict(zip((a.name for a in func._schema.arguments), args))
                values.update(kwargs)
                args = tuple(values[a.name] for a in func._schema.arguments)
            if namespace != '_c10d_functional':
                raise NotImplementedError('functional autograd collectives are not modeled')
            if name == 'wait_tensor':
                return self._wait_tensor(args[0])
            if name not in ('all_reduce', 'all_gather_into_tensor', 'reduce_scatter_tensor', 'all_to_all_single'):
                raise NotImplementedError(f'functional collective {name} is not modeled')
            tensor = args[0]
            from torch.distributed._functional_collectives import AsyncCollectiveTensor
            if isinstance(tensor, AsyncCollectiveTensor):
                tensor = self._wait_tensor(tensor)
            group = dist.distributed_c10d._resolve_process_group(args[-1])
            pg_class, _ = _torch_classes()
            if not isinstance(group, pg_class):
                raise ValueError('functional simulation requires a registered titan_sim ProcessGroup')
            # Allocate only metadata; ProcessGroup validation rejects numeric tensors.
            if tensor.device.type != 'meta':
                from torch._subclasses.fake_tensor import FakeTensor
                if not isinstance(tensor, FakeTensor):
                    raise ValueError('functional simulation requires meta or FakeTensor')
            if name == 'all_reduce':
                if args[1] != 'sum':
                    raise NotImplementedError('functional all_reduce currently supports sum only')
                output = torch.empty_like(tensor)
                work = group.allreduce([output])
            elif name == 'all_gather_into_tensor':
                if args[1] != group.size() or tensor.dim() == 0:
                    raise ValueError('functional all_gather requires matching group size and nonscalar input')
                output = torch.empty((tensor.shape[0] * group.size(), *tensor.shape[1:]),
                                     device=tensor.device, dtype=tensor.dtype)
                work = group.all_gather_single(output, tensor)
            elif name == 'reduce_scatter_tensor':
                if args[1] != 'sum':
                    raise NotImplementedError('functional reduce_scatter currently supports sum only')
                if args[2] != group.size() or tensor.dim() == 0 or tensor.shape[0] % group.size():
                    raise ValueError('functional reduce_scatter requires evenly divisible dimension zero')
                output = torch.empty((tensor.shape[0] // group.size(), *tensor.shape[1:]),
                                     device=tensor.device, dtype=tensor.dtype)
                work = group.reduce_scatter_single(output, tensor)
            else:
                outputs, inputs = args[1], args[2]
                output = torch.empty((sum(outputs), *tensor.shape[1:]),
                                     device=tensor.device, dtype=tensor.dtype)
                work = group.all_to_all_single(output, tensor, outputs, inputs)
            self.pending[output.untyped_storage()._cdata] = (output, work, tensor)
            return output

        def __exit__(self, exc_type, exc_value, traceback):
            result = super().__exit__(exc_type, exc_value, traceback)
            if exc_type is None and self.pending:
                raise RuntimeError('functional collective outputs must be waited before leaving simulation scope')
            return result

    return FunctionalCollectiveMode()
