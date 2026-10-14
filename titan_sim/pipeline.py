"""Extract actual PyTorch pipeline schedules and lower their metadata actions.

This executes upstream schedule construction and communication lowering on CPU.
Stage compute durations/buffer sizes are explicit scenarios, not captured model
kernels. It does not claim to execute a TorchTitan model or FSDP stage hooks.
"""
from types import SimpleNamespace
from .distributed import lower_distributed_events


def capture_pipeline_schedule(*, num_ranks=2, num_stages=4, microbatches=4,
                              activation_bytes=1024, forward_us=10, backward_us=20):
    """Construct upstream interleaved 1F1B, preserving compute and P2P actions.

    ``forward_us`` and ``backward_us`` may be scalars or per-stage sequences.
    The schedule uses round-robin stage placement and at least two stages/rank.
    UNSHARD/RESHARD/REDUCE_GRAD are control markers for the explicitly unsharded,
    DP=1 scenario. No distributed optimizer/model numerical work is performed.
    """
    import torch
    from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

    for name, value in [('num_ranks', num_ranks), ('num_stages', num_stages),
                        ('microbatches', microbatches), ('activation_bytes', activation_bytes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if num_ranks < 2 or num_stages % num_ranks or num_stages < 2 * num_ranks:
        raise ValueError('interleaved schedule requires >=2 ranks and >=2 evenly placed stages per rank')
    def durations(value, name):
        from .events import nonnegative
        result = list(value) if isinstance(value, (list, tuple)) else [value] * num_stages
        if len(result) != num_stages:
            raise ValueError(f'{name} needs one duration per stage')
        return [nonnegative(x, name) for x in result]
    forwards, backwards = durations(forward_us, 'forward_us'), durations(backward_us, 'backward_us')
    stages = [SimpleNamespace(group_size=num_ranks, group_rank=0,
                              num_stages=num_stages, stage_index=i, is_last=i == num_stages - 1)
              for i in range(0, num_stages, num_ranks)]
    schedule = ScheduleInterleaved1F1B(stages, microbatches, loss_fn=lambda output, target: None)
    if not hasattr(schedule, 'pipeline_order_with_comms'):
        raise RuntimeError('this PyTorch version does not expose lowered pipeline communication actions')
    events, action_log = [], {}
    for rank, actions in schedule.pipeline_order_with_comms.items():
        previous_compute = None
        available = {}
        action_log[str(rank)] = [str(action) for action in actions]
        def emit(event_id, kind, *, deps=(), stream='compute', **kwargs):
            event = dict(event_id=event_id, kind=kind, rank=rank, stream=stream,
                         deps=list(deps), **kwargs)
            events.append(event)
            return event_id
        def allocate(storage, deps=()):
            return emit(f'alloc:{storage}', 'allocate', storage_id=storage,
                        size_bytes=activation_bytes, category='pipeline_activation', deps=deps,
                        stream=f'memory:{storage}')
        def release(storage, deps=()):
            emit(f'release:{storage}', 'release', storage_id=storage, deps=deps,
                 stream=f'memory:{storage}')
        for ordinal, action in enumerate(actions):
            kind = action.computation_type.name
            stage, mb = action.stage_index, action.microbatch_index
            event_id = f'r{rank}:{ordinal}:{action}'
            base = f's{stage}:m{mb}'
            prior = [previous_compute] if previous_compute else []
            if kind in ('UNSHARD', 'RESHARD', 'REDUCE_GRAD'):
                emit(event_id, 'wait', deps=prior, stream=f'control:{stage}',
                     name=kind, metadata={'scenario': 'unsharded stages and DP=1; no hook work'})
            elif kind.startswith(('SEND_', 'RECV_')):
                send = kind.startswith('SEND_')
                forward = kind.endswith('_F')
                peer_stage = stage + (1 if forward == send else -1)
                peer = peer_stage % num_ranks
                boundary = min(stage, peer_stage)
                tag = 2 * boundary + (0 if forward else 1)
                storage = f'{base}:output' if forward and send else f'{base}:grad_out' if send else f'{base}:input' if forward else f'{base}:grad_in'
                dependencies = prior.copy()
                if not send:
                    dependencies.append(allocate(storage, prior))
                else:
                    dependencies.append(available[(stage, mb, 'F' if forward else 'B')])
                emit(event_id, 'collective', deps=dependencies, stream=f'p2p:{event_id}',
                     read_storage_ids=[storage] if send else [],
                     write_storage_ids=[] if send else [storage], dtype='float32',
                     metadata={'distributed': {'group': 'pipeline', 'sequence': mb,
                               'op': 'send' if send else 'recv', 'peer': peer, 'tag': tag,
                               'payload_bytes': activation_bytes}})
                if not send:
                    available[(stage, mb, 'recv_F' if forward else 'recv_B')] = event_id
                elif not forward:
                    release(storage, [event_id])
            elif kind in ('FORWARD', 'FULL_BACKWARD'):
                forward = kind == 'FORWARD'
                storage = f'{base}:output' if forward else f'{base}:grad_out'
                dependencies = prior.copy()
                reads = []
                if stage > 0:
                    reads.append(f'{base}:input')
                if forward:
                    if stage > 0:
                        dependencies.append(available[(stage, mb, 'recv_F')])
                else:
                    reads.append(f'{base}:output')
                    dependencies.append(available[(stage, mb, 'F')])
                    if stage < num_stages - 1:
                        reads.append(f'{base}:grad_in')
                        dependencies.append(available[(stage, mb, 'recv_B')])
                receive_key = (stage, mb, 'recv_F' if forward else 'recv_B')
                if receive_key in available:
                    # Upstream runtime waits on recv Work before invoking stage compute.
                    wait_id = emit(f'host_wait:{event_id}', 'host_sync',
                                   deps=[available[receive_key]], stream='host',
                                   metadata={'sync_scope': 'event', 'pipeline_recv_wait': True})
                    dependencies.append(wait_id)
                dependencies.append(allocate(storage, dependencies))
                previous_compute = emit(event_id, 'compute', deps=dependencies,
                                        name=f'stage{stage}.{kind.lower()}',
                                        duration_us=(forwards if forward else backwards)[stage],
                                        read_storage_ids=reads, write_storage_ids=[storage],
                                        metadata={'duration_provenance': 'explicit pipeline stage scenario',
                                                  'stage': stage, 'microbatch': mb})
                available[(stage, mb, 'F' if forward else 'B')] = event_id
                if not forward:
                    for stored in reads:
                        release(stored, [event_id])
                    if stage == 0:
                        release(storage, [event_id])
            else:
                raise NotImplementedError(f'unsupported upstream pipeline action {kind}')
    lowered = lower_distributed_events(events, {'pipeline': list(range(num_ranks))})
    return {'schema_version': 1, 'source': 'torch.distributed.pipelining.ScheduleInterleaved1F1B',
            'metadata': {'torch_version': torch.__version__, 'num_ranks': num_ranks,
                         'num_stages': num_stages, 'microbatches': microbatches,
                         'stage_mapping': dict(schedule.stage_index_to_group_rank),
                         'upstream_actions': action_log, 'dp': 1,
                         'stage_execution': 'metadata schedule extraction; explicit duration/buffer scenarios'},
            'events': lowered,
            'limitations': ['Upstream schedule and P2P actions execute as metadata; model kernels are scenario durations.',
                            'Unsharded stages, DP=1; no FSDP hooks, optimizer, or TorchTitan model execution.',
                            'Buffer sizes are explicit scenarios; no caching allocator or kernel workspaces.']}
