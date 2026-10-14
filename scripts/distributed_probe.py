#!/usr/bin/env python3
"""CPU-only upstream pipeline + real torch.distributed metadata control probe."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan_sim.distributed import SimulatedWorld
from titan_sim.live import LiveExecution
from titan_sim.pipeline import capture_pipeline_schedule
from titan_sim.runtime import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--multiprocess', action='store_true', help='also spawn two registered-backend CPU workers')
    args = parser.parse_args()
    import torch
    import torch.distributed as dist

    hardware = {'name': 'explicit synthetic two-rank network scenario',
                'peak_flops': {'float32': 1e12}, 'memory_bandwidth_bytes_s': 1e12,
                'memory_capacity_bytes': 1_000_000,
                'network_topology': {'links': [{'id': 'r0-r1', 'src': 0, 'dst': 1,
                    'bandwidth_bytes_s': 1e9, 'latency_us': 2, 'bidirectional': True}]}}
    capture = capture_pipeline_schedule(activation_bytes=1024,
                                        forward_us=[10, 30, 10, 10], backward_us=[20, 60, 20, 20])
    pipeline_result = simulate(capture['events'], hardware)
    backend = LiveExecution(hardware)
    backend.submit([{'event_id': 'fake_gpu_producer', 'kind': 'compute', 'rank': 0,
                     'duration_us': 100, 'duration_provenance': 'explicit demonstration scenario'}])
    world = SimulatedWorld(backend=backend)
    pg0, pg1 = world.process_group(0), world.process_group(1)
    pg0.set_dependencies('fake_gpu_producer')
    tensor = torch.empty(256, device='meta')
    first = dist.all_reduce(tensor, group=pg0, async_op=True)
    try:
        first.wait()
    except RuntimeError as error:
        missing_peer_diagnostic = str(error)
    else:
        raise AssertionError('missing peer must not complete')
    second = dist.all_reduce(tensor, group=pg1, async_op=True)
    if first.is_completed():
        raise AssertionError('asynchronous operation must not advance host clock')
    future = first.get_future()
    second.wait()
    world.assert_complete()
    if not first.is_completed() or not future.done() or backend.host_time_us <= 100:
        raise AssertionError('CPU wait must observe fake-GPU producer and communication completion')
    report = {'torch_version': torch.__version__, 'cpu_only': True,
              'real_torch_distributed_api': True, 'numerical_transport': False,
              'missing_peer_diagnostic': missing_peer_diagnostic,
              'completed_work': first.is_completed(), 'completed_future': future.done(),
              'live_result': backend.document(), 'pipeline_summary': pipeline_result['summary'],
              'scope': 'CPU metadata ProcessGroup and upstream pipeline schedule extraction; no TorchTitan model execution'}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, document in [('hardware.json', hardware), ('pipeline-capture.json', capture),
                           ('pipeline-simulation.json', pipeline_result), ('distributed-probe.json', report)]:
        (args.out / name).write_text(json.dumps(document, indent=2) + '\n')
    if args.multiprocess:
        run_multiprocess(args.out, hardware)
    print(json.dumps({'out': str(args.out.resolve()), 'pipeline_events': len(capture['events']),
                      'pipeline_runtime_us': pipeline_result['summary']['runtime_us'],
                      'cpu_wait_completion_us': backend.host_time_us, 'checks_passed': True}))


def _registered_worker(rank, store_path, output_path, hardware):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from titan_sim.distributed import register_simulated_backend

    register_simulated_backend(hardware)
    dist.init_process_group('titan_sim', init_method='file://' + store_path,
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    pg = dist.distributed_c10d._get_default_group()
    tensor = torch.empty(256, device='meta')
    first = dist.all_reduce(tensor, async_op=True)
    first.get_future().wait()
    second = dist.all_gather_single(torch.empty(512, device='meta'), tensor, async_op=True)
    second.wait()
    third = dist.all_to_all_single(torch.empty_like(tensor), tensor, async_op=True)
    third.wait()
    pg.world.assert_complete()
    before = len(pg.world.backend.events)
    try:
        native_output = torch.ops._c10d_functional.all_reduce(tensor, 'sum', pg.group_name)
        torch.ops._c10d_functional.wait_tensor(native_output)
    except Exception as error:
        functional = {'status': 'unsupported', 'diagnostic': str(error)}
    else:
        after = len(pg.world.backend.events)
        functional = {'status': 'native_wait_reaches_simulator' if after > before else 'unsupported_native_wait_does_not_advance_virtual_completion',
                      'new_simulated_events': after - before}
    # Resolve the diagnostic call through the supported Python Work interface.
    # Native C++ Work timeout/completion semantics are outside this adapter's scope.
    for pending_work in pg.world._store_pending.values():
        if not pending_work.submitted:
            pending_work.wait()
    adapter_before = sum(event['kind'] == 'collective' for event in pg.world.backend.events)
    from torch._subclasses.fake_tensor import FakeTensorMode
    import torch.distributed._functional_collectives as funcol
    from titan_sim.distributed import functional_collectives
    with FakeTensorMode():
        with functional_collectives():
            fake = torch.empty((4, 2), device='cpu')
            reduced = funcol.all_reduce(fake, 'sum', pg)
            reduced = funcol.wait_tensor(reduced)
            gathered = funcol.all_gather_tensor(fake, 0, pg)
            gathered = funcol.wait_tensor(gathered)
            scattered = funcol.reduce_scatter_tensor(gathered, 'sum', 0, pg)
            scattered = funcol.wait_tensor(scattered)
            if reduced.shape != fake.shape or gathered.shape != (8, 2) or scattered.shape != fake.shape:
                raise AssertionError('functional metadata shapes are incorrect')
    functional_adapter = {'status': 'supported_eager_dispatch_mode',
                          'fake_tensor_mode': True,
                          'operations': ['all_reduce', 'all_gather_into_tensor', 'reduce_scatter_tensor'],
                          'new_simulated_transfers': sum(event['kind'] == 'collective' for event in pg.world.backend.events) - adapter_before}
    report = {'rank': rank, 'torch_version': torch.__version__,
              'backend': dist.get_backend(), 'real_init_process_group': True,
              'host_clock_scope': 'rank-local host clocks; common matched collective timeline',
              'numerical_payload_transport': False, 'control_transport': 'PyTorch Store JSON',
              'functional_collective_probe': functional,
              'functional_adapter_probe': functional_adapter,
              'result': pg.world.backend.document()}
    Path(output_path, f'registered-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
    dist.destroy_process_group()


def run_multiprocess(output, hardware):
    import tempfile
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix='titan-sim-store-') as directory:
        mp.spawn(_registered_worker, args=(str(Path(directory, 'rendezvous')), str(output.resolve()), hardware),
                 nprocs=2, join=True)
    reports = [json.loads((output / f'registered-rank{rank}.json').read_text()) for rank in (0, 1)]
    clocks = [report['result']['live_execution']['host_time_us'] for report in reports]
    if clocks[0] != clocks[1] or clocks[0] <= 0:
        raise AssertionError(f'registered workers have inconsistent clocks: {clocks}')
    print(json.dumps({'registered_workers': 2, 'virtual_completion_us': clocks[0],
                      'functional_collectives_require_dispatch_mode': True, 'checks_passed': True}))


if __name__ == '__main__':
    main()
