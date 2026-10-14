import importlib.util
import unittest

from titan_sim.distributed import lower_distributed_events, SimulatedWorld
from titan_sim.runtime import simulate

HW = {'peak_flops': {'float32': 1e6}, 'memory_bandwidth_bytes_s': 1e6,
      'network_bandwidth_bytes_s': 1e6, 'memory_capacity_bytes': 1000000,
      'network_topology': {'links': [{'id': 'link', 'src': 0, 'dst': 1,
                            'bandwidth_bytes_s': 1e6, 'latency_us': 1, 'bidirectional': True}]}}


def participant(rank, sequence=0, op='all_reduce', payload=8, **extra):
    return {'event_id': f'r{rank}:{sequence}', 'kind': 'collective', 'rank': rank,
            'stream': 'comm', 'dtype': 'float32',
            'metadata': {'distributed': {'group': 'world', 'sequence': sequence,
                         'op': op, 'payload_bytes': payload, **extra}}}


class RendezvousTests(unittest.TestCase):
    def test_waits_for_late_participant_and_prices_once(self):
        inputs = [{'event_id': 'late', 'kind': 'compute', 'rank': 1, 'flops': 20},
                  participant(0), dict(participant(1), deps=['late']),
                  {'event_id': 'consumer', 'kind': 'compute', 'rank': 0, 'flops': 1, 'deps': ['r0:0']}]
        lowered = lower_distributed_events(inputs, {'world': [1, 0]})
        transfers = [e for e in lowered if e['kind'] == 'collective']
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0]['metadata']['participants'], [1, 0])
        rows = {r['event_id']: r for r in simulate(lowered, HW)['timeline']}
        self.assertEqual(rows[transfers[0]['event_id']]['start_us'], 20)
        self.assertEqual(rows['r0:0']['end_us'], rows['r1:0']['end_us'])
        self.assertEqual(rows['consumer']['start_us'], rows['r1:0']['end_us'])

    def test_collective_sequence_not_global_merge_order(self):
        events = [participant(1, 0), participant(1, 1), participant(0, 0), participant(0, 1)]
        result = simulate(lower_distributed_events(events, {'world': [0, 1]}), HW)
        rows = {r['event_id']: r for r in result['timeline']}
        self.assertGreaterEqual(rows['__dist_transfer_1']['start_us'], rows['r0:0']['end_us'])

    def test_missing_duplicate_and_mismatch_rejected(self):
        cases = [[participant(0)], [participant(0), participant(1, op='all_gather')],
                 [participant(0), dict(participant(1), dtype='float16')],
                 [participant(0), participant(1, payload=12)],
                 [participant(0, 1), participant(0, 0), participant(1, 0), participant(1, 1)]]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ValueError):
                lower_distributed_events(events, {'world': [0, 1]})

    def test_reduction_operator_preserved_and_matched(self):
        lowered = lower_distributed_events([participant(0, reduce_op='max'),
                                             participant(1, reduce_op='max')], {'world': [0, 1]})
        transfer = next(event for event in lowered if event['kind'] == 'collective')
        self.assertEqual(transfer['metadata']['reduce_op'], 'max')
        with self.assertRaisesRegex(ValueError, 'mismatched reduction'):
            lower_distributed_events([participant(0, reduce_op='sum'),
                                      participant(1, reduce_op='max')], {'world': [0, 1]})

    def test_p2p_matches_peer_tag_and_sequence(self):
        a = participant(0, op='send', peer=1, tag=7)
        b = participant(1, op='recv', peer=0, tag=7)
        result = lower_distributed_events([a, b], {'world': [0, 1]})
        self.assertEqual(sum(e['kind'] == 'collective' for e in result), 1)
        b['metadata']['distributed']['tag'] = 8
        with self.assertRaisesRegex(ValueError, 'missing'):
            lower_distributed_events([a, b], {'world': [0, 1]})

    def test_variable_alltoall_splits(self):
        a = participant(0, op='all_to_all', payload=12, input_splits=[4, 8], output_splits=[4, 12])
        b = participant(1, op='all_to_all', payload=16, input_splits=[12, 4], output_splits=[8, 4])
        lowered = lower_distributed_events([a, b], {'world': [0, 1]})
        transfer = next(e for e in lowered if e['kind'] == 'collective')
        self.assertEqual(transfer['metadata']['input_splits'], [[4, 8], [12, 4]])
        b['metadata']['distributed']['output_splits'] = [4, 8]
        with self.assertRaisesRegex(ValueError, 'splits mismatch'):
            lower_distributed_events([a, b], {'world': [0, 1]})

    def test_deadlock_diagnosed(self):
        a, b = participant(0), participant(1)
        a['deps'], b['deps'] = ['r1:0'], ['r0:0']
        with self.assertRaisesRegex(ValueError, 'cycle'):
            simulate(lower_distributed_events([a, b], {'world': [0, 1]}), HW)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class ProcessGroupTests(unittest.TestCase):
    def test_actual_torch_distributed_calls_and_virtual_work(self):
        import torch
        import torch.distributed as dist
        world = SimulatedWorld(HW)
        tensor = torch.empty(2, device='meta')
        a = dist.all_reduce(tensor, group=world.process_group(0), async_op=True)
        self.assertFalse(a.is_completed())
        with self.assertRaisesRegex(RuntimeError, 'missing distributed participants'):
            a.wait()
        b = dist.all_reduce(tensor, group=world.process_group(1), async_op=True)
        future = a.get_future()
        self.assertFalse(future.done())
        self.assertFalse(a.is_completed())
        self.assertTrue(b.wait())
        self.assertTrue(a.is_completed())
        self.assertTrue(future.done())
        self.assertGreater(world.backend.host_time_us, 0)
        world.assert_complete()
        a2 = dist.all_reduce(tensor, group=world.process_group(0), async_op=True)
        b2 = dist.all_reduce(tensor, group=world.process_group(1), async_op=True)
        self.assertFalse(a2.is_completed())
        self.assertTrue(b2.wait())

    def test_shared_backend_waits_for_compute_producers(self):
        import torch
        import torch.distributed as dist
        from titan_sim.live import LiveExecution
        backend = LiveExecution(HW)
        backend.submit([{'event_id': 'producer', 'kind': 'compute', 'rank': 0,
                         'duration_us': 100, 'duration_provenance': 'test fixture'}])
        world = SimulatedWorld(backend=backend)
        pg0, pg1 = world.process_group(0), world.process_group(1)
        pg0.set_dependencies('producer')
        tensor = torch.empty(2, device='meta')
        a = dist.all_reduce(tensor, group=pg0, async_op=True)
        b = dist.all_reduce(tensor, group=pg1, async_op=True)
        self.assertEqual(backend.host_time_us, 0)
        b.wait()
        self.assertGreater(backend.host_time_us, 100)
        self.assertTrue(a.is_completed())

    def test_future_wait_advances_clock_and_unsupported_op_fails(self):
        import torch
        import torch.distributed as dist
        world = SimulatedWorld(HW)
        tensor = torch.empty(2, device='meta')
        a = dist.all_reduce(tensor, group=world.process_group(0), async_op=True)
        dist.all_reduce(tensor, group=world.process_group(1), async_op=True)
        self.assertEqual(a.get_future().wait(), [tensor])
        self.assertGreater(world.backend.host_time_us, 0)
        with self.assertRaisesRegex(NotImplementedError, 'not modeled'):
            dist.broadcast(tensor, group_src=0, group=world.process_group(0), async_op=True)

    def test_explicit_capture_storage_stays_live_until_work_completes(self):
        import torch
        import torch.distributed as dist
        world = SimulatedWorld(HW)
        world.backend.submit([{'event_id': 'alloc', 'kind': 'allocate', 'rank': 0,
                               'storage_id': 'captured_tensor', 'size_bytes': 8}])
        pg0 = world.process_group(0)
        pg0.set_storage_accesses(reads=['captured_tensor'], writes=['captured_tensor'])
        tensor = torch.empty(2, device='meta')
        a = dist.all_reduce(tensor, group=pg0, async_op=True)
        world.backend.submit([{'event_id': 'release', 'kind': 'release', 'rank': 0,
                               'storage_id': 'captured_tensor', 'stream': 'host'}])
        self.assertIsNone(world.backend.document()['storage_lifetimes'][0]['reclaim_us'])
        dist.all_reduce(tensor, group=world.process_group(1), async_op=True)
        self.assertFalse(a.is_completed())
        lifetime = world.backend.document()['storage_lifetimes'][0]
        self.assertGreater(lifetime['reclaim_us'], 0)
        a.wait()
        self.assertEqual(lifetime['reclaim_us'], world.backend.host_time_us)

    def test_explicit_work_timeout_is_rejected(self):
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        world = SimulatedWorld(HW)
        work = dist.all_reduce(torch.empty(2, device='meta'), group=world.process_group(0), async_op=True)
        with self.assertRaisesRegex(NotImplementedError, 'per-call Work timeout'):
            work.wait(timedelta(seconds=1))

    def test_functional_mode_rejects_unmodeled_collective(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.distributed import functional_collectives
        with FakeTensorMode(), functional_collectives():
            tensor = torch.empty(2)
            with self.assertRaisesRegex(NotImplementedError, 'not modeled'):
                torch.ops._c10d_functional.all_reduce_coalesced([tensor], 'sum', 'unsupported')

    def test_p2p_actual_torch_calls(self):
        import torch
        import torch.distributed as dist
        world = SimulatedWorld(HW)
        tensor = torch.empty(2, device='meta')
        a = dist.isend(tensor, group_dst=1, group=world.process_group(0), tag=9)
        b = dist.irecv(tensor, group_src=0, group=world.process_group(1), tag=9)
        self.assertTrue(a.wait())
        self.assertTrue(b.is_completed())

    def test_rejects_numerical_tensor_and_missing_rank(self):
        import torch
        import torch.distributed as dist
        world = SimulatedWorld(HW)
        with self.assertRaisesRegex(ValueError, 'meta or FakeTensor'):
            dist.all_reduce(torch.empty(2), group=world.process_group(0), async_op=True)
        dist.all_reduce(torch.empty(2, device='meta'), group=world.process_group(0), async_op=True)
        with self.assertRaisesRegex(RuntimeError, 'missing distributed participants'):
            world.assert_complete()

    def test_allgather_reduce_scatter_and_alltoall(self):
        import torch
        import torch.distributed as dist
        for op in ('allgather', 'reduce_scatter', 'alltoall'):
            world = SimulatedWorld(HW)
            works = []
            for rank in (0, 1):
                pg = world.process_group(rank)
                if op == 'allgather':
                    work = dist.all_gather_into_tensor(torch.empty(4, device='meta'), torch.empty(2, device='meta'), group=pg, async_op=True)
                elif op == 'reduce_scatter':
                    work = dist.reduce_scatter_tensor(torch.empty(2, device='meta'), torch.empty(4, device='meta'), group=pg, async_op=True)
                else:
                    work = dist.all_to_all_single(torch.empty(4, device='meta'), torch.empty(4, device='meta'), group=pg, async_op=True)
                works.append(work)
            self.assertTrue(works[0].wait())
            self.assertTrue(works[1].is_completed())
            expected_elements = 2 if op == 'reduce_scatter' else 4
            self.assertEqual(works[0].get_future().wait()[0].numel(), expected_elements)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class RegisteredBackendTests(unittest.TestCase):
    def test_real_two_process_init_and_eager_collectives(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(root / 'scripts/distributed_probe.py'),
                                     '--out', directory, '--multiprocess'],
                                    capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reports = [json.loads(Path(directory, f'registered-rank{rank}.json').read_text()) for rank in (0, 1)]
            for report in reports:
                self.assertTrue(report['real_init_process_group'])
                self.assertEqual(report['backend'], 'titan_sim')
                self.assertEqual(sum(event['kind'] == 'collective' for event in report['result']['timeline']), 7)
                self.assertEqual(report['functional_adapter_probe']['new_simulated_transfers'], 3)
                self.assertTrue(report['functional_adapter_probe']['fake_tensor_mode'])
                self.assertTrue(report['functional_collective_probe']['status'].startswith('unsupported'))
            self.assertEqual(reports[0]['result']['live_execution']['host_time_us'],
                             reports[1]['result']['live_execution']['host_time_us'])

    def test_store_resolution_preserves_enqueue_time_after_host_work(self):
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        from titan_sim.distributed import _StoreWorld
        store = dist.HashStore()
        worlds = [_StoreWorld(store, HW, [0, 1], rank, timedelta(seconds=1)) for rank in (0, 1)]
        works = [dist.all_reduce(torch.empty(2, device='meta'), group=world.process_group(rank), async_op=True)
                 for rank, world in enumerate(worlds)]
        worlds[0].backend.advance_host(100)
        for work in works:
            work.wait()
        transfers = [next(event for event in world.backend.document()['timeline'] if event['kind'] == 'collective')
                     for world in worlds]
        self.assertEqual(transfers[0]['start_us'], 0)
        self.assertEqual(transfers[0]['end_us'], transfers[1]['end_us'])
        self.assertEqual(worlds[0].backend.host_time_us, 100)
        self.assertLess(worlds[1].backend.host_time_us, 100)

    def test_store_mismatch_and_timeout_fail_explicitly(self):
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        from titan_sim.distributed import _StoreWorld
        store = dist.HashStore()
        worlds = [_StoreWorld(store, HW, [0, 1], rank, timedelta(milliseconds=10)) for rank in (0, 1)]
        first = dist.all_reduce(torch.empty(2, device='meta'), group=worlds[0].process_group(0), async_op=True)
        dist.all_reduce(torch.empty(3, device='meta'), group=worlds[1].process_group(1), async_op=True)
        with self.assertRaisesRegex(RuntimeError, 'mismatched payload'):
            first.wait()
        lonely = _StoreWorld(dist.HashStore(), HW, [0, 1], 0, timedelta(milliseconds=10))
        pending = dist.all_reduce(torch.empty(2, device='meta'), group=lonely.process_group(0), async_op=True)
        with self.assertRaisesRegex(RuntimeError, 'rendezvous failed'):
            pending.wait()


if __name__ == '__main__':
    unittest.main()
