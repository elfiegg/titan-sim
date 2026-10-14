import importlib.util
import unittest
from titan_sim.pipeline import capture_pipeline_schedule
from titan_sim.runtime import simulate
from tests.test_distributed import HW


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class PipelineTests(unittest.TestCase):
    def test_upstream_schedule_p2p_and_lifetimes(self):
        document = capture_pipeline_schedule(activation_bytes=8)
        self.assertEqual(document['source'], 'torch.distributed.pipelining.ScheduleInterleaved1F1B')
        actions = document['metadata']['upstream_actions']
        self.assertEqual(len(actions), 2)
        self.assertTrue(any('SEND_F' in action for action in actions['0']))
        transfers = [e for e in document['events'] if e['kind'] == 'collective']
        self.assertEqual(len(transfers), 24)
        self.assertEqual(sum(e['kind'] == 'compute' for e in document['events']), 32)
        result = simulate(document['events'], HW)
        self.assertEqual(set(result['summary']['per_rank']), {'0', '1'})
        self.assertTrue(all(r['final_allocated_bytes'] == 0 for r in result['summary']['per_rank'].values()))
        self.assertTrue(all(x['reclaim_us'] is not None for x in result['storage_lifetimes']))
        self.assertTrue(any(x['reclaim_us'] > x['allocate_us'] for x in result['storage_lifetimes']))

    def test_imbalance_changes_critical_path(self):
        balanced = simulate(capture_pipeline_schedule(activation_bytes=8)['events'], HW)
        slow = simulate(capture_pipeline_schedule(activation_bytes=8, forward_us=[10, 100, 10, 10])['events'], HW)
        self.assertGreater(slow['summary']['runtime_us'], balanced['summary']['runtime_us'])

    def test_configuration_rejected(self):
        with self.assertRaises(ValueError):
            capture_pipeline_schedule(num_stages=3)
        with self.assertRaises(ValueError):
            capture_pipeline_schedule(forward_us=[1, 2])


if __name__ == '__main__':
    unittest.main()
