import os
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class LocalExecutionTests(unittest.TestCase):
    def test_virtual_groups_and_finite_representatives(self):
        from titan_sim.local_execution import virtual_groups, communication_mode
        dist = torch.distributed
        prior = os.environ.get('WORLD_SIZE')
        groups, communication = [], []
        with virtual_groups(rank=1, world_size=4, records=groups):
            dist.init_process_group('gloo')
            self.assertEqual(dist.get_rank(), 1)
            self.assertEqual(dist.get_world_size(), 4)
            self.assertEqual(dist.get_process_group_ranks(dist.group.WORLD), [0, 1, 2, 3])
            with communication_mode(communication):
                source = torch.tensor([2., 3.])
                gathered = torch.empty(8)
                dist.all_gather_into_tensor(gathered, source)
                self.assertTrue(torch.equal(gathered, source.repeat(4)))
                reduced = torch.empty(2)
                dist.reduce_scatter_tensor(reduced, gathered)
                self.assertTrue(torch.equal(reduced, source))
                dist.all_reduce(source)
                self.assertTrue(torch.isfinite(source).all())
        self.assertEqual(groups[0]['requested_backend'], 'gloo')
        self.assertFalse(dist.is_initialized())
        self.assertEqual(os.environ.get('WORLD_SIZE'), prior)
        self.assertEqual(len(communication), 3)
        self.assertTrue(all(c['representative_outputs_finite'] for c in communication))

    def test_nonfinite_source_rejected(self):
        from titan_sim.local_execution import virtual_groups, communication_mode
        with virtual_groups(rank=0, world_size=2, records=[]):
            torch.distributed.init_process_group('gloo')
            with communication_mode([]), self.assertRaisesRegex(ValueError, 'not finite'):
                torch.distributed.all_reduce(torch.tensor([float('nan')]))

    def test_target_mismatch_rejected_and_restored(self):
        from titan_sim.local_execution import virtual_groups
        with virtual_groups(rank=0, world_size=2, records=[]):
            with self.assertRaisesRegex(ValueError, 'differs'):
                torch.distributed.init_process_group('gloo', rank=0, world_size=1)
        self.assertFalse(torch.distributed.is_initialized())

    def test_local_cuda_requires_actual_device(self):
        from titan_sim.local_execution import capture
        if torch.cuda.is_available():
            self.skipTest('CPU preflight check')
        with self.assertRaisesRegex(ValueError, 'actual CUDA'):
            capture(dict(execution='local_cuda', target_rank=0, target_world_size=2,
                         recipe_args=[], hardware={'id': 'unit'}))

    def test_environment_restored_when_group_cleanup_raises(self):
        from unittest.mock import patch
        from titan_sim.local_execution import virtual_groups
        import torch.testing._internal.distributed.fake_pg
        original = {key: os.environ.get(key) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')}
        with patch.object(torch.distributed, 'is_initialized', side_effect=[False, True]), \
             patch.object(torch.distributed, 'destroy_process_group', side_effect=RuntimeError('cleanup failed')):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                with virtual_groups(rank=1, world_size=4, records=[]):
                    self.assertEqual(os.environ['WORLD_SIZE'], '4')
        self.assertEqual({key: os.environ.get(key) for key in original}, original)
