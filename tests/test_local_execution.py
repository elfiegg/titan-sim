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

    def test_custom_dispatch_keeps_below_autograd_context_and_exact_source_node(self):
        from titan_sim.local_execution import communication_mode
        from titan_sim.executable_regions import LiveRegionRegistry, region_provider
        from titan_sim.torch_capture import TorchCapture
        setup_calls = []
        @torch.library.custom_op('titan_test::observed_square', mutates_args=())
        def square(x: torch.Tensor) -> torch.Tensor:
            return x * x
        def setup(ctx, inputs, output):
            setup_calls.append(True)
        square.register_autograd(lambda ctx, grad: grad, setup_context=setup)
        recorder, records = TorchCapture(), []
        value = torch.ones(2, requires_grad=True)
        with communication_mode([], native_records=records, retain_arguments=True,
                recorder_getter=lambda: recorder), recorder:
            output = square(value)
        self.assertTrue(output.requires_grad)
        record = records[0]
        self.assertEqual(record['source_occurrences'], [[0]])
        self.assertEqual(record['status'], 'returned')
        self.assertEqual(recorder.nodes[0].attrs['native_region_member_node_ids'], [0])
        registry = LiveRegionRegistry()
        with torch.no_grad():
            registry.register_observed(record, native_callable_index=0, source_workload_sha256='a'*64,
                context={'hardware_id':'unit'}, state_contract={'mutation':'read_only','communication':'none'}, replay_safe=True)
            self.assertIs(registry.entries[0]['native_handle'], torch.ops.titan_test.observed_square.default)
            replay = region_provider(registry.calibration_workload(hardware={'id':'unit'}), {})['regions'][0]['run']()
        self.assertTrue(torch.equal(replay, output))
        self.assertFalse(replay.requires_grad)
        self.assertEqual(len(setup_calls), 1)
        record['observation_error'] = 'output contract unavailable'
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'observation is incomplete'):
            registry.register_observed(record, native_callable_index=0, source_workload_sha256='a'*64,
                context={'hardware_id':'unit'},
                state_contract={'mutation':'read_only','communication':'none'}, replay_safe=True)
        registry.close()

    def test_custom_observation_without_matching_issued_node_has_no_occurrence(self):
        from titan_sim.local_execution import communication_mode
        from titan_sim.torch_capture import TorchCapture
        @torch.library.custom_op('titan_test::unmatched', mutates_args=())
        def native(x: torch.Tensor) -> torch.Tensor:
            return x.clone()
        recorder, records = TorchCapture(), []
        recorder._node('unrelated', operator='other', status='issued')
        with communication_mode([], native_records=records, retain_arguments=True, recorder_getter=lambda: recorder):
            native(torch.ones(2))
        self.assertEqual(records[0]['source_occurrences'], [])
        self.assertNotIn('native_callable_boundary', recorder.nodes[0].attrs)

    def test_custom_failure_and_selective_retention(self):
        from titan_sim.local_execution import communication_mode
        from titan_sim.executable_regions import ArgumentRetention
        from titan_sim.torch_capture import TorchCapture
        @torch.library.custom_op('titan_test::bounded', mutates_args=())
        def native(x: torch.Tensor) -> torch.Tensor:
            return x.clone()
        recorder, records = TorchCapture(), []
        policy = ArgumentRetention(True, ['titan_test.bounded.default'], 1)
        with communication_mode([], native_records=records, retention=policy, recorder_getter=lambda: recorder), recorder:
            for _ in range(3):
                native(torch.ones(2))
        self.assertIsNotNone(records[0]['source_arguments'][0])
        self.assertTrue(all(r['source_arguments'] == [None] for r in records[1:]))
        self.assertTrue(all(r['source_occurrences'] for r in records))

    def test_mutable_custom_requires_reset_even_with_unchanged_shape(self):
        from titan_sim.local_execution import communication_mode
        from titan_sim.executable_regions import LiveRegionRegistry
        from titan_sim.torch_capture import TorchCapture
        @torch.library.custom_op('titan_test::mutating', mutates_args=('x',))
        def native(x: torch.Tensor) -> None:
            x.add_(1)
        recorder, records = TorchCapture(), []
        with torch.no_grad(), communication_mode([], native_records=records, retain_arguments=True,
                recorder_getter=lambda: recorder), recorder:
            native(torch.ones(2))
        registry = LiveRegionRegistry()
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'state changed|mutable custom schema'):
            registry.register_observed(records[0], native_callable_index=0, source_workload_sha256='a'*64,
                context={'hardware_id':'unit'}, state_contract={'mutation':'read_only','communication':'none'}, replay_safe=True)
        registry.close()
