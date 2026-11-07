import copy
import unittest

from titan_sim.core import Node, Storage, simulate
from titan_sim.memory import COVERAGE_CATEGORIES, assess_memory


class PhysicalMemoryTests(unittest.TestCase):
    def fixture(self):
        nodes = [Node(0, 'left', 0), Node(1, 'right', 1)]
        storage = [Storage('a', 0, 60, 0), Storage('b', 1, 70, 1)]
        timing = {'0': (0, 2), '1': (1, 3)}
        hardware = {'devices': {'gpu': {'capacity_bytes': 200}}, 'reserve_fraction': 0}
        manifest = {'ranks': [0, 1], 'rank_placement': {'0': 'gpu', '1': 'gpu'},
                    'memory': {'coverage': dict.fromkeys(COVERAGE_CATEGORIES, True)}}
        return nodes, storage, timing, hardware, manifest

    def test_collocated_overlap_and_nonoverlap_are_not_sum_of_rank_peaks(self):
        args = self.fixture()
        result = assess_memory(*args)
        self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 130)
        self.assertEqual(result['per_device']['gpu']['peak_upper_bytes'], 130)
        self.assertEqual(result['memory_verdict'], 'estimated_fit')
        args[2]['1'] = (2, 3)
        result = assess_memory(*args)
        self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 70)
        self.assertEqual(result['per_device']['gpu']['peak_upper_bytes'], 70)

    def test_persistent_state_survives_last_user(self):
        args = self.fixture()
        args[1][0].persistent = True
        args[2]['1'] = (2, 3)
        result = assess_memory(*args)
        self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 130)

    def test_aliases_union_lifetimes_and_count_backing_once(self):
        args = self.fixture()
        args[1][1].size = 60
        args[4]['memory']['storage_aliases'] = {'b': 'a'}
        result = assess_memory(*args)
        self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 60)
        self.assertEqual(result['per_device']['gpu']['peak_upper_bytes'], 60)
        args[4]['memory']['allocations'] = [dict(id='workspace-view', device='gpu',
            category='workspaces', bytes=60, storage_id='a')]
        self.assertEqual(assess_memory(*args)['per_device']['gpu']['peak_upper_bytes'], 60)

    def test_heterogeneous_capacity_checks_each_device(self):
        args = self.fixture()
        args[3]['devices']['small'] = {'capacity_bytes': 50}
        args[4]['rank_placement']['1'] = 'small'
        result = assess_memory(*args)
        self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 60)
        self.assertEqual(result['per_device']['small']['peak_lower_bytes'], 70)
        self.assertEqual(result['memory_verdict'], 'estimated_not_fit')

    def test_external_bounds_and_allocator_slack(self):
        args = self.fixture()
        args[4]['memory']['allocations'] = [
            dict(id='workspace', rank=0, category='workspaces', lower_bytes=10,
                 upper_bytes=30, start_seconds=0, end_seconds=2),
            dict(id='slack', device='gpu', category='allocator_slack', bytes=20),
        ]
        result = assess_memory(*args)['per_device']['gpu']
        self.assertEqual(result['peak_lower_bytes'], 160)
        self.assertEqual(result['peak_upper_bytes'], 180)
        self.assertEqual(result['peak_upper_categories_bytes']['allocator_slack'], 20)
        args[3]['devices']['gpu']['capacity_bytes'] = 170
        self.assertEqual(assess_memory(*args)['memory_verdict'], 'inconclusive')

    def test_missing_coverage_blocks_fit_but_not_known_overflow(self):
        args = self.fixture()
        del args[4]['memory']['coverage']['graph_pools']
        result = assess_memory(*args)
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        self.assertIsNone(result['per_device']['gpu']['peak_upper_bytes'])
        self.assertEqual(result['per_device']['gpu']['modeled_peak_upper_bytes'], 130)
        args[3]['devices']['gpu']['capacity_bytes'] = 100
        self.assertEqual(assess_memory(*args)['memory_verdict'], 'estimated_not_fit')

    def test_unknown_external_upper_bound_is_local(self):
        args = self.fixture()
        args[3]['devices']['other'] = {'capacity_bytes': 200}
        args[4]['rank_placement']['1'] = 'other'
        args[4]['memory']['allocations'] = [dict(id='unknown', device='gpu',
            category='workspaces', lower_bytes=0, upper_bytes=None)]
        result = assess_memory(*args)
        self.assertIsNone(result['per_device']['gpu']['peak_upper_bytes'])
        self.assertEqual(result['per_device']['other']['peak_upper_bytes'], 70)
        self.assertEqual(result['memory_verdict'], 'inconclusive')

    def test_placement_is_required_for_physical_verdict(self):
        args = self.fixture()
        args[3].pop('devices')
        args[3]['capacity_bytes'] = 200
        del args[4]['rank_placement']
        result = assess_memory(*args)
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        self.assertTrue(result['memory_assumptions'])

    def test_named_devices_without_placement_leave_ranks_unassigned(self):
        args = self.fixture()
        del args[4]['rank_placement']
        result = assess_memory(*args)
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        provisional = result['per_device']['unplaced-rank:0']
        self.assertIsNone(provisional['capacity_bytes'])
        self.assertEqual(provisional['peak_lower_bytes'], 60)
        self.assertFalse(provisional['placement_verified'])
        self.assertIsNone(result['per_device']['gpu']['peak_upper_bytes'])

    def test_lifecycle_peak_is_not_added_to_training_peak(self):
        args = self.fixture()
        args[4]['memory']['lifecycle_peaks'] = [dict(phase='initialization',
            device='gpu', lower_bytes=180, upper_bytes=190)]
        result = assess_memory(*args)
        self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 180)
        self.assertEqual(result['per_device']['gpu']['peak_upper_bytes'], 190)
        self.assertEqual(result['memory_verdict'], 'estimated_fit')
        args[4]['memory']['lifecycle_peaks'][0]['upper_bytes'] = None
        self.assertEqual(assess_memory(*args)['memory_verdict'], 'inconclusive')

    def test_lifecycle_rank_peaks_cannot_be_summed_under_collocation(self):
        args = self.fixture()
        args[4]['memory']['lifecycle_peaks'] = [dict(phase='initialization',
            rank=0, lower_bytes=180, upper_bytes=190)]
        with self.assertRaisesRegex(ValueError, 'device-wide'):
            assess_memory(*args)

    def test_recipe_and_schedule_scopes_require_different_coverage(self):
        args = self.fixture()
        del args[4]['memory']['coverage']['initialization']
        self.assertEqual(assess_memory(*args)['memory_verdict'], 'inconclusive')
        args[4]['memory']['scope'] = 'schedule'
        self.assertEqual(assess_memory(*args)['memory_verdict'], 'estimated_fit')
        self.assertEqual(assess_memory(*args)['memory_scope'], 'schedule')

    def test_zero_duration_allocations_produce_safe_bounds(self):
        args = self.fixture()
        args[2].update({'0': (1, 1), '1': (1, 1)})
        result = assess_memory(*args)['per_device']['gpu']
        self.assertEqual(result['peak_lower_bytes'], 70)
        self.assertEqual(result['peak_upper_bytes'], 130)

    def test_zero_duration_last_user_keeps_safe_upper_at_boundary(self):
        args = self.fixture()
        args[0].append(Node(2, 'instant-user', 0))
        args[1][0].consumers = [2]
        args[2].update({'0': (0, 1), '1': (1, 2), '2': (1, 1)})
        result = assess_memory(*args)['per_device']['gpu']
        self.assertEqual(result['peak_lower_bytes'], 70)
        self.assertEqual(result['peak_upper_bytes'], 130)

    def test_invalid_aliases_rejected(self):
        for aliases in ({'a': 'b', 'b': 'a'}, {'a': 'missing'}, {'a': 'b'}):
            with self.subTest(aliases=aliases):
                args = self.fixture()
                args[4]['memory']['storage_aliases'] = aliases
                with self.assertRaises(ValueError):
                    assess_memory(*args)
        args = self.fixture()
        args[1][1].size = 60
        args[3]['devices']['other'] = {'capacity_bytes': 200}
        args[4]['rank_placement']['1'] = 'other'
        args[4]['memory']['storage_aliases'] = {'a': 'b'}
        with self.assertRaisesRegex(ValueError, 'physical device'):
            assess_memory(*args)

    def test_invalid_quantities_rejected(self):
        for bad in (-1, float('nan'), float('inf'), True, 1.5, '10'):
            with self.subTest(bad=bad):
                args = self.fixture()
                args[1][0].size = bad
                with self.assertRaises(ValueError):
                    assess_memory(*args)
        for bad in (-1, float('nan'), float('inf'), True):
            with self.subTest(timing=bad):
                args = self.fixture()
                args[2]['0'] = (0, bad)
                with self.assertRaises(ValueError):
                    assess_memory(*args)
        for bad in (-1, 1, float('nan'), float('inf'), True):
            with self.subTest(reserve=bad):
                args = self.fixture()
                args[3]['reserve_fraction'] = bad
                with self.assertRaises(ValueError):
                    assess_memory(*args)

    def test_rejects_reserved_double_count_and_invalid_bounds(self):
        base = dict(id='extra', device='gpu', category='allocator_slack', lower_bytes=10, upper_bytes=20)
        for change in ({'category': 'allocator_reserved'}, {'upper_bytes': 9},
                       {'bytes': 15}, {'start_seconds': 5, 'end_seconds': 2},
                       {'upper_bytes': float('nan')}, {'device': 'unknown'},
                       {'rank': 0}, {'id': 'a'}):
            with self.subTest(change=change):
                args = self.fixture()
                args[4]['memory']['allocations'] = [dict(base, **change)]
                with self.assertRaises(ValueError):
                    assess_memory(*args)

    def test_simulation_maps_default_resources_to_physical_device(self):
        nodes, storage, _, hardware, manifest = self.fixture()
        hardware['id'] = 'test'
        for node in nodes:
            node.attrs['legacy_seconds'] = 2
        result = simulate(nodes, storage, manifest, hardware)
        self.assertEqual(result['seconds'], 4)
        self.assertEqual(result['timing']['0'], (0, 2))
        self.assertEqual(result['timing']['1'], (2, 4))
        self.assertEqual(result['physical_memory']['per_device']['gpu']['peak_lower_bytes'], 70)
        self.assertEqual(result['memory_verdict'], 'estimated_fit')
        for node in nodes:
            node.attrs['resources'] = ['stream:' + str(node.rank)]
        result = simulate(nodes, storage, manifest, hardware)
        self.assertEqual(result['seconds'], 2)
        self.assertEqual(result['physical_memory']['per_device']['gpu']['peak_lower_bytes'], 130)

    def test_timing_status_does_not_hide_memory_coverage_gaps(self):
        nodes, storage, _, hardware, manifest = self.fixture()
        hardware['id'] = 'test'
        for node in nodes:
            node.attrs['zero_cost_semantic'] = True
        del manifest['memory']['coverage']['graph_pools']
        result = simulate(nodes, storage, manifest, hardware, strict=True)
        self.assertEqual(result['status'], 'calibrated_prediction')
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        self.assertEqual(result['memory_unresolved'], result['physical_memory']['memory_unresolved'])
        self.assertIn('memory coverage not established: graph_pools', result['memory_unresolved'])

    def test_unverified_retained_projection_cannot_prove_out_of_memory(self):
        args = self.fixture()
        args[1][0].persistent = True
        args[2]['1'] = (2, 3)
        args[3]['devices']['gpu']['capacity_bytes'] = 100
        args[4]['memory']['storage_evidence'] = 'unverified'
        args[4]['unresolved'] = ['Observed extents retained conservatively to step end']
        result = assess_memory(*args)
        device = result['per_device']['gpu']
        self.assertEqual(device['modeled_peak_lower_bytes'], 130)
        self.assertEqual(device['modeled_peak_upper_bytes'], 130)
        self.assertEqual(device['peak_lower_bytes'], 0)
        self.assertIsNone(device['peak_upper_bytes'])
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        self.assertEqual(result['storage_evidence'], 'unverified')

    def test_unverified_storage_does_not_invalidate_independent_allocation_lower(self):
        args = self.fixture()
        args[3]['devices']['gpu']['capacity_bytes'] = 100
        args[4]['memory']['storage_evidence'] = 'unverified'
        args[4]['memory']['allocations'] = [dict(id='known-workspace', device='gpu',
            category='workspaces', lower_bytes=110, upper_bytes=120)]
        result = assess_memory(*args)
        self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 110)
        self.assertEqual(result['memory_verdict'], 'estimated_not_fit')

    def test_unverified_storage_requires_more_than_coverage_flags(self):
        args = self.fixture()
        args[4]['memory']['storage_evidence'] = 'unverified'
        self.assertEqual(assess_memory(*args)['memory_verdict'], 'inconclusive')
        args[4]['memory']['storage_evidence'] = 'exact'
        self.assertEqual(assess_memory(*args)['memory_verdict'], 'estimated_fit')
        args[4]['memory']['storage_evidence'] = 'observed'
        with self.assertRaisesRegex(ValueError, 'storage_evidence'):
            assess_memory(*args)

    def test_external_alias_proves_only_its_interval_under_unverified_storage(self):
        args = self.fixture()
        args[4]['memory']['storage_evidence'] = 'unverified'
        args[4]['memory']['allocations'] = [
            dict(id='known-a', device='gpu', category='workspaces', bytes=60,
                 storage_id='a', start_seconds=0, end_seconds=1),
            dict(id='known-a-alias', device='gpu', category='workspaces', bytes=60,
                 storage_id='a', start_seconds=0, end_seconds=1),
            dict(id='known-b', device='gpu', category='workspaces', bytes=70,
                 storage_id='b', start_seconds=2, end_seconds=3),
        ]
        result = assess_memory(*args)['per_device']['gpu']
        self.assertEqual(result['peak_lower_bytes'], 70)
        self.assertEqual(result['modeled_peak_upper_bytes'], 130)

    def test_legacy_profiler_projection_defaults_unverified_through_preparation(self):
        for provenance in (
            {'observer': 'pytorch_execution_trace_and_profiler'},
            {'observer': 'pytorch_profiler_chrome_only'},
            {'source_manifests': [{'source_manifests': [
                {'observer': 'pytorch_execution_trace_and_profiler'}]}]},
        ):
            with self.subTest(provenance=provenance):
                args = self.fixture()
                args[4].update(provenance)
                args[3]['devices']['gpu']['capacity_bytes'] = 100
                result = assess_memory(*args)
                self.assertEqual(result['memory_verdict'], 'inconclusive')
                self.assertEqual(result['per_device']['gpu']['peak_lower_bytes'], 0)
                self.assertEqual(result['per_device']['gpu']['modeled_peak_lower_bytes'], 130)
                args[4]['memory']['storage_evidence'] = 'exact'
                self.assertEqual(assess_memory(*args)['memory_verdict'], 'estimated_not_fit')

    def test_inputs_not_mutated(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        assess_memory(*args)
        self.assertEqual(args, before)


if __name__ == '__main__':
    unittest.main()
