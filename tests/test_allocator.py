import unittest
from titan_sim.allocator import replay_allocator
from titan_sim.runtime import simulate


def life(g, size, start, end=None, **extra):
    return dict(rank=0, generation=g, storage_id=str(g), size_bytes=size,
                allocate_us=start, reclaim_us=end, **extra)


def replay(lives, capacity=100, **config):
    return replay_allocator({'storage_lifetimes': lives}, {'memory_capacity_bytes': capacity},
                            dict(alignment_bytes=1, segment_bytes=100, release_cached_on_oom=False) | config)


class AllocatorTests(unittest.TestCase):
    def test_fragmentation_oom_even_when_live_storage_fits(self):
        lives = [life(0, 40, 0, 2), life(1, 20, 0), life(2, 40, 0, 2), life(3, 60, 3)]
        self.assertFalse(replay(lives, policy='ideal')['summary']['capacity_exceeded'])
        failure = replay(lives)['oom'][0]
        self.assertEqual(failure['requested_bytes'], 60)
        self.assertEqual(failure['total_pool_free_bytes'], 80)
        self.assertEqual(failure['largest_free_block_bytes'], 40)
        self.assertEqual(failure['before_allocated_bytes'], 20)

    def test_coalescing_enables_larger_reuse(self):
        lives = [life(0, 40, 0, 1), life(1, 60, 0, 1), life(2, 100, 2)]
        self.assertFalse(replay(lives)['summary']['capacity_exceeded'])
        self.assertTrue(replay(lives, coalesce=False)['summary']['capacity_exceeded'])

    def test_graph_pool_isolated_and_pinned(self):
        lives = [life(0, 100, 0, 1, metadata={'graph_pool': 'G'}), life(1, 100, 2)]
        result = replay(lives, release_cached_on_oom=True)
        self.assertEqual(result['oom'][0]['total_pool_free_bytes'], 0)
        self.assertEqual(result['oom'][0]['before_reserved_bytes'], 100)
        self.assertFalse(replay([lives[0], life(1, 100, 2, metadata={'graph_pool': 'G'})])['summary']['capacity_exceeded'])

    def test_whole_default_segment_evicted_on_retry(self):
        lives = [life(0, 100, 0, 1), life(1, 150, 2)]
        self.assertTrue(replay(lives, capacity=200)['summary']['capacity_exceeded'])
        result = replay(lives, capacity=200, release_cached_on_oom=True)
        self.assertFalse(result['summary']['capacity_exceeded'])
        self.assertEqual(result['summary']['per_rank']['0']['released_cached_bytes'], 100)

    def test_rounding_bins_and_external_counters(self):
        count = replay([life(0, 17, 0)], capacity=120, alignment_bytes=8,
                       bin_sizes_bytes=[32, 64], external_memory_bytes=10)['samples'][-1]
        self.assertEqual([count[k] for k in ('requested_bytes', 'allocated_bytes', 'reserved_bytes',
                                           'inactive_split_bytes', 'external_bytes', 'device_used_bytes')],
                         [17, 32, 104, 72, 10, 114])
        self.assertEqual(replay([life(0, 17, 0)], split=False)['samples'][-1]['allocated_bytes'], 100)

    def test_external_storage_never_cached(self):
        result = replay([life(0, 80, 0, 1, external_memory=True)])
        self.assertEqual(result['samples'][1]['external_bytes'], 80)
        self.assertEqual(result['samples'][1]['reserved_bytes'], 0)
        self.assertEqual(result['samples'][-1]['external_bytes'], 0)

    def test_async_use_defers_reuse_after_host_release(self):
        hw = dict(memory_capacity_bytes=100, memory_bandwidth_bytes_s=1e6, peak_flops={'float32': 1e6})
        events = [dict(event_id='a', kind='allocate', storage_id='a', size_bytes=80, host_time_us=0),
                  dict(event_id='use', kind='compute', flops=10, reads=['a'], stream='async', host_time_us=0),
                  dict(event_id='drop', kind='release', storage_id='a', host_time_us=1),
                  dict(event_id='b', kind='allocate', storage_id='b', size_bytes=80, host_time_us=2)]
        result = simulate(events, hw)
        self.assertEqual(result['storage_lifetimes'][0]['reclaim_us'], 10)
        allocator = replay_allocator(result, hw, {'policy': 'ideal'}, events)
        self.assertEqual(allocator['oom'][0]['before_allocated_bytes'], 80)

    def test_allocation_metadata_follows_cpu_and_alias_generations(self):
        hw = dict(memory_capacity_bytes=100, memory_bandwidth_bytes_s=1e6, peak_flops={'float32': 1e6})
        events = [dict(event_id='cpu', kind='allocate', storage_id='cpu', size_bytes=1, device='cpu'),
                  dict(event_id='a', kind='allocate', storage_id='a', size_bytes=80, graph_pool='G'),
                  dict(event_id='alias', kind='allocate', storage_id='a', size_bytes=80)]
        allocator = replay_allocator(simulate(events, hw), hw, {'segment_bytes': 100, 'alignment_bytes': 1}, events)
        self.assertEqual(len(allocator['allocations']), 1)
        self.assertEqual(allocator['allocations'][0]['pool'], 'G')

    def test_ties_and_invalid_configuration(self):
        self.assertTrue(replay([life(0, 100, 0, 1), life(1, 100, 1)], policy='ideal')['summary']['capacity_exceeded'])
        for config in ({'alignment_bytes': 0}, {'bin_sizes_bytes': [16, 8]}, {'typo': 1}, {'policy': 'cuda'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                replay([], **config)
        with self.assertRaisesRegex(ValueError, 'precedes'):
            replay([life(0, 1, 3, 2)])

    def test_external_capacity_without_tensor_lifetimes(self):
        result = replay([], external_memory_bytes=101)
        self.assertTrue(result['summary']['capacity_exceeded'])
        self.assertEqual(result['oom'][0]['rank'], 0)
        self.assertEqual(result['oom'][0]['reason'], 'external_memory_exceeds_capacity')
        self.assertFalse(replay([], external_memory_bytes=0)['summary']['capacity_exceeded'])

    def test_missing_categories_cannot_establish_full_target_fit(self):
        result = replay([life(0, 10, 0)], policy='ideal')
        self.assertEqual(result['summary']['capacity_status'], 'fits_recorded_allocations_only')
        self.assertFalse(result['summary']['full_target_fit_proven'])
        self.assertEqual(result['summary']['coverage']['unclassified_lifetimes'], 1)

    def test_external_metadata_validation(self):
        with self.assertRaisesRegex(ValueError, 'rank keys'):
            replay([], external_memory_bytes={'-1': 10})
        with self.assertRaisesRegex(ValueError, 'boolean'):
            replay([life(0, 10, 0, external_memory='false')])
