import unittest
from titan_sim.runtime import simulate
from titan_sim.events import Event

HW = {'name': 'synthetic', 'peak_flops': {'float32': 1e6},
      'memory_bandwidth_bytes_s': 1e6, 'memory_capacity_bytes': 100,
      'network_bandwidth_bytes_s': 1e6}


def op(event_id, **kwargs):
    return {'event_id': event_id, 'kind': 'compute', 'flops': 10, **kwargs}


def allocation(event_id, storage_id='x', size_bytes=80, **kwargs):
    return {'event_id': event_id, 'kind': 'allocate', 'storage_id': storage_id,
            'size_bytes': size_bytes, **kwargs}


def release(event_id, storage_id='x', **kwargs):
    return {'event_id': event_id, 'kind': 'release', 'storage_id': storage_id, **kwargs}


def indexed(result):
    return {e['event_id']: e for e in result['timeline']}


class RuntimeTests(unittest.TestCase):
    def test_stream_order_and_explicit_dependency(self):
        result = simulate([op('a'), op('b'),
                           {'event_id': 'c', 'kind': 'collective', 'stream': 'comm',
                            'deps': ['b'], 'bytes_read': 4}], HW)
        records = indexed(result)
        self.assertEqual(records['a']['start_us'], 0)
        self.assertEqual(records['b']['start_us'], 10)
        self.assertEqual(records['c']['start_us'], 20)
        self.assertEqual(result['summary']['runtime_us'], 24)

    def test_host_submission_async_and_rank_sync(self):
        events = [op('a'), {'event_id': 'sync', 'kind': 'host_sync', 'stream': 'host'},
                  op('b', stream='other')]
        records = indexed(simulate(events, HW))
        self.assertEqual(records['a']['submit_us'], 0)
        self.assertEqual(records['sync']['submit_us'], 0)
        self.assertEqual(records['sync']['end_us'], 10)
        self.assertEqual(records['b']['submit_us'], 10)

    def test_scoped_sync_does_not_drain_unrelated_communication(self):
        for scope in ('event', 'stream'):
            with self.subTest(scope=scope):
                events = [{'event_id': 'long_comm', 'kind': 'collective', 'stream': 'comm', 'bytes_read': 100},
                          op('short_compute', stream='short'),
                          {'event_id': 'sync', 'kind': 'host_sync', 'stream': 'short',
                           'deps': ['short_compute'], 'sync_scope': scope},
                          allocation('after_sync', stream='host')]
                records = indexed(simulate(events, HW))
                self.assertEqual(records['sync']['end_us'], 10)
                self.assertEqual(records['sync']['sync_scope'], scope)
                self.assertEqual(records['after_sync']['submit_us'], 10)
                self.assertEqual(records['long_comm']['end_us'], 100)
                self.assertNotIn('long_comm', records['sync']['deps'])

    def test_device_sync_drains_unrelated_work_and_rejects_unknown_scope(self):
        events = [{'event_id': 'long_comm', 'kind': 'collective', 'stream': 'comm', 'bytes_read': 100},
                  op('short_compute', stream='short'),
                  {'event_id': 'sync', 'kind': 'host_sync', 'stream': 'host', 'sync_scope': 'device'}]
        self.assertEqual(indexed(simulate(events, HW))['sync']['end_us'], 100)
        events[-1]['sync_scope'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'sync_scope'):
            simulate(events, HW)

    def test_wait_does_not_block_host(self):
        events = [op('a'), {'event_id': 'wait', 'kind': 'wait', 'stream': 'other', 'deps': ['a']},
                  allocation('alloc', stream='third')]
        records = indexed(simulate(events, HW))
        self.assertEqual(records['wait']['end_us'], 10)
        self.assertEqual(records['alloc']['submit_us'], 0)

    def test_compute_serial_comm_ideal_overlap(self):
        result = simulate([op('a'), op('b', stream='second'),
                           {'event_id': 'comm', 'kind': 'collective', 'stream': 'comm', 'bytes_read': 8}], HW)
        records = indexed(result)
        self.assertEqual(records['b']['start_us'], 10)
        self.assertEqual(records['comm']['start_us'], 0)
        self.assertIn('ideal', result['summary']['overlap_model'])

    def test_waiting_stream_does_not_reserve_idle_compute_lane(self):
        events = [{'event_id': 'comm', 'kind': 'collective', 'stream': 'comm', 'bytes_read': 100},
                  op('dependent', stream='first', deps=['comm']),
                  op('ready', stream='second')]
        records = indexed(simulate(events, HW))
        self.assertEqual(records['ready']['start_us'], 0)
        self.assertEqual(records['dependent']['start_us'], 100)

    def test_alias_storage_and_inflight_release(self):
        result = simulate([allocation('alloc'), allocation('alias'),
                           op('a', read_storage_ids=['x']), release('drop_alias'), release('drop_last')], HW)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 80)
        self.assertEqual(len(result['memory']), 2)
        self.assertEqual(result['memory'][-1]['time_us'], 10)
        self.assertEqual(result['memory'][-1]['allocated_bytes'], 0)
        self.assertEqual(result['storage_lifetimes'][0]['remaining_refs'], 0)

    def test_partial_alias_release_retains_storage(self):
        result = simulate([allocation('alloc'), allocation('alias'), release('drop_one')], HW)
        self.assertEqual(result['summary']['per_rank']['0']['final_allocated_bytes'], 80)
        self.assertIsNone(result['storage_lifetimes'][0]['reclaim_us'])

    def test_reused_storage_identifier_is_new_generation(self):
        result = simulate([allocation('alloc'), op('a', reads=['x']), release('drop'),
                           allocation('reuse'), op('b', writes=['x'])], HW)
        self.assertEqual(len(result['storage_lifetimes']), 2)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 160)
        self.assertTrue(result['summary']['scheduled_capacity_exceeded'])
        self.assertIsNone(result['summary']['capacity_exceeded'])
        self.assertEqual(result['summary']['logical_peak_allocated_bytes'], 80)
        self.assertEqual(result['summary']['per_rank']['0']['final_allocated_bytes'], 80)

    def test_explicit_host_schedule_memory_status(self):
        result = simulate([allocation('alloc', size_bytes=150, host_time_us=0),
                           release('drop', host_time_us=2)], HW)
        self.assertTrue(result['summary']['capacity_exceeded'])
        self.assertEqual(result['summary']['memory_confidence'], 'explicit_host_schedule_scenario')
        self.assertEqual([sample['allocated_bytes'] for sample in result['logical_memory']], [150, 0])
        self.assertEqual(result['logical_memory'][1]['event_index'], 1)

    def test_cpu_storage_and_work_excluded_from_gpu_projection(self):
        result = simulate([allocation('cpu_alloc', size_bytes=200, device='cpu'),
                           op('cpu_op', reads=['x'], device='cpu'), release('cpu_drop')], HW)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 0)
        self.assertEqual(result['summary']['logical_peak_allocated_bytes'], 0)
        self.assertEqual(result['summary']['runtime_us'], 0)
        self.assertEqual(result['summary']['coverage']['work_events'], 0)
        self.assertEqual(indexed(result)['cpu_op']['source'], 'host_unmodeled')

    def test_meta_initialization_excluded_but_fake_cuda_stays_counted(self):
        events = [allocation('meta_alloc', storage_id='meta', size_bytes=200000, device='meta'),
                  op('meta_op', reads=['meta'], device='meta'),
                  allocation('cuda_alloc', storage_id='cuda', size_bytes=80, device='cuda:0'),
                  op('cuda_op', reads=['cuda'], device='cuda:0'),
                  release('meta_drop', storage_id='meta')]
        result = simulate(events, HW)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 80)
        self.assertEqual(result['summary']['logical_peak_allocated_bytes'], 80)
        self.assertEqual(result['summary']['runtime_us'], 10)
        self.assertEqual(result['summary']['coverage']['work_events'], 1)
        self.assertEqual(indexed(result)['meta_op']['source'], 'meta_unmodeled')

    def test_rank_local_storage_and_sync(self):
        result = simulate([allocation('alloc0'), allocation('alloc1', rank=1),
                           op('rank0', reads=['x']), op('rank1', rank=1, reads=['x'])], HW)
        self.assertEqual(result['summary']['runtime_us'], 10)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 80)
        self.assertEqual(result['summary']['per_rank']['1']['peak_allocated_bytes'], 80)

    def test_cycles_missing_and_duplicates(self):
        invalid = [[op('a', deps=['missing'])], [op('a', deps=['b']), op('b', deps=['a'])],
                   [op('a'), op('a')]]
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(ValueError):
                simulate(events, HW)

    def test_storage_errors(self):
        invalid = [[release('drop')], [op('a', reads=['unknown'])],
                   [allocation('alloc'), release('drop'), op('a', reads=['x'])],
                   [allocation('alloc'), allocation('alias', size_bytes=20)]]
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(ValueError):
                simulate(events, HW)

    def test_host_timestamp_is_monotonic(self):
        records = indexed(simulate([op('a', host_time_us=10), op('b', host_time_us=0)], HW))
        self.assertEqual(records['b']['submit_us'], 10)

    def test_zero_work_empty_and_metadata(self):
        result = simulate([], HW)
        self.assertEqual(result['summary']['runtime_us'], 0)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 0)
        event = Event.from_dict(op('a', flops=0, arbitrary_metadata={'key': 1}))
        self.assertEqual(event.metadata['arbitrary_metadata'], {'key': 1})
        self.assertEqual(simulate([event], HW)['summary']['runtime_us'], 0)

    def test_hybrid_coverage(self):
        result = simulate([op('a'), op('b', duration_us=5, duration_provenance='fixture')], HW, 'hybrid')
        self.assertEqual(result['summary']['coverage']['fallback_events'], 1)
        self.assertEqual(result['summary']['coverage']['calibrated_events'], 0)
        self.assertEqual(result['summary']['coverage']['explicit_events'], 1)
        self.assertEqual(result['summary']['coverage']['fallback_flop_fraction'], 0.5)
        self.assertAlmostEqual(result['summary']['coverage']['fallback_duration_fraction'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
