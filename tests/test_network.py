import unittest
from titan_sim.network import estimate_network


def hardware():
    return {'network_topology': {'links': [
        dict(id='a', src=0, dst=1, bandwidth_bytes_s=1e6, latency_us=2),
        dict(id='b', src=1, dst=0, bandwidth_bytes_s=2e6, latency_us=3)]}}


def event(op='all_reduce', **metadata):
    return dict(event_id='x', kind='collective', collective=op,
                participants=[0, 1], message_bytes=100) | metadata


class NetworkTests(unittest.TestCase):
    def test_heterogeneous_allreduce_counts_payload_once(self):
        result = estimate_network(event(bytes_read=100, bytes_write=100), hardware())
        self.assertEqual(result['traffic_bytes'], 200)
        self.assertEqual(result['duration_us'], 106)
        self.assertEqual(result['network_resources'], ['network:a', 'network:b'])
        self.assertEqual(len(result['network_phases']), 2)

    def test_allgather_and_reduce_scatter(self):
        for op in ('all_gather', 'reduce_scatter'):
            result = estimate_network(event(op), hardware())
            self.assertEqual(result['duration_us'], 53)
            self.assertEqual(result['traffic_bytes'], 100)

    def test_shared_channel_contention(self):
        hw = hardware()
        for link in hw['network_topology']['links']:
            link.update(resource='shared', resource_bandwidth_bytes_s=1e6)
        result = estimate_network(event('all_gather'), hw)
        self.assertEqual(result['duration_us'], 103)
        self.assertEqual(result['network_resources'], ['network:shared'])

    def test_p2p_multihop_bottleneck(self):
        hw = {'network_topology': {'links': [
            dict(id='a', src=0, dst='s', bandwidth_bytes_s=2e6, latency_us=1),
            dict(id='b', src='s', dst=1, bandwidth_bytes_s=1e6, latency_us=2)]}}
        result = estimate_network(event('p2p', src=0, dst=1), hw)
        self.assertEqual(result['duration_us'], 103)
        self.assertEqual(result['network_phases'][0]['transfers'][0]['route'], ['a', 'b'])

    def test_disconnected_and_invalid_routes_fail(self):
        hw = hardware()
        hw['network_topology']['links'].pop()
        with self.assertRaisesRegex(ValueError, 'unknown route'):
            estimate_network(event(), hw)
        hw = hardware()
        hw['network_topology']['routes'] = {'0->1': ['b']}
        with self.assertRaisesRegex(ValueError, 'noncontiguous'):
            estimate_network(event(), hw)

    def test_uniform_and_variable_alltoall(self):
        self.assertEqual(estimate_network(event('all_to_all'), hardware())['duration_us'], 53)
        result = estimate_network(event('all_to_all', input_splits=[[0, 30], [80, 0]],
                                        output_splits=[[0, 80], [30, 0]]), hardware())
        self.assertEqual(result['traffic_bytes'], 110)
        self.assertEqual(result['duration_us'], 43)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            estimate_network(event('all_to_all', input_splits=[[0, 30], [80, 0]],
                                   output_splits=[[0, 30], [80, 0]]), hardware())

    def test_barrier_and_single_rank(self):
        self.assertEqual(estimate_network(event('barrier', message_bytes=0), hardware())['duration_us'], 3)
        self.assertEqual(estimate_network(event(participants=[0]), hardware())['duration_us'], 0)

    def test_missing_payload_and_unsupported_algorithm_fail(self):
        with self.assertRaisesRegex(ValueError, 'algorithm'):
            estimate_network(event(algorithm='tree'), hardware())
        bad = event()
        del bad['message_bytes']
        with self.assertRaisesRegex(ValueError, 'message_bytes'):
            estimate_network(bad, hardware())

    def test_scheduler_serializes_shared_route_and_overlaps_disjoint_routes(self):
        from titan_sim.runtime import simulate
        hw = dict(memory_capacity_bytes=1000, memory_bandwidth_bytes_s=1e6,
                  peak_flops={'float32': 1e6}, network_topology={'links': [
                      dict(id='left', src=0, dst=1, bandwidth_bytes_s=1e6, resource='fabric'),
                      dict(id='right', src=2, dst=3, bandwidth_bytes_s=1e6, resource='fabric')]})
        events = [event('p2p', event_id='a', rank=0, src=0, dst=1, participants=[0, 1]),
                  event('p2p', event_id='b', rank=2, src=2, dst=3, participants=[2, 3])]
        result = simulate(events, hw)
        self.assertEqual(result['summary']['runtime_us'], 200)
        self.assertEqual([r['start_us'] for r in result['timeline']], [0, 100])
        hw['network_topology']['links'][1]['resource'] = 'independent'
        result = simulate(events, hw)
        self.assertEqual(result['summary']['runtime_us'], 100)
        self.assertEqual([r['start_us'] for r in result['timeline']], [0, 0])

    def test_explicit_network_duration_keeps_contention_resources(self):
        from titan_sim.runtime import simulate
        hw = dict(memory_capacity_bytes=1000, memory_bandwidth_bytes_s=1e6,
                  peak_flops={'float32': 1e6}, **hardware())
        result = simulate([event(duration_us=7, duration_provenance='independent calibration')], hw)
        self.assertEqual(result['summary']['runtime_us'], 7)
        self.assertEqual(result['timeline'][0]['network_resources'], ['network:a', 'network:b'])
        self.assertEqual(result['timeline'][0]['source'], 'explicit')

    def test_p2p_participant_mismatch_and_shared_endpoint_fail(self):
        with self.assertRaisesRegex(ValueError, 'participants must match'):
            estimate_network(event('p2p', src=2, dst=3), hardware())
        hw = hardware()
        hw['network_topology']['rank_nodes'] = {'0': 'same_host', '1': 'same_host'}
        with self.assertRaisesRegex(ValueError, 'distinct topology endpoints'):
            estimate_network(event(), hw)

    def test_participant_aliases_must_agree(self):
        with self.assertRaisesRegex(ValueError, 'must agree'):
            estimate_network(event(participant_ranks=[0, 2]), hardware())
