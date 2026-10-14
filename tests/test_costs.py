import unittest
import json
from pathlib import Path
from titan_sim.costs import estimate_cost, calibration_key, validate_hardware

HW = {'name': 'synthetic', 'peak_flops': {'float32': 1e12, 'float16': 2e12},
      'memory_bandwidth_bytes_s': 1e9, 'memory_capacity_bytes': 1000,
      'network_bandwidth_bytes_s': 1e8}


def compute(**kwargs):
    return {'event_id': 'op', 'kind': 'compute', 'name': 'matmul', **kwargs}


class CostTests(unittest.TestCase):
    def test_roofline_units_and_bound(self):
        cost = estimate_cost(compute(flops=2e6, bytes_read=3000, bytes_write=1000), HW)
        self.assertEqual(cost['compute_us'], 2)
        self.assertEqual(cost['memory_us'], 4)
        self.assertEqual(cost['duration_us'], 4)
        self.assertEqual(cost['bound'], 'memory')

    def test_dtype_peak_and_alias(self):
        cost = estimate_cost(compute(flops=2e6, dtype='fp16'), HW)
        self.assertEqual(cost['duration_us'], 1)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            estimate_cost(compute(flops=1, dtype='fp8'), HW)

    def test_zero_work_in_strict_mode(self):
        self.assertEqual(estimate_cost(compute(), HW, 'calibrated')['duration_us'], 0)

    def test_exact_calibration_and_miss(self):
        event = compute(shape=[2, 3, 4], dtype='float16', flops=10)
        calibration = {'entries': [{'name': 'matmul', 'shape': [2, 3, 4], 'dtype': 'float16',
                                    'duration_us': 4.25, 'provenance': 'fixture'}]}
        result = estimate_cost(event, HW, 'calibrated', calibration)
        self.assertEqual(result['duration_us'], 4.25)
        self.assertEqual(result['source'], 'calibrated')
        mapped = {calibration_key(event): calibration['entries'][0]}
        self.assertEqual(estimate_cost(event, HW, 'calibrated', mapped), result)
        with self.assertRaisesRegex(ValueError, 'missing exact calibration'):
            estimate_cost({**event, 'shape': [3, 3, 4]}, HW, 'calibrated', calibration)
        self.assertEqual(estimate_cost({**event, 'shape': [3, 3, 4]}, HW, 'hybrid', calibration)['source'],
                         'roofline_fallback')

    def test_full_signature_separates_strides_and_scalar_arguments(self):
        signature = {'inputs': [{'shape': [2, 3], 'stride': [3, 1], 'layout': 'strided', 'dtype': 'float32'}],
                     'scalars': {'alpha': 1}}
        event = compute(flops=10, shape=[2, 3], signature=signature)
        entry = {'name': 'matmul', 'signature': signature, 'duration_us': 8, 'provenance': 'GPU fixture'}
        self.assertEqual(estimate_cost(event, HW, 'calibrated', [entry])['duration_us'], 8)
        self.assertEqual(estimate_cost(event, HW, 'calibrated', {calibration_key(event): entry})['duration_us'], 8)
        alternatives = [
            {'inputs': [{'shape': [2, 3], 'stride': [1, 2], 'layout': 'strided', 'dtype': 'float32'}], 'scalars': {'alpha': 1}},
            {**signature, 'scalars': {'alpha': 2}},
            {**signature, 'scalars': {'alpha': True}},
        ]
        for alternative in alternatives:
            changed = {**event, 'signature': alternative}
            with self.subTest(alternative=alternative):
                self.assertNotEqual(calibration_key(event), calibration_key(changed))
                with self.assertRaisesRegex(ValueError, 'missing exact calibration'):
                    estimate_cost(changed, HW, 'calibrated', [entry])
        legacy = {'name': 'matmul', 'shape': [2, 3], 'dtype': 'float32',
                  'duration_us': 8, 'provenance': 'old shape-only fixture'}
        with self.assertRaisesRegex(ValueError, 'missing exact calibration'):
            estimate_cost(event, HW, 'calibrated', [legacy])

    def test_unsupported_signature_cannot_match_calibration(self):
        signature = {'status': 'unsupported', 'arguments': {'opaque_type': 'Example'}}
        event = compute(flops=1, signature=signature)
        calibration = [{'name': 'matmul', 'signature': signature, 'duration_us': 2, 'provenance': 'fixture'}]
        with self.assertRaisesRegex(ValueError, 'unsupported operator signature'):
            estimate_cost(event, HW, 'calibrated', calibration)

    def test_metadata_operation_has_no_device_launch_duration(self):
        event = compute(metadata_only=True, duration_us=9, duration_provenance='host wrapper timing')
        result = estimate_cost(event, HW, 'calibrated')
        self.assertEqual(result['duration_us'], 0)
        self.assertEqual(result['source'], 'metadata_only')

    def test_explicit_duration_requires_provenance(self):
        with self.assertRaisesRegex(ValueError, 'provenance'):
            estimate_cost(compute(duration_us=3), HW)
        result = estimate_cost(compute(duration_us=3, duration_provenance='measured fixture'), HW)
        self.assertEqual(result['duration_us'], 3)
        self.assertEqual(result['source'], 'explicit')

    def test_sparse_mixed_precision_rejected(self):
        for fields in ({'sparse': True}, {'sparsity': 0.5}, {'mixed_precision': True},
                       {'work_components': [{'dtype': 'float16', 'flops': 1}]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                estimate_cost(compute(flops=1, **fields), HW)

    def test_network_separate_from_hbm(self):
        event = {'event_id': 'comm', 'kind': 'collective', 'bytes_read': 1000}
        self.assertEqual(estimate_cost(event, HW)['duration_us'], 10)
        with self.assertRaisesRegex(ValueError, 'network_bandwidth'):
            estimate_cost(event, {k: v for k, v in HW.items() if not k.startswith('network')})

    def test_host_device_transfer_never_uses_hbm_bandwidth(self):
        event = compute(traffic_domain='host_device', bytes_read=1000, bytes_write=1000)
        with self.assertRaisesRegex(ValueError, 'host_device_bandwidth'):
            estimate_cost(event, HW)
        hardware = {**HW, 'host_device_bandwidth_bytes_s': 1e8}
        result = estimate_cost(event, hardware)
        self.assertEqual(result['duration_us'], 10)
        self.assertEqual(result['transfer_bytes'], 1000)
        self.assertEqual(result['bound'], 'host_device')
        self.assertIn('max(read,write)', result['traffic_basis'])
        exact_payload = estimate_cost({**event, 'host_device_transfer_bytes': 500}, hardware)
        self.assertEqual(exact_payload['duration_us'], 5)
        calibration = {calibration_key(event): {'duration_us': 3, 'provenance': 'measured transfer'}}
        self.assertEqual(estimate_cost(event, HW, 'calibrated', calibration)['duration_us'], 3)
        with self.assertRaisesRegex(ValueError, 'arithmetic'):
            estimate_cost({**event, 'flops': 1}, hardware)

    def test_gb200_profile_dense_units_capacity_and_network_scope(self):
        path = Path(__file__).resolve().parents[1] / 'profiles' / 'gb200.json'
        profile = json.loads(path.read_text())
        validate_hardware(profile)
        self.assertEqual(profile['peak_flops']['bfloat16'], 2.5e15)
        self.assertEqual(profile['peak_flops']['fp8'], 5e15)
        self.assertEqual(profile['sparse_tensor_core_peak_flops_reference_only']['bfloat16'], 5e15)
        self.assertEqual(profile['sparse_tensor_core_peak_flops_reference_only']['fp8'], 10e15)
        self.assertEqual(profile['memory_bandwidth_bytes_s'], 8e12)
        self.assertEqual(profile['memory_capacity_bytes'], 197897617408)
        self.assertNotIn('network_bandwidth_bytes_s', profile)
        dense = estimate_cost(compute(flops=2.5e9, dtype='bfloat16'), profile)
        self.assertEqual(dense['duration_us'], 1)
        with self.assertRaisesRegex(ValueError, 'sparse'):
            estimate_cost(compute(flops=2.5e9, dtype='bfloat16', sparse=True), profile)
        with self.assertRaisesRegex(ValueError, 'network_bandwidth'):
            estimate_cost({'event_id': 'network', 'kind': 'collective', 'bytes_read': 1}, profile)

    def test_invalid_numeric_fields(self):
        for fields in ({'flops': float('nan')}, {'flops': -1}, {'bytes_read': 1.5}, {'rank': True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                estimate_cost(compute(**fields), HW)
        with self.assertRaises(ValueError):
            validate_hardware({**HW, 'memory_bandwidth_bytes_s': 0})


if __name__ == '__main__':
    unittest.main()
