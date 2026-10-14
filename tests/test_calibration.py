import unittest
from titan_sim.calibration import timing_entry, heldout_error
from titan_sim.costs import calibration_key, estimate_cost


class CalibrationTests(unittest.TestCase):
    def test_execution_dtype_is_part_of_even_an_explicit_signature(self):
        event = dict(event_id='x', kind='compute', name='mm', dtype='float32', signature={'shape': [8]})
        other = {**event, 'dtype': 'float16'}
        self.assertNotEqual(calibration_key(event), calibration_key(other))

    def test_measurement_summary_and_independent_holdout(self):
        event = dict(event_id='x', kind='compute', name='mm', dtype='float32', flops=10, shape=[2, 3])
        key, entry = timing_entry(event, [1, 100, 3], provenance={'source': 'cuda_event_measurement'})
        self.assertEqual(key, calibration_key(event))
        self.assertEqual(entry['duration_us'], 3)
        result = heldout_error(3, [4, 5, 6])
        self.assertEqual(result['relative_error'], .4)

    def test_invalid_measurements_fail(self):
        event = dict(event_id='x', kind='compute', name='mm')
        for samples in ([1, 2], [0, 1, 2], [1, 2, float('nan')]):
            with self.assertRaises(ValueError):
                timing_entry(event, samples, provenance={'source': 'cuda_event_measurement'})
        with self.assertRaises(ValueError):
            timing_entry(event, [1, 2, 3], provenance={'source': 'roofline'})

    def test_collective_calibration_requires_participant_and_payload_match(self):
        event = dict(event_id='x', kind='collective', name='all_reduce', participants=[0, 1],
                     collective='all_reduce', message_bytes=64, algorithm='ring')
        other = {**event, 'participants': [0, 1, 2, 3]}
        key, entry = timing_entry(event, [2, 3, 4], provenance={'source': 'cuda_event_measurement'})
        hw = dict(memory_capacity_bytes=1000, memory_bandwidth_bytes_s=1e9, peak_flops={'float32': 1e12})
        self.assertEqual(estimate_cost(event, hw, 'calibrated', [entry])['duration_us'], 3)
        self.assertNotEqual(key, calibration_key(other))
        with self.assertRaises(ValueError):
            estimate_cost(other, hw, 'calibrated', [entry])
