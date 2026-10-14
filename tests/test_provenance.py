import unittest
from titan_sim.provenance import validate_calibration_context

class ProvenanceTests(unittest.TestCase):
    def test_wrong_software_or_hardware_rejected(self):
        metadata = {'torch_version': 'a', 'torchtitan_commit': 'b'}
        hardware = {'name': 'c'}
        with self.assertRaises(ValueError):
            validate_calibration_context(metadata, hardware, {'context': {'torch_version': 'older'}})
        validate_calibration_context(metadata, hardware, {'context': {**metadata, 'hardware': 'c'}})

    def test_partial_preserves_bound_configuration(self):
        import functools
        from titan_sim.provenance import config_value
        first = config_value(functools.partial(pow, exp=2))
        second = config_value(functools.partial(pow, exp=3))
        self.assertNotEqual(first, second)
        self.assertEqual(first['keywords'], {'exp': 2})
