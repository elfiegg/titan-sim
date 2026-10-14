import unittest
from titan_sim.provenance import validate_calibration_context

class ProvenanceTests(unittest.TestCase):
    def test_wrong_software_or_hardware_rejected(self):
        metadata = {'torch_version': 'a', 'torchtitan_commit': 'b'}
        hardware = {'name': 'c'}
        with self.assertRaises(ValueError):
            validate_calibration_context(metadata, hardware, {'context': {'torch_version': 'older'}})
        validate_calibration_context(metadata, hardware, {'context': {**metadata, 'hardware': 'c'}})
