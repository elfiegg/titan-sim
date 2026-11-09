import unittest
from titan_sim.hardware import analytical_target, validate_hardware


class HardwareTests(unittest.TestCase):
    def test_no_rack_capacity_or_peak_rate_inference(self):
        spec={'source_url':'https://example.invalid/spec', 'rack_memory_bytes':20_700_000_000_000}
        with self.assertRaises(ValueError):
            analytical_target(spec, {'id':'rubin'})
        result=analytical_target(spec, {'id':'scenario', 'devices':{'gpu0':{'capacity_bytes':100}},
                                      'assumptions':['Explicit user-supplied device capacity']})
        self.assertNotIn('effective_flops_per_second',result)
        self.assertEqual(result['evidence_level'],'analytical_assumptions')

    def test_invalid_rates_and_capacity(self):
        for value in [True, float('nan'), float('inf'), -1, 0]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_hardware({'id':'test','capacity_bytes':100,'effective_flops_per_second':value})
        with self.assertRaises(ValueError):
            validate_hardware({'id':'test','capacity_bytes':100,'reserve_fraction':1})

    def test_capacity_requires_integer_bytes_for_uniform_and_device_specs(self):
        for value in (True, 1.5, 100.0, float('nan'), float('inf'), -1):
            for capacity in ({'capacity_bytes': value},
                             {'devices': {'gpu': {'capacity_bytes': value}}}):
                with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                    validate_hardware(dict(id='test', **capacity))

    def test_heterogeneous_capacities(self):
        hardware={'id':'cluster', 'devices':{'a':{'capacity_bytes':100},'b':{'capacity_bytes':200}}}
        self.assertEqual(validate_hardware(hardware),hardware)
