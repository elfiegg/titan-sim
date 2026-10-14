import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from titan_sim.roofline import roofline_svg

HW = {'name': 'Fixture <GPU>', 'peak_flops': {'float32': 1e12, 'bfloat16': 2e12},
      'memory_bandwidth_bytes_s': 1e9, 'memory_capacity_bytes': 1000}
NS = {'svg': 'http://www.w3.org/2000/svg'}


def row(**kwargs):
    return {'event_id': 'matmul<&', 'name': '<matmul>', 'kind': 'compute', 'dtype': 'float32',
            'flops': 2e6, 'bytes_read': 1000, 'bytes_write': 1000,
            'duration_us': 2, 'source': 'roofline', **kwargs}


def elements(svg, class_name):
    tree = ET.fromstring(svg)
    return [node for node in tree.iter() if node.get('class') == class_name]


class RooflineTests(unittest.TestCase):
    def test_analytical_event_is_ceiling_never_achieved_marker(self):
        svg = roofline_svg({'timeline': [row()]}, HW)
        self.assertEqual(len(elements(svg, 'analytical-ceiling')), 1)
        self.assertEqual(len(elements(svg, 'calibrated-measurement')), 0)
        self.assertIn('intensity=1000 FLOP/logical byte', svg)
        self.assertIn('analytical ceiling=1 TFLOP/s (not achieved)', svg)
        self.assertIn('logical bytes, not observed DRAM traffic', svg)

    def test_calibrated_throughput_distinct_from_ceiling(self):
        svg = roofline_svg({'timeline': [row(source='calibrated', duration_us=8, provenance='GPU fixture')]}, HW)
        self.assertEqual(len(elements(svg, 'analytical-ceiling')), 1)
        self.assertEqual(len(elements(svg, 'calibrated-measurement')), 1)
        self.assertIn('calibrated throughput=0.25 TFLOP/s', svg)
        self.assertIn('provenance=GPU fixture', svg)
        self.assertEqual(elements(svg, 'calibrated-measurement')[0].get('data-event-id'), 'matmul<&')

    def test_explicit_synthetic_timing_not_called_measured(self):
        svg = roofline_svg({'timeline': [row(source='explicit')]}, HW)
        self.assertEqual(len(elements(svg, 'calibrated-measurement')), 0)

    def test_zero_work_unknown_flops_and_cpu_exclusions(self):
        result = {'timeline': [row(flops=0), row(bytes_read=0, bytes_write=0),
                               row(source='host_unmodeled'), row(kind='collective')],
                  'unknown_flop_ops': {'aten.example': 2}}
        svg = roofline_svg(result, HW)
        self.assertIn('GPU compute events: 2', svg)
        self.assertIn('zero/unreported FLOPs: 1', svg)
        self.assertIn('zero traffic: 1', svg)
        self.assertIn('2 events across 1 operators without a FLOP formula', svg)
        self.assertEqual(len(elements(svg, 'analytical-ceiling')), 0)

    def test_empty_plot_remains_valid_and_honest(self):
        svg = roofline_svg({'timeline': []}, HW)
        ET.fromstring(svg)
        self.assertIn('Operator FLOP coverage unavailable', svg)
        self.assertIn('GPU compute events: 0', svg)
        self.assertNotIn('nan', svg.lower())

    def test_gb200_execution_dtype(self):
        hardware = json.loads((Path(__file__).resolve().parents[1] / 'profiles' / 'gb200.json').read_text())
        svg = roofline_svg({'timeline': [row(dtype='bf16', flops=1e9)]}, hardware)
        self.assertIn('bfloat16: dense theoretical peak 2500 TFLOP/s', svg)
        self.assertEqual(len(elements(svg, 'roofline-ceiling')), 1)

    def test_invalid_fields_rejected(self):
        for bad in ({'flops': -1}, {'bytes_write': float('nan')}, {'duration_us': -2, 'source': 'calibrated'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                roofline_svg({'timeline': [row(**bad)]}, HW)


if __name__ == '__main__':
    unittest.main()
