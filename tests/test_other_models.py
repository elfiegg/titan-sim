import json
from pathlib import Path
import unittest

from scripts.validate_other_models import config_dump, frozen_ratio_transfer

ROOT = Path(__file__).resolve().parents[1]


class OtherModelTests(unittest.TestCase):
    def test_config_sections_do_not_confuse_model_and_other_tp(self):
        text = '[model] Non-default values:\n  tensor_model_parallel_size: 2  (Mcore default: 1)\n[checkpoint]:\n  tensor_model_parallel_size: 1\n'
        self.assertEqual(config_dump(text, section='model')['tensor_model_parallel_size']['value'], '2')
        with self.assertRaises(ValueError):
            config_dump(text)

    def test_dotted_config_and_conflicts(self):
        self.assertEqual(config_dump('  num_layers ........ 15', dotted=True)['num_layers']['value'], '15')
        self.assertEqual(config_dump('0:   num_layers ........ 5', dotted=True)['num_layers']['value'], '5')
        with self.assertRaises(ValueError):
            config_dump('  num_layers .... 15\n  num_layers .... 61', dotted=True)

    def test_transfer_is_conditional_and_rejects_misalignment(self):
        a = {'rank': 0, 'step': 1, 'printed_unit': 'MB', 'line': 1,
             'printed_values': {'allocated': 100, 'max-allocated': 150, 'max-reserved': 200},
             'high_water_reserved_over_current': 2}
        b = dict(a, line=2, printed_values={'allocated': 200, 'max-allocated': 300, 'max-reserved': 400})
        out = frozen_ratio_transfer([a], [b])[0]
        self.assertEqual(out['predicted_reserved'], 400)
        self.assertIn('measured', out['conditioning'])
        for bad in (dict(b, rank=1), dict(b, step=2), dict(b, printed_unit='GB')):
            with self.assertRaises(ValueError): frozen_ratio_transfer([a], [bad])
        with self.assertRaises(ValueError): frozen_ratio_transfer([a, a], [b])
        with self.assertRaises(ValueError): frozen_ratio_transfer([a], [b, b])

    def test_recorded_other_model_results_preserve_scope(self):
        data = json.loads((ROOT / 'docs/validation/other_models/results.json').read_text())
        self.assertFalse(data['independent_full_model_validation'])
        for precision in ('mxfp8', 'nvfp4'):
            rows = data['v3_16b'][precision]['rows']
            fit = [r for r in rows if r['role'] == 'fit']
            evaluation = [r for r in rows if r['role'] == 'evaluation']
            self.assertEqual([r['local_batch'] for r in fit], [1, 8, 64])
            self.assertEqual([r['local_batch'] for r in evaluation], [2, 32])
            self.assertFalse({r['evidence']['sha256'] for r in fit} & {r['evidence']['sha256'] for r in evaluation})
            self.assertLess(max(abs(r['error_percent']) for r in evaluation), 2.21)
        v4 = data['v4_long_context']
        self.assertEqual(len(v4['ratio_transfer']), 32)
        self.assertLess(v4['max_absolute_error_percent'], .56)
        self.assertFalse(v4['full_prediction'])
        self.assertEqual(set(v4['reviewed_config_differences']), {'global_batch_size', 'varlen_mock_dataset_config_json'})


if __name__ == '__main__':
    unittest.main()
