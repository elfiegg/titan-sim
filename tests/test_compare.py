import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from titan_sim.compare import compare_documents, export_comparison, main


def documents():
    metadata = {'status': 'complete', 'torchtitan_commit': 'abc', 'torch_version': '2.14',
                'world_size': 1, 'flavor': 'test', 'sequence_length': 128,
                'tokens_per_microbatch_per_dp_rank': 128, 'rank': 0,
                'policy_fingerprint': 'same-verified-policy'}
    projection = {'source': 'cpu_projection', 'metadata': metadata,
                  'logical_memory': [
                      {'event_id': 'e1', 'event_index': 10, 'rank': 0, 'phase': 'forward', 'step': 1, 'allocated_bytes': 100},
                      {'event_id': 'e2', 'event_index': 20, 'rank': 0, 'phase': 'forward', 'step': 1, 'allocated_bytes': 160},
                      {'event_id': 'e3', 'event_index': 30, 'rank': 0, 'phase': 'optimizer', 'step': 1, 'allocated_bytes': 80}]}
    observed = {'source': 'gpu_observation', 'status': 'complete', 'metadata': copy.deepcopy(metadata),
                'samples': [
                    {'rank': 0, 'phase': 'forward', 'step': 1, 'module': '', 'time_us': 13, 'allocated_bytes': 200, 'peak_allocated_bytes': 9999},
                    {'rank': 0, 'phase': 'forward', 'step': 1, 'module': '', 'time_us': 19, 'allocated_bytes': 240, 'peak_allocated_bytes': 9999},
                    {'rank': 0, 'phase': 'optimizer', 'step': 1, 'module': '', 'time_us': 100, 'allocated_bytes': 90, 'peak_allocated_bytes': 9999}]}
    return projection, observed


def mapping():
    return [{'label': 'declared boundary', 'mode': 'boundary',
             'prediction': {'event_id': 'e2'}, 'observation': {'sample_index': 1}}]


class ComparisonTests(unittest.TestCase):
    def test_named_phase_sampled_peaks_keep_different_metric_definitions(self):
        projection, observed = documents()
        report = compare_documents(projection, observed)
        row = report['comparisons'][0]
        self.assertEqual(row['predicted_bytes'], 160)
        self.assertEqual(row['observed_bytes'], 240)
        self.assertEqual(row['logical_minus_allocator_bytes'], -80)
        self.assertEqual(row['observation_sample_indices'], [0, 1])
        self.assertFalse(row['metric_definitions_match'])
        self.assertIsNone(row['peak_error_bytes'])
        self.assertNotEqual(row['predicted_metric'], row['observed_metric'])
        self.assertNotEqual(row['observed_bytes'], 9999)

    def test_explicit_mapping_is_exact_no_time_interpolation(self):
        projection, observed = documents()
        projection['logical_memory'][1]['phase'] = 'forward_backward_step_0'
        observed['samples'][1]['phase'] = 'after_forward_backward'
        observed['samples'][1]['time_us'] = 1e12
        report = compare_documents(projection, observed, mapping())
        row = report['comparisons'][0]
        self.assertEqual(row['predicted_bytes'], 160)
        self.assertEqual(row['observed_bytes'], 240)
        self.assertEqual(row['prediction_event_indices'], [20])
        self.assertEqual(row['observation_sample_indices'], [1])

    def test_mismatched_fingerprint_rejected_even_with_explicit_mapping(self):
        projection, observed = documents()
        observed['metadata']['policy_fingerprint'] = 'different'
        with self.assertRaisesRegex(ValueError, 'comparison_summary'):
            compare_documents(projection, observed, mapping())

    def test_post_initialization_scope_cannot_claim_full_trajectory_coverage(self):
        for scope_metadata in (
            {'capture_scope': 'post_initialization'},
            {'initialization_coverage': {'initialization_time_included': False,
                                         'initialization_peak_included': False}},
        ):
            with self.subTest(metadata=scope_metadata):
                projection, observed = documents()
                projection['metadata'].update(scope_metadata)
                with self.assertRaisesRegex(ValueError, 'coverage_limitations'):
                    compare_documents(projection, observed, mapping())

    def test_every_configuration_gate_field_is_enforced(self):
        for key in ('torchtitan_commit', 'torch_version', 'world_size', 'flavor',
                    'sequence_length', 'tokens_per_microbatch_per_dp_rank', 'rank', 'policy_fingerprint'):
            with self.subTest(key=key):
                projection, observed = documents()
                del observed['metadata'][key]
                with self.assertRaises(ValueError):
                    compare_documents(projection, observed)

    def test_incomplete_blocked_and_partial_rejected(self):
        for side, field, value in [('prediction', 'metadata', {'status': 'blocked'}),
                                   ('observation', 'status', 'running'),
                                   ('prediction', 'partial_trace', True),
                                   ('prediction', 'status', 'incomplete')]:
            with self.subTest(side=side, field=field):
                projection, observed = documents()
                target = projection if side == 'prediction' else observed
                if field == 'metadata':
                    target['metadata'].update(value)
                else:
                    target[field] = value
                with self.assertRaises(ValueError):
                    compare_documents(projection, observed, mapping())

    def test_matching_summary_gate_is_always_called(self):
        projection, observed = documents()
        denied = {'eligible_for_phase_comparison': False, 'both_complete': True, 'config_mismatches': {}}
        with patch('titan_sim.compare.comparison_summary', return_value=denied) as gate:
            with self.assertRaises(ValueError):
                compare_documents(projection, observed, mapping())
            gate.assert_called_once_with(projection, observed)

    def test_no_names_match_requires_explicit_mapping(self):
        projection, observed = documents()
        for row in observed['samples']:
            row['phase'] = 'unrelated'
        with self.assertRaisesRegex(ValueError, 'no exact named-phase'):
            compare_documents(projection, observed)

    def test_step_and_module_identity_not_invented(self):
        for changed in ('step', 'module'):
            projection, observed = documents()
            for row in observed['samples']:
                row[changed] = 2 if changed == 'step' else 'layer0'
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'no exact named-phase'):
                compare_documents(projection, observed)

    def test_ambiguous_boundary_and_unmatched_selector_rejected(self):
        projection, observed = documents()
        for selector in ({'phase': 'forward'}, {'event_id': 'missing'}):
            correspondence = mapping()
            correspondence[0]['prediction'] = selector
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                compare_documents(projection, observed, correspondence)

    def test_explicit_selected_range_maxima(self):
        projection, observed = documents()
        correspondence = [{'label': 'specified intervals', 'mode': 'sampled_peak',
                           'prediction': {'event_index_start': 10, 'event_index_end': 20},
                           'observation': {'sample_index_start': 0, 'sample_index_end': 1}}]
        report = compare_documents(projection, observed, correspondence)
        self.assertEqual(report['comparisons'][0]['prediction_event_indices'], [10, 20])
        self.assertEqual(report['comparisons'][0]['observed_bytes'], 240)

    def test_predictions_and_observations_never_mutated(self):
        projection, observed = documents()
        before = copy.deepcopy((projection, observed))
        compare_documents(projection, observed)
        self.assertEqual((projection, observed), before)

    def test_zero_observation_has_no_relative_error(self):
        projection, observed = documents()
        observed['samples'][1]['allocated_bytes'] = 0
        row = compare_documents(projection, observed, mapping())['comparisons'][0]
        self.assertIsNone(row['difference_over_observed'])
        self.assertEqual(row['logical_minus_allocator_bytes'], 160)

    def test_nested_simulation_supported_but_blocked_nested_metadata_rejected(self):
        projection, observed = documents()
        wrapped = {'metadata': projection['metadata'], 'simulation': projection}
        self.assertEqual(len(compare_documents(wrapped, observed)['comparisons']), 2)
        wrapped['simulation']['metadata'] = {'status': 'blocked'}
        with self.assertRaises(ValueError):
            compare_documents(wrapped, observed)

    def test_outer_metadata_cannot_hide_nested_policy_mismatch(self):
        projection, observed = documents()
        wrapped = {'metadata': copy.deepcopy(projection['metadata']), 'simulation': projection}
        wrapped['simulation']['metadata']['policy_fingerprint'] = 'different'
        with self.assertRaisesRegex(ValueError, 'nested simulation'):
            compare_documents(wrapped, observed)

    def test_cli_exports_json_and_csv(self):
        projection, observed = documents()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'p.json').write_text(json.dumps(projection))
            (folder / 'o.json').write_text(json.dumps(observed))
            main([str(folder / 'p.json'), str(folder / 'o.json'), '--output', str(folder / 'out')])
            report = json.loads((folder / 'out' / 'comparison.json').read_text())
            with (folder / 'out' / 'comparison.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(rows[0]['predicted_metric'], 'logical_gpu_tensor_storage_bytes')

    def test_rejected_export_writes_no_output(self):
        projection, observed = documents()
        observed['status'] = 'failed'
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'out'
            with self.assertRaises(ValueError):
                export_comparison(projection, observed, target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
