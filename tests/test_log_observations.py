import json
from pathlib import Path
import tempfile
import unittest
from titan_sim.log_observations import parse_log, summarize, assess_cross_validation, MATCH_FIELDS


class LogObservationTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.log'
            path.write_text(text)
            return parse_log(path)

    def test_ansi_unknown_ranks_and_repeated_messages_remain_observations(self):
        doc = self.parse('\x1b[31mstep: 1 loss: 2 grad_norm: nan memory: 10.50GiB(20%)\x1b[0m\n' * 2)
        self.assertEqual(len(doc['samples']), 2)
        self.assertIsNone(doc['samples'][0]['rank'])
        self.assertEqual(doc['samples'][0]['bytes'], int(10.5 * 2**30))
        self.assertEqual(doc['samples'][0]['grad_norm'], 'nan')
        self.assertEqual(summarize(doc)[0]['unidentified_rank_samples'], 2)
        self.assertFalse(doc['samples'][0]['metric_semantics_verified'])
        json.dumps(doc, allow_nan=False)

    def test_explicit_rank_and_peak_units_are_not_guessed(self):
        doc = self.parse('[Rank 64] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 179.27 | mem-max-allocated-gigabytes: 227.13 | mem-reserved-gigabytes: 236.64\n')
        self.assertEqual(len(doc['samples']), 3)
        self.assertEqual(doc['samples'][0]['rank'], 64)
        self.assertTrue(all(s['bytes'] is None for s in doc['samples']))
        self.assertEqual(doc['samples'][1]['window'], 'peak_reset_unverified')
        self.assertEqual(doc['samples'][2]['value'], 236.64)

    def test_legacy_mb_allocator_rows_keep_rank_units_and_support_ratios(self):
        from titan_sim.empirical_memory import allocator_ratios
        doc = self.parse('[Rank 0] (after 2 iterations) memory (MB) | allocated: 100 | max allocated: 150 | reserved: 180 | max reserved: 180'
                         '[Rank 15] (after 2 iterations) memory (MB) | allocated: 200 | max allocated: 250 | reserved: 300 | max reserved: 300')
        self.assertEqual(len(doc['samples']), 8)
        self.assertTrue(all(s['unit'] == 'MB' and s['bytes'] is None for s in doc['samples']))
        self.assertEqual(doc['samples'][1]['window'], 'peak_reset_unverified')
        rows = allocator_ratios(doc)
        self.assertEqual([(r['rank'], r['step']) for r in rows], [(0, 2), (15, 2)])
        self.assertEqual(rows[0]['high_water_reserved_over_current'], 1.8)
        self.assertEqual(rows[1]['high_water_reserved_over_current'], 1.5)

    def test_conflicting_facts_preserved_and_commands_not_samples(self):
        doc = self.parse('CUDA capacity: NVIDIA GB200 with 184.00GiB memory\nCUDA capacity: NVIDIA GB300 with 276.62GiB memory\n+ echo "step: 1 memory: 999GiB"\n[rank0]: step: 10 memory: 102.21GiB\n')
        self.assertEqual(len(doc['facts']['hardware']), 2)
        self.assertEqual(len(doc['samples']), 1)
        self.assertEqual(doc['samples'][0]['rank'], 0)
        self.assertEqual(doc['samples'][0]['line'], 4)

    def test_configuration_gates_missing_and_mismatched_identity(self):
        complete = {key: 'same' for key in MATCH_FIELDS}
        self.assertTrue(assess_cross_validation(complete, complete)['eligible'])
        observed = {**complete, 'pp': 4, 'memory_metric': 'reserved'}
        observed.pop('torch_version')
        result = assess_cross_validation(complete, observed)
        self.assertFalse(result['eligible'])
        self.assertIn('pp', result['mismatches'])
        self.assertIn('memory_metric', result['mismatches'])
        self.assertIn('torch_version', result['missing'])

    def test_mesh_and_training_override_facts_are_extracted(self):
        doc = self.parse('Building device mesh with parallelism: pp=1, dp_shard=256, tp=1, ep=64\nTrainer is initialized with local batch size 8, global batch size 2048, gradient accumulation steps 1, sequence length 4096, total steps 50\n')
        self.assertEqual(doc['facts']['mesh'][0]['value']['ep'], 64)
        self.assertEqual(doc['facts']['training'][0]['value']['local_batch'], 8)

    def test_mesh_and_empty_evidence_do_not_pass_gate(self):
        context = {key: 'same' for key in MATCH_FIELDS}
        for key in ('cp', 'dp_shard', 'dp_replicate', 'expert_tp'):
            self.assertFalse(assess_cross_validation(context, {**context, key: 'different'})['eligible'])
        for value in ('', 'unknown', 'training_memory_unspecified', None):
            self.assertFalse(assess_cross_validation(context, {**context, 'memory_metric': value})['eligible'])

    def test_signed_nonfinite_and_overflow_metrics_are_json_safe(self):
        doc = self.parse('[Rank 2] step: 1 loss: 1e309 grad_norm: -nan memory: 1GiB\n')
        self.assertEqual(doc['samples'][0]['rank'], 2)
        self.assertEqual(doc['samples'][0]['loss'], '1e309')
        self.assertEqual(doc['samples'][0]['grad_norm'], '-nan')
        self.assertTrue(doc['indicators']['nan_gradient'])
        json.dumps(doc, allow_nan=False)
        with self.assertRaises(ValueError):
            self.parse('step: 1 memory: 1e309GiB')

    def test_concatenated_allocator_reports_keep_their_own_rank(self):
        doc = self.parse('[Rank 0] (after 1 iterations) memory (GB) | mem-allocated-gigabytes: 10 [Rank 1] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 20')
        self.assertEqual([(s['rank'], s['step'], s['value']) for s in doc['samples']], [(0, 1, 10), (1, 2, 20)])

    def test_stash_payload_uses_storage_dtype_and_separates_host(self):
        doc = self.parse('  0: allocate_stash_buffers cuda: torch.Size([10, 32]) host: torch.Size([5, 32]) dtype=torch.uint8 (torch.float8_e8m0fnu)\n')
        gpu, host = doc['observed_components']
        self.assertEqual((gpu['device'], gpu['payload_bytes']), ('cuda', 320))
        self.assertEqual((host['device'], host['payload_bytes']), ('cpu', 160))
        self.assertEqual(gpu['logical_dtype'], 'float8_e8m0fnu')
        self.assertEqual(gpu['launcher_label'], 0)
        self.assertIsNone(gpu['rank'])
        self.assertIsNone(gpu['storage_identity'])
        self.assertEqual(doc['samples'], [])

    def test_missing_host_and_unknown_storage_dtype_remain_unknown(self):
        doc = self.parse('[rank64] allocate_stash_buffers cuda: torch.Size([10, 32]) dtype=torch.new_dtype (torch.float8_e4m3fn)\n')
        self.assertEqual(len(doc['observed_components']), 1)
        component = doc['observed_components'][0]
        self.assertEqual(component['rank'], 64)
        self.assertIsNone(component['payload_bytes'])
        self.assertEqual(component['source_kind'], 'observed_buffer_geometry')

    def test_concatenated_rank_records_cannot_steal_rank_identity(self):
        doc = self.parse('[rank0] ready [rank1] step: 2 memory: 10GiB [rank2] step: 3 memory: 11GiB [rank3] allocate_stash_buffers cuda: torch.Size([2]) dtype=torch.uint8 (torch.uint8)')
        self.assertEqual([(s['rank'], s['step']) for s in doc['samples']], [(1, 2), (2, 3)])
        self.assertEqual(doc['observed_components'][0]['rank'], 3)
        self.assertTrue(all(s['line'] == 1 for s in doc['samples']))

    def test_negative_memory_is_rejected_for_every_counter_family(self):
        for text in ('step: 1 memory: -1GiB', 'CUDA memory usage for model: -1GiB',
                     '[Rank 0] (after 1 iterations) memory (MB) | allocated: -1',
                     '[Rank 0] (after 1 iterations) memory (GB) | mem-allocated-gigabytes: -1'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_counter_scope_and_units_are_required_comparison_evidence(self):
        context = {key: 'same' for key in MATCH_FIELDS}
        for key in ('rank_scope', 'memory_unit_basis', 'counter_window'):
            self.assertFalse(assess_cross_validation(context, {**context, key: None})['eligible'])
            self.assertFalse(assess_cross_validation(context, {**context, key: 'different'})['eligible'])

    def test_quoted_rank_markers_in_commands_are_not_observations(self):
        doc = self.parse('+ echo "[rank1] step: 99 memory: 999GiB"\n# [rank2] step: 3 memory: 2GiB')
        self.assertEqual(doc['samples'], [])
