import unittest
import xml.etree.ElementTree as ET
from titan_sim.reporting import memory_svg, comparison_summary


class ReportingTests(unittest.TestCase):
    def test_plot_escapes_title_and_uses_selected_coordinate(self):
        plot = memory_svg([{'event_index': 3, 'allocated_bytes': 1024**3}], coordinate='event_index', title='A < B')
        ET.fromstring(plot)
        self.assertIn('A &lt; B', plot)
        self.assertIn('event_index', plot)

    def test_incomplete_or_missing_metadata_cannot_claim_match(self):
        result = comparison_summary({'metadata': {'status': 'blocked'}}, {'metadata': {}, 'status': 'complete'})
        self.assertFalse(result['eligible_for_phase_comparison'])
        self.assertIn('world_size', result['config_mismatches'])

    def test_identical_dimensions_do_not_certify_execution_policy(self):
        metadata = {key: 'same' for key in ('torchtitan_commit', 'torch_version', 'world_size', 'flavor', 'sequence_length', 'tokens_per_microbatch_per_dp_rank', 'rank')}
        metadata['status'] = 'complete'
        result = comparison_summary({'metadata': metadata}, {'metadata': metadata, 'status': 'complete'})
        self.assertFalse(result['eligible_for_phase_comparison'])
        self.assertIn('policy_fingerprint', result['config_mismatches'])
