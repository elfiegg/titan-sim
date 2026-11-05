import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
from titan_sim.core import Node, Storage, simulate
from titan_sim.kimi import hardware
from titan_sim.cli import checks
from titan_sim.workload import read_bundle, write_bundle


class ToolchainTests(unittest.TestCase):
    def test_scheduler_and_storage_contracts(self):
        result = checks()
        self.assertTrue(result['collective_waits_for_all_ranks'])
        self.assertTrue(result['allocation_last_use_and_aliasing'])
        self.assertEqual(len(result['rejected_cases']), 5)

    def test_direct_workload_round_trip_and_schema_rejection(self):
        nodes = [Node(1, 'producer', 0, attrs={'legacy_seconds': 2}),
                 Node(2, 'consumer', 0, deps=[1], attrs={'legacy_seconds': 3})]
        storages = [Storage('shared_backing', 0, 128, 1, [2])]
        manifest = {'ranks': [0], 'unresolved': ['synthetic timing scenario']}
        with tempfile.TemporaryDirectory() as temp:
            write_bundle(temp, nodes, storages, manifest, hardware())
            result = simulate(*read_bundle(temp))
            self.assertEqual(result['seconds'], 5)
            self.assertEqual(result['modeled_peak_bytes'][0], 128)
            self.assertEqual(result['memory_verdict'], 'inconclusive')
            with self.assertRaisesRegex(ValueError, 'incomplete workload'):
                simulate(*read_bundle(temp), strict=True)
            path = Path(temp)/'workload.json'
            data = json.loads(path.read_text())
            data['version'] = 999
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'unsupported workload'):
                read_bundle(temp)

    def test_calibration_requires_hardware_and_sample_evidence(self):
        nodes = [Node(1, 'measured_component', 0, attrs={'cost_key': 'case'})]
        manifest = {'ranks': [0], 'unresolved': []}
        h = hardware()
        samples = [0.01, 0.02, 0.03]
        record = dict(hardware_id=h['id'], seconds=0.02, samples_seconds=samples,
                      evidence_sha256=hashlib.sha256(json.dumps(samples, separators=(',', ':')).encode()).hexdigest())
        result = simulate(nodes, [], manifest, h, {'case': record}, strict=True)
        self.assertEqual(result['seconds'], 0.02)
        with self.assertRaisesRegex(ValueError, 'hardware/evidence mismatch'):
            simulate(nodes, [], manifest, h, {'case': dict(record, hardware_id='another-device')}, strict=True)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            simulate(nodes, [], manifest, h, {'case': dict(record, samples_seconds=[1])}, strict=True)

    def test_cpu_entry_point_without_site_packages(self):
        # -S excludes third-party site packages even in a Torch-enabled venv.
        with tempfile.TemporaryDirectory() as temp:
            p = subprocess.run([sys.executable, '-S', '-m', 'titan_sim', '--case',
                                'checks', '--output-dir', temp], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(json.loads((Path(temp)/'summary.json').read_text())['checks'])


if __name__ == '__main__':
    unittest.main()
