"""Native CPU observer tests plus explicitly synthetic Chrome GPU fixtures."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'native observer tests require existing training PyTorch')
class ChromeOnlyCaptureTests(unittest.TestCase):
    def test_no_execution_trace_observer_registration_and_native_numerics(self):
        import torch
        from titan_sim.torchtitan import capture_trainer
        class Engine:
            def __init__(self):
                self.device = torch.device('cpu')
                self.model_parts = [torch.nn.Linear(4, 3)]
                self.optim = types.SimpleNamespace(optimizers=[torch.optim.SGD(self.model_parts[0].parameters(), lr=.01)])
                self.completed = 0
            def forward_backward(self):
                self.model_parts[0](torch.ones(2, 4)).sum().backward()
            def optim_step(self):
                self.optim.optimizers[0].step()
                self.optim.optimizers[0].zero_grad(set_to_none=True)
                self.completed += 1
        class Trainer:
            def __init__(self): self.engine = Engine()
            def train_step(self):
                self.engine.forward_backward()
                self.engine.optim_step()
            def should_continue_training(self): return self.engine.completed < 4
            def train(self):
                while self.should_continue_training(): self.train_step()
        trainer = Trainer()
        reference = copy.deepcopy(trainer)
        with tempfile.TemporaryDirectory() as root, patch.object(
                torch.profiler, 'ExecutionTraceObserver', side_effect=AssertionError('ET must never be constructed')):
            workload = capture_trainer(trainer, hardware={'id': 'cpu-test', 'capacity_bytes': 1024},
                steps=2, warmup_steps=1, trace_dir=root, execution_trace=False)
            for _ in range(3): reference.train_step()
            for actual, expected in zip(trainer.engine.model_parts[0].parameters(), reference.engine.model_parts[0].parameters()):
                torch.testing.assert_close(actual, expected)
            self.assertEqual(workload.manifest['captured_steps'], 2)
            self.assertFalse(workload.manifest['execution_trace'])
            self.assertEqual(workload.manifest['observer'], 'pytorch_profiler_chrome_only')
            self.assertEqual(workload.manifest['memory']['storage_evidence'], 'unverified')
            self.assertEqual(workload.storage, [])
            self.assertTrue(any('backing allocations' in gap for gap in workload.manifest['unresolved']))
            self.assertFalse(list(Path(root).rglob('*.et.json')))
            self.assertEqual(len(workload.manifest['native_trace_artifacts']), 2)
            for artifact in workload.manifest['native_trace_artifacts']:
                data = Path(artifact['path']).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), artifact['sha256'])
                self.assertTrue(json.loads(data)['traceEvents'])

    def test_synthetic_gpu_projection_has_unique_occurrence_costs_and_valid_bindings(self):
        from titan_sim.profiler_capture import project_chrome_trace
        from titan_sim.native_runner import measure
        from titan_sim.prepare import prepare
        from titan_sim.workload import Workload
        from titan_sim.core import simulate
        trace = {'traceEvents': [
            dict(ph='X', cat='kernel', name='synthetic same kernel', ts=i*10, dur=2+i,
                 pid=0, args={'stream': 7, 'grid': [1, 1, 1]}) for i in range(2)]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'step0.chrome.json'
            path.write_text(json.dumps(trace))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            nodes = project_chrome_trace(trace, rank=0, step=0, offset=0,
                artifact_sha256=sha, provenance_key='synthetic-source', hardware_id='synthetic-device')
            self.assertNotEqual(nodes[0].attrs['cost_key'], nodes[1].attrs['cost_key'])
            self.assertEqual(nodes[1].deps, [nodes[0].id])
            for index, node in enumerate(nodes):
                binding = node.attrs['native_measurement']
                self.assertEqual(binding['event_index'], index)
                self.assertEqual(binding['artifact_sha256'], sha)
                self.assertTrue(binding['context']['occurrence_only'])
            workload = Workload(nodes, [], dict(ranks=[0], world_size=1,
                observer='pytorch_profiler_chrome_only', execution_trace=False,
                native_trace_artifacts=[dict(path=str(path), sha256=sha, step=0)],
                memory={'storage_evidence': 'unverified'},
                unresolved=['Synthetic Chrome fixture has no physical allocation evidence.']),
                dict(id='synthetic-device', capacity_bytes=1024))
            prepared = prepare([workload])
            self.assertEqual(len(prepared.nodes), 2)
            self.assertEqual({n.attrs['cost_key'] for n in prepared.nodes}, {n.attrs['cost_key'] for n in nodes})
            catalog = measure(prepared, dict(mode='trace', hardware_id='synthetic-device', accept_profiler_overhead=True))
            self.assertEqual(sorted(r['seconds'] for r in catalog.values()), [2e-6, 3e-6])
            result = simulate(*prepared.arguments(), calibration=catalog)
            self.assertAlmostEqual(result['seconds'], 5e-6)
            self.assertEqual(result['memory_verdict'], 'inconclusive')

    def test_nonboolean_option_rejected_before_native_execution(self):
        from titan_sim.profiler_capture import ProfilerCapture
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'boolean'):
                ProfilerCapture(trace_dir=root, execution_trace='false')


if __name__ == '__main__':
    unittest.main()
