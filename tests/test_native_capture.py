"""Optional PyTorch tests; the framework-free suite still runs without PyTorch."""
import copy
import gc
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import weakref

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'native capture tests require the training PyTorch environment')
class NativeCaptureTests(unittest.TestCase):
    def test_alias_storage_and_observer_does_not_retain_tensors(self):
        import torch
        from titan_sim.torch_capture import TorchCapture
        from titan_sim.core import validate
        recorder = TorchCapture()
        value = torch.ones(8)
        reference = weakref.ref(value)
        with recorder:
            alias = value.view(2, 4)
            result = alias + 1
        self.assertEqual(len(recorder.storage), 2)
        self.assertEqual(recorder.nodes[0].attrs['arguments']['args'][0]['storage_id'],
                         recorder.nodes[0].attrs['outputs']['storage_id'])
        del value, alias, result
        gc.collect()
        self.assertIsNone(reference())
        recorder.sweep()
        self.assertEqual(recorder._live, {})
        workload = recorder.finish(hardware={'id':'test', 'capacity_bytes':1024})
        validate(*workload.arguments()[:3])
        json.dumps(workload.manifest, allow_nan=False)

    def test_resize_creates_an_allocation_generation(self):
        import torch
        from titan_sim.torch_capture import TorchCapture
        recorder = TorchCapture()
        value = torch.ones(8)
        old = recorder.register_tensor(value)
        value.untyped_storage().resize_(0)
        recorder.sweep()
        self.assertIn('release_boundary', recorder.storage_metadata[old])
        value.untyped_storage().resize_(32)
        new = recorder.register_tensor(value)
        self.assertNotEqual(old, new)

    def test_signature_reuse_depends_on_shape_and_scalars_not_allocations(self):
        import torch
        from titan_sim.torch_capture import TorchCapture
        recorder = TorchCapture()
        with recorder:
            a = torch.ones(8) + 1
            b = torch.ones(8) + 1
            c = torch.ones(8) + 2
        additions = [n for n in recorder.nodes if n.name == 'aten.add.Tensor']
        self.assertEqual(additions[0].attrs['cost_key'], additions[1].attrs['cost_key'])
        self.assertNotEqual(additions[0].attrs['cost_key'], additions[2].attrs['cost_key'])

    def trainer(self, *, fail=False):
        import torch

        class Engine:
            def __init__(self):
                self.device = torch.device('cpu')
                self.model_parts = [torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.SiLU())]
                self.optim = types.SimpleNamespace(optimizers=[torch.optim.AdamW(self.model_parts[0].parameters())])
                self.completed = 0
            def forward_backward(self):
                self.model_parts[0](torch.ones(2, 4)).sum().backward()
            def optim_step(self):
                self.optim.optimizers[0].step()
                self.optim.optimizers[0].zero_grad(set_to_none=True)
                self.completed += 1

        class Trainer:
            def __init__(self):
                self.engine = Engine()
            def train_step(self):
                self.engine.forward_backward()
                if fail:
                    raise RuntimeError('injected native failure')
                self.engine.optim_step()
            def should_continue_training(self):
                return self.engine.completed < 5
            def train(self):
                while self.should_continue_training():
                    self.train_step()
        return Trainer()

    def test_native_contract_preserves_numerics_and_restores_overrides(self):
        import torch
        from titan_sim.torchtitan import capture_trainer
        from titan_sim.core import validate
        trainer = self.trainer()
        reference = copy.deepcopy(trainer)
        original_step = trainer.train_step
        original_forward_backward = trainer.engine.forward_backward
        workload = capture_trainer(trainer, observer='dispatch', hardware={'id':'test','capacity_bytes':4096},
                                   steps=2, warmup_steps=1)
        for _ in range(3):
            reference.train_step()
        for actual, expected in zip(trainer.engine.model_parts[0].parameters(),
                                    reference.engine.model_parts[0].parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(workload.manifest['capture_status'], 'complete')
        self.assertEqual(workload.manifest['captured_steps'], 2)
        self.assertEqual(trainer.engine.completed, 3)
        self.assertEqual(trainer.train_step, original_step)
        self.assertEqual(trainer.engine.forward_backward, original_forward_backward)
        self.assertNotIn('train_step', vars(trainer))
        self.assertTrue(any(n.attrs.get('in_backward') for n in workload.nodes))
        self.assertTrue(any(n.attrs.get('phase') == 'optim_step' for n in workload.nodes))
        self.assertTrue(any(s.category == 'optimizer' for s in workload.storage))
        validate(*workload.arguments()[:3])

    def test_native_failure_produces_partial_trace_and_restores_methods(self):
        from titan_sim.torchtitan import capture_trainer
        trainer = self.trainer(fail=True)
        original = trainer.train_step
        workload = capture_trainer(trainer, observer='dispatch', hardware={'id':'test','capacity_bytes':4096})
        self.assertEqual(workload.manifest['capture_status'], 'blocked')
        self.assertIn('injected native failure', workload.manifest['error'])
        self.assertTrue(workload.nodes)
        self.assertEqual(trainer.train_step, original)

    def test_short_training_is_incomplete(self):
        from titan_sim.torchtitan import capture_trainer
        trainer = self.trainer()
        trainer.engine.completed = 5
        workload = capture_trainer(trainer, observer='dispatch', hardware={'id':'test','capacity_bytes':4096})
        self.assertEqual(workload.manifest['capture_status'], 'incomplete')

    def test_fake_tensors_and_fake_process_group(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.torch_capture import TorchCapture
        if torch.distributed.is_initialized():
            self.skipTest('test must own its fake process group')
        torch.distributed.init_process_group('fake', store=torch.distributed.HashStore(), rank=0, world_size=2)
        try:
            with FakeTensorMode():
                recorder = TorchCapture(world_size=2)
                with recorder:
                    tensor = torch.ones(4, 4)
                    torch.distributed.all_reduce(tensor)
                workload = recorder.finish(hardware={'id':'fake','capacity_bytes':4096})
            self.assertTrue(any(n.attrs.get('semantics') == 'unresolved_communication' for n in workload.nodes))
            self.assertTrue(workload.manifest['unresolved'])
        finally:
            torch.distributed.destroy_process_group()

    def test_native_profiler_handles_compiled_steps_and_preserves_artifacts(self):
        import torch
        from titan_sim.torchtitan import capture_trainer
        from titan_sim.core import validate
        trainer = self.trainer()
        reference = copy.deepcopy(trainer)
        # Exercise Dynamo fullgraph without depending on a platform C++ compiler.
        # This checks instrumentation compatibility, not Inductor kernel selection.
        trainer.engine.model_parts[0] = torch.compile(trainer.engine.model_parts[0], backend='eager', fullgraph=True)
        with tempfile.TemporaryDirectory() as root:
            workload = capture_trainer(trainer, hardware={'id':'cpu','capacity_bytes':4096},
                                       steps=2, warmup_steps=1, trace_dir=root)
            self.assertEqual(workload.manifest['capture_status'], 'complete', workload.manifest)
            self.assertEqual(len(workload.manifest['native_trace_artifacts']), 4)
            for artifact in workload.manifest['native_trace_artifacts']:
                self.assertTrue(Path(artifact['path']).is_file())
            self.assertTrue(any(n.attrs['phase'] == 'optim_step' for n in workload.nodes))
            validate(*workload.arguments()[:3])
            for _ in range(3):
                reference.train_step()
            for actual, expected in zip(trainer.engine.model_parts[0].parameters(),
                                        reference.engine.model_parts[0].parameters()):
                torch.testing.assert_close(actual, expected)

    def test_unknown_execution_trace_schema_is_rejected(self):
        from titan_sim.profiler_capture import project_trace
        with self.assertRaisesRegex(ValueError, 'unsupported PyTorch execution-trace schema'):
            project_trace({'schema':'future', 'nodes':[]}, rank=0, step=0,
                          device='cpu', offset=0, provenance_key='test')

    def test_fake_mode_refuses_real_process_groups_and_restores_api(self):
        import torch.distributed as dist
        from titan_sim.torchtitan import _fake_groups_only
        original = dist.init_process_group
        with _fake_groups_only():
            with self.assertRaisesRegex(ValueError, 'refuses process-group backend'):
                dist.init_process_group('nccl')
            with self.assertRaisesRegex(ValueError, 'refuses process-group backend'):
                dist.new_group(backend='gloo')
        self.assertIs(dist.init_process_group, original)


if __name__ == '__main__':
    unittest.main()
