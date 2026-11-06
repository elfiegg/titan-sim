"""Opt-in source integration: original feed-forward implementations, not full training."""
import importlib
import os
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('TITAN_SIM_TEST_TORCHTITAN') == '1',
                     'set TITAN_SIM_TEST_TORCHTITAN=1 with the pinned TorchTitan environment')
class TorchTitanBlockTests(unittest.TestCase):
    def test_native_llama_and_qwen_feed_forward_backward(self):
        import torch
        from torchtitan.config.parallelism import ParallelismConfig
        from titan_sim.profiler_capture import ProfilerCapture
        from titan_sim.core import validate
        for name in ('llama3', 'qwen3'):
            with self.subTest(model=name):
                model_module = importlib.import_module('torchtitan.models.'+name)
                config = model_module.build_model_config('debugmodel', parallelism=ParallelismConfig(),
                                                         seq_len=8, attn_backend='flex')
                block = config.layers[0].feed_forward.build()
                block.init_states()
                inputs = torch.randn(8, config.dim)
                with tempfile.TemporaryDirectory() as root:
                    observer = ProfilerCapture(trace_dir=root)
                    observer.register_model(block)
                    with observer:
                        actual = block(inputs)
                        actual.sum().backward()
                    gradients = [p.grad.clone() for p in block.parameters()]
                    block.zero_grad(set_to_none=True)
                    expected = block(inputs)
                    expected.sum().backward()
                    torch.testing.assert_close(actual, expected)
                    for parameter, gradient in zip(block.parameters(), gradients):
                        torch.testing.assert_close(parameter.grad, gradient)
                    workload = observer.finish(hardware={'id':'cpu-test','capacity_bytes':0})
                    validate(*workload.arguments()[:3])
                    self.assertTrue(workload.nodes)
                    self.assertTrue(workload.storage)


if __name__ == '__main__':
    unittest.main()
