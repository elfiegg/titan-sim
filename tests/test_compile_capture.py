"""Runtime AOT graph execution is distinct from compiler tracing and CUDA plans."""
import functools
import gc
import unittest

try:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    from titan_sim.compile_capture import AOTGraphCapture
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for AOT graph execution")
class CompileCaptureTests(unittest.TestCase):
    @staticmethod
    def swiglu(gate, up):
        return torch.nn.functional.silu(gate) * up

    def test_fake_full_ac_swiglu_backward_has_no_tracing_operations(self):
        capture = AOTGraphCapture()
        contexts = functools.partial(create_selective_checkpoint_contexts,
                                     lambda *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE)
        with capture.compile_scope():
            compiled = torch.compile(self.swiglu, fullgraph=True)
            with FakeTensorMode(allow_fallback_kernels=False), capture:
                gate = torch.empty(4, 8, requires_grad=True)
                up = torch.empty(4, 8, requires_grad=True)
                with capture.phase("forward_backward_step_0"):
                    output = checkpoint(compiled, gate, up, use_reentrant=False, context_fn=contexts)
                    output.sum().backward()
                self.assertEqual(gate.grad.shape, gate.shape)
                self.assertEqual(up.grad.shape, up.shape)
        silus = [event for event in capture.events if event["name"] == "aten.silu.default"]
        self.assertEqual([event["execution_phase"] for event in silus], ["forward", "recompute"])
        self.assertEqual(len([event for event in capture.events if event["name"] == "aten.silu_backward.default"]), 1)
        self.assertEqual([entry["kind"] for entry in capture.compilation["graphs"]], ["forward", "backward"])
        self.assertEqual([entry["execution_phase"] for entry in capture.compilation["executions"]],
                         ["forward", "recompute", "backward"])
        self.assertTrue(all(entry["status"] == "complete" for entry in capture.compilation["executions"]))
        self.assertTrue(capture.compilation["compile_requests"][0]["fullgraph"])
        self.assertFalse(capture.compilation["compiled_storage_lifetimes_equivalent"])
        self.assertFalse(capture.compilation["original_backend_preserved"])

    def test_native_aot_saved_storage_survives_forward_until_backward(self):
        capture = AOTGraphCapture()
        compiled = torch.compile(self.swiglu, fullgraph=True, backend=capture.backend)
        gate = torch.randn(4, 8, requires_grad=True)
        up = torch.randn(4, 8, requires_grad=True)
        reference_gate = gate.detach().clone().requires_grad_()
        reference_up = up.detach().clone().requires_grad_()
        reference = self.swiglu(reference_gate, reference_up)
        reference.sum().backward()
        with capture:
            with capture.phase("forward"):
                output = compiled(gate, up)
            forward = capture.compilation["executions"][0]
            output_id = capture.register_tensor(output)
            saved_ids = set(forward["output_storage_ids"]) - set(forward["input_storage_ids"]) - {output_id}
            self.assertTrue(saved_ids)
            gc.collect()
            capture.sweep()
            self.assertTrue(all(capture.storages[sid]["release_event"] is None for sid in saved_ids))
            with capture.phase("backward"):
                output.sum().backward()
            gc.collect()
            capture.sweep()
            self.assertTrue(all(capture.storages[sid]["release_event"] is not None for sid in saved_ids))
        self.assertTrue(torch.allclose(output, reference))
        self.assertTrue(torch.allclose(gate.grad, reference_gate.grad))
        self.assertTrue(torch.allclose(up.grad, reference_up.grad))

    def test_backend_replacement_is_explicit_and_restored_on_failure(self):
        capture = AOTGraphCapture()
        original = torch.compile
        with self.assertRaisesRegex(NotImplementedError, "default Inductor"):
            with capture.compile_scope():
                torch.compile(self.swiglu, backend="eager", fullgraph=True)
        self.assertIs(torch.compile, original)
        with capture.compile_scope():
            with self.assertRaisesRegex(ValueError, "enabled"):
                torch.compile(self.swiglu, disable=True)
            with self.assertRaisesRegex(NotImplementedError, "options"):
                torch.compile(self.swiglu, mode="reduce-overhead")
        self.assertIs(torch.compile, original)
        self.assertEqual(capture.compilation["compile_requests"], [])

    def test_cuda_inputs_rejected_before_aot_compiler_runs(self):
        capture = AOTGraphCapture()
        with FakeTensorMode(allow_fallback_kernels=False):
            tensor = torch.empty(4, 8, device="cuda")
            with self.assertRaisesRegex(NotImplementedError, "CPU semantic"):
                capture.backend(None, [tensor])
        self.assertIsNone(capture._backend)
        self.assertEqual(capture.compilation["graphs"], [])


if __name__ == "__main__":
    unittest.main()
