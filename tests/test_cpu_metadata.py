import unittest

try:
    import torch
    import torch._dynamo
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

from titan_sim.cpu_metadata import cpu_metadata_execution


@unittest.skipIf(torch is None, "PyTorch is required")
class CpuMetadataTests(unittest.TestCase):
    def test_original_module_backward_and_optimizer_need_no_cuda_runtime(self):
        from unittest.mock import patch
        from titan_sim.capture import TensorCapture

        capture = TensorCapture()
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
            with cpu_metadata_execution() as report, capture:
                model = torch.nn.Sequential(torch.nn.Linear(3, 4, device="cuda"), torch.nn.GELU())
                optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
                inputs = torch.empty(2, 3, device="cuda:0", requires_grad=True)
                with capture.phase("forward"):
                    loss = model(inputs).square().mean()
                with capture.phase("backward"):
                    loss.backward()
                self.assertEqual(inputs.grad.shape, inputs.shape)
                self.assertTrue(all(isinstance(p.grad, FakeTensor) for p in model.parameters()))
                with capture.phase("optimizer"):
                    optimizer.step()
                self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
                self.assertTrue(all(state["exp_avg"].shape == p.shape for p, state in optimizer.state.items()))
        operations = [event["name"] for event in capture.events if event["kind"] == "compute"]
        self.assertIn("aten.gelu_backward.default", operations)
        self.assertIn("aten.mm.default", operations)
        self.assertTrue(report["device_normalizations"])
        self.assertFalse(report["cuda_dispatch_equivalent"])
        self.assertFalse(report["cuda_memory_equivalent"])

    def test_device_transfer_normalization_exposes_changed_alias_semantics(self):
        with cpu_metadata_execution() as report:
            value = torch.empty(3)
            # These aliases are intentionally not passed off as CUDA transfers.
            self.assertIs(value.to("cuda:0"), value)
            self.assertIs(value.cuda(), value)
            self.assertIs(value.to(device=torch.device("cuda")), value)
            converted = value.to("cuda", torch.float64)
            self.assertEqual(converted.dtype, torch.float64)
            self.assertEqual(converted.device.type, "cpu")
        self.assertEqual(len(report["device_normalizations"]), 4)
        self.assertIn("copy and alias", " ".join(report["limitations"]))

    def test_real_inputs_and_data_dependent_reads_stay_unsupported(self):
        real = torch.ones(2)
        with cpu_metadata_execution():
            with self.assertRaisesRegex(AssertionError, "FakeTensor"):
                real.sin()
            with self.assertRaises(torch._subclasses.fake_tensor.DataDependentOutputException):
                torch.empty(()).item()

    def test_existing_accelerator_tensor_rejected_before_native_autograd(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            value = torch.empty(2, device="cuda", requires_grad=True)
        with cpu_metadata_execution():
            with self.assertRaisesRegex(NotImplementedError, "existing accelerator"):
                value.square()

    def test_scope_restores_default_device_and_function_mode_after_error(self):
        original = torch.get_default_device()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with cpu_metadata_execution():
                raise RuntimeError("test")
        self.assertEqual(torch.get_default_device(), original)
        self.assertNotIsInstance(torch.empty(2), FakeTensor)

    def test_backward_callbacks_keep_device_normalization_and_input_selection(self):
        from unittest.mock import patch
        requested_devices = []
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")), cpu_metadata_execution() as report:
            first = torch.empty(3, requires_grad=True)
            second = torch.empty(3, requires_grad=True)

            def callback(gradient):
                buffer = torch.empty(gradient.shape, device="cuda")
                requested_devices.append(buffer.device.type)
                return gradient

            first.register_hook(callback)
            output = first.square() + second.square()
            seed = torch.empty_like(output)
            output.backward(seed, retain_graph=True, inputs=[first])
            self.assertIsNotNone(first.grad)
            self.assertIsNone(second.grad)
            first.grad = None
            torch.autograd.backward(output, grad_tensors=seed, inputs={"first": first})
            self.assertIsNotNone(first.grad)
        self.assertEqual(requested_devices, ["cpu", "cpu"])
        self.assertEqual(report["backward_calls"], 2)
        self.assertEqual(len(report["device_normalizations"]), 2)

    def test_backward_preserves_create_graph_and_rejects_real_tensor(self):
        import warnings
        real = torch.ones((), requires_grad=True)
        with cpu_metadata_execution():
            with self.assertRaisesRegex(NotImplementedError, "CPU FakeTensor"):
                real.backward()
            leaf = torch.empty(3, requires_grad=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                leaf.square().sum().backward(create_graph=True)
            self.assertTrue(leaf.grad.requires_grad)
            derivative = leaf.grad
            leaf.grad = None  # Break the documented create_graph reference cycle.
            derivative.sum().backward()
            self.assertEqual(leaf.grad.shape, leaf.shape)

    def test_backward_requires_public_reentrant_dispatch_support(self):
        from unittest.mock import patch
        with cpu_metadata_execution():
            leaf = torch.empty((), requires_grad=True)
            with patch.object(torch.overrides, "redispatch_function", None):
                with self.assertRaisesRegex(NotImplementedError, "redispatch_function"):
                    leaf.backward()

    def test_device_mode_remains_traceable_by_explicit_aot_backend(self):
        from titan_sim.compile_capture import AOTGraphCapture
        capture = AOTGraphCapture()
        with cpu_metadata_execution(), capture:
            compiled = torch.compile(lambda value: value.sin().square(),
                                     backend=capture.backend, fullgraph=True)
            leaf = torch.empty(3, requires_grad=True)
            compiled(leaf).sum().backward()
            self.assertEqual(leaf.grad.device.type, "cpu")
        kinds = {graph["kind"] for graph in capture.compilation["graphs"]}
        self.assertTrue({"forward", "backward"} <= kinds)
