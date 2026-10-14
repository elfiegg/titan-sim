"""Check the explicit steady-state boundary without replacing model policies."""
from functools import partial
import os
from types import MethodType
import unittest


@unittest.skipUnless(os.environ.get("TORCHTITAN_ROOT"),
                     "set TORCHTITAN_ROOT for upstream initialization integration")
class InitializationScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from titan_sim.recipes import import_torchtitan
        import_torchtitan(os.environ["TORCHTITAN_ROOT"])

    def make_model(self):
        import torch
        from torchtitan.models.common.linear import Linear
        return Linear.Config(
            in_features=8, out_features=4, bias=True,
            param_init={"weight": partial(torch.nn.init.trunc_normal_, std=0.02),
                        "bias": torch.nn.init.ones_},
        ).build()

    def test_preserves_materialization_deterministic_parameters_and_buffers(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.capture import TensorCapture
        from titan_sim.torchtitan_adapter import post_initialization_scope
        with FakeTensorMode(allow_fallback_kernels=False), TensorCapture() as capture:
            model = self.make_model()
            parameter = model.weight
            def initialize_buffers(module, *, buffer_device=None):
                module.register_buffer("probe_buffer", torch.ones(7, device=buffer_device))
            model._init_self_buffers = MethodType(initialize_buffers, model)
            before = len(capture.events)
            with post_initialization_scope() as report:
                model.init_states(buffer_device=torch.device("cpu"))
            self.assertIs(model.weight, parameter)
            self.assertEqual(tuple(model.weight.shape), (4, 8))
            self.assertEqual(tuple(model.probe_buffer.shape), (7,))
        names = [event["name"] for event in capture.events[before:]]
        self.assertIn("aten.fill_.Scalar", names)
        self.assertIn("aten.ones.default", names)
        self.assertNotIn("aten.normal_.default", names)
        self.assertEqual(report["skipped_parameter_initializers"], 1)
        self.assertEqual(report["initializers"], {"trunc_normal_": 1})
        self.assertEqual(report["random_parameter_values"], "unknown")
        self.assertFalse(report["initialization_time_included"])
        self.assertFalse(report["initialization_peak_included"])

    def test_rejects_real_parameters_and_restores_upstream_method(self):
        from titan_sim.torchtitan_adapter import post_initialization_scope
        from torchtitan.protocols.module import Module
        model = self.make_model()
        original = Module._init_param
        with self.assertRaisesRegex(RuntimeError, "only supports FakeTensor"):
            with post_initialization_scope():
                model.init_states()
        self.assertIs(Module._init_param, original)

    def test_composite_parameter_callback_keeps_nonrandom_operations(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.capture import TensorCapture
        from titan_sim.torchtitan_adapter import post_initialization_scope
        def composite(parameter):
            torch.nn.init.trunc_normal_(parameter[0])
            with torch.no_grad():
                parameter[1:].zero_()
        with FakeTensorMode(allow_fallback_kernels=False), TensorCapture() as capture:
            model = self.make_model()
            model._param_init["weight"] = composite
            before = len(capture.events)
            with post_initialization_scope() as report:
                model.init_states()
        names = [event["name"] for event in capture.events[before:]]
        self.assertIn("aten.select.int", names)
        self.assertIn("aten.zero_.default", names)
        self.assertEqual(report["initializers"], {"trunc_normal_": 1})
