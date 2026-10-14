"""Exercise routing host controls against the actual pinned upstream method."""
import os
import unittest


@unittest.skipUnless(os.environ.get("TORCHTITAN_ROOT"),
                     "set TORCHTITAN_ROOT for upstream routing integration")
class RoutingScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from titan_sim.recipes import import_torchtitan
        import_torchtitan(os.environ["TORCHTITAN_ROOT"])

    def setUp(self):
        from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
        self.dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=8, top_k=2)
        )

    def test_strict_preserves_data_dependent_failure(self):
        import torch
        from torch._subclasses.fake_tensor import DataDependentOutputException, FakeTensorMode
        with FakeTensorMode(allow_fallback_kernels=False):
            local = torch.empty(8, dtype=torch.int64)
            peer = torch.empty(4, 2, dtype=torch.int64)
            with self.assertRaises(DataDependentOutputException):
                self.dispatcher._sync_token_count_exchange(local, peer, 4)

    def test_balanced_preserves_tensor_ops_kwargs_and_restores_host_reads(self):
        import torch
        from torch._subclasses.fake_tensor import DataDependentOutputException, FakeTensor, FakeTensorMode
        from titan_sim.capture import TensorCapture
        from titan_sim.project_engine import _balanced_routing_scenario
        original_tolist = FakeTensor.tolist
        with FakeTensorMode(allow_fallback_kernels=False), TensorCapture() as capture:
            local = torch.empty(8, dtype=torch.int64)
            peer = torch.empty(4, 2, dtype=torch.int64)
            with _balanced_routing_scenario(tokens=16, capture=capture):
                global_counts, inputs, outputs = self.dispatcher._sync_token_count_exchange(
                    num_local_tokens_per_expert_E=local,
                    num_global_tokens_per_local_expert_EP_e=peer,
                    ep_size=4,
                )
                self.assertEqual(global_counts.shape, (8,))
                self.assertEqual(inputs, [8] * 4)
                self.assertEqual(outputs, [8] * 4)
                self.assertIs(FakeTensor.tolist, original_tolist)
                with self.assertRaises(DataDependentOutputException):
                    local.tolist()
        self.assertIs(FakeTensor.tolist, original_tolist)
        names = [event["name"] for event in capture.events]
        self.assertEqual(names.count("aten.sum.dim_IntList"), 2)
        self.assertTrue(any("wait_tensor" in name for name in names))
        self.assertEqual(names.count("scenario.balanced_ep_splits"), 1)

    def test_balanced_rejects_nonintegral_split_scenario(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.capture import TensorCapture
        from titan_sim.project_engine import _balanced_routing_scenario
        with FakeTensorMode(allow_fallback_kernels=False), TensorCapture() as capture:
            local = torch.empty(8, dtype=torch.int64)
            peer = torch.empty(4, 2, dtype=torch.int64)
            with _balanced_routing_scenario(tokens=15, capture=capture):
                with self.assertRaisesRegex(ValueError, "T\\*K divisible"):
                    self.dispatcher._sync_token_count_exchange(local, peer, 4)
