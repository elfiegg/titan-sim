"""Original upstream attention/MoE contracts with driverless native autograd."""
import gc
import importlib
import os
import unittest


@unittest.skipUnless(os.environ.get("TORCHTITAN_ROOT"),
                     "set TORCHTITAN_ROOT for original DeepSeek integration")
class AttentionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from titan_sim.recipes import import_torchtitan
        import_torchtitan(os.environ["TORCHTITAN_ROOT"])

    def test_original_flex_hop_saves_qkv_and_stats_until_backward(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.attention_metadata import flex_attention_metadata_scope
        from titan_sim.capture import TensorCapture
        from torchtitan.models.common.attention import (
            FlexInnerAttention, create_attention_mask,
        )

        def causal(batch, head, query, key):
            return query >= key

        with FakeTensorMode(allow_fallback_kernels=False), flex_attention_metadata_scope() as report:
            with TensorCapture() as capture:
                mask = create_attention_mask(causal, 1, 1, 16, 16, device="cpu")
                q = torch.empty(16, 2, 32, dtype=torch.bfloat16, requires_grad=True)
                k = torch.empty_like(q, requires_grad=True)
                v = torch.empty(16, 2, 16, dtype=torch.bfloat16, requires_grad=True)
                qkv_ids = {capture.register_tensor(t) for t in (q, k, v)}
                inner = FlexInnerAttention.Config().build()
                with capture.phase("forward"):
                    out = inner(q, k, v, attention_masks=mask)
                saved_ids = {item["storage_id"] for item in capture.saved_tensors
                             if item["action"] == "save"}
                self.assertTrue(qkv_ids <= saved_ids)
                # Original HOP saves both logsumexp and max_scores as FP32 BHT.
                stats = [sid for sid in saved_ids
                         if capture.storages[sid]["dtype"] == "float32"
                         and capture.storages[sid]["size_bytes"] == 1 * 2 * 16 * 4]
                self.assertGreaterEqual(len(stats), 2)
                self.assertTrue(all(capture.storages[sid]["release_event"] is None for sid in stats))
                with capture.phase("backward"):
                    out.backward(torch.empty_like(out))
                gc.collect()
                capture.sweep()
                self.assertTrue(all(capture.storages[sid]["release_event"] is not None for sid in stats))
                for tensor in (q, k, v):
                    self.assertEqual(tensor.grad.shape, tensor.shape)
                self.assertEqual(tuple(out.shape), (16, 2, 16))
        names = [event["name"] for event in capture.events]
        self.assertIn("flex_attention", names)
        self.assertIn("flex_attention_backward", names)
        self.assertNotIn("aten.bmm.default", names)
        hops = [event for event in capture.events if event["name"].startswith("flex_attention")]
        self.assertTrue(all(event["kernel_workspace_bytes"] is None for event in hops))
        self.assertTrue(all(event["signature"]["status"] == "unsupported" for event in hops))
        self.assertEqual(report["attention_calls"], 1)
        self.assertEqual(report["mask_calls"], 1)

    def test_real_inputs_rejected_and_global_wrappers_restored_on_failure(self):
        import torch
        from titan_sim.attention_metadata import flex_attention_metadata_scope
        from torchtitan.models.common import attention
        flex = importlib.import_module("torch.nn.attention.flex_attention")
        original_validate = flex._validate_device
        original_attention = attention.FlexInnerAttention._compiled_flex_attn
        original_mask = attention._compiled_create_block_mask
        original_debug = flex._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG
        q = torch.empty(1, 1, 16, 16, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "requires CPU FakeTensors"):
            with flex_attention_metadata_scope():
                attention.FlexInnerAttention._compiled_flex_attn(q, q, q)
        self.assertIs(flex._validate_device, original_validate)
        self.assertIs(attention.FlexInnerAttention._compiled_flex_attn, original_attention)
        self.assertIs(attention._compiled_create_block_mask, original_mask)
        self.assertEqual(flex._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG, original_debug)

    def test_fallback_enabled_fake_mode_is_rejected(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.attention_metadata import flex_attention_metadata_scope
        from torchtitan.models.common.attention import FlexInnerAttention
        with FakeTensorMode(allow_fallback_kernels=True), flex_attention_metadata_scope():
            q = torch.empty(1, 1, 16, 16, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "fallback kernels disabled"):
                FlexInnerAttention._compiled_flex_attn(q, q, q)

    def test_score_callback_rejected_before_losing_captured_gradient(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.attention_metadata import flex_attention_metadata_scope
        from torchtitan.models.common.attention import FlexInnerAttention
        with FakeTensorMode(allow_fallback_kernels=False), flex_attention_metadata_scope():
            q = torch.empty(1, 1, 16, 16, requires_grad=True)
            bias = torch.empty((), requires_grad=True)
            def score_mod(score, batch, head, query, key):
                return score + bias
            with self.assertRaisesRegex(NotImplementedError, "identity score_mod only"):
                FlexInnerAttention._compiled_flex_attn(q, q, q, score_mod=score_mod)

    def test_unknown_hop_fails_closed(self):
        import torch
        from titan_sim.capture import TensorCapture
        with self.assertRaisesRegex(NotImplementedError, "Higher-order storage capture unsupported"):
            TensorCapture().__torch_dispatch__(torch.ops.higher_order.cond, (), (), {})

    def test_original_moe_router_dispatch_and_grouped_backward(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.capture import TensorCapture
        from titan_sim.recipes import build_smoke_model_config
        from titan_sim.torchtitan_adapter import post_initialization_scope
        from torchtitan.models.common.aux_loss import AuxLoss
        from torchtitan.tools.utils import set_default_dtype
        for flavor in ("debugmodel", "671B"):
            with self.subTest(flavor=flavor):
                config = build_smoke_model_config(flavor=flavor)
                moe_config = next(layer.moe for layer in config.layers if layer.moe)
                with FakeTensorMode(allow_fallback_kernels=False), TensorCapture() as capture:
                    with set_default_dtype(torch.bfloat16):
                        moe = moe_config.build()
                    with post_initialization_scope():
                        moe.init_states(buffer_device=torch.device("cpu"))
                    AuxLoss.set_step_denominator(torch.full((), 16.0, dtype=torch.float32))
                    x = torch.empty(16, config.dim, dtype=torch.bfloat16, requires_grad=True)
                    with capture.phase("forward"):
                        out = moe(x)
                    with capture.phase("backward"):
                        out.backward(torch.empty_like(out))
                    self.assertEqual(x.grad.shape, x.shape)
                    self.assertTrue(all(parameter.grad is not None for parameter in moe.parameters()))
                names = [event["name"] for event in capture.events]
                self.assertIn("aten.topk.default", names)
                self.assertIn("aten._grouped_mm.default", names)
                self.assertIn("torchtitan.deterministic_scatter_add.default", names)
                self.assertNotIn("aten._local_scalar_dense.default", names)
                self.assertTrue(any(event["name"] == "aten._grouped_mm.default"
                                    and event["phase"] == "backward" for event in capture.events))
