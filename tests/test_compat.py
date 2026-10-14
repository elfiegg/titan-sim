import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class CompatibilityTests(unittest.TestCase):
    def test_optional_api_placeholders_raise_and_restore(self):
        import torch.distributed.pipelining as pipeline
        from titan_sim.compat import unused_pipeline_imports
        before = {key: hasattr(pipeline, key) for key in ('PipelineStageInfo', 'analyze_pipeline_activation_liveness')}
        with unused_pipeline_imports() as metadata:
            for name in metadata['missing_exports']:
                with self.assertRaises(NotImplementedError):
                    getattr(pipeline, name)()
        for key, exists in before.items():
            self.assertEqual(hasattr(pipeline, key), exists)

    def test_fsdp_callback_cannot_be_silently_ignored(self):
        from titan_sim.compat import nvidia2609_compat
        from torch.distributed.fsdp import MixedPrecisionPolicy
        with nvidia2609_compat() as metadata:
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, param_dtype_override_fn=None)
            self.assertEqual(policy.param_dtype, torch.bfloat16)
            if metadata['none_only_arguments']:
                with self.assertRaises(NotImplementedError):
                    MixedPrecisionPolicy(param_dtype_override_fn=lambda parameter: torch.float32)
