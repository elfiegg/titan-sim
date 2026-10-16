"""A complete original reduced model step, with explicit metadata exclusions."""
import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("TORCHTITAN_ROOT"), "Pinned upstream checkout required")
class StepCaptureTests(unittest.TestCase):
    def test_original_fullac_forward_backward_and_optimizer(self):
        from titan_sim.torchtitan_adapter import capture_smoke

        with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA must not initialize")):
            result = capture_smoke(
                torchtitan_root=os.environ["TORCHTITAN_ROOT"], flavor="debugmodel",
                seq_len=16, tokens=16, capture_scope="post_initialization",
                attention_contract="flex_metadata", full_ac=True,
            )
        metadata = result["metadata"]
        self.assertEqual(metadata["status"], "complete", metadata.get("traceback"))
        self.assertEqual(metadata["completed_steps"], 1)
        check = metadata["step_checks"][0]
        self.assertEqual(check["parameter_tensors_with_grad"], metadata["trainable_parameter_tensors"])
        self.assertTrue(check["optimizer_completed"])
        phases = {event["execution_phase"] for event in result["events"]}
        self.assertTrue({"forward", "backward", "recompute", "optimizer"} <= phases)
        self.assertTrue(any(e["name"] == "flex_attention_backward" for e in result["events"]))
        self.assertTrue(any(s["category"] == "optimizer" for s in result["storages"]))
        self.assertFalse(metadata["production_training_capture"])
        self.assertFalse(metadata["cuda_memory_equivalent"])
        self.assertFalse(metadata["initialization_coverage"]["initialization_peak_included"])
        self.assertEqual(metadata["attention_contract"], "flex_metadata")


if __name__ == "__main__":
    unittest.main()
