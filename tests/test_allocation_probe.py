import unittest
from titan_sim.allocation_probe import summarize_snapshot, summarize_external_allocations


class AllocationAttributionTests(unittest.TestCase):
    def snapshot(self):
        return {"segments": [{"address": 100, "total_size": 160, "device": 0,
            "segment_pool_id": (0, 1), "blocks": [
                {"address": 100, "size": 100, "state": "active_allocated", "frames": [{"name": "quantize"}]},
                {"address": 200, "size": 20, "state": "active_awaiting_free"},
                {"address": 220, "size": 40, "state": "inactive"}]}]}

    def test_aliases_are_counted_once_and_partial_coverage_stays_unknown(self):
        inventory = [{"address": 100, "size_bytes": 60, "categories": ["parameters"]},
                     {"address": 100, "size_bytes": 60, "categories": ["parameters"]},
                     {"address": 140, "size_bytes": 20, "categories": ["gradients"]}]
        result = summarize_snapshot(self.snapshot(), inventory, 0)
        self.assertEqual(result["exact_storage_categories_bytes"], {
            "parameters": 40, "shared:gradients+parameters": 20, "unattributed": 40})
        self.assertEqual(result["unattributed_stack_hints_bytes"], {"quantization_candidate": 40})
        self.assertTrue(result["attributed_plus_unattributed_equals_active"])
        self.assertEqual(result["totals_bytes"]["active_awaiting_free"], 20)
        self.assertEqual(result["totals_bytes"]["reserved_bytes"], 160)

    def test_other_device_inventory_cannot_attribute_matching_address(self):
        result = summarize_snapshot(self.snapshot(), [{"address": 100, "size_bytes": 100,
            "device_index": 1, "categories": ["parameters"]}], 0)
        self.assertEqual(result["exact_storage_categories_bytes"], {"unattributed": 100})
        self.assertEqual(summarize_snapshot(self.snapshot(), [], 1)["exact_storage_categories_bytes"], {})

    def test_allocator_block_slack_is_distinct_from_unknown_tensor_storage(self):
        snapshot = self.snapshot()
        snapshot["segments"][0]["blocks"][0]["requested_size"] = 60
        result = summarize_snapshot(snapshot, [{"address": 100, "size_bytes": 55,
            "device_index": 0, "categories": ["parameters"]}], 0)
        self.assertEqual(result["exact_storage_categories_bytes"], {
            "parameters": 55, "unattributed": 5, "allocator_block_slack": 40})
        self.assertEqual(result["unattributed_stack_hints_bytes"], {"quantization_candidate": 5})
        self.assertTrue(result["attributed_plus_unattributed_equals_active"])

    def test_workspace_classification_uses_function_evidence_not_workspace_directory(self):
        snapshot = self.snapshot()
        block = snapshot["segments"][0]["blocks"][0]
        block["frames"] = [{"filename": "/workspace/train.py", "name": "something"}]
        self.assertEqual(summarize_snapshot(snapshot, [], 0)["unattributed_stack_hints_bytes"], {"unknown": 100})
        block["frames"] = [{"name": "at::cuda::setWorkspaceForHandle"}]
        self.assertEqual(summarize_snapshot(snapshot, [], 0)["unattributed_stack_hints_bytes"], {"gemm_workspace_candidate": 100})
        block["frames"] = [{"name": "get_dummy_wgrad"}]
        self.assertEqual(summarize_snapshot(snapshot, [], 0)["unattributed_stack_hints_bytes"], {"te_dummy_weight_gradient_candidate": 100})

    def test_external_replay_accounts_for_alignment_and_reused_addresses(self):
        result = summarize_external_allocations([
            {"event": "allocate_fabric", "address": "0x100", "requested_bytes": 513, "physical_bytes": 1024},
            {"event": "allocate_cuda", "address": "0x200", "requested_bytes": 32, "physical_bytes": 32},
            {"event": "free", "address": "0x100"},
            {"event": "allocate_fabric", "address": "0x100", "requested_bytes": 5, "physical_bytes": 512},
        ])
        self.assertEqual(result["requested_live_bytes"], 37)
        self.assertIsNone(result["physical_live_bytes"])
        self.assertEqual(result["accounted_live_bytes"], 544)
        self.assertEqual(result["fabric_alignment_live_bytes"], 507)
        self.assertEqual(result["peak_accounted_bytes"], 1056)
        self.assertEqual(result["unmatched_events"], [])
        self.assertFalse(result["complete_external_memory_coverage"])


if __name__ == "__main__":
    unittest.main()
