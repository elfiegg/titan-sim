import copy
import json
from pathlib import Path
import unittest

from titan_sim.gpt_oss_memory import project_state
from scripts.validate_gpt_oss import current_allocation_transfer, full_config_fields

ROOT = Path(__file__).resolve().parents[1]


class GptOssMemoryTests(unittest.TestCase):
    def test_optimizer_uses_distinct_dense_and_expert_groups(self):
        inventory = [
            {"shape": [128], "elements": 128, "placement": "dense_sharded_tp"},
            {"shape": [16], "elements": 16, "placement": "dense_replicated_tp"},
            {"shape": [512], "elements": 512, "placement": "expert"},
        ]
        result = project_state(inventory, world_size=32, tp=2, cp=4, ep=8)
        # Dense local 80 / dense DP+CP 16; expert local 64 / expert DP 4.
        self.assertEqual(result["local_elements_by_group"], {"dense": 80, "expert": 64})
        self.assertEqual(result["optimizer_group_sizes"], {"dense": 16, "expert": 4})
        self.assertEqual(result["persistent_state_bytes"], 4 * 144 + 12 * (5 + 16))
        self.assertEqual(result["end_step_state_bytes"], 4 * 144 + 16 * (5 + 16))
        self.assertEqual(result["trajectory"][-1]["allocated_bytes"], sum(result["components_bytes"].values()))
        at_zero = [r for r in result["trajectory"] if r["phase"] == "iteration_2_zero_grad"][-1]
        self.assertEqual(at_zero["allocated_bytes"], result["persistent_state_bytes"])
        self.assertEqual(result["capacity_verdict"], "unknown")

    def test_recorded_inventory_reconciles_every_logged_parameter_count(self):
        inventory = json.loads((ROOT / "docs/validation/gpt_oss/parameter_inventory.json").read_text())
        for world, tp, cp, ep, expected in (
            (8, 1, 1, 1, 20495244864), (8, 1, 1, 2, 10936778304),
            (72, 1, 2, 4, 6157545024), (512, 2, 4, 8, 3079983936),
        ):
            with self.subTest(world=world, ep=ep):
                result = project_state(inventory, world_size=world, tp=tp, cp=cp, ep=ep)
                self.assertEqual(result["local_parameter_elements"], expected)

    def test_rejects_unreviewed_topology_and_observation_inputs(self):
        for change in ({"pp": 2}, {"etp": 2}, {"ep": 3}, {"world_size": 7}, {"tp": True}):
            args = dict(world_size=8, tp=1, cp=1, ep=2, **{})
            args.update(change)
            with self.assertRaises(ValueError):
                project_state([], **args)
        with self.assertRaises(TypeError):
            project_state([], world_size=8, tp=1, cp=1, ep=1, observed_allocated_gb=100)

    def test_config_rejects_conflicting_or_missing_storage_policy(self):
        with self.assertRaises(ValueError):
            full_config_fields("Config(x=True)\nConfig(x=False)", "Config", {"x": "True"})
        with self.assertRaises(ValueError):
            full_config_fields("Config(y=True)", "Config", {"x": "True"})

    def test_calibrated_prediction_does_not_depend_on_target_observation(self):
        runs = json.loads((ROOT / "docs/validation/gpt_oss/results.json").read_text())["runs"]
        before = current_allocation_transfer(runs)
        changed = copy.deepcopy(runs)
        for run in changed:
            if run["config"]["world_size"] != 8:
                for observation in run["observations"]:
                    observation["allocated_gb"] *= 2
        after = current_allocation_transfer(changed)
        for a, b in zip(before, after):
            self.assertEqual(a["factor"], b["factor"])
            for x, y in zip(a["evaluations"], b["evaluations"]):
                self.assertEqual(x["predicted_current_gb"], y["predicted_current_gb"])
                self.assertNotEqual(x["error_percent"], y["error_percent"])


if __name__ == "__main__":
    unittest.main()
