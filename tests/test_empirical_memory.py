import copy
import json
from pathlib import Path
import unittest

from titan_sim.empirical_memory import (
    allocator_ratios, estimate, fit_profile, project_allocator_envelope,
)
from scripts.calibrate_memory_from_logs import build_catalog
from titan_sim.memory_budget import component_ledger

ROOT = Path(__file__).resolve().parents[1]


class EmpiricalMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "docs/validation/runtime_logs"
        cls.observations = json.loads((source / "observations.json").read_text())
        cls.audit = json.loads((source / "run_audit.json").read_text())
        cls.catalog, cls.requests = build_catalog(cls.observations, cls.audit)

    def test_calibration_reconstructs_logged_counter_without_capacity_claim(self):
        for key, total in [("titan_v3_mxfp8_batch16", 164.54),
                           ("titan_v3_expert_mxfp8_batch8", 102.21),
                           ("titan_v3_nvfp4_batch16", 153.75)]:
            with self.subTest(key=key):
                p = self.catalog["profiles"][key]
                result = estimate(p, self.requests[key])
                self.assertAlmostEqual(result["estimated_gib"], total, places=8)
                self.assertFalse(result["independent_validation"])
                self.assertIsNone(result["required_capacity_bytes"])
                self.assertEqual(result["capacity_verdict"], "unknown")
        planning = self.catalog["estimates"]["titan_v3_mxfp8_batch16"]["planning"]
        self.assertAlmostEqual(planning["estimated_gib"], 167.1588774085045)

    def test_changes_in_hardware_precision_sequence_and_parallelism_need_opt_in(self):
        key = "titan_v3_mxfp8_batch16"
        p = self.catalog["profiles"][key]
        for field, value in [("sequence_length", 16384), ("hardware", "NVIDIA GB200"),
                             ("pp", 4), ("precision", "BF16 compute")]:
            req = copy.deepcopy(self.requests[key])
            req["context"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Outside calibration"):
                    estimate(p, req)
                result = estimate(p, req, allow_extrapolation=True)
                self.assertTrue(result["extrapolation"])
                self.assertIn(field, result["configuration_mismatches"])
                self.assertEqual(result["capacity_verdict"], "unknown")

    def test_changed_baseline_requires_extrapolation_even_with_matching_context(self):
        key = "titan_v3_mxfp8_batch16"
        req = copy.deepcopy(self.requests[key])
        req["baseline"]["state_bytes"] /= 2
        with self.assertRaisesRegex(ValueError, "baseline.state_bytes"):
            estimate(self.catalog["profiles"][key], req)

    def test_double_counting_and_independent_provenance_are_rejected(self):
        key = "titan_v3_mxfp8_batch16"
        req = copy.deepcopy(self.requests[key])
        req["workspace_bytes"] = 1024
        with self.assertRaisesRegex(ValueError, "stacked"):
            estimate(self.catalog["profiles"][key], req)
        with self.assertRaises(ValueError):
            component_ledger(self.catalog["profiles"][key])

    def test_audit_hash_mismatch_blocks_calibration(self):
        obs = copy.deepcopy(self.observations)
        obs["runs"]["titan_v3_mxfp8_batch16"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_catalog(obs, self.audit)

    def test_unverified_unit_and_invalid_values_block_fit(self):
        key = "titan_v3_mxfp8_batch16"
        p = self.catalog["profiles"][key]
        for field, value in [("unit", "GB"), ("bytes", float("nan")), ("bytes", -1)]:
            obs = copy.deepcopy(self.observations["runs"][key])
            sample = next(s for s in obs["samples"] if s["metric"] == "training_memory_unspecified")
            sample[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                fit_profile(obs, p["context"], p["baseline"], name=key, assumptions=p["assumptions"])

    def test_v4_ratios_pair_same_rank_step_and_do_not_stack_maxima(self):
        rows = self.catalog["v4_allocator"]["rows"]
        self.assertEqual(len(rows), 8)
        row = next(r for r in rows if r["rank"] == 0 and r["step"] == 2)
        self.assertAlmostEqual(row["high_water_reserved_over_current"], 266.13 / 187.98)
        result = project_allocator_envelope(187.98 * 10**9, row)
        self.assertAlmostEqual(result["projected_high_water_reserved_bytes"] / 10**9, 266.13)
        self.assertAlmostEqual(result["projected_high_water_allocated_bytes"] / 10**9, 243.74)
        self.assertEqual(result["capacity_verdict"], "unknown")

    def test_incomplete_rows_do_not_borrow_metrics_from_other_ranks(self):
        obs = copy.deepcopy(self.observations["runs"]["megatron_v4"])
        obs["samples"] = [s for s in obs["samples"] if not (
            s["rank"] == 0 and s["step"] == 2 and s["metric"] == "mem-allocated-gigabytes")]
        self.assertEqual(len(allocator_ratios(obs)), 7)

    def test_allocator_ratios_require_consistent_units(self):
        obs = copy.deepcopy(self.observations["runs"]["megatron_v4"])
        obs["samples"][0]["unit"] = "GiB"
        with self.assertRaisesRegex(ValueError, "same unit"):
            allocator_ratios(obs)

    def test_stash_geometry_separates_host_and_gpu_with_missing_not_zero(self):
        stash = self.catalog["observed_stash"]
        self.assertEqual(stash["megatron_v3"]["gpu_payload_bytes"], 23924736000)
        self.assertEqual(stash["megatron_v3"]["host_payload_bytes"], 15949824000)
        self.assertEqual(stash["megatron_v4"]["gpu_payload_bytes"], 51219582208)
        self.assertIsNone(stash["megatron_v4"]["host_payload_bytes"])

    def test_rankless_samples_are_not_claimed_as_all_rank_coverage(self):
        p = self.catalog["profiles"]["titan_v3_expert_mxfp8_batch8"]
        self.assertTrue(all(r["has_unranked_samples"] for r in p["sampled_counter_history"]))
        self.assertTrue(all(not r["known_ranks"] for r in p["sampled_counter_history"]))


if __name__ == "__main__":
    unittest.main()
