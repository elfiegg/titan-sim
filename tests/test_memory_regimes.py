import copy
import json
from pathlib import Path
import unittest

from titan_sim.memory_regimes import estimate, fit_profile

ROOT = Path(__file__).resolve().parents[1]


class BatchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "profiles/memory/titan_v3_mxfp8_batch_regimes.json").read_text())
        self.request = {"source": "batch_memory_request", "context": copy.deepcopy(self.profile["context"]), "local_batch": 2}

    def refit(self, anchors=None, context=None):
        return fit_profile(name="test", context=context or self.profile["context"],
                           anchors=self.profile["anchors"] if anchors is None else anchors,
                           assumed_state_bytes=self.profile["assumed_state_bytes"],
                           metric_evidence=self.profile["metric_evidence"])

    def test_retrospective_nonfit_batches_meet_target_for_both_precisions(self):
        # Observations were inspected before split selection: not blind validation.
        original = json.loads((ROOT / "docs/validation/expanded_logs/results.json").read_text())
        for kind, precision in (("mxfp8", "MXFP8 compute"), ("nvfp4", "mixed NVFP4/MXFP8 compute")):
            profile = json.loads((ROOT / f"profiles/memory/titan_v3_{kind}_batch_regimes.json").read_text())
            self.assertEqual([a["local_batch"] for a in profile["anchors"]], [1, 4, 16])
            checked = 0
            for row in original["v3_evaluations"]:
                if row["cohort"] != "aws_expA" or row["local_batch"] not in (2, 8) or row["context"]["precision"] != precision:
                    continue
                with self.subTest(kind=kind, batch=row["local_batch"]):
                    checked += 1
                    self.assertNotIn(row["sha256"], [a["evidence"]["sha256"] for a in profile["anchors"]])
                    result = estimate(profile, {"source": "batch_memory_request", "context": profile["context"], "local_batch": row["local_batch"]})
                    self.assertLess(abs(100 * (result["estimated_gib"] / row["observed_max_gib"] - 1)), 5)
                    self.assertIsNone(result["required_capacity_bytes"])
                    self.assertFalse(result["independent_validation"])
            self.assertEqual(checked, 2)

    def test_same_frozen_baseline_still_exposes_small_batch_failure(self):
        from titan_sim.empirical_memory import estimate as old_estimate
        p = json.loads((ROOT / "profiles/memory/titan_v3_mxfp8_batch16.json").read_text())
        baseline = dict(p["baseline"], checkpoint_proxy_bytes=p["baseline"]["checkpoint_proxy_bytes"] / 16)
        context = dict(p["context"], tokens_per_rank=4096, global_batch=256)
        result = old_estimate(p, {"source": "empirical_memory_request", "context": context, "baseline": baseline}, allow_extrapolation=True)
        self.assertLess(100 * (result["estimated_gib"] / 57.43 - 1), -18)

    def test_rejects_hardware_mesh_precision_sequence_and_software_transfer(self):
        for key, value in (("hardware", "NVIDIA GB200"), ("ep", 32), ("pp", 4),
                           ("sequence_length", 8192), ("precision", "BF16"), ("revision", "other"),
                           ("torch_version", "new"), ("compile_policy", "eager")):
            request = copy.deepcopy(self.request)
            request["context"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "configuration"):
                estimate(self.profile, request)

    def test_rejects_extrapolation_invalid_batches_and_double_counting(self):
        for batch in (0, 17, 2.5, True, float("nan")):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                estimate(self.profile, dict(self.request, local_batch=batch))
        with self.assertRaisesRegex(ValueError, "stack"):
            estimate(self.profile, dict(self.request, extra_workspace_bytes=1024))

    def test_interpolation_reproduces_anchors_and_retains_fixed_allowance(self):
        for anchor in self.profile["anchors"]:
            out = estimate(self.profile, dict(self.request, local_batch=anchor["local_batch"]))
            self.assertAlmostEqual(out["estimated_bytes"], anchor["observed_bytes"], delta=1)
            self.assertAlmostEqual(sum(out["decomposition"].values()), out["estimated_bytes"], delta=1)
        self.assertGreater(estimate(self.profile, self.request)["decomposition"]["effective_fixed_allowance_bytes"], 10 * 2**30)

    def test_bad_calibration_inputs_are_rejected(self):
        for mutation in ("duplicate_batch", "duplicate_source", "negative", "nonfinite", "nonmonotonic", "missing_hash"):
            anchors = copy.deepcopy(self.profile["anchors"])
            if mutation == "duplicate_batch": anchors[1]["local_batch"] = anchors[0]["local_batch"]
            if mutation == "duplicate_source": anchors[1]["evidence"] = anchors[0]["evidence"]
            if mutation == "negative": anchors[1]["observed_bytes"] = -1
            if mutation == "nonfinite": anchors[1]["observed_bytes"] = float("nan")
            if mutation == "nonmonotonic": anchors[1]["observed_bytes"] = 1
            if mutation == "missing_hash": anchors[1]["evidence"]["sha256"] = ""
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.refit(anchors)
        with self.assertRaises(ValueError): self.refit([])

    def test_metric_revision_and_fixed_context_are_required(self):
        for key, value in (("revision", "other"), ("memory_metric", "allocated"), ("global_batch", 256)):
            context = dict(self.profile["context"], **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): self.refit(context=context)


if __name__ == "__main__":
    unittest.main()
