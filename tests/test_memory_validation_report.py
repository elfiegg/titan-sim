"""Prevent partial/calibrated evidence being presented as complete prediction."""
import importlib.util
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validation_report", ROOT / "scripts/report_memory_validation.py")
report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_module)


class MemoryValidationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = report_module.build_report()

    def test_delta_sign_and_missing_prediction(self):
        self.assertEqual(report_module.delta(90, 100), {"delta": -10, "delta_percent": -10})
        self.assertEqual(report_module.delta(None, 100), {"delta": None, "delta_percent": None})
        with self.assertRaises(ValueError):
            report_module.delta(100, 0)

    def test_missing_full_peak_prediction_is_not_zero_error(self):
        observed = [r for r in self.report["rows"] if r["method"] == "observation_only"]
        self.assertTrue(observed)
        self.assertTrue(all(r["predicted"] is None and r["delta_percent"] is None for r in observed))
        self.assertTrue(all(not r["full_allocation_prediction"] for r in self.report["rows"]))
        self.assertTrue(all(r["delta"] is None for r in self.report["unavailable"]))

    def test_v4_keeps_conditioning_and_unverified_units(self):
        v4 = [r for r in self.report["rows"] if r["model"].startswith("DeepSeek V4")]
        self.assertEqual(len(v4), 40)
        self.assertTrue(all(r["target_measurement_input"] for r in v4))
        self.assertTrue(all("unverified" in r["unit"] for r in v4))
        self.assertTrue(all("observed_bytes" not in r for r in v4))

    def test_frozen_transfer_reproduces_and_remains_point_specific(self):
        transfers = [r for r in self.report["rows"] if r["method"] == "frozen_additive_components"]
        self.assertEqual(len(transfers), 2)
        for row, expected in zip(transfers, (0.302093696, 0.424589184)):
            self.assertAlmostEqual(row["delta"], expected, places=8)
        self.assertTrue(all(r["rank"] == 0 and r["step"] == 2 for r in transfers))
        self.assertTrue(all(not r["target_measurement_input"] for r in transfers))

    def test_sources_are_not_claimed_rechecked_without_archive(self):
        self.assertEqual(len(self.report["raw_sources"]), 29)
        self.assertFalse(any(s["verified_this_run"] for s in self.report["raw_sources"].values()))
        evaluation = [r for r in self.report["rows"] if r.get("role") == "evaluation"]
        self.assertEqual(len(evaluation), 8)
        self.assertAlmostEqual(max(abs(r["delta_percent"]) for r in evaluation), 2.84955848032)
