import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_expanded_logs import inspect_archive, v4_config
from titan_sim.memory_validation import evaluate_v3, training_signature

ROOT = Path(__file__).resolve().parents[1]


class FrozenValidationTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "profiles/memory/titan_v3_mxfp8_batch16.json").read_text())
        source = "synthetic-held-out.log"
        # Explicit one-row fixture: the test does not require private raw logs.
        facts = {"model": "deepseek_v3 671B", "parameter_elements": 671026404352,
                 "mesh": {"pp": 1, "tp": 1, "cp": 1, "ep": 64, "dp_shard": 256, "dp_replicate": 1},
                 "training": {"local_batch": 1, "global_batch": 256, "accumulation": 1, "sequence_length": 4096, "steps": 50},
                 "hardware": "NVIDIA GB300", "torchtitan_version": "0.2.2+gita615ec2"}
        self.observation = {"source_id": source, "sha256": "synthetic-held-out",
                            "facts": {k: [{"value": v}] for k, v in facts.items()},
                            "indicators": {"full_checkpointing": [1], "mxfp8_conversion": [1]},
                            "samples": [{"metric": "training_memory_unspecified", "value": 57.43,
                                         "bytes": round(57.43 * 2**30), "unit": "GiB", "step": 50,
                                         "source": source, "line": 1, "rank": 0}]}

    def evaluate(self):
        return evaluate_v3(self.profile, self.observation, compile_policy="loss and each TransformerBlock")

    def test_frozen_factor_exposes_error_without_refitting(self):
        before = copy.deepcopy(self.profile)
        result = self.evaluate()
        self.assertEqual(before, self.profile)
        self.assertAlmostEqual(result["frozen_estimate_gib"], 46.9014475705, places=7)
        self.assertLess(result["frozen_error_percent"], -18)
        self.assertLess(result["planning_error_percent"], -18)
        self.assertTrue(result["configuration_extrapolation"])
        self.assertFalse(result["end_to_end_validation"])

    def test_rejects_reduced_model_wrong_hardware_and_unknown_version(self):
        for field, value in [("parameter_elements", 16_000_000_000), ("hardware", "NVIDIA GB200"),
                             ("torchtitan_version", "unknown")]:
            with self.subTest(field=field):
                old = self.observation["facts"][field]
                self.observation["facts"][field] = [{"value": value}]
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.observation["facts"][field] = old

    def test_conflicting_config_is_not_first_value_wins(self):
        self.observation["facts"]["hardware"].append({"value": "NVIDIA GB200"})
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            self.evaluate()

    def test_changed_precision_or_missing_checkpoint_evidence_rejected(self):
        self.observation["indicators"]["nvfp4_conversion"] = [1]
        with self.assertRaisesRegex(ValueError, "precision"):
            self.evaluate()
        del self.observation["indicators"]["nvfp4_conversion"]
        del self.observation["indicators"]["full_checkpointing"]
        with self.assertRaisesRegex(ValueError, "FullAC"):
            self.evaluate()

    def test_calibration_hash_reuse_is_not_held_out(self):
        self.observation["sha256"] = self.profile["source_evidence"]["sha256"]
        result = self.evaluate()
        self.assertTrue(result["calibration_source_reused"])
        self.assertEqual(result["evaluation_kind"], "calibration_replay")

    def test_signature_ignores_fanout_but_retains_metrics(self):
        before = training_signature(self.observation)
        row = copy.deepcopy(self.observation["samples"][0])
        row.update(source="rewrapped.log", line=55, rank=None)
        self.observation["samples"].append(row)
        self.assertEqual(before, training_signature(self.observation))
        row["value"] = 58
        self.assertNotEqual(before, training_signature(self.observation))

    def test_manifest_declared_duplicate_resolves_to_stored_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"unclassified test log\n"
            (root / "stored.log").write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            manifest = {"files": [{"archive_path": "stored.log", "sha256": digest},
                                  {"archive_path": "absent-copy.log", "duplicate_of": "stored.log", "sha256": digest}]}
            (root / "MANIFEST.json").write_text(json.dumps(manifest))
            result = inspect_archive(root)
            self.assertEqual(result["manifest_entries"], 2)
            self.assertEqual(result["unique_stored_files"], 1)
            manifest["files"][1]["sha256"] = "wrong"
            (root / "MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                inspect_archive(root)

    def test_missing_v4_configuration_cannot_be_compared(self):
        with self.assertRaisesRegex(ValueError, "Missing or conflicting"):
            v4_config("num_layers: 61\n")


if __name__ == "__main__":
    unittest.main()
