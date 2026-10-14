import copy
import json
from pathlib import Path
import tempfile
import unittest

from titan_sim.memory_budget import CATEGORIES, budget_scenarios, capacity_bytes, component_ledger, main


def component(name, category, size, start="forward", end=None, rank=0, storage_id=None):
    return dict(name=name, category=category, rank=rank, storage_id=storage_id or name,
                size_bytes=size, live_range=dict(start=start, end=end),
                provenance=dict(kind="unknown" if size is None else "declared_assumption",
                                detail="Independent test scenario"))


def spec(components=(), ranks=(0,)):
    return dict(source="independent_component_spec", ranks=list(ranks),
                phases=["forward", "backward", "optimizer"],
                coverage={category: "complete" for category in CATEGORIES},
                components=list(components))


class MemoryBudgetTests(unittest.TestCase):
    def test_nonsimultaneous_buffers_do_not_sum_their_peaks(self):
        declared = spec([
            component("params", "parameters", 100),
            component("activation", "activations", 80, end="backward"),
            component("gradient", "gradients", 60, start="backward", end="optimizer"),
            component("workspace", "workspace", 20, start="optimizer"),
        ])
        rank = component_ledger(declared)["per_rank"]["0"]
        self.assertEqual([row["known_bytes"] for row in rank["phases"]], [180, 160, 120])
        self.assertEqual(rank["required_peak_bytes"], 180)
        self.assertEqual(rank["peak_known_phases"], ["forward"])

    def test_aliases_union_lifetimes_without_double_counting(self):
        declared = spec([
            component("original", "activations", 80, end="optimizer", storage_id="shared"),
            component("saved_view", "activations", 80, start="backward", storage_id="shared"),
        ])
        rows = component_ledger(declared)["per_rank"]["0"]["phases"]
        self.assertEqual([row["known_bytes"] for row in rows], [80, 80, 80])
        self.assertTrue(all(row["live_storage_ids"] == ["shared"] for row in rows))

    def test_conflicting_alias_sizes_or_categories_rejected(self):
        for changed in ({"size_bytes": 81}, {"category": "gradients"}, {"device": "cpu"}):
            first = component("original", "activations", 80, storage_id="shared")
            second = {**first, "name": "view", **changed}
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "aliases"):
                component_ledger(spec([first, second]))

    def test_cpu_offload_buffers_do_not_raise_gpu_peak(self):
        offload = component("offloaded_attention", "activations", 2 * 10**9, end="optimizer")
        offload.update(device="cpu", pinned=True)
        declared = spec([component("params", "parameters", 100), offload])
        result = budget_scenarios(declared, [{"name": "capacity", "memory_capacity_bytes": 150}])
        ledger = result["ledger"]
        self.assertEqual(ledger["max_over_ranks_known_bytes"], 100)
        self.assertEqual(ledger["max_over_ranks_known_host_bytes"], 2 * 10**9)
        rows = ledger["per_rank"]["0"]["phases"]
        self.assertEqual([row["known_host_bytes"] for row in rows], [2 * 10**9, 2 * 10**9, 0])
        self.assertEqual(rows[0]["by_category"]["activations"], 0)
        rank = result["capacity_scenarios"][0]["per_rank"]["0"]
        self.assertEqual(rank["capacity_verdict"], "conditional_fits_capacity")
        self.assertFalse(rank["host_capacity_evaluated"])

    def test_unknown_host_component_is_preserved_without_invented_gpu_bytes(self):
        offload = component("unknown_host", "activations", None)
        offload["device"] = "cpu"
        row = component_ledger(spec([offload]))["per_rank"]["0"]["phases"][0]
        self.assertEqual(row["required_memory_bytes"], 0)
        self.assertEqual(row["known_host_bytes"], 0)
        self.assertEqual(len(row["unknown_host_components"]), 1)
        self.assertEqual(row["unknown_components"], [])

    def test_heterogeneous_ranks_and_rank_local_storage_ids(self):
        declared = spec([
            component("params", "parameters", 100, rank=0),
            component("params", "parameters", 300, rank=1),
            component("pipeline", "pipeline_buffers", 80, start="backward", rank=0),
        ], ranks=(0, 1))
        ledger = component_ledger(declared)
        self.assertEqual(ledger["per_rank"]["0"]["max_known_bytes"], 180)
        self.assertEqual(ledger["per_rank"]["1"]["max_known_bytes"], 300)
        self.assertEqual(ledger["required_max_over_ranks_bytes"], 300)
        self.assertEqual([row["max_per_rank_known_bytes"] for row in ledger["max_per_rank_by_phase"]], [300]*3)

    def test_shape_uses_storage_dtype_not_compute_precision(self):
        item = component("weight", "parameters", 0)
        del item["size_bytes"]
        item.update(shape=[3, 5], dtype="float32", compute_precision="fp8")
        declared = spec([item])
        self.assertEqual(component_ledger(declared)["max_over_ranks_known_bytes"], 60)
        item["dtype"] = "float8_e4m3fn"
        self.assertEqual(component_ledger(declared)["max_over_ranks_known_bytes"], 15)
        item["dtype"] = "fp8"
        with self.assertRaisesRegex(ValueError, "storage dtype"):
            component_ledger(declared)

    def test_missing_dtype_cannot_be_inferred_from_precision(self):
        item = component("weight", "parameters", 0)
        del item["size_bytes"]
        item.update(shape=[15], precision="bf16")
        with self.assertRaises(ValueError):
            component_ledger(spec([item]))

    def test_unknown_component_is_unknown_only_while_live(self):
        declared = spec([component("workspace", "workspace", None, start="backward", end="optimizer")])
        ledger = component_ledger(declared)
        self.assertEqual([row["required_memory_bytes"] for row in ledger["per_rank"]["0"]["phases"]], [0, None, 0])
        self.assertIsNone(ledger["required_max_over_ranks_bytes"])
        self.assertFalse(ledger["coverage_complete"])

    def test_missing_coverage_stays_unknown_even_when_below_capacity(self):
        declared = spec([component("params", "parameters", 100)])
        del declared["coverage"]["workspace"]
        result = budget_scenarios(declared, [{"name": "capacity", "memory_capacity_bytes": 1000}])
        rank = result["capacity_scenarios"][0]["per_rank"]["0"]
        self.assertIsNone(rank["required_peak_bytes"])
        self.assertEqual(rank["capacity_verdict"], "unknown")
        self.assertFalse(rank["gpu_fit_validated"])
        self.assertIn("workspace", result["ledger"]["per_rank"]["0"]["phases"][0]["uncovered_categories"])

    def test_incomplete_known_overflow_is_explicitly_conditional(self):
        declared = spec([component("params", "parameters", 100)])
        declared["coverage"] = {"parameters": "complete"}
        result = budget_scenarios(declared, [{"name": "capacity", "memory_capacity_bytes": 90}])
        rank = result["capacity_scenarios"][0]["per_rank"]["0"]
        self.assertEqual(rank["capacity_verdict"], "declared_known_components_exceed_capacity")
        self.assertIsNone(rank["required_peak_bytes"])
        self.assertEqual(rank["known_headroom_bytes"], -10)
        self.assertFalse(rank["gpu_fit_validated"])

    def test_rank_specific_coverage_does_not_hide_incomplete_rank(self):
        declared = spec(ranks=(0, 1))
        declared["coverage_by_rank"] = {"1": {"external": "unknown"}}
        ledger = component_ledger(declared)
        self.assertEqual(ledger["per_rank"]["0"]["required_peak_bytes"], 0)
        self.assertIsNone(ledger["per_rank"]["1"]["required_peak_bytes"])
        self.assertIsNone(ledger["required_max_over_ranks_bytes"])

    def test_allocator_cache_is_unused_reserved_only(self):
        cache = component("cache", "allocator_cache", 20)
        declared = spec([component("params", "parameters", 100), cache])
        with self.assertRaisesRegex(ValueError, "unused_reserved"):
            component_ledger(declared)
        cache["accounting"] = "reserved"
        with self.assertRaises(ValueError):
            component_ledger(declared)
        cache["accounting"] = "unused_reserved"
        self.assertEqual(component_ledger(declared)["required_max_over_ranks_bytes"], 120)

    def test_hardware_capacity_changes_do_not_change_component_bytes(self):
        declared = spec([component("params", "parameters", 150)])
        before = copy.deepcopy(declared)
        result = budget_scenarios(declared, [
            {"name": "large", "memory_capacity_bytes": 200},
            {"name": "small", "memory_capacity_bytes": 100},
        ])
        rows = [scenario["per_rank"]["0"] for scenario in result["capacity_scenarios"]]
        self.assertEqual([row["known_bytes"] for row in rows], [150, 150])
        self.assertEqual(rows[0]["capacity_verdict"], "conditional_fits_capacity")
        self.assertEqual(rows[1]["capacity_verdict"], "declared_known_components_exceed_capacity")
        self.assertEqual(declared, before)

    def test_decimal_and_binary_capacity_units_are_distinct(self):
        self.assertEqual(capacity_bytes({"name": "decimal", "memory_capacity": {"value": 1.5, "unit": "GB"}}), 1500000000)
        self.assertEqual(capacity_bytes({"name": "binary", "memory_capacity": {"value": "1.5", "unit": "GiB"}}), 1610612736)
        for invalid in ({"value": 1, "unit": "gb"}, {"value": .5, "unit": "B"}, {"value": "NaN", "unit": "GB"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                capacity_bytes({"name": "bad", "memory_capacity": invalid})

    def test_observation_or_residual_cannot_be_labeled_a_prediction(self):
        declared = spec([component("params", "parameters", 100)])
        declared["source"] = "gpu_observation"
        with self.assertRaises(ValueError):
            component_ledger(declared)
        declared["source"] = "independent_component_spec"
        for kind in ("gpu_observation", "peak_fit", "residual"):
            declared["components"][0]["provenance"]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                component_ledger(declared)

    def test_invalid_lifetime_and_unsupported_coverage_rejected(self):
        item = component("params", "parameters", 100, start="optimizer", end="forward")
        with self.assertRaises(ValueError):
            component_ledger(spec([item]))
        declared = spec()
        declared["coverage"]["parameters"] = "estimated_complete"
        with self.assertRaises(ValueError):
            component_ledger(declared)

    def test_scalar_and_empty_shapes_are_exact(self):
        scalar = component("scalar", "optimizer", 0)
        del scalar["size_bytes"]
        scalar.update(shape=[], dtype="float32")
        empty = {**scalar, "name": "empty", "storage_id": "empty", "shape": [0, 99]}
        self.assertEqual(component_ledger(spec([scalar, empty]))["max_over_ranks_known_bytes"], 4)

    def test_cli_supports_multiple_capacity_only_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "spec.json").write_text(json.dumps(spec([component("params", "parameters", 100)])))
            for name, size in (("a", 90), ("b", 110)):
                (folder / f"{name}.json").write_text(json.dumps({"name": name, "memory_capacity_bytes": size}))
            main([str(folder / "spec.json"), "--hardware", str(folder / "a.json"),
                  "--hardware", str(folder / "b.json"), "--output", str(folder / "budget.json")])
            result = json.loads((folder / "budget.json").read_text())
            self.assertEqual(len(result["capacity_scenarios"]), 2)
            self.assertEqual(result["ledger"]["max_over_ranks_known_bytes"], 100)


if __name__ == "__main__":
    unittest.main()
