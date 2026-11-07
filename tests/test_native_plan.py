import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

from titan_sim.native_plan import extract_plan, merge_plans, read_plan, seal_plan, validate_plan, write_plan

try:
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
except ImportError:
    torch = None


@dataclass
class Modules:
    fw: object
    full_bw: object = None
    bw_di: object = None
    bw_dw: object = None
    unshard: object = None
    reduce_grad: object = None


@dataclass
class Meta:
    num_user_outputs: int = 1
    num_saved_for_backward: int = 1
    num_fw_param_inputs: int = 1
    partition: object = None


class PlanEnvelopeTests(unittest.TestCase):
    def test_read_does_not_import_torch(self):
        code = "import sys; import titan_sim.native_plan; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_missing_graph_api_is_explicit(self):
        plan = extract_plan(object(), {0: object()}, provenance={"recipe": "exact"})
        self.assertEqual(plan["status"], "incomplete")
        self.assertFalse(plan["graphs"])
        self.assertTrue(any("no supported action table" in g for g in plan["unresolved"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            write_plan(path, plan)
            self.assertEqual(read_plan(path), plan)
        changed = copy.deepcopy(plan)
        changed["provenance"]["recipe"] = "changed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_plan(changed)

    def test_unknown_action_is_preserved_as_unsupported(self):
        plan = extract_plan(NS(pipeline_order_with_comms={0: [object()]}))
        self.assertIn("unsupported_type", plan["schedule"]["rank_actions"]["0"][0])
        self.assertTrue(any("unsupported native schedule action" in g for g in plan["unresolved"]))


@unittest.skipIf(torch is None, "optional native torch CPU environment unavailable")
class NativeObjectTests(unittest.TestCase):
    def fixture(self, stage=0, rank=0, *, compiled=False):
        from torch.distributed.pipelining.schedules import _Action, _ComputationType
        x = torch.randn(2, 3)
        gm = make_fx(lambda a: (a.sin(), a.view(-1), a.view(-1), None), tracing_mode="fake")(x)
        # Use actual dataclass as upstream GraphMeta does.
        @dataclass
        class Partition:
            saved_for_backward_names: tuple = ("view",)
            bwd_input_names: tuple = ("view",)
        bundle = NS(modules=Modules(gm), meta=Meta(partition=Partition()), compiled=compiled)
        state = NS(sharded_param_values=[x], buffer_values=[])
        native_stage = NS(stage_index=stage, graphs=bundle, state=state,
                          stage_index_to_group_rank={0: 0, 1: 1})
        schedule = NS(_stages=[native_stage], _n_microbatches=2,
                      pipeline_order_with_comms={
                          0: [_Action(0, _ComputationType.FORWARD, 0)],
                          1: [_Action(1, _ComputationType.FORWARD, 0)]})
        runtime = NS(schedule=schedule, stage_graphs={stage: bundle}, overlap_graphs={})
        return runtime, gm

    def test_actual_fx_alias_storage_shapes_and_schedule_actions(self):
        runtime, gm = self.fixture()
        plan = extract_plan(runtime, provenance={"recipe": "submitted"})
        graph = plan["graphs"]["stage0.fw"]
        nodes = {n["name"]: n for n in graph["nodes"]}
        placeholder = next(n for n in nodes.values() if n["op"] == "placeholder")
        views = [n for n in nodes.values() if n["target"] == "aten.view.default"]
        sin = next(n for n in nodes.values() if n["target"] == "aten.sin.default")
        self.assertEqual(placeholder["outputs"][0]["shape"], [2, 3])
        self.assertEqual(placeholder["outputs"][0]["storage_bytes"], 24)
        self.assertEqual(views[0]["outputs"][0]["storage_id"], placeholder["outputs"][0]["storage_id"])
        self.assertNotEqual(sin["outputs"][0]["storage_id"], placeholder["outputs"][0]["storage_id"])
        self.assertEqual(graph["outputs"][-1], None)
        self.assertEqual(len(graph["outputs"]), 4)
        self.assertEqual(plan["schedule"]["rank_actions"]["0"][0]["computation_type"], "F")
        self.assertEqual(plan["stages"]["0"]["metadata"]["partition"]["saved_for_backward_names"], ["view"])
        self.assertEqual(plan["stages"]["0"]["state"]["sharded_param_values"][0]["device"], "cpu")
        self.assertTrue(any(v["sha256"] for v in plan["source_files"].values()))
        validate_plan(plan)
        json.dumps(plan, allow_nan=False)

    def test_real_mode_snapshot_alias_loss_is_explicit(self):
        runtime, _ = self.fixture()
        runtime.stage_graphs[0].modules.fw = make_fx(lambda a: a.view(-1))(torch.ones(2, 3))
        plan = extract_plan(runtime)
        graph = plan["graphs"]["stage0.fw"]
        storages = [n["outputs"][0]["storage_id"] for n in graph["nodes"] if n["outputs"]]
        # Some future Torch versions may preserve real-trace metadata aliases.
        if len(set(storages)) > 1:
            self.assertTrue(any("alias schema conflicts" in g for g in plan["unresolved"]))

    def test_compiled_graph_is_not_promoted_to_execution_graph(self):
        runtime, _ = self.fixture(compiled=True)
        plan = extract_plan(runtime)
        self.assertTrue(any("compiled execution correspondence" in g for g in plan["unresolved"]))

    def test_real_native_gpipe_generated_order_and_lowered_communications(self):
        from torch.distributed.pipelining.schedules import ScheduleGPipe, _add_send_recv
        # Exercise native schedule generation without constructing a process group.
        schedule = object.__new__(ScheduleGPipe)
        schedule._num_stages = 2
        schedule._n_microbatches = 2
        order = schedule._get_pipeline_order()
        runtime, _ = self.fixture()
        runtime.schedule.pipeline_order_with_comms = _add_send_recv(order, lambda stage: stage, 2)
        plan = extract_plan(runtime)
        actions = plan["schedule"]["rank_actions"]
        self.assertTrue(any(a and a["computation_type"] == "SEND_F" for a in actions["0"]))
        self.assertTrue(any(a and a["computation_type"] == "RECV_F" for a in actions["1"]))

    def test_rank_merge_namespaces_storage_and_rejects_mismatch(self):
        a, _ = self.fixture(0, 0)
        b, _ = self.fixture(1, 1)
        p0 = extract_plan(a, provenance={"recipe": "same"}, rank=0)
        p1 = extract_plan(b, provenance={"recipe": "same"}, rank=1)
        merged = merge_plans([p0, p1])
        self.assertEqual(set(merged["stages"]), {"0", "1"})
        s0 = merged["stages"]["0"]["state"]["sharded_param_values"][0]["storage_id"]
        s1 = merged["stages"]["1"]["state"]["sharded_param_values"][0]["storage_id"]
        self.assertNotEqual(s0, s1)
        validate_plan(merged)
        with self.assertRaisesRegex(ValueError, "unique"):
            merge_plans([p0, p0])
        other = extract_plan(b, provenance={"recipe": "changed"}, rank=1)
        with self.assertRaisesRegex(ValueError, "provenance"):
            merge_plans([p0, other])

    def test_native_nonfinite_mask_arguments_and_metadata_have_exact_tags(self):
        for scalar, label in ((float("-inf"), "-inf"), (float("inf"), "inf"), (float("nan"), "nan")):
            with self.subTest(label=label):
                runtime, _ = self.fixture()
                gm = make_fx(lambda a: a.masked_fill(a < 0, scalar), tracing_mode="fake")(torch.ones(2, 3))
                runtime.stage_graphs[0].modules.fw = gm
                runtime.stage_graphs[0].meta.partition = {"native_scalar": scalar, "finite_scalar": 1.25}
                plan = extract_plan(runtime)
                masked = next(n for n in plan["graphs"]["stage0.fw"]["nodes"]
                              if n["target"] == "aten.masked_fill.Scalar")
                self.assertEqual(masked["args"][-1], {"float": label})
                metadata = plan["stages"]["0"]["metadata"]["partition"]
                self.assertEqual(metadata["native_scalar"], {"float": label})
                self.assertEqual(metadata["finite_scalar"], 1.25)
                validate_plan(plan)
                self.assertEqual(json.loads(json.dumps(plan, allow_nan=False)), plan)

    def test_symbolic_shapes_remain_unresolved_without_concretization(self):
        runtime, _ = self.fixture()
        runtime.stage_graphs[0].modules.fw = make_fx(lambda a: a.sin(), tracing_mode="symbolic")(torch.ones(2, 3))
        plan = extract_plan(runtime)
        self.assertTrue(any("symbolic or unavailable integer" in g for g in plan["unresolved"]))
        first = plan["graphs"]["stage0.fw"]["nodes"][0]["outputs"][0]
        self.assertIn(None, first["shape"])
        self.assertIsNone(first["logical_bytes"])

    def test_plain_joint_alias_matches_native_full_forward_backward_contract(self):
        runtime, gm = self.fixture()
        runtime.stage_graphs[0] = NS(traced=NS(gm=gm), num_param_grads=1)
        plan = extract_plan(runtime)
        roles = plan["stages"]["0"]["graphs"]
        self.assertEqual(roles["FULL_FORWARD_BACKWARD"], roles["joint"])
        self.assertNotIn("FORWARD_BACKWARD_NOGRADACCUM", roles)
        self.assertEqual(plan["graphs"][roles["joint"]]["role"], "joint")

    def test_merge_rejects_incomplete_capture_and_stale_graph_hash(self):
        runtime, _ = self.fixture()
        plan = extract_plan(runtime)
        short = copy.deepcopy(plan)
        short["capture_status"] = "incomplete"
        short = seal_plan(short)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_plans([short])
        changed = copy.deepcopy(plan)
        changed.pop("plan_sha256")
        changed["graphs"]["stage0.fw"]["nodes"][0]["target"] = "changed"
        with self.assertRaisesRegex(ValueError, "graph .* checksum"):
            validate_plan(changed)

    def test_mutation_schema_and_arguments_are_extracted(self):
        x = torch.ones(3)
        gm = make_fx(lambda a: a.add_(2.0))(x)
        runtime, _ = self.fixture()
        runtime.stage_graphs[0].modules.fw = gm
        plan = extract_plan(runtime)
        add = next(n for n in plan["graphs"]["stage0.fw"]["nodes"] if n["target"] == "aten.add_.Tensor")
        self.assertEqual(add["mutates_arguments"], ["self"])
        self.assertEqual(add["args"][1], 2.0)


if __name__ == "__main__":
    unittest.main()
