"""Storage aliases/lifetime and genuine autograd coverage without a GPU."""
import gc
import unittest

try:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from titan_sim.capture import TensorCapture
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for dispatcher tests")
class CaptureTests(unittest.TestCase):
    def test_view_shares_storage_and_last_owner_controls_release(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(16)
            alias = tensor.view(4, 4)
            sid = capture.register_tensor(tensor)
            self.assertEqual(sid, capture.register_tensor(alias))
            del tensor
            gc.collect()
            capture.sweep()
            self.assertIsNone(capture.storages[sid]["release_event"])
            del alias
            gc.collect()
            capture.sweep()
            self.assertIsNotNone(capture.storages[sid]["release_event"])
        alloc = [e for e in capture.events if e["kind"] == "allocate" and e["storage_id"] == sid]
        self.assertEqual(len(alloc), 1)
        self.assertEqual(alloc[0]["size_bytes"], 64)

    def test_storage_resize_releases_even_with_live_tensor_alias(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(16)
            sid = capture.register_tensor(tensor)
            tensor.untyped_storage().resize_(0)
            capture.sweep()
            self.assertIsNotNone(capture.storages[sid]["release_event"])
            tensor.untyped_storage().resize_(64)
            replacement = capture.register_tensor(tensor)
            self.assertNotEqual(sid, replacement)
            self.assertEqual(capture.storages[replacement]["size_bytes"], 64)

    @staticmethod
    def _simulate(events):
        from titan_sim.runtime import simulate
        hardware = {"name": "test", "peak_flops": {"float32": 1e6},
                    "memory_bandwidth_bytes_s": 1e6, "memory_capacity_bytes": 4096,
                    "network_bandwidth_bytes_s": 1e6, "host_device_bandwidth_bytes_s": 1e6}
        return {event["event_id"]: event for event in simulate(events, hardware)["timeline"]}

    def test_cpu_scalar_read_does_not_drain_unrelated_communication(self):
        with TensorCapture() as capture:
            tensor = torch.tensor(1.0)
            communication = capture._event("collective", "long_communication", stream="comm",
                                           device="cuda:0", bytes_read=100)
            self.assertEqual(tensor.item(), 1.0)
            next_submit = capture._event("compute", "next_submit", stream="next", device="cuda:0")
        scalar = next(event for event in capture.events if "_local_scalar_dense" in event["name"])
        self.assertEqual(scalar["kind"], "compute")
        self.assertTrue(scalar["metadata_only"])
        timeline = self._simulate(capture.events)
        self.assertEqual(timeline[communication["event_id"]]["end_us"], 100)
        self.assertEqual(timeline[next_submit["event_id"]]["submit_us"], 0)

    def test_blocking_copy_waits_for_transfer_before_next_submission(self):
        for non_blocking in (False, True):
            with self.subTest(non_blocking=non_blocking):
                with FakeTensorMode(), TensorCapture() as capture:
                    tensor = torch.empty(16, device="cuda")
                    copied = tensor.to("cpu", non_blocking=non_blocking)
                    next_submit = capture._event("compute", "next_submit", stream="next", device="cuda:0")
                timeline = self._simulate(capture.events)
                self.assertEqual(timeline[next_submit["event_id"]]["submit_us"], 0 if non_blocking else 64)
                syncs = [event for event in capture.events if event["kind"] == "host_sync"]
                self.assertEqual(len(syncs), 0 if non_blocking else 1)
                if syncs:
                    self.assertEqual(syncs[0]["sync_scope"], "event")

    def test_signature_distinguishes_scalar_controls(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(8)
            torch.add(tensor, tensor, alpha=1)
            torch.add(tensor, tensor, alpha=2)
        adds = [event for event in capture.events if event["name"] == "aten.add.Tensor"]
        self.assertNotEqual(adds[0]["signature"], adds[1]["signature"])
        self.assertEqual(adds[0]["signature"]["status"], "complete")

    def test_empty_and_device_query_are_metadata_only(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(16)
            torch.ops.prim.device.default(tensor)
        operations = [event for event in capture.events if event["kind"] == "compute"]
        self.assertTrue(operations)
        self.assertTrue(all(event["metadata_only"] for event in operations))
        self.assertTrue(all(event["bytes_read"] == event["bytes_write"] == 0 for event in operations))

    def test_cuda_to_cpu_copy_retains_host_device_traffic(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(16, device="cuda")
            tensor.cpu()
        copy = next(event for event in capture.events if event["name"] == "aten._to_copy.default")
        self.assertEqual(copy["traffic_domain"], "host_device")
        self.assertTrue(copy["device"].startswith("cuda"))
        self.assertEqual(copy["host_device_transfer_bytes"], 64)

    def test_disjoint_views_contribute_both_read_regions(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(8)
            left, right = tensor[:4], tensor[4:]
            torch.cat([left, right])
        cat = next(e for e in capture.events if e["name"].startswith("aten.cat"))
        self.assertEqual(cat["bytes_read"], 32)

    def test_tensor_resize_fails_without_invalid_trace(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(8)
            with self.assertRaisesRegex(NotImplementedError, "resize_"):
                tensor.resize_(16)
        self.assertFalse(any("resize_" in e["name"] for e in capture.events))

    def test_void_mutation_reports_writes_once_per_storage(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(8)
            torch.ops.aten._foreach_add_.Scalar([tensor], 1)
        operation = next(e for e in capture.events if "_foreach_add_" in e["name"])
        self.assertEqual(operation["bytes_write"], 32)
        self.assertEqual(len(operation["writes"]), 1)
        self.assertNotIn("host_time_us", operation)
        self.assertIn("capture_wall_us", operation)

    def test_backward_and_optimizer_are_metadata_only(self):
        with FakeTensorMode(allow_fallback_kernels=False), TensorCapture() as capture:
            model = torch.nn.Linear(8, 4)
            capture.register_model(model)
            optimizer = torch.optim.AdamW(model.parameters(), fused=True)
            with capture.phase("forward"):
                loss = model(torch.empty(2, 8)).square().sum()
            with capture.phase("backward"):
                loss.backward()
                capture.register_model(model)
            with capture.phase("optimizer"):
                optimizer.step()
                capture.register_optimizer(optimizer)
            self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
        doc = capture.document()
        self.assertTrue(any(e["flops"] > 0 for e in doc["events"]))
        self.assertTrue(any(s["action"] == "save" for s in doc["saved_tensors"]))
        self.assertTrue(any(s["category"] == "optimizer" for s in doc["storages"]))
        self.assertTrue(any(s["category"] == "gradients" for s in doc["storages"]))
        self.assertEqual(doc["source"], "cpu_projection")

    def test_saved_activation_survives_python_reference(self):
        with FakeTensorMode(), TensorCapture() as capture:
            tensor = torch.empty(4, requires_grad=True)
            activation = tensor.sin()
            sid = capture.register_tensor(activation)
            loss = activation.square().sum()
            del activation
            gc.collect()
            capture.sweep()
            self.assertIsNone(capture.storages[sid]["release_event"])
            loss.backward()
            del loss
            gc.collect()
            capture.sweep()
            self.assertIsNotNone(capture.storages[sid]["release_event"])


if __name__ == "__main__":
    unittest.main()
