import unittest

try:
    import torch
    from titan_sim.capture import TensorCapture
    from titan_sim.fake_device import FakeCudaRuntime, fake_tensor_parameter_swaps
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class FakeDeviceTests(unittest.TestCase):
    def test_fake_parameter_swap_preserves_identity_and_materializes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode(), fake_tensor_parameter_swaps() as report:
            with torch.device("meta"):
                model = torch.nn.Linear(8, 4)
            parameter = model.weight
            model.to_empty(device="cpu")
            self.assertIs(parameter, model.weight)
            self.assertEqual(model.weight.device.type, "cpu")
            self.assertGreater(report["memo_entries_evicted"], 0)

    def test_fake_parameter_swap_still_rejects_external_weakref(self):
        import weakref
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode(), fake_tensor_parameter_swaps():
            model = torch.nn.Linear(8, 4)
            reference = weakref.ref(model.weight)
            with self.assertRaisesRegex(RuntimeError, "swap"):
                model.to(torch.bfloat16)
            self.assertIs(reference(), model.weight)

    def test_event_synchronize_does_not_join_unrelated_current_stream(self):
        capture = TensorCapture()
        properties = dict(name="logical GB200", total_memory=197_897_617_408,
                          major=10, minor=0, multi_processor_count=152)
        with FakeCudaRuntime(capture, properties=properties):
            producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
            with torch.cuda.stream(producer):
                event = torch.cuda.Event()
                event.record()
            with torch.cuda.stream(consumer):
                unrelated = capture._event("compute", "unrelated")
                event.synchronize()
                sync = capture.events[-1]
                self.assertEqual(sync["sync_scope"], "event")
                self.assertIn(event.recorded, sync["deps"])
                self.assertNotIn(unrelated["event_id"], sync["deps"])

    def test_cross_stream_waits_have_dependency_edges(self):
        capture = TensorCapture()
        properties = dict(name="logical GB200", total_memory=197_897_617_408,
                          major=10, minor=0, multi_processor_count=152)
        with FakeCudaRuntime(capture, properties=properties):
            producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
            with torch.cuda.stream(producer):
                produced = capture._event("compute", "produce")
                event = torch.cuda.Event()
                event.record()
            with torch.cuda.stream(consumer):
                independent = capture._event("compute", "independent")
                self.assertNotIn(produced["event_id"], independent["deps"])
                consumer.wait_event(event)
                self.assertIn(event.recorded, capture.events[-1]["deps"])
                consumer.wait_stream(producer)
                self.assertIn(event.recorded, capture.events[-1]["deps"])

    def test_stream_scope_restores_runtime_on_exception(self):
        original = torch.cuda.current_stream
        capture = TensorCapture()
        properties = dict(name="logical GB200", total_memory=180_000_000_000,
                          major=10, minor=0, multi_processor_count=148)
        with self.assertRaisesRegex(ValueError, "test"):
            with FakeCudaRuntime(capture, properties=properties):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    self.assertIs(torch.cuda.current_stream(), stream)
                    event = torch.cuda.Event()
                    event.record()
                    stream.wait_event(event)
                self.assertEqual(len(capture.events), 2)
                raise ValueError("test")
        self.assertIs(torch.cuda.current_stream, original)
