"""Observe actual checkpoint replay while preserving native save semantics."""
import functools
import gc
import unittest

try:
    import torch
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    from titan_sim.capture import TensorCapture
    from titan_sim.checkpoint import checkpoint_phase_tracking, execution_phase
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for checkpoint tests")
class CheckpointCaptureTests(unittest.TestCase):
    def test_observer_preserves_aot_checkpoint_context_contract(self):
        # This checks Dynamo/AOTAutograd integration on CPU, not a GPU
        # allocation plan or an Inductor fusion prediction.
        def policy(*args, **kwargs):
            return CheckpointPolicy.PREFER_RECOMPUTE

        contexts = functools.partial(create_selective_checkpoint_contexts, policy)

        def function(value):
            return checkpoint(lambda tensor: tensor.sin().square(), value,
                              use_reentrant=False, context_fn=contexts)

        parameter = torch.randn(8, requires_grad=True)
        with checkpoint_phase_tracking():
            compiled = torch.compile(function, backend="aot_eager", fullgraph=True)
            output = compiled(parameter)
            output.sum().backward()
        self.assertTrue(torch.allclose(output, parameter.sin().square()))
        self.assertTrue(torch.allclose(parameter.grad, 2 * parameter.sin() * parameter.cos()))
        self.assertEqual(execution_phase("setup"), "setup")

    def test_native_recomputation_and_gradients_with_optimizer(self):
        records = []
        visits = []
        parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 8))
        initial = parameter.detach().clone()
        optimizer = torch.optim.AdamW([parameter], lr=0.01)

        def policy(ctx, op, *args, **kwargs):
            visits.append(execution_phase("forward_backward_step_0", in_backward=True))
            return CheckpointPolicy.PREFER_RECOMPUTE

        def function(value):
            records.append(execution_phase("forward_backward_step_0", in_backward=True))
            return value.sin().square()

        contexts = functools.partial(create_selective_checkpoint_contexts, policy)
        with checkpoint_phase_tracking(), TensorCapture() as capture:
            with capture.phase("forward_backward_step_0"):
                loss = checkpoint(function, parameter, use_reentrant=False, context_fn=contexts).sum()
                loss.backward()
            self.assertTrue(torch.allclose(parameter.grad, 2 * initial.sin() * initial.cos()))
            self.assertEqual(execution_phase("forward_backward_step_0", in_backward=True), "backward")
            with capture.phase("optimizer_step_0"):
                optimizer.step()
                capture.register_optimizer(optimizer)
                self.assertEqual(execution_phase("optimizer_step_0"), "optimizer")
        self.assertEqual(records, ["forward", "recompute"])
        self.assertEqual(set(visits), {"forward"})
        sine_phases = [event["execution_phase"] for event in capture.events
                       if event["name"] == "aten.sin.default" and event["phase"] == "forward_backward_step_0"]
        self.assertEqual(sine_phases, ["forward", "recompute"])
        cosine = next(event for event in capture.events if event["name"] == "aten.cos.default")
        self.assertEqual(cosine["execution_phase"], "backward")
        optimizer_events = [event for event in capture.events if event["phase"] == "optimizer_step_0"]
        self.assertTrue(optimizer_events)
        self.assertEqual({event["execution_phase"] for event in optimizer_events}, {"optimizer"})
        self.assertTrue(any(entry["category"] == "optimizer" for entry in capture.storages.values()))
        self.assertFalse(torch.equal(initial, parameter.detach()))

    def test_cached_storage_and_alias_lifetime_are_preserved(self):
        contexts = functools.partial(create_selective_checkpoint_contexts, [torch.ops.aten.mm.default])
        saved_storage = []
        parameter = torch.randn(4, 4, requires_grad=True)
        with checkpoint_phase_tracking(), TensorCapture() as capture:
            def function(value):
                product = value @ value
                saved_storage.append(capture.register_tensor(product))
                return product.sin()

            output = checkpoint(function, parameter, use_reentrant=False, context_fn=contexts)
            sid = saved_storage[0]
            gc.collect()
            capture.sweep()
            self.assertIsNone(capture.storages[sid]["release_event"])
            output.sum().backward()
            self.assertEqual(saved_storage, [sid, sid])
            del output
            gc.collect()
            capture.sweep()
            self.assertIsNotNone(capture.storages[sid]["release_event"])

    def test_native_contexts_and_state_restored_after_error(self):
        contexts = create_selective_checkpoint_contexts(lambda *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE)
        mode_type = type(contexts[0])
        original_enter = mode_type.__enter__
        with checkpoint_phase_tracking():
            with contexts[0]:
                self.assertEqual(execution_phase("setup"), "forward")
                with self.assertRaisesRegex(RuntimeError, "probe"):
                    with contexts[1]:
                        self.assertEqual(execution_phase("setup"), "recompute")
                        raise RuntimeError("probe")
                self.assertEqual(execution_phase("setup"), "forward")
            self.assertEqual(execution_phase("setup"), "setup")
        self.assertIs(mode_type.__enter__, original_enter)
        self.assertEqual(execution_phase("setup"), "setup")

    def test_nested_forward_context_during_replay_remains_recompute(self):
        policy = lambda *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE
        outer = create_selective_checkpoint_contexts(policy)
        inner = create_selective_checkpoint_contexts(policy)
        with checkpoint_phase_tracking():
            with outer[1]:
                with checkpoint_phase_tracking(), inner[0]:
                    self.assertEqual(execution_phase("forward_backward_step_0"), "recompute")
                self.assertEqual(execution_phase("forward_backward_step_0"), "recompute")
            self.assertEqual(execution_phase("setup"), "setup")


if __name__ == "__main__":
    unittest.main()
