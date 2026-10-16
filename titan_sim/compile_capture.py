"""Explicit AOTAutograd graph execution for CPU semantic storage diagnostics.

Dynamo and AOTAutograd really run; the resulting ATen graphs execute through
the dispatcher. This backend does not run Inductor or predict its GPU buffers.
Compiler tracing is hidden from the observer using PyTorch's supported mode
contract, while actual graph executions retain normal storage observation.
"""
from __future__ import annotations

import contextlib
from unittest.mock import patch

from .capture import TensorCapture, _tensors
from .checkpoint import execution_phase


class AOTGraphCapture(TensorCapture):
    """Capture executed AOT graphs separately from compiler tracing metadata.

    Pass ``backend=capture.backend`` to torch.compile explicitly, or use the
    opt-in compile_scope when upstream functions bind torch.compile internally.
    CPU FakeTensors support metadata integration; real CPU tensors support
    numerical regression tests. The resulting allocation lifetimes describe
    this ATen graph backend, not the original CUDA/Inductor backend.
    """

    @classmethod
    def ignore_compile_internals(cls):
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compilation = dict(
            backend="aot_autograd_aten_graph_execution",
            original_backend_preserved=False,
            compiler_tracing_is_execution=False,
            cuda_dispatch_equivalent=False,
            compiled_storage_lifetimes_equivalent=False,
            graphs=[], executions=[], compile_requests=[],
            limitations=[
                "ATen graph execution preserves AOTAutograd saved values and native checkpoint policy.",
                "Graph intermediates belong to the selected semantic backend; they are not Inductor buffers.",
                "CUDA fusion, workspaces, allocator behavior and generated-kernel execution are not covered.",
            ],
        )
        self._backend = None

    def backend(self, graph, example_inputs):
        """Dynamo backend entry point; specialize real forward/backward graphs."""
        if any(tensor.device.type != "cpu" for tensor in _tensors(example_inputs)):
            raise NotImplementedError("AOTGraphCapture supports CPU semantic execution only")
        if self._backend is None:
            from torch._dynamo.backends.common import aot_autograd
            self._backend = aot_autograd(
                fw_compiler=lambda gm, inputs: self._compiler("forward", gm, inputs),
                bw_compiler=lambda gm, inputs: self._compiler("backward", gm, inputs),
                inference_compiler=lambda gm, inputs: self._compiler("inference", gm, inputs),
            )
        return self._backend(graph, example_inputs)

    def _compiler(self, kind, graph, example_inputs):
        from torch._functorch.aot_autograd import make_boxed_func
        graph_id = len(self.compilation["graphs"])
        self.compilation["graphs"].append(dict(
            id=graph_id, kind=kind, code=graph.code,
            operations=[str(node.target) for node in graph.graph.nodes if node.op == "call_function"],
            provenance="actual AOTAutograd compiler callback; graph metadata, not execution",
        ))

        def run(*args):
            inputs = _tensors(args)
            if any(tensor.device.type != "cpu" for tensor in inputs):
                raise NotImplementedError("AOTGraphCapture supports CPU semantic execution only")
            record = dict(graph_id=graph_id, graph_kind=kind,
                          execution_phase=execution_phase(self._phase, in_backward=self._tracker.is_bw),
                          first_event=len(self.events),
                          input_storage_ids=[self.register_tensor(tensor) for tensor in inputs])
            self.compilation["executions"].append(record)
            try:
                output = graph.forward(*args)
                # Forward outputs include AOT's saved values for backward.
                # Register existing storages; never manufacture a save lifetime.
                record["output_storage_ids"] = [self.register_tensor(tensor) for tensor in _tensors(output)]
                record["status"] = "complete"
                return output
            except Exception:
                record["status"] = "failed"
                raise
            finally:
                record["last_event_exclusive"] = len(self.events)

        return make_boxed_func(run)

    @contextlib.contextmanager
    def compile_scope(self):
        """Explicitly replace upstream's default backend, preserving region flags.

        Installation patches torch.compile process-wide and must not overlap another
        session in a different thread. This scope does not enter the capture mode;
        enter ``capture`` around runtime execution separately.
        """
        import torch
        original = torch.compile

        def compile(model=None, **kwargs):
            requested = kwargs.get("backend", "inductor")
            if requested != "inductor":
                raise NotImplementedError("Only replacement of the default Inductor backend is supported")
            if kwargs.get("disable", False):
                raise ValueError("AOT graph capture requires compilation enabled")
            if kwargs.get("options") or kwargs.get("mode") not in (None, "default"):
                raise NotImplementedError("Inductor-specific compile options cannot be preserved by the AOT backend")
            self.compilation["compile_requests"].append(dict(
                function=getattr(model, "__qualname__", None),
                requested_backend="inductor", selected_backend=self.compilation["backend"],
                fullgraph=kwargs.get("fullgraph", False), dynamic=kwargs.get("dynamic"),
            ))
            kwargs["backend"] = self.backend
            return original(model, **kwargs)

        with patch.object(torch, "compile", compile):
            yield self.compilation

    def document(self, *, metadata=None):
        return super().document(metadata={**(metadata or {}), "aot_graph_capture": self.compilation})
