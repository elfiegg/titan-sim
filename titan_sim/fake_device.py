"""Scoped Python CUDA control-plane emulation for metadata execution.

This cannot replace C++ CUDA DeviceGuard. Run the subprocess autograd probe
before attempting a full engine on a host without a usable CUDA driver.
"""
from __future__ import annotations

import contextlib
import itertools
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch


class FakeCudaRuntime:
    """Emulate CUDA streams/events, with all assumptions visible in the trace.

    Tensor kernels must independently run under FakeTensorMode with fallback
    disabled. This context never emulates a numerical tensor kernel. Runtime
    memory counters are unavailable placeholders used only by upstream logging;
    they are not exported as observations or used as prediction inputs.
    """

    def __init__(self, capture, *, properties: dict, execution=None):
        self.capture = capture
        self.properties = SimpleNamespace(**properties)
        self.actions: list[dict] = []
        self._stack = contextlib.ExitStack()
        self._ids = itertools.count(1)
        self._current = None
        self._device_index = 0
        self.execution = execution

    def _record(self, action, *, stream=None, **fields):
        deps = []
        if fields.get("recorded_event") is not None:
            deps.append(fields["recorded_event"])
        if fields.get("waited_stream") in self.capture._last_by_stream:
            deps.append(self.capture._last_by_stream[fields["waited_stream"]])
        if action == "synchronize" and fields.get("sync_scope", "device") == "device":
            deps.extend(self.capture._last_by_stream.values())
        self.actions.append(dict(action=action, stream=stream, event_index=len(self.capture.events), **fields))
        event = self.capture._event("host_sync" if action == "synchronize" else "wait",
                                   "cuda_runtime." + action, stream=stream or self.capture.stream, deps=list(set(deps)),
                                   **fields)
        return event

    def _query(self, event_id):
        if self.execution is None:
            raise NotImplementedError('Completion queries require a LiveExecution backend')
        self.execution.ingest_capture(self.capture)
        return event_id is None or self.execution.query(event_id)

    def __enter__(self):
        import torch
        runtime = self
        if self.execution is not None:
            def on_event(event):
                self.execution.ingest_capture(self.capture)
                if event['kind'] == 'allocate':
                    self.execution.revise_storage_metadata(event)
                if event['kind'] == 'host_sync':
                    self.execution.wait(event['event_id'])
            self._stack.enter_context(patch.object(self.capture, '_host_time_provider',
                                                   lambda: self.execution.host_time_us))
            self._stack.enter_context(patch.object(self.capture, '_event_listener', on_event))

        class Stream:
            def __init__(self, device=None, priority=0, **kwargs):
                self.device = torch.device("cuda", runtime._device_index)
                self.stream_id = next(runtime._ids)
                self.cuda_stream = self.stream_id
                self.priority = priority
                self.device_index = runtime._device_index
                self.device_type = 1
                self.name = f"cuda_stream_{self.stream_id}"

            def wait_event(self, event):
                runtime._record("wait_event", stream=self.name, recorded_event=getattr(event, "recorded", None))

            def wait_stream(self, stream):
                runtime._record("wait_stream", stream=self.name, waited_stream=stream.name)

            def record_event(self, event=None):
                event = event or Event()
                event.record(self)
                return event

            def query(self):
                return runtime._query(runtime.capture._last_by_stream.get(self.name))

            def synchronize(self):
                runtime._record("synchronize", stream=self.name, sync_scope="stream")

            def __eq__(self, other):
                return isinstance(other, Stream) and self.stream_id == other.stream_id

            def __hash__(self):
                return self.stream_id

        class Event:
            def __init__(self, enable_timing=False, blocking=False, interprocess=False, **kwargs):
                self.event_id = next(runtime._ids)
                self.recorded = None
                self.enable_timing = enable_timing

            def record(self, stream=None):
                stream = stream or runtime._current
                self.recorded = runtime._record("record_event", stream=stream.name)["event_id"]

            def wait(self, stream=None):
                (stream or runtime._current).wait_event(self)

            def query(self):
                return runtime._query(self.recorded)

            def synchronize(self):
                runtime._record("synchronize", stream="__host_sync__", recorded_event=self.recorded, sync_scope="event")

            def elapsed_time(self, end_event):
                if runtime.execution is None:
                    raise NotImplementedError('Modeled elapsed time requires a LiveExecution backend')
                if not self.enable_timing or not end_event.enable_timing:
                    raise ValueError('Timing must be enabled on both events')
                runtime.execution.ingest_capture(runtime.capture)
                return runtime.execution.elapsed_time_ms(self.recorded, end_event.recorded)

        self.Stream, self.Event = Stream, Event
        self._current = Stream()
        self.capture.stream = self._current.name
        patches = {
            "Stream": Stream, "Event": Event,
            "current_stream": lambda device=None: self._current,
            "default_stream": lambda device=None: self._default,
            "stream": self.stream,
            "set_stream": self.set_stream,
            "set_device": self.set_device,
            "current_device": lambda: self._device_index,
            "device_count": lambda: 1,
            "is_available": lambda: True,
            "is_initialized": lambda: True,
            "_lazy_init": lambda: None,
            "_exchange_device": self.exchange_device,
            "_maybe_exchange_device": self.exchange_device,
            "get_device_properties": lambda device=None: self.properties,
            "get_device_name": lambda device=None: self.properties.name,
            "get_device_capability": lambda device=None: (self.properties.major, self.properties.minor),
            "synchronize": lambda device=None: self._record("synchronize", sync_scope="device"),
            "empty_cache": lambda: None,
            "memory_stats": lambda device=None: defaultdict(int),
            "memory_allocated": lambda device=None: 0,
            "memory_reserved": lambda device=None: 0,
            "max_memory_allocated": lambda device=None: 0,
            "max_memory_reserved": lambda device=None: 0,
            "reset_peak_memory_stats": lambda device=None: None,
            "reset_accumulated_memory_stats": lambda device=None: None,
            "manual_seed_all": lambda seed: None,
            "manual_seed": lambda seed: None,
        }
        self._default = self._current
        for name, replacement in patches.items():
            self._stack.enter_context(patch.object(torch.cuda, name, replacement))
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self.capture.stream = "compute"

    @contextlib.contextmanager
    def stream(self, stream):
        previous = self._current
        self.set_stream(stream)
        try:
            yield
        finally:
            self.set_stream(previous)

    def set_stream(self, stream):
        if stream is not None:
            self._current = stream
            self.capture.stream = stream.name

    def set_device(self, device):
        import torch
        self._device_index = device if isinstance(device, int) else (torch.device(device).index or 0)

    def exchange_device(self, device):
        previous = self._device_index
        self.set_device(device)
        return previous


def probe_fake_cuda_autograd(*, timeout=30, cpu_only=True):
    """Probe Python emulation and native autograd separately in a child process.

    Hide GPUs by default so successful execution establishes CPU-only coverage.
    A visible-device probe is an explicit diagnostic and does not establish that
    coverage. Native aborts remain contained in the child process.
    """
    import json
    import os
    import subprocess
    import sys
    code = """
import json
import torch
import torch._dynamo
from torch._subclasses.fake_tensor import FakeTensorMode
from titan_sim.capture import TensorCapture
from titan_sim.fake_device import FakeCudaRuntime
def phase(name):
    print(json.dumps({'phase': name}), flush=True)
phase('python_runtime')
properties = dict(name='logical NVIDIA GB200', major=10, minor=0,
                  total_memory=197897617408, multi_processor_count=152)
try:
    with FakeCudaRuntime(TensorCapture(), properties=properties):
        with FakeTensorMode(allow_fallback_kernels=False):
            phase('tensor_creation')
            value = torch.empty(2, device='cuda', requires_grad=True)
            phase('forward')
            loss = value.square().sum()
            phase('backward')
            loss.backward()
    print(json.dumps({'status': 'complete', 'torch': torch.__version__}), flush=True)
except Exception as error:
    print(json.dumps({'status': 'blocked', 'error_type': type(error).__name__,
                      'error': str(error)}), flush=True)
    raise
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    if cpu_only:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=timeout, env=environment)
    except subprocess.TimeoutExpired as error:
        return dict(status="blocked", reason="fake CUDA autograd probe timed out",
                    cpu_only=cpu_only, stderr=str(error))
    diagnostics = []
    for line in result.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            diagnostics.append(row)
    return dict(status="complete" if result.returncode == 0 else "blocked",
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                cpu_only=cpu_only, diagnostics=diagnostics,
                limitation="Python runtime emulation cannot replace native C++ CUDA DeviceGuard")


@contextlib.contextmanager
def fake_tensor_parameter_swaps():
    """Invalidate only stale FakeTensor converter entries before native swaps.

    PyTorch Module._apply chooses swap_tensors for FakeTensor parameters, while
    FakeTensor's own WeakValueDictionary memo can prevent that swap. Conversion
    changes dtype/device/storage metadata, so these particular memo entries
    cannot remain valid. External weak references are not removed: native
    swap_tensors still rejects them. This context preserves parameter identity
    and runs the original swap implementation and its remaining checks.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    original = torch.utils.swap_tensors
    report = dict(policy="invalidate FakeTensor converter memo before parameter swaps",
                  memo_entries_evicted=0)

    def swap(first, second):
        for tensor in (first, second):
            if isinstance(tensor, FakeTensor):
                memo = tensor.fake_mode.fake_tensor_converter.tensor_memo
                keys = [key for key, value in memo.items() if value is tensor]
                for key in keys:
                    del memo[key]
                report["memo_entries_evicted"] += len(keys)
        return original(first, second)

    with patch.object(torch.utils, "swap_tensors", swap):
        yield report
