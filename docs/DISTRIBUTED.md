# Distributed CPU metadata execution

The simulator can run supported eager `torch.distributed` calls through actual
Python `ProcessGroup`/`Work` subclasses, and can register those groups with
`init_process_group` in independent CPU processes. Tensor payloads must be meta
tensors or `FakeTensor` objects. No tensor values are transported or computed.

The integration has three distinct scopes:

| Entry point | Implemented behavior | Boundary |
|---|---|---|
| `lower_distributed_events(events, groups)` | Validate participant matching and lower each rendezvous into one network transfer plus rank-local arrivals/completions | Requires an explicit complete participant trace |
| `SimulatedWorld(...).process_group(rank)` | Cooperative asynchronous ranks; supported eager collective/P2P calls; Work/future waits; explicit compute and storage bindings | Every participant must submit before waiting; not a thread-blocking backend |
| `register_simulated_backend(hardware)` | Real `init_process_group` with multiple CPU processes; PyTorch Store exchanges small JSON control records | Collective-only; captured compute/storage graphs and P2P across worker processes are not implemented |

Supported eager operations are SUM all-reduce, all-gather (list or single output),
SUM reduce-scatter with a single input/output tensor, all-to-all-single, and
barrier. The cooperative world also supports `isend`/`irecv`. Unimplemented
broadcast, reduce, gather/scatter, coalesced operations, and list reduce-scatter
raise `NotImplementedError`. Numeric CPU/CUDA tensors are rejected. Per-call
`Work.wait(timeout=...)` is rejected; registered Store rendezvous uses the
`init_process_group` timeout.

## Reproduce the CPU probe

From the repository root, using the pinned local environment:

```sh
../../work/local-env/bin/python scripts/distributed_probe.py --out examples/distributed --multiprocess
../../work/local-env/bin/python -m unittest tests.test_distributed tests.test_pipeline
```

The probe runs cooperative `torch.distributed` calls, extracts an upstream
pipeline schedule, and spawns two registered-backend workers. It writes the
hardware scenario, pipeline capture/simulation, and one report per worker.
The recorded durations are synthetic scenarios, not GPU measurements.

## Cooperative groups and fake-GPU waits

```python
import torch
import torch.distributed as dist
from titan_sim.distributed import SimulatedWorld
from titan_sim.live import LiveExecution

backend = LiveExecution(hardware)
backend.submit([{
    'event_id': 'producer', 'kind': 'compute', 'rank': 0,
    'duration_us': 100, 'duration_provenance': 'explicit scenario',
}])
world = SimulatedWorld(backend=backend)
pg0, pg1 = world.process_group(0), world.process_group(1)
pg0.set_dependencies('producer')
tensor = torch.empty(256, device='meta')
a = dist.all_reduce(tensor, group=pg0, async_op=True)
b = dist.all_reduce(tensor, group=pg1, async_op=True)
b.wait()  # advances virtual host time through producer + communication
assert a.is_completed()
world.assert_complete()
```

Use `pg.set_storage_accesses(reads=[...], writes=[...])` to bind storage IDs
already submitted by `TensorCapture` to the same backend. These bindings persist
until replaced or cleared with `pg.set_storage_accesses()`. Tensor-to-storage
correspondence is explicit; it is not inferred automatically. Storage is pinned
at asynchronous submission, including before other participants arrive, and the
pin is released after communication completion. Thus dropping the original host
reference before rendezvous cannot reclaim its backing storage early.

`Work.wait()` and `Work.get_future().wait()` advance the virtual clock. Querying
a pending work object does not advance that clock. Futures return output tensors
for all-gather, reduce-scatter, and all-to-all. Optional wall-clock pacing belongs
to the shared `LiveExecution` backend; metadata tracing time is not device time.

## Registered workers and functional collectives

Each independently launched worker must register the same hardware scenario,
then initialize its rank using a normal PyTorch Store rendezvous:

```python
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._subclasses.fake_tensor import FakeTensorMode
from titan_sim.distributed import register_simulated_backend, functional_collectives

register_simulated_backend(hardware)
dist.init_process_group('titan_sim', init_method=init_method,
                        rank=rank, world_size=world_size)
pg = dist.distributed_c10d._get_default_group()
with FakeTensorMode():
    with functional_collectives():
        x = torch.empty((4, 2), device='cpu')
        reduced = funcol.all_reduce(x, 'sum', pg)
        reduced = funcol.wait_tensor(reduced)
        gathered = funcol.all_gather_tensor(x, 0, pg)
        gathered = funcol.wait_tensor(gathered)
        scattered = funcol.reduce_scatter_tensor(gathered, 'sum', 0, pg)
        scattered = funcol.wait_tensor(scattered)
dist.destroy_process_group()
```

The explicit dispatch mode is required for functional APIs. The final PyTorch
2.14.1 probe rejects native `wait_tensor` outside the mode because its C++ Work
bridge supplies a per-call timeout, whose semantics this backend does not model.
The diagnostic probe records that exception and explicitly resolves its queued
operation before testing the supported adapter. The dispatch adapter routes the
actual functional operator family and `wait_tensor` into the modeled Work
interface; it is not a blanket meta no-op. It rejects unimplemented functional and autograd collective families
and rejects leaving the scope with unawaited outputs. Input/output references
remain held until completion; backing-storage matching supports output aliases.
The two-process smoke checks functional all-reduce, all-gather, and
reduce-scatter inside `FakeTensorMode`.

Registered workers preserve each participant's recorded enqueue timestamp.
Delayed rendezvous resolution must not shift already issued communication to
the later CPU wait time. Every worker models the same matched transfer timeline;
its local host clock can differ if that worker performs independent host work.
Hardware mismatches, missing participants/timeouts, dtype/payload mismatches,
and inconsistent collective operations fail explicitly.

## Upstream pipeline schedule

`capture_pipeline_schedule()` constructs the installed PyTorch
`ScheduleInterleaved1F1B` and executes its upstream communication lowering. It
accepts at least two ranks and at least two evenly placed stages per rank.
The generated action list determines stage order, matched forward/backward P2P,
and receive Work waits. Stage compute durations and activation-buffer sizes are
explicit inputs. The default four-stage/four-microbatch case contains 32 stage
compute actions and 24 shared P2P transfers; buffer lifetimes span asynchronous
uses through backward/release. Tests verify imbalance changes the critical path
and all buffers are reclaimed.

This is upstream schedule extraction using metadata stage descriptors. It does
not execute `PipelineStage` numerical forward/backward or a TorchTitan model.
UNSHARD/RESHARD/REDUCE_GRAD actions are labeled control markers for the declared
unsharded, DP=1 scenario. FSDP hooks, optimizer execution, automatic full-model
capture/storage exchange across processes, native CUDA DeviceGuard/autograd,
DeviceMesh compatibility, and full TorchTitan distributed training remain
unvalidated. These probes do not establish full-model memory or GPU timing
accuracy.
