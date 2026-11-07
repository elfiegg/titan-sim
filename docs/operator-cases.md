# Captured native operator cases

`titan_sim.operator_cases` extracts reusable dense-matmul cases from actual
`TorchCapture` dispatch records. It calls the captured ATen overload directly,
with the captured shapes, strides, storage offsets, floating dtype and scalar
arguments. No model registry, replacement forward or hand-written GEMM is used.
Importing the module and extracting catalogs require no PyTorch installation.

The supported overloads are `aten.mm.default`, `aten.bmm.default`,
`aten.addmm.default` and `aten.addmm_.default`. Eligibility is intentionally a
semantic allowlist, not every operator with a tensor-shaped schema. Data-dependent
indexing/masks, random operators, allocation/uninitialized-value operators,
custom/compiled operators, complex/sparse/distributed tensors, empty tensors and
expanded/overlapping inputs are skipped with per-node reasons. Aliased input
operands are rejected. The mutating addmm overload must return the exact captured
first-input alias; its initialized backing storage is reset before every warmup
and timed sample, outside the CUDA events.

Tensor values are **synthetic finite constants** (operand index plus one divided
by 16), not the captured weights/activations. This reproduces eligible native
operator metadata, not original numerical results. Values, cache state, autotuner
state and isolated execution may change performance. Matching overload/runtime
identities does not prove that capture and replay select an identical CUDA kernel.
There is no claim of compiled-kernel fidelity, training overlap, or held-out
end-to-end validation.

## Capture and extraction

The native recipe adapter records the effective math context after trainer
construction. For an already captured dispatch workload, select every eligible
case and retain the skip inventory with:

```sh
python -m titan_sim calibration-cases workload.rank0.json --kind compute \
  --report operator-inventory.json --output operator-cases.workload.json
```

For callers using `capture_trainer` directly, record these identities after
trainer setup and before capture, alongside source hashes and resolved recipe:

```python
from titan_sim.operator_cases import execution_context

provenance.update(
    torch_version=str(torch.__version__),
    cuda_version=torch.version.cuda,
    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    float32_matmul_precision=torch.get_float32_matmul_precision(),
    operator_execution_context=execution_context(torch),
)
```

`execution_context` records the framework git/build fingerprint, reduced-precision
math flags, deterministic settings, selected BLAS preference and relevant BLAS
workspace/TF32 environment settings. The provider checks exact equality against
the live measurement runtime and rejects missing identities. Capture and replay
must use the same installed framework. CPU-captured cases cannot be relabeled as
CUDA cases. A fake CUDA dispatch capture is signature evidence only; it supplies
no measured duration.

```python
from titan_sim.operator_cases import prepare_operator_cases, operator_case_workload
from titan_sim.workload import read_workload, write_workload

source = read_workload('workload.rank0.json')
catalog = prepare_operator_cases(source)
# catalog has schema/version, cases, skipped, source provenance and limitations.
# Select any subset by exact case_id; omitted case_ids selects every eligible case.
measurement = operator_case_workload(source, case_ids=[catalog['cases'][0]['case_id']])
write_workload('operator-cases.workload.json', measurement)
```

Every case retains all originating rank/node IDs and dispatch cost keys. Its
identity includes source-provenance digest and complete input/output signature;
allocation occurrence IDs are normalized while alias topology is preserved.
Different scalar values, offsets, strides or source identities stay distinct.
Only successful returned dispatches are eligible. A partial capture can supply
individual cases, but extraction never promotes it to a complete training trace.

`operator_case_workload` returns a **separate single-rank calibration workload**.
It does not rewrite the source training workload or apply these costs to its
original dispatch cost keys. The workload retains the complete extraction catalog,
including skipped operators. Cases from a multi-GPU source retain their original
rank provenance, while mapping the captured CUDA ordinal to the current
measurement GPU explicitly in each cost context. Communication is unsupported by
this provider and needs its own identified process-group measurement.

## Native measurement

Use one explicitly authorized GPU allocation in the original runtime, with an
initial five-minute scheduler hard limit. Do not benchmark a login node.

```json
{
  "mode": "execute",
  "hardware_id": "the-exact-workload-hardware-id",
  "expected_device": {"name": "the-actual-GPU-name", "total_memory": 123456789},
  "region_provider": "titan_sim.operator_cases:region_provider",
  "provider_config": {},
  "warmup": 5,
  "repeats": 20
}
```

`expected_device.total_memory` must be the actual device property in bytes, not
the illustrative value above. `native_runner` checks physical device properties,
records CUDA-event samples, and binds each sample record to the exact
implementation/context key. This provider validates the installed overload
schema, executes it once to check output shape/stride/dtype/offset/alias, and
materializes one case at a time so a large catalog does not retain every case's
operands simultaneously. Initialization and validation occur outside timing.
Fresh output allocation from the native operator remains part of each call.
CUDA graph measurement is currently rejected.

```sh
python -m titan_sim calibrate operator-cases.workload.json \
  --runner titan_sim.native_runner:measure --config measurement.json \
  --output operator-costs.json
```

These durations are native executable-region samples with synthetic values.
Original training gaps, unknown operators, workspace/memory coverage and
full-scale scheduling remain unresolved; a catalog is not a strict full-training
simulation result or physical memory-fit verdict.
