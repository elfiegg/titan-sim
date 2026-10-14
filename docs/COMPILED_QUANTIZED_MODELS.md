# Compiled regions and quantized storage

The simulator now lowers explicit compiled-region plans and quantized training-state layouts into the same scheduled storage ledger. These are executable model contracts with validation. The included FX artifact comes from an actual CPU `torch.export` invocation. A separate static Inductor importer now derives storage from an actual `torch.compile` CPU wrapper, including compiler-selected reuse. The GPU fusion, allocation, conversion, and graph-pool plans remain declared inputs; no native Transformer Engine training step or GPU compiler allocation plan was captured.

## Reproduce the example

From the repository root, using a Python environment with PyTorch:

```sh
python scripts/generate_compiled_quantized_example.py --output-dir /tmp/titan-compiled-example
python -m unittest tests.test_compiled tests.test_quantization -v
```

The generator reads the two saved declared specifications, obtains a fresh graph, binds its SHA-256 digest, lowers the storage and compute events, and runs runtime plus allocator replay. It rejects an unexpected exported graph. Files written to the output directory:

| File | Evidence or contract |
| --- | --- |
| `compiled_quantized_fx.json` | Actual exported matmul/relu graph, node shapes/dtypes, generated graph code, PyTorch version |
| `compiled_quantized_training.json` | Declared persistent state, packed MXFP8 layout, and conversion cost/scratch requirements |
| `compiled_quantized_manifest.json` | Declared GPU kernels, external storage IDs, intermediate/workspace lifetimes, private pool and replay count |
| `compiled_quantized_capture.json` | Lowered ordinary allocation/compute/release events with storage read/write IDs |
| `compiled_quantized_simulation.json` | Scheduled work, logical and asynchronous storage trajectories, allocator replay |

The fixture has 19 events and 198 declared FLOPs. Its program-order logical peak is 416 bytes; its scheduled physical and allocator-reserved peaks are 480 bytes. Host launches have no supplied delay, so conversion scratch remains physically resident while later buffers are submitted. These tiny sizes and hardware rates are semantic test fixtures, not GB200 or model performance estimates.

The example retains 192 bytes of BF16 parameters/gradients and FP32 master parameters/moments, 132 bytes of padded packed weight/scales, 28 bytes of input/output, and a 64-byte graph pool. The conversion scratch adds 64 bytes to the asynchronous peak. The graph pool is allocated once for capture plus two replays. Intermediate and workspace views do not allocate the pool a second time.

## Compiled-region contract

`lower_compiled_manifest(manifest, artifact=artifact)` returns a capture document. `plan_source` is either `declared` or `compiler_artifact`. A supplied artifact must match `artifact_sha256`. An actual FX artifact also requires complete, unique coverage of its computational node names in `logical_ops`. A `compiler_artifact` must have kind `gpu_compiler_allocation_plan` or `inductor_wrapper_allocation_plan` and contain an exactly matching plan. The manifest preserves its source device; importing CPU code does not turn it into a GPU allocation/timing claim. FX metadata cannot be labeled a GPU allocation plan.

Kernels declare execution dtype, FLOPs, HBM read/write bytes, and logical-buffer/external-storage references. Fused kernel traffic is authoritative: the lowerer does not add eliminated eager intermediate traffic. The test suite checks a declared fused matmul/relu kernel. The exported quantized example instead declares separate MXFP8 matmul and BF16 ReLU kernels, because their execution precision differs. Mixed arithmetic must be split into separately costed events with the correct execution dtypes and explicitly assigned traffic. The lowerer does not discover fusion or derive GPU kernel arithmetic from an FX graph.

Physical slots are allocations. Logical buffers have slot offsets, byte sizes, and first/last kernel uses. Disjoint lifetimes can reuse a slot; overlapping live byte ranges require `alias_of`, and the alias must fit inside its owner's allocation and lifetime. Unknown storage, out-of-lifetime accesses, alias cycles, oversized views, and overlapping undeclared reuse are rejected. A slot remains allocated across the lifetimes of all buffers assigned to it. Optional `retained_buffers` keep returned physical storage alive after the region. Their byte ranges and alias owners remain live through the final kernel, so a later unrelated reuse cannot silently overwrite a retained result.

`graph.mode=capture_replay` requires an explicit private pool size, per-slot offsets, replay count, and `release_pool` flag. Capture executes the declared work once. Replays repeat work without allocating new slots. The graph pool contributes its complete declared capacity, including slack. `release_pool` releases logical ownership; the optional caching allocator may retain graph-owned cached segments, and currently does not simulate graph destruction.

Kernel order is serial on the declared stream. External buffers must be allocated by the caller; concatenating quantization events and region events on the same stream supplies conversion ordering. Other streams require explicit dependencies. Compiler latency, GPU launch overhead, allocator stream arenas, and hidden workspaces are not inferred. Calibration signatures include the full plan hash, kernel declaration, artifact hash, and caller-supplied `calibration_context`; changing compiler/backend/layout policy changes the lookup key.

## Packed quantization contract

`packed_layout(spec)` calculates payload and scale storage for a declared two-dimensional layout. Rowwise and columnwise representations are independent physical copies; both are charged when requested. Payload padding and scale-row/scale-column alignment are explicit options.

| Format/layout | Payload | Scaling |
| --- | --- | --- |
| FP8 `tensorwise` | 8 bits per padded element | One FP32 scale per stored orientation |
| MXFP8 `rowwise_block` | 8 bits per padded element | One E8M0 byte per 32-element block, plus declared scale padding |
| NVFP4 `rowwise_block` | Two E2M1 elements per byte | One E4M3 byte per 16-element block and one FP32 tensor scale per orientation |
| NVFP4 `tile_16x16` | Two E2M1 elements per byte, padded to complete tiles | Explicit `scale_storage`: compact tile scales or scales replicated across rows; additional declared scale padding |

These geometry choices follow the format definitions in NVIDIA's [MXFP8 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html) and [NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html). The model distinguishes logical scale sharing from physical scale replication. A particular backend's swizzling, padded dimensions, replicated scales, and shared storage must be supplied accurately. It does not assume that a compact generic layout matches native TE allocation.

`lower_quantized_training(spec)` independently declares parameter, master parameter, gradient, first moment, and second moment dtypes; each may explicitly be null. It allocates packed payload/scales separately from those states. Changing compute precision never scales the entire training state. Conversion events read their source state, write all declared packed components, and include explicitly supplied arithmetic and scratch traffic. Scratch has its own allocation/use/release lifetime. Conversion signatures bind layout, source dtype, declared work, and calibration context.

The implementation models sizes and work, not numerical quantization values or training accuracy. It does not synthesize backward/optimizer execution, sharding, amax collectives, stochastic rounding, random Hadamard transforms, extra amax histories, or scale-swizzle kernels. Those require additional explicit work/storage events. Unknown layout names and unsupported schema fields fail rather than silently receiving a dense-byte approximation.


## Import an actual Inductor storage plan

The additional `titan_sim.inductor.import_inductor_wrapper` API parses an actual static `output_code.py` and constructs its allocation/view/reuse/lifetime plan. It never imports or executes the supplied source. `examples/inductor/cpu_output_code.py` was obtained by running `torch.compile` on `relu(x @ weight) + 1`, with inputs shaped 2×4 and 4×3. The compiled function was executed numerically and every output was verified to be 5. This is actual compiler evidence, unlike the separately declared plan attached to the earlier FX export example.

Reproduce the CPU compile, numerical check, compiler-wrapper capture, import, and runtime validation:

```sh
python scripts/generate_inductor_example.py --output-dir /tmp/titan-inductor-example
python -m unittest tests.test_inductor tests.test_compiled -v
```

On an actual CUDA GPU, collect the GPU compiler wrapper using:

```sh
python scripts/generate_inductor_example.py --device cuda --output-dir /tmp/titan-inductor-cuda
```

The CUDA command saves the actual wrapper and numerical-execution provenance before attempting strict import. If compiler-version-specific syntax is unsupported, `cuda_import_status.json` records the exact failure and the captured source remains available for inspection. Supported CUDA kernels are the benchmark’s external GEMM and fused add/ReLU Triton call. The generator validates pointer access against the generated `tl.load`/`tl.store` AST; arithmetic and traffic stay explicitly declared benchmark formulas.

The recorded run uses PyTorch 2.14.1 on CPU. It yields one 24-byte compiler allocation, `buf0`, reused as `buf1` for a fused add/ReLU kernel. The returned output keeps that backing allocation alive. The importer obtains these storage facts from the wrapper. Kernel FLOPs and traffic come from the separate explicit declaration in `cpu_kernel_costs.json`: 48 GEMM FLOPs and 12 fused pointwise operations, with one pointwise read/write pass. Those costs are not measurements or compiler-inferred cache traffic.

The generator emits `cpu_provenance.json`, `cpu_output_code.py`, the exact-source-bound cost declaration, `cpu_import.json` (artifact, derived manifest and import report), `cpu_capture.json`, and `cpu_simulation.json`. CPU events retain `device=cpu`, so GPU cost/memory reporting remains zero/unmodeled for this fixture. The compiler-derived physical storage report separately records the 24 bytes. `cuda_syntax_fixture.py` is prominently labeled as a handwritten parser test; it provides no evidence that a GPU compiler produced that particular plan.

The programmatic API is:

```python
from titan_sim.inductor import import_inductor_wrapper

imported = import_inductor_wrapper(
    wrapper_source,
    kernel_costs=cost_document,
    provenance={"source": "actual_torch_compile_debug_output", "device": "cuda"},
)
capture = imported["capture"]
```

`cost_document` must contain the wrapper's exact `source_sha256` and a `kernels` mapping keyed `k0`, `k1`, … in call order. Each entry provides `target`, `dtype`, `flops`, `bytes_read`, `bytes_write`, `reads`, `writes`, and `provenance`. Access selectors are positional argument indices or keyword names such as `"out"`. Every tensor argument must have an explicit access mode. A changed wrapper, unknown call, unused cost entry, or missing arithmetic/traffic declaration fails. The complete imported source, input metadata and cost declaration are retained in the hashed compiler artifact.

Supported syntax is intentionally bounded:

| Wrapper operation | Interpretation |
| --- | --- |
| Global `call(args)` or `Runner.call(self, args)` | One static region |
| Flat argument unpacking, `args.clear()` | External inputs; caller owns their lifetime |
| `assert_size_stride` / grouped equivalent | Static input geometry guards |
| `empty_strided_cpu` / `empty_strided_cuda` | Physical allocation span derived from shape, strides and dtype |
| Tensor assignment and `del` | Aliases/reuse and reference lifetime |
| `reinterpret_tensor` | Validated backing-storage view, byte offset and strided span |
| Costed kernel calls | Explicit tensor access, dtype/arithmetic and boundary traffic |
| Device-0 CUDA guard, `set_device(0)`, raw stream and static grid | Supported launch metadata; no source execution |
| Flat tensor return | Returned backing slots retained until caller release |

Input dtype/device metadata is read from static `get_args`/`rand_strided` declarations when present. Otherwise the caller must provide `input_specs` with shape, stride, dtype and device. The cost document must then also bind `input_specs_sha256`; shapes and strides must still match wrapper guards. When static `get_args` declarations are present, supplied input metadata must also match their dtype, device and complete geometry; an explicit declaration cannot override conflicting compiler metadata. Generated external storage IDs and returned storage IDs are reported for composition with caller allocation and release events.

Dynamic shapes/control flow, loops, unknown memory planners/pools, mixed devices, nonzero CUDA device indices, unknown allocation functions, symbolic launch geometry, and zero-byte local allocations are rejected. The importer does not read C++/Triton code to invent arithmetic or access modes. A CUDA compiler wrapper can be imported when it fits this subset and has an exact cost declaration; an actual NV26.09 CUDA wrapper has now been captured and numerically executed on NVIDIA GB200, as described below. This does not establish full-model compiler coverage.


### Actual NV26.09 CUDA wrapper and alignment-dependent paths

The `nv2609_cuda_*` fixtures were captured on NVIDIA GB200 using PyTorch `2.14.0a0+b2c75dd062.nv26.09` and CUDA 13.4. Actual CUDA compilation and numerical execution of the tiny function succeeded. The wrapper contains an external GEMM, one 24-byte output allocation, and an in-place fused Triton add/ReLU kernel. Its exact source hash is bound by the cost declaration. The CPU parser regression imports this real CUDA source and verifies the 24-byte compiler allocation and retained output.

This wrapper also contains `copy_if_misaligned` before GEMM. That helper can allocate conditionally. Its first import was rejected rather than silently omitting possible copies. The importer now supports the aligned no-copy branch only when `kernel_costs.input_alignment` explicitly describes each affected input. Every entry supplies `alignment_bytes`, `pointer_mod_alignment`, `copy_if_misaligned_returns_same_storage`, and provenance. The remainder must be zero and the same-storage condition true; unknown or misaligned paths remain unsupported.

The generator reads the target's alignment constant and invokes the exact `torch._C._dynamo.guards.copy_if_misaligned` helper on the same retained real input tensors used for the verified compiled call. It records pointer remainders, same-pointer/same-storage booleans, helper name and target PyTorch version. It stores no raw addresses. The imported allocation plan is explicitly conditional on this observed input path. The CPU regression uses an explicitly declared aligned scenario and does not mislabel it as an observed GPU pointer.

The CUDA generator also writes a small steady-state memory crosscheck after compilation/warmup. It keeps inputs and the prior output alive, writes compiler-derived fresh-output payload and a fixed 512-byte allocator-rounding expectation before measurement, then records baseline/current/peak allocated and reserved bytes around one more compiled call. It checks that prior and new outputs do not share storage. Cached reserve and hidden library workspaces are not fitted to the observation or claimed as predicted. This is one tiny compiled forward; it does not validate full training memory, arbitrary allocator behavior or native quantized training.

The final GB200 run completed this path: the 24-byte compiler output payload
predicted a 512-byte allocator increase under the declared rounding scenario,
and both current and peak allocated memory increased by 512 bytes. The
[validation report](validation/EXECUTION_EXTENSION_RESULTS.md) includes the
independent counters, excluded baseline, exact source hashes, and scope limits.
