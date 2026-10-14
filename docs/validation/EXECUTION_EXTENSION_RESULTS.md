# Execution extension validation

This record separates CPU semantics, container compatibility, GPU timing
calibration, compiler artifacts, and native runtime feasibility. None is a
matched full DeepSeek-V3 training comparison.

## Independent CPU checks

The final reviewed suite passes **226 tests, no skips**, both locally on macOS
PyTorch 2.14.1 and in public NVIDIA PyTorch 26.09 on Ptyche (job 3045309).
The [target test log](extensions-3045309/tests.log) and
[source manifest](extensions-3045309/source-manifest.json) are retained. Every
recorded source hash matches the released implementation, tests, and scripts.

The test suite exercises collective matching, asynchronous storage pins,
delayed rendezvous timestamps, supported functional APIs, upstream pipeline
scheduling, allocation fragmentation, network resource sharing, exact calibration
signatures, compiler slot reuse, retained outputs, and quantization layouts.
Independent review found and fixed errors in pending storage retention,
late communication timing, Future outputs, storage reclassification, calibration
matching, shared-collective synchronization, and compiler input/output lifetimes.

The reproducible two-worker distributed example completes at 27.36 microseconds
on both virtual clocks. Its explicit hardware scenario is synthetic. The
four-stage, four-microbatch pipeline example has 300 events, 32 stage compute
actions, 24 shared P2P transfers, and a 528.144-microsecond modeled critical path.
Its declared activation buffers peak at 12,288 and 14,336 bytes by rank, then
all return to zero live bytes. These are component examples, not DeepSeek sizes.

The compiler-plus-quantization example has 19 events and 198 declared FLOPs.
Logical ownership peaks at 416 bytes; the scheduled storage peak is 480 bytes
because unfinished work retains storage. Its selected allocator scenario also
reserves 480 bytes. This demonstrates why host ownership and device storage
lifetimes must be accounted separately.

## First NVIDIA-container rerun

The target uses public NVIDIA PyTorch 26.09, PyTorch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA 13.4, and an actual NVIDIA GB200. The
TorchTitan context remains `607852350b41b4b280d8b8de6340f4847ed71c5e`.

The initial container attempt stopped because the test directory was shadowed
by another installed `tests` package. Adding an explicit local package fixed
test discovery. Rerun 3045179 ran 204 tests and exposed one diagnostic-count
failure: a native functional-collective probe left pending work that was counted
with the following adapter probe. The corrected probe explicitly resolves that
pending work before measuring adapter behavior. The failing snapshot and its
log remain visible in [the recorded results](extensions-3045179/tests.log).

Independent stages of that rerun completed three eager BF16 GEMM calibration
checks. CPU FakeTensor capture and actual CUDA capture produced identical exact
operation signatures for all three shapes.

| GEMM M × K × N | Calibration median (µs) | Independent held-out median (µs) | Absolute relative error |
|---|---:|---:|---:|
| 1024 × 1024 × 1024 | 29.328 | 31.408 | 6.62% |
| 4096 × 7168 × 512 | 39.168 | 40.144 | 2.43% |
| 4096 × 2048 × 7168 | 90.368 | 90.320 | 0.053% |

[Raw calibration samples and context](extensions-3045179/calibration.json) are
retained. Calibration uses 20 measurements after 10 warmups; held-out evaluation
uses a separate 10 measurements after three warmups. The CUDA-event interval
around an eager call can include device idle while the CPU launches work. These
results test the same operator boundary, not an isolated kernel service time,
novel-shape generalization, quantized kernels, communication or full training.

## Native CPU-only boundary

The [native metadata guard probe](extensions-3045179/native-guard.json) hides
GPUs, builds a CPU extension, traps actual generic CUDA allocation, and tests
virtual stream/event bookkeeping. Those checks and fake CUDA forward execution
pass. Backward fails at `hasPrimaryContext`: PyTorch's CUDA hooks reject device
index zero when no device is visible. The original guard is restored.

A successful diagnostic process exit only means the probe recorded its result;
the structured result is **blocked**. A DeviceGuard replacement alone does not
provide CUDA accelerator/hooks behavior. Full original-engine CPU execution
therefore remains unsupported pending additional reviewed framework integration.

## Final target run: compiler and memory comparison

Job 3045309 completed successfully after all review fixes. Both registered CPU
workers passed eager and adapted functional collective checks. The actual
GB200 compiler probe compiled and numerically executed `relu(x @ weight) + 1`,
then imported its [generated CUDA wrapper](extensions-3045309/inductor/cuda_output_code.py).
The wrapper supplies one 24-byte allocation reused by the fused pointwise
kernel and retained as output. Its alignment-dependent input-copy helpers were
checked on the same real inputs; the imported plan is conditional on that
observed no-copy path. Unknown or misaligned paths remain unsupported.

The probe wrote its compiler-derived output payload and a fixed 512-byte
allocator-rounding assumption **before** measuring a separate warmed-up
invocation. Inputs and the previous output remained alive, preventing the new
output from reusing their storage.

| Quantity | Prediction or assumption | Independent observation |
|---|---:|---:|
| Fresh output payload | 24 bytes from compiler plan | Allocator counters do not expose unrounded payload |
| Additional allocated memory | 512 bytes with declared rounding | 512 bytes |
| Additional peak allocated memory | 512 bytes | 512 bytes |
| Additional reserved memory | Not predicted | 0 bytes |

The [memory crosscheck](extensions-3045309/inductor/cuda_memory_crosscheck.json)
preserves both sides. The pre-existing baseline was 33,555,968 allocated bytes
and 171,966,464 reserved bytes. That baseline is measured, unattributed by this
small model, and excluded from the prediction claim. Matching a 512-byte delta
does not establish total-process memory accuracy, arbitrary allocator fidelity,
backward/optimizer coverage, or full DeepSeek training accuracy.

Final independent eager-GEMM timing checks were:

| GEMM M × K × N | Calibration median (µs) | Held-out median (µs) | Absolute relative error |
|---|---:|---:|---:|
| 1024 × 1024 × 1024 | 28.352 | 28.560 | 0.728% |
| 4096 × 7168 × 512 | 40.208 | 40.432 | 0.554% |
| 4096 × 2048 × 7168 | 92.064 | 93.728 | 1.775% |

[Final timing samples](extensions-3045309/calibration.json) are separate from
the earlier run above. These repeat measurements do not validate unseen shapes
or full-step timing. The [native probe](extensions-3045309/native-guard.json)
again reports blocked backward execution despite successful diagnostic process
completion. The full selected 671B CPU/GPU comparison remains outstanding.
