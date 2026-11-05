# titan-sim

Experimental recipe prescreening for TorchTitan: describe a distributed workload,
measure representative operations on real hardware, then estimate execution time
and per-device peak memory on CPU.

**Current status:** the event/storage simulator, small fake-execution captures,
GPU component benchmarks, calibration import, and held-out two-GEMM validation
work. The automatic full-model capture → matched calibration → prediction pipeline
is **not complete**. The included full Kimi graph is a generated analytical
scenario, not captured native execution or validated throughput/memory fit.

This implementation replaces the previous repository contents; earlier work remains
in Git history. The old event API and microsecond-based files are not compatible
with this workload schema. Agent instructions are in [AGENTS.md](AGENTS.md).

## 1. Motivation

A training recipe combines model structure, precision, optimizer state, activation
checkpointing, parallelism, routing, communication and scheduling. Allocating a
large GPU cluster for each candidate is expensive. Prescreening should help reject
obviously unsuitable recipes and identify what needs hardware validation first.

The questions are:

- Does the recipe's **peak on every device** fit, including simultaneous tensors
  and implementation buffers? Aggregate cluster capacity is insufficient.
- What step time and throughput should we expect under the specified placement
  and schedule, and which assumptions dominate that estimate?
- Which operations, memory allocations and performance effects remain unsupported
  or uncalibrated?

Fake tensors and fake process groups help explore the program's structure without
real parameter storage or real communication. They do not supply GPU timings,
network contention, achieved overlap, or native workspace sizes. Those require
explicit models and measurements. Prescreening does not establish measured
performance or successful execution at scale.

## 2. Basic structure: how agents use the toolchain

The intended workflow is:

```text
Resolved TorchTitan recipe + execution scenario
                    |
             TorchTitan capture adapter
                    |
          Workload data + calibration targets
             |                       |
             |             Native GPU calibration runner
             |                       |
             |                Measured cost catalog
             |                       |
             +-----------> CPU simulator <----- Hardware/topology model
                                      |
                         Timing, per-rank memory, coverage/gaps
```

The two outputs of the proposed adapter have different consumers: the simulator
reads workload descriptions; the calibration runner needs references to executable
native implementations. There is no conversion from a simulator graph back into
PyTorch. **The generic adapter and automatic target generation are still planned.**
Current runnable cases and their narrower scopes are listed below.

### What an agent prepares

| Input | Required information |
|---|---|
| Recipe | Model/flavor and source revision, optimizer, precision, checkpointing, routing, parallelism and compilation settings |
| Execution scenario | Sequence length, microbatch size/count, schedule and rank placement; an explicit scenario for data-dependent routing |
| Capture environment | Compatible pinned TorchTitan/PyTorch/custom-op dependencies, preferably the intended training container |
| Hardware description | Capacity per device, topology, resource assumptions and provenance; assumed rates must be labelled |
| Calibration evidence | Matching operation/region signatures, software/hardware settings, timing samples, workspace observations and measurement scope |

The repository is the shared codebase, not a hosted service or an implicit cluster
allocation. Agents clone a pinned revision and invoke its Python CLI/modules. The
CPU simulator does not import TorchTitan, PyTorch, CUDA or protobuf. Capture and
calibration depend on the training environment; calibration additionally uses real
target GPUs. Fake capture may still encounter native extension or CUDA control-plane
requirements, so a GPU-free full-model capture is not guaranteed.

### Install and CPU smoke test

Python 3.10+ is sufficient for the simulator and generated scenarios:

```sh
git clone https://github.com/elfiegg/titan-sim.git
cd titan-sim
git rev-parse HEAD                 # record/pin this revision for the experiment
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
titan-sim --help
titan-sim --case checks --output-dir results/checks
python -m unittest discover -s tests -v
```

`python -m titan_sim` is equivalent to `titan-sim` and works directly from the
checkout without installation. Installation has no mandatory runtime dependencies.

### Commands available today

| Command/case | Required environment | Actual scope/output |
|---|---|---|
| `--case checks` | CPU Python | Deterministic dependency, communication-rendezvous, resource and storage checks; `summary.json` |
| `--case dense` | Compatible PyTorch with FakeTensor/FakeProcessGroup internals | Small dense forward/backward/SGD plus fake all-reduce; 102 nodes on two logical ranks |
| `--case hybridep` | PyTorch and the pinned TorchTitan source | Four native custom-op fake-function shape probes; no native buffer initialization or GPU transfers |
| `--case kimi` | CPU Python | Generated PP8/DP32/EP32 analytical stage graph; memory verdict remains inconclusive |
| `--case calibration` | CPU PyTorch | Local CPU two-matmul calibration/validation plumbing; not GPU evidence |
| `--simulate-workload PATH` | CPU Python | Load direct workload JSON and simulate with optional matched calibration |
| `python -m titan_sim.calibration.benchmark` | Authorized GB200 allocation and compatible training runtime | Explicit component measurements, per-rank JSONL |
| `python -m titan_sim.calibration.summarize RAW OUT` | CPU Python | Validated cost catalogs, held-out workloads/results and a report |

For a quick generated scheduling example:

```sh
titan-sim --case kimi --microbatches 2 --output-dir results/kimi-demo
titan-sim --simulate-workload results/kimi-demo/kimi/workload.json
# Expected to fail: this workload contains declared coverage gaps.
titan-sim --simulate-workload results/kimi-demo/kimi/workload.json --strict
```

Use `--microbatches 128` to exercise the full generated 256-rank scenario. This
changes scenario size, not its evidence quality: historical analytical stage costs
are not new measured Kimi performance predictions.

For actual small fake capture, use the existing compatible PyTorch environment:

```sh
python -m titan_sim --case dense --output-dir results/dense
TORCHTITAN_PROTOTYPE_SOURCE=/path/to/pinned/torchtitan \
  python -m titan_sim --case hybridep --output-dir results/hybridep
```

The source pin and hashes are in
[`torchtitan-provenance.json`](titan_sim/fixtures/torchtitan-provenance.json).
Do not replace a training container's PyTorch merely to run these examples.
`--case all` additionally attempts native Kimi dependency/build probes; it is not
the default and does not perform a complete native Kimi training step.

### GPU calibration and result import

The current harness benchmarks manually specified full-size GEMM cases, selected
backward GEMMs, batched expert proxies, HBM copy, NCCL collectives, paired transfers
and a native HybridEP probe. It does not discover the full compiled model's kernels.

Inside an authorized four-GB200 allocation with the intended runtime:

```sh
CALIBRATION_PHASES=compute CALIBRATION_OUTPUT=results/gb200-compute \
  torchrun --standalone --nproc-per-node=4 --module titan_sim.calibration.benchmark
python -m titan_sim.calibration.summarize \
  results/gb200-compute results/gb200-compute-summary
titan-sim --simulate-workload results/gb200-compute-summary/heldout-cuda_graph/workload.json \
  --calibration results/gb200-compute-summary/rank-calibration.json --strict
```

Use the site's normal multi-node launcher for 32 GPUs and set
`CALIBRATION_PHASES=communication` to skip compute. `all` runs both phases. Native
HybridEP requires the compatible DeepEP implementation and runtime dependencies.
Use a **five-minute scheduler hard limit** for initial calibration. These commands
do not allocate GPUs or enforce the limit themselves. Completed samples are flushed
per rank and survive a later failure/timeout.

The HybridEP case uses 2,048 tokens/rank, hidden 3,584, 896 experts, top-16 forced
round-robin, BF16, padding multiple **32**, capacity 33,152 rows and 32 ranks. This
differs from the padding-128 fake-shape/legacy fixture scenario: capacity and cost
claims are not interchangeable. Match actual rank placement/topology; an intra-domain
measurement does not calibrate inter-domain communication. The importer currently
accepts four- or 32-rank measurements and requires all expected hardware records.

### Output an agent reports

Case commands save `summary.json` and per-case `workload.json`/`result.json` under
`--output-dir` (default `results/titan-sim`). Simulation of an existing workload
prints a JSON summary. Calibration import saves `calibration.json`,
`rank-calibration.json`, `summary.json`, `REPORT.md` and held-out workload/results.

Report elapsed time/throughput **with** `status`, `cost_coverage` and `unresolved`.
Report the worst rank and modeled memory categories, not an aggregate cluster
capacity comparison. `memory_verdict=inconclusive` must remain inconclusive.
Allocated memory is contained in reserved memory; never add the two.

`--strict` rejects declared workload gaps and uncalibrated operation costs. It does
not independently prove that the producer captured every allocation or operation.
The simulator's storage output is a modeled logical peak, not automatically a CUDA
allocator-reserved or device-wide physical peak.

## 3. Design structure

The simulator's direct interface is ordinary Python data:

```python
from titan_sim.core import Node, Storage, simulate

result = simulate(nodes, storages, manifest, hardware, calibration, strict=True)
```

| Component | Source | Input | Output/responsibility |
|---|---|---|---|
| Capture probes | `capture.py` | Example code or pinned native fake definitions | Operation occurrences, tensor shape/dtype/stride metadata, backing storages, manifests and explicit gaps |
| Analytical scenario builder | `kimi.py` | Recipe fixture and microbatch count | Generated stage regions, PP ordering, transfers, FSDP rendezvous and coarse storage lifetimes |
| Workload contract | `core.py`: `Node`, `Storage` | Producer-created Python records | Framework-independent simulator input; node occurrence IDs and cost lookup signatures have distinct roles |
| Persistence | `workload.py` | Nodes/storages/manifest/hardware | Versioned JSON and reconstruction; rejects unsupported schema/time units; no executable code |
| Calibration runner | `calibration/benchmark.py` | Explicit operation cases and real GPU runtime | Independent timings and allocator observations, retaining raw samples and unsupported operations |
| Calibration importer | `calibration/summarize.py` | Rank JSONL files | Descriptor/evidence validation, per-rank and aggregate cost maps, held-out comparison |
| Cost resolver | `core.py`: `CostModel` | Operation attributes, hardware and optional catalog | Exact-key measured cost or explicitly labelled analytical/historical fallback |
| Event/storage simulator | `core.py`: `simulate` | Workload plus costs/resources | Dependency-aware timings and storage lifetimes/peaks on each rank |
| Optional transport | `bundle.py` | Same workload records | Chakra protobuf files plus custom attributes/sidecars; not used for kernel discovery |
| CLI | `cli.py` | Case or workload path, calibration and output options | Reproducible invocation and reports |

The workload fields are:

- **Node:** occurrence ID, operation/region name, rank, kind, dependency IDs and
  attributes such as cost key, tensor metadata, communication peers/participants,
  rendezvous identity and required resources.
- **Storage:** backing-allocation ID, rank, byte size, producer, consumers,
  persistence and category. Aliases share the same backing-allocation identity.
- **Manifest:** recipe/scope/provenance, ranks, useful tokens and unresolved gaps.
- **Hardware:** device capacity, reserve allowance and explicit rate/resource
  assumptions. Placement and communication groups are declared by the producer.
- **Calibration:** cost signature → duration/samples, hardware identity and
  evidence checksum. The current signatures are not a complete cross-version
  implementation-fingerprint registry.

All durations use **seconds**, byte quantities use **bytes**, and modeled rates
use FLOP/s or bytes/s. The direct JSON schema is `recipe-prescreen-workload`,
version 1. The simulator does not need a graph-to-PyTorch conversion.

The engine waits for all participants in a communication rendezvous, respects
dependencies, and reserves declared resources until an event completes. Storage
starts at its declared producer and survives all declared consumers. These rules
only model the information actually supplied. Resource exclusion is not a full
bandwidth-sharing or compute/communication-interference model.

Chakra is optional: install `.[chakra]` and use `--format chakra` for the historical
transport round trip. Vendored schema license and provenance are in `titan_sim/vendor`.
Our custom attributes/sidecars require a compatible consumer; generic Chakra
simulator compatibility is not asserted. The default CPU path uses neither
protobuf nor PyTorch.

## Evidence, limitations and development sequence

The initial experiment captured the small dense example and four HybridEP fake
operations, and exercised a generated 180,736-node, 256-rank Kimi scenario at 128
microbatches. Native full-model capture was blocked by runtime/native dependency
requirements. The generated Kimi graph uses historical analytical stage costs;
its memory verdict is inconclusive and strict mode rejects it.

A four-GB200 component run completed 39 operation configurations. An independently
measured two-GEMM chain, excluded from the calibration catalog, differed from the
component-cost prediction by **+3.2% eager** and **+4.0% CUDA graph replay**. This
validates that narrow composition experiment only. The software stack was PyTorch
`2.14.0a0+b2c75dd062.nv26.09` and CUDA 13.4. Private raw logs/device identifiers and
cluster launch files are not published here; run calibration to obtain a usable
catalog for your environment. No EP32 measurements are included in this package.

Important gaps:

- The generic native TorchTitan adapter and automatic calibration-target generation
  are not implemented. Current capture and GPU benchmark signatures are not yet
  joined automatically for the full model.
- Fake capture currently inserts conservative sequential ordering; it does not
  faithfully capture all asynchronous streams, waits, or runtime overlap.
- Compiled/fused implementation identities, native MLA/KDA and grouped/SiTU expert
  kernels, and complete backward/optimizer coverage remain unresolved.
- Full-step native communication buffers, CUDA graph pools, Muon workspaces,
  allocator reservation, and gradient/activation lifetimes are not fully modeled.
- Full-scale placement, contention and pipeline overlap need independent validation.

Development should proceed by capturing **one native Kimi block**, retaining its
original executable calibration targets, matching measured costs back to its graph,
and validating time and memory independently. Then extend to communication and
pipeline scheduling before claiming full-model accuracy. A successful simulator
test or small held-out chain does not close these coverage gaps.
