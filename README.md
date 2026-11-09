# titan-sim

titan-sim prescreens a native TorchTitan training recipe before you commit its
full target cluster. It predicts **peak memory on each physical GPU** and
**step time and useful-token throughput**, with the assumptions and missing
evidence attached to each result.

The intended user is a simulation agent working after `prepare-runtime`, or an
engineer using the same tools directly. You supply the prepared training runtime,
resolved recipe, target hardware and rank placement. The recipe stays authoritative:
TorchTitan owns the model, optimizer, parallelism and execution settings.

## How the program works

Prescreening combines three kinds of information:

1. **Execution structure:** native plans and observed execution describe which
   operations run, their dependencies, rank roles and storage lifetimes.
2. **GPU costs:** real GPUs measure selected native compute regions,
   communication operations and short concurrent windows. Measurements are saved
   in local files and matched to the original workload.
3. **Memory evidence:** allocation and lifecycle contracts describe how much
   physical memory can be live on each device, including temporary and external
   allocations.

A CPU simulator combines these into a target timeline and per-device memory
bounds. Fake process groups help explore target layouts without real transport;
their wall time does not predict GPU performance. Explicit analytical models can
supply costs when measurements are unavailable, with their assumptions reported.

The approach can cost less than full training because repeated work can share
measurements, local compute can use one GPU, and communication can use its
identified group. Some native regions still require distributed execution or
custom integration. One GPU must have enough memory for the local case being run.

Read the [system design](docs/architecture.md) for component responsibilities,
artifact relationships and the distinction between a measurement and a prediction.

## Start here

| Your task | Read |
|---|---|
| Run a recipe prescreen in a prepared runtime | [GPU-assisted workflow](docs/gpu-assisted.md) |
| Delegate that task to an agent | [Simulation skill](skills/simulate-torchtitan/SKILL.md) |
| Understand or extend the program | [System design](docs/architecture.md), then [command and API reference](docs/reference.md) |
| Integrate an unfamiliar custom library or compiled region | [Custom-library workflow](docs/custom-library-workflow.md) |
| Try the CPU simulator without TorchTitan | The example below |

### Try the CPU simulator

CPU validation and simulation require Python 3.10+ with no mandatory third-party
dependencies. From a pinned checkout:

```sh
git clone https://github.com/elfiegg/titan-sim.git
cd titan-sim
git rev-parse HEAD
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m titan_sim --help

python -m titan_sim capture --adapter examples.minimal:capture \
  --config examples/minimal.json --output results/minimal/workload.json
python -m titan_sim validate results/minimal/workload.json
python -m titan_sim simulate results/minimal/workload.json \
  --output results/minimal/result.json
```

This example constructs two synthetic operations with declared analytical rates.
Its result is exploratory and memory is `inconclusive` because workspace coverage
is missing. Adding `--strict` rejects it. It demonstrates artifact flow, not model
performance. Run example modules from the checkout; they are not included in the wheel.

For native capture and calibration, use the original prepared training environment
with its pinned PyTorch, CUDA and extensions. The caller supplies GPU allocation
and launch commands. Read [AGENTS.md](AGENTS.md) before executing workloads.

## Read the result

Timing and memory are separate assessments:

| Question | Result fields | Interpretation |
|---|---|---|
| How long will the declared work take? | `seconds`, `useful_tokens_per_second` | Predictions for the declared scope; count unique tokens across model-parallel replicas correctly |
| What supports that timing? | `status`, `cost_coverage`, `cost_evidence`, `cost_limitations`, `unresolved` | Distinguish measured costs, analytical assumptions and missing execution semantics |
| Will every GPU have sufficient memory? | `memory_verdict`, `physical_memory`, `memory_unresolved` | Per-device bounds, capacity, headroom and missing physical-memory evidence |
| What storage does the schedule account for? | `modeled_peak_bytes`, `worst_rank`, `peak_categories_bytes` | Logical per-rank storage under the supplied lifetime model; narrower than physical device usage |

`--strict` requires complete declared timing coverage and exact measured costs.
`--require-memory-fit` separately requires an unconditional `estimated_fit`.
For a full-recipe decision, also require `physical_memory.memory_scope="recipe"`:
a schedule-only fit does not cover initialization and loading. See the
[memory contract](docs/native-memory.md) for the evidence behind these verdicts.

Structural validation, matching cost keys and complete declared coverage cannot
establish that an integration captured every behavior. Prediction accuracy requires
comparison with a separate real run over the same scope.

## Current boundaries

This is an experimental integration toolkit with an agent workflow. It is not yet
a universal, automatic recipe-to-prediction pipeline. Bundled paths support native
profiler capture, selected host plans and dispatch cases, local CUDA discovery,
explicit native region/window providers, exact source binding and per-device
memory envelopes.

Automatic global schedule recovery, arbitrary custom/compiled implementations and
complete physical-allocation evidence still require integration. Distributed
contention needs measured windows or explicit assumptions. Independent full-model
accuracy and hardware coverage must be established for the scope being claimed.

The [design and its limits](docs/architecture.md) explain where these gaps enter.
The [validation record](docs/validation/native-workflow.md) separates component
checks from full-training accuracy. Prescreening reports a prediction; it does
not establish successful execution or measured performance at target scale.
