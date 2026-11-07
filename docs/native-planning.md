# Simulate the submitted native execution plan

The plan path reads the schedules and graph metadata that a native runtime has
already constructed. It preserves the submitted recipe's stages, microbatches,
graph transformations and action ordering. It does not select a different model,
change checkpointing, or convert an eager recipe to GraphTrainer.

This path answers a narrower question than complete recipe qualification:
what execution dependencies and logical storage lifetimes follow from these
native artifacts? Kernel costs and complete physical allocations remain separate
inputs. The current implementation cannot automatically predict every TorchTitan
recipe offline.

## Components and boundaries

```text
prepare-runtime + unchanged native recipe
                    |
       native schedule and stage graph construction
                    |
        export-plan: inspect native objects
                    |
       versioned plan, source hashes, unresolved gaps
                    |
        lower-plan: expand schedule + storage lifetimes
                    |
        workload + placement + component cost bindings
                    |
     calibrate / analytical assumptions -> simulate
```

| Component | Responsibility |
|---|---|
| Native TorchTitan/PyTorch runtime | Build the selected implementation, graph transformations, stage calling conventions and action order |
| `native_plan` | Read action tables, FX graphs, tensor metadata and initialized state; preserve source identities and unsupported cases |
| `plan_capture` | Export prepared trainer objects, or observe lazy graph preparation during bounded original training steps |
| `plan_lowering` | Expand supported native actions into dependencies, communication rendezvous and component cost identities |
| `plan_memory` | Bind graph values to state, saved backward values and microbatch instances; account for backing aliases and modeled completion |
| Existing simulator | Schedule the workload using explicit costs and resources; report timing and physical-memory uncertainty separately |

Exported tensor sizes are logical metadata. They do not reveal all compiler
buffer reuse, library workspaces, graph pools or allocator behavior. A complete
set of calibrated operation costs does not close those memory gaps.

## Export from an existing prepared runtime object

Inside the caller-owned training environment:

```python
from titan_sim.plan_capture import export_trainer_plan
from titan_sim.native_plan import write_plan

# trainer is the original, already-built native trainer.
# provenance records the exact recipe, sources and prepared runtime identity.
plan = export_trainer_plan(trainer, provenance=provenance, rank=rank)
write_plan(output_path, plan)
```

The default `observe_steps=0` only inspects objects. GraphTrainer commonly builds
graphs lazily. If they do not exist yet, the exporter retains that limitation;
it does not invent graphs or change compile settings.

For a supported GraphRuntime, `observe_steps=1` observes its first `ensure_ready`
return during one original training step. This executes the native step and its
communication, optimizer update and configured side effects. Every participating
rank must run the bounded observation. The observer never stops one rank halfway
through a pipeline step, which could strand peers. It restores instrumented
methods on success and failure. Different exported plans across observed steps
are rejected rather than collapsed into one static workload.
Observer-only errors are reported after the bounded native steps finish; failures
from native preparation or training itself still propagate normally.

The readiness hook is a version-dependent integration point. Current tests cover
native PyTorch graphs/schedules and the hook contract; full GraphTrainer runtime
compatibility has not been established by those tests. Unsupported eager trainers
require an additional extraction implementation, not an automatic trainer swap.

## Use the native recipe loader

`export-plan` defaults to `titan_sim.plan_capture:export_recipe`. Its configuration
accepts the same `execution`, unchanged `recipe_args`, `runtime_manifest`,
`capability_report`, `expected_runtime`, and `runtime_provenance` fields as native
capture, plus `observe_steps` (default zero). Fake execution requires the recipe's
explicit `--comm-backend fake` argument. Missing fake kernels or incompatible
native initialization remain errors. Fake mode does not emulate CUDA libraries.

After creating this configuration from the caller's prepared runtime and recipe:

```sh
python -m titan_sim export-plan --config plan-export.json \
  --output 'results/plan.rank{rank}.json'
```

Run that invocation through the prepared runtime's existing launcher when ranks
are required. Native initialization, compilation or observed steps can require
GPUs and memory even before useful training work. This command is not a promise
of full-target offline planning. A trusted alternative adapter can return the
same plan schema from already-available native artifacts.

## Lower a complete pipeline group

```sh
python -m titan_sim lower-plan results/plan.rank*.json \
  --hardware hardware.json --config plan-bindings.json \
  --output results/planned-workload.json

python -m titan_sim validate results/planned-workload.json
```

`plan-bindings.json` supplies explicit placement and component cost/communication
bindings described in [schedule lowering](plan-scheduling.md). It cannot replace
the extracted schedule or contradict known tensor payloads. Supply a global
useful-token count for the exact modeled interval if requesting throughput.

Rank exports must describe the same pipeline schedule and compatible provenance.
Duplicate stages, source disagreements, stale checksums and blocked/incomplete
captures are rejected. The current merge covers one pipeline group. Its local
rank namespace is not automatically a complete PP/DP/TP/EP global mesh; other
groups and their collectives need further integration.

Use the existing [calibration workflow](native-calibration.md) for implemented
regions, or explicit analytical quantities and rates for an exploratory scenario:

```sh
python -m titan_sim simulate results/planned-workload.json \
  --calibration results/costs.json --output results/prediction.json
```

FX instruction keys identify candidates. They do not establish selected fused
kernel identity. Missing costs are errors; unknown semantics remain gaps.
Physical memory from this path currently remains `inconclusive`: its scope is
the extracted schedule, its storage evidence is unverified, and optimizer,
workspace, allocator and lifecycle coverage is incomplete. The detailed
[memory guide](plan-memory.md) explains what is derived and what is retained
conservatively.

## What this closes and what remains

Implemented boundaries now connect native action tables and FX storage metadata
to the simulator. Tests exercise actual PyTorch schedule generation and fake FX
tensor metadata, aliasing, microbatch liveness, communication matching, malformed
inputs, CLI flow and recipe-preserving observation contracts. These are
implementation tests, not a new full-model GPU accuracy result.

Remaining work includes native runtime qualification, construction of missing
plans without full-target execution, complete global-mesh lowering, compiler
region/workspace binding, optimizer/lifecycle allocations, distributed graph and
subgroup calibration, bandwidth sharing and held-out accuracy measurements.
Calibration artifacts remain ordinary local files; selecting a central service
is independent of the plan representation and is intentionally deferred.

Validation varies implementation features to detect simulator bugs. A requested
simulation always retains that workload's exact settings. A validation fixture
never becomes a model default or a reason to alter the submitted recipe.
