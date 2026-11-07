# Native schedule and graph plans

`titan_sim.native_plan` exports already-materialized native schedule and graph
objects as versioned JSON. It does not construct models, trace an eager recipe,
change compile settings, initialize distributed state, or execute graph nodes.
The caller must prepare the **submitted recipe's own** objects in its matched
runtime. If that recipe does not expose these objects, export reports the gap.
An eager Trainer is not converted into GraphTrainer.

```python
from titan_sim.native_plan import extract_plan, write_plan

# runtime is the original GraphRuntime, after its ensure_ready(ctx) returned.
plan = extract_plan(runtime, provenance=source_and_recipe_identity, rank=global_rank)
write_plan("plan.rank0.json", plan)
```

A native schedule plus its `stage_graphs` mapping can be passed separately.
Extraction itself is read-only. Preparing those objects in TorchTitan can compile,
autotune, allocate tensors and load data: the surrounding export adapter is not
an allocation-free substitute for native initialization. It must use the caller's
authorized execution environment and preserve native preparation settings.

## Object compatibility

The exporter reads these inspected source APIs by their attributes:

- PyTorch pipeline schedules: `pipeline_order_with_comms`, or the explicitly
  incomplete `pipeline_order`; `_n_microbatches`; `_stages` / `_stage`; each stage's
  `stage_index_to_group_rank`. Native `_Action` fields preserve stage, enum value,
  microbatch, and recursive overlap subactions. Idle `None` slots are retained.
- TorchTitan GraphRuntime: `schedule`, `stage_graphs`, `graph_provider.plan`,
  `overlap_graphs`, and the stages' initialized `state` tensor lists.
- GraphTrainerStageGraphs: `modules.{fw,full_bw,bw_di,bw_dw,unshard,reduce_grad}`,
  `meta` including `partition`, compiled flag and gradient accumulator metadata.
- GraphTrainerJointStageGraphs: `traced.gm` and traced state/input counts. Its
  `joint` graph is also bound to native `FULL_FORWARD_BACKWARD`; accumulation
  variants require their own native call specs.
- GraphTrainerScheduledFwdBwdStageGraphs: `graphs.call_specs`, retaining each
  computation enum value, its FX module and native argument/accumulator indices.
- FX GraphModules: graph node order, target, schema, input references, scalar
  arguments, mutation argument names, positional graph output references, and
  tensor leaves in `meta['val']` or `meta['example_value']`.

These interfaces were inspected in the local GraphTrainer source snapshot
(`graph_pp/pp_graph_builder.py`, `spmd_graph_builder.py`,
`spmd_gradient_accumulation_graph_builder.py`, `make_fx_tracer.py`, and
`graph_pp/runner.py`) and PyTorch 2.15 CPU sources. The GraphTrainer snapshot had
no Git metadata; no upstream commit compatibility is inferred. The exporter
records available SHA-256 source-file digests for the actual runtime classes,
plus caller-supplied recipe/source provenance. A matching attribute layout is
necessary, but does not establish executable semantic compatibility. Unknown
bundle interfaces, missing graphs, and opaque targets remain explicit gaps.

## JSON contract

`schema_version` is `titan_sim.native_plan.v1`. `read_plan`, `write_plan`, and
`validate_plan` use only Python's standard library and never import TorchTitan or
PyTorch. `plan_sha256` hashes the canonical JSON envelope excluding itself.
`seal_plan` allows an adapter to add observation metadata and recompute the hash.
Each present graph checksum is verified as well. Checksums detect artifact
changes; they do not certify model fidelity.

| Field | Meaning |
|---|---|
| `schedule.rank_actions` | Rank-string to ordered native action objects or idle nulls |
| `schedule.stage_to_rank` | Native stage ownership; ranks are **pipeline-group ranks** |
| `schedule.source_attribute` | Whether the actual source supplied communication actions |
| `graphs` | Graph ID to stage, role, native nodes and positional output references |
| `stages.*.graphs` | Native role or scheduled action enum to graph ID |
| `stages.*.metadata` | Native calling convention, partition and call-spec metadata |
| `stages.*.state` | Initialized parameter, buffer and gradient storage tensor leaves |
| `source_files` | Native class source filenames and available SHA-256 digests |
| `unresolved` | Missing or unsupported semantics, retained by downstream lowering |

Graph nodes have `name`, `op`, `target`, ordered `inputs`, recursive `args` and
`kwargs`, and tensor `outputs`. In arguments, `{"node": "name"}` is a graph
reference. Native nonfinite floating-point arguments and metadata use exact
`{"float": "-inf"}`, `{"float": "inf"}`, or `{"float": "nan"}` tags; finite numbers
remain ordinary JSON numbers. Caller-supplied provenance must already be strict
JSON and is not silently normalized. Graph `outputs` preserve duplicate references and use null for
non-node constant slots. This preserves the positions required for saved-value
bindings; consumers must not drop or deduplicate those slots.

Tensor leaves contain `path`, `shape`, `stride`, `dtype`, `device`,
`storage_offset`, `element_size`, `logical_bytes`, `storage_id`, `storage_bytes`
and `requires_grad` when the native object exposes them. Storage identities come
from backing storage identity, never a data pointer; extraction does not read
tensor contents or request CUDA initialization. IDs are local to one export.
`path` retains the leaf's position in a structured value or native state list.
Symbolic integers are not concretized. Data-dependent scalar extraction and
symbolic or opaque output values retain explicit unsupported gaps. Opaque tensor subclasses are not
silently treated as local dense tensors.

Fake storage IDs describe trace metadata, not physical runtime addresses.
Repeated graph invocations require separate temporary allocations unless native
calling conventions bind saved values, persistent state or mutation aliases.
A real-mode `make_fx` snapshot can give a view a different fake backing storage
from its input. When operator alias schemas contradict metadata identities, the
exporter records a gap. It does not promote those IDs to proven runtime aliases.

`merge_plans` combines exports from the same native pipeline schedule. It
requires exact provenance apart from an explicit provenance `rank`, unique
extraction ranks, identical schedules, unique stage ownership and consistent
source digests. Storage IDs are namespaced by export rank. Missing stage exports
remain gaps. Replica groups and global/group-rank remapping are not inferred;
DP/TP replica merging requires a separate explicit placement contract. This
merge intentionally rejects conflicting or differently configured exports and
exports marked `capture_status: blocked` or `incomplete` (for example, a native
observation that ended before its requested steps).

## Limits and checks

A transformed FX graph is not automatically a compiled-kernel execution graph.
If the bundle is compiled, or its compilation state is unavailable, an explicit
execution-correspondence gap remains. Custom operator workspaces, nested opaque
regions, data-dependent control flow, multiprocessing/subgroup mapping,
multiplexed overlap graphs, optimizer steps, initialization and checkpoint peaks,
CUDA graph pools and allocator behavior are not established by this exporter.
With no native graphs or action table, `capture_status` is `blocked`. Otherwise
`capture_status: complete` means the supported objects were read, not that their
semantics cover the whole recipe. Consequently, a successful export remains
`status: incomplete` for full-recipe memory
and timing claims. Downstream preparation must retain those gaps.

The optional CPU tests use actual `make_fx` GraphModules, real native `_Action`
objects, and native GPipe schedule generation plus communication insertion. They
check shapes/storage aliases, mutation schemas, positional outputs, checksums,
rank merge and fail-closed interfaces. They do not run a full GraphTrainer,
initialize a GPU or validate distributed/native-kernel execution.

```sh
python -m unittest discover -s tests -p test_native_plan.py -v
```

Without PyTorch, transport and unsupported-interface tests still run and native
CPU tests skip. Use the existing pinned PyTorch environment for native checks;
do not replace its Torch/CUDA/custom-op versions.
