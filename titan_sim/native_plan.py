"""Read-only projection of already-built native schedule and FX graph objects.

Importing this module, reading a plan, and validating its envelope require only
Python. Extraction never builds a trainer, runs a graph, or rewrites a recipe.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "titan_sim.native_plan.v1"
_GRAPH_ROLES = ("fw", "full_bw", "bw_di", "bw_dw", "unshard", "reduce_grad")
_STATE_FIELDS = ("sharded_param_values", "buffer_values", "unsharded_param_values",
                 "unsharded_param_grads", "sharded_param_grads", "grad_accumulators")


def _class(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class _Extractor:
    def __init__(self) -> None:
        self.gaps: list[str] = []
        self.storages: dict[Any, str] = {}
        self.storage_refs: list[Any] = []  # Keep storage identities alive while reading.
        self.sources: dict[str, dict] = {}

    def gap(self, message: str) -> None:
        if message not in self.gaps:
            self.gaps.append(message)

    def source(self, value: Any) -> None:
        cls = _class(value)
        if cls in self.sources:
            return
        try:
            filename = inspect.getsourcefile(type(value))
            if filename and Path(filename).is_file():
                self.sources[cls] = {"file": Path(filename).name,
                                     "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()}
                return
        except (OSError, TypeError):
            pass
        self.sources[cls] = {"file": None, "sha256": None}

    def plain(self, value: Any, path: str = "metadata") -> Any:
        if type(value) is float and not math.isfinite(value):
            return {"float": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
        if value is None or type(value) in (str, bool, int, float):
            return value
        if isinstance(value, Enum):
            return self.plain(value.value, path)
        if isinstance(value, Mapping):
            return {str(_enum(k)): self.plain(v, f"{path}.{k}") for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.plain(v, f"{path}.{i}") for i, v in enumerate(value)]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: self.plain(getattr(value, f.name), f"{path}.{f.name}")
                    for f in dataclasses.fields(value) if f.name not in ("module", "gm", "_run")}
        # No repr() of arbitrary native objects: it may embed addresses or tensors.
        self.gap(f"{path}: unsupported native metadata {_class(value)}")
        return {"unsupported_type": _class(value)}

    def integer(self, value: Any, path: str) -> int | None:
        # Do not concretize SymInt/SymFloat: that could install new trace guards.
        if type(value) is int:
            return value
        self.gap(f"{path}: symbolic or unavailable integer metadata")
        return None

    def tensors(self, value: Any, context: str, path: list | None = None) -> list[dict]:
        path = [] if path is None else path
        if isinstance(value, Mapping):
            return [t for k, v in value.items() for t in self.tensors(v, context, path + [str(k)])]
        if isinstance(value, (tuple, list)):
            return [t for i, v in enumerate(value) for t in self.tensors(v, context, path + [i])]
        if not all(hasattr(value, attr) for attr in ("shape", "dtype", "untyped_storage", "stride")):
            return []
        result: dict[str, Any] = {"path": path, "dtype": str(value.dtype),
                                  "device": str(getattr(value, "device", "unknown")),
                                  "storage_id": None, "storage_bytes": None}
        try:
            result["shape"] = [self.integer(x, context) for x in value.shape]
            result["stride"] = [self.integer(x, context) for x in value.stride()]
            result["storage_offset"] = self.integer(value.storage_offset(), context)
            result["element_size"] = self.integer(value.element_size(), context)
            result["logical_bytes"] = (value.numel() * value.element_size()
                                        if all(x is not None for x in result["shape"]) else None)
            result["requires_grad"] = bool(value.requires_grad)
            # DTensor and opaque subclasses may not have physical local storage.
            storage = value.untyped_storage()
            key = (result["device"], storage._cdata)
            if key not in self.storages:
                self.storages[key] = f"storage_{len(self.storages)}"
                self.storage_refs.append(storage)
            result["storage_id"] = self.storages[key]
            result["storage_bytes"] = self.integer(storage.nbytes(), context)
            result["storage_evidence"] = "native_tensor_metadata"
        except Exception as exc:
            self.gap(f"{context}: tensor storage metadata unavailable ({type(exc).__name__})")
        return [result]

    def action(self, value: Any) -> dict | None:
        if value is None:
            return None
        required = ("stage_index", "computation_type", "microbatch_index")
        if not all(hasattr(value, key) for key in required):
            self.gap(f"unsupported native schedule action {_class(value)}")
            return {"unsupported_type": _class(value)}
        self.source(value)
        result = {"stage_index": self.integer(value.stage_index, "action.stage_index"),
                  "computation_type": _enum(value.computation_type),
                  "microbatch_index": None if value.microbatch_index is None else
                  self.integer(value.microbatch_index, "action.microbatch_index")}
        if getattr(value, "sub_actions", None) is not None:
            result["sub_actions"] = [self.action(v) for v in value.sub_actions]
        return result

    def graph(self, module: Any, graph_id: str, stage: int, role: str) -> dict | None:
        graph = getattr(module, "graph", None)
        if graph is None or not hasattr(graph, "nodes"):
            self.gap(f"{graph_id}: opaque callable; no native FX graph")
            return None
        self.source(module)
        native_nodes = list(graph.nodes)
        node_names = {id(n): n.name for n in native_nodes}

        def refs(value: Any, *, positional: bool = False) -> list:
            if id(value) in node_names:
                return [node_names[id(value)]]
            if isinstance(value, Mapping):
                return [r for v in value.values() for r in refs(v, positional=positional)]
            if isinstance(value, (tuple, list)):
                return [r for v in value for r in refs(v, positional=positional)]
            return [None] if positional else []

        def argument(value: Any) -> Any:
            if id(value) in node_names:
                return {"node": node_names[id(value)]}
            if isinstance(value, Mapping):
                return {str(k): argument(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [argument(v) for v in value]
            if isinstance(value, slice):
                return {"slice": [argument(value.start), argument(value.stop), argument(value.step)]}
            if type(value) in (str, bool, int, float) or value is None:
                return self.plain(value)
            return {"type": _class(value), "value": str(value) if type(value).__module__ == "torch" else None}

        result = {"stage_index": stage, "role": role, "class": _class(module),
                  "nodes": [], "outputs": [], "metadata": {}}
        by_name = {}
        for node in native_nodes:
            target = node.target
            schema = getattr(target, "_schema", None)
            target_name = str(target) if isinstance(target, str) or schema is not None else (
                f"{getattr(target, '__module__', '')}.{getattr(target, '__qualname__', type(target).__qualname__)}")
            meta = getattr(node, "meta", {})
            native_value = meta.get("val", meta.get("example_value"))
            leaves = self.tensors(native_value, f"{graph_id}:{node.name}")
            def opaque_value(value: Any) -> bool:
                if value is None or type(value) in (str, bool, int, float):
                    return False
                if isinstance(value, Mapping):
                    return any(opaque_value(v) for v in value.values())
                if isinstance(value, (list, tuple)):
                    return any(opaque_value(v) for v in value)
                return not all(hasattr(value, a) for a in ("shape", "dtype", "untyped_storage"))
            if opaque_value(native_value):
                self.gap(f"{graph_id}:{node.name}: symbolic or opaque native output value is unsupported")
            if target_name in ("aten._local_scalar_dense.default", "aten.item.default", "aten.is_nonzero.default"):
                self.gap(f"{graph_id}:{node.name}: data-dependent scalar extraction is unsupported")
            if node.op not in ("output",) and not leaves:
                if "tensor_meta" in meta:
                    self.gap(f"{graph_id}:{node.name}: shape-only metadata has no backing-storage evidence")
                elif not any(k in meta for k in ("val", "example_value")):
                    self.gap(f"{graph_id}:{node.name}: native output metadata unavailable")
            record = {"name": node.name, "op": node.op, "target": target_name,
                      "inputs": list(dict.fromkeys(refs((node.args, node.kwargs)))),
                      "args": argument(node.args), "kwargs": argument(node.kwargs), "outputs": leaves}
            if schema is not None:
                record["operator_schema"] = str(schema)
                record["mutates_arguments"] = [a.name for a in schema.arguments
                    if a.alias_info is not None and a.alias_info.is_write]
                alias_args = [a for a in schema.arguments if a.alias_info is not None]
                if any(r.alias_info is not None for r in schema.returns) and alias_args and leaves:
                    # Snapshot fake metadata from real-mode make_fx may discard
                    # alias identity. Flag the contradiction instead of treating
                    # those independent metadata storages as real allocations.
                    candidates = []
                    for i, arg in enumerate(schema.arguments):
                        if arg.alias_info is None:
                            continue
                        value = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
                        for name in refs(value):
                            candidates.extend(by_name.get(name, {}).get("outputs", []))
                    input_ids = {x.get("storage_id") for x in candidates} - {None}
                    output_ids = {x.get("storage_id") for x in leaves} - {None}
                    if input_ids and output_ids and not output_ids.issubset(input_ids):
                        self.gap(f"{graph_id}:{node.name}: alias schema conflicts with tensor storage metadata")
            if node.op in ("call_module", "call_method") or (node.op == "call_function" and schema is None and
                    not (getattr(target, "__module__", "") == "_operator" and getattr(target, "__name__", "") == "getitem")):
                self.gap(f"{graph_id}:{node.name}: opaque or higher-order target {target_name}")
            if node.op == "output":
                result["outputs"] = refs(node.args, positional=True)
            result["nodes"].append(record)
            by_name[node.name] = record
        result["graph_sha256"] = _digest(result)
        return result


def extract_plan(schedule: Any, stage_graphs: Mapping[int, Any] | None = None, *,
                 provenance: Mapping | None = None, rank: int = 0) -> dict:
    """Project an initialized GraphRuntime or schedule plus bound graph objects.

    GraphRuntime.ensure_ready must already have run in the caller's authorized
    native execution. No missing graph is constructed by this function.
    Rank keys are native pipeline-group ranks, not inferred global ranks.
    """
    ex = _Extractor()
    runtime = schedule if hasattr(schedule, "schedule") else None
    if runtime is not None:
        stage_graphs = stage_graphs if stage_graphs is not None else getattr(runtime, "stage_graphs", None)
        schedule = runtime.schedule
        ex.source(runtime)
    ex.source(schedule)
    native_stages = list(getattr(schedule, "_stages", []))
    if not native_stages and getattr(schedule, "_stage", None) is not None:
        native_stages = [schedule._stage]
    if stage_graphs is None:
        stage_graphs = {s.stage_index: s.graphs for s in native_stages if getattr(s, "graphs", None) is not None}
    stage_graphs = stage_graphs or {}
    order = getattr(schedule, "pipeline_order_with_comms", None)
    source_attribute = "pipeline_order_with_comms"
    if not order:
        order = getattr(schedule, "pipeline_order", None)
        source_attribute = "pipeline_order"
        ex.gap("native schedule lacks materialized communication actions; compute-only order is incomplete")
    if not isinstance(order, Mapping):
        order = {}
        ex.gap("native schedule has no supported action table; schedule execution was not synthesized")
    rank_actions = {str(r): [ex.action(a) for a in actions] for r, actions in order.items()}
    placement: dict[str, int] = {}
    for stage in native_stages:
        mapping = getattr(stage, "stage_index_to_group_rank", None)
        if isinstance(mapping, Mapping):
            for key, value in mapping.items():
                if str(key) in placement and placement[str(key)] != value:
                    ex.gap(f"conflicting native placement for stage {key}")
                placement[str(key)] = value
    # Action ownership itself is native evidence, including remote stages.
    def owners(action: dict | None, owner: int) -> None:
        if not action or "stage_index" not in action:
            return
        for sub in action.get("sub_actions", []):
            owners(sub, owner)
        index = str(action["stage_index"])
        if action["computation_type"] in ("SEND_F", "RECV_F", "SEND_B", "RECV_B", "WAIT_SEND_F", "WAIT_SEND_B"):
            return
        if index in placement and placement[index] != owner:
            ex.gap(f"conflicting native action ownership for stage {index}")
        placement.setdefault(index, owner)
    for owner, actions in rank_actions.items():
        for action in actions:
            owners(action, int(owner))
    plan = {"schema_version": SCHEMA_VERSION, "provenance": dict(provenance or {}),
            "extraction": {"rank": rank, "mode": "native_object_introspection", "executes_graphs": False},
            "schedule": {"class": _class(schedule), "source_attribute": source_attribute,
                         "rank_namespace": "pipeline_group", "rank_actions": rank_actions,
                         "stage_to_rank": placement, "num_microbatches": getattr(schedule, "_n_microbatches", None)},
            "graphs": {}, "stages": {}, "unresolved": ex.gaps}
    by_index = {s.stage_index: s for s in native_stages}
    for index, bundle in stage_graphs.items():
        ex.source(bundle)
        entry = {"class": _class(bundle), "graphs": {},
                 "metadata": ex.plain(getattr(bundle, "meta", None)), "state": {}}
        if entry["metadata"] is None:
            entry["metadata"] = {}
        for name in ("full_bw_grad_accumulator_indices", "bw_dw_grad_accumulator_indices"):
            if hasattr(bundle, name):
                entry["metadata"][name] = ex.plain(getattr(bundle, name))
        for name in ("full_bw_initial_grad_accumulators", "bw_dw_initial_grad_accumulators"):
            if hasattr(bundle, name):
                entry["state"][name] = ex.tensors(getattr(bundle, name), f"stage{index}.{name}")
        state = getattr(by_index.get(index), "state", None)
        for name in _STATE_FIELDS:
            if state is not None and hasattr(state, name):
                entry["state"][name] = ex.tensors(getattr(state, name), f"stage{index}.{name}")
        if not entry["state"].get("sharded_param_values"):
            ex.gap(f"stage{index}: initialized native parameter state unavailable or empty")
        modules = getattr(bundle, "modules", None)
        candidates = {role: getattr(modules, role, None) for role in _GRAPH_ROLES} if modules is not None else {}
        traced = getattr(bundle, "traced", None)
        if traced is not None:
            candidates["joint"] = getattr(traced, "gm", None)
            entry["metadata"]["traced"] = {name: ex.plain(getattr(traced, name)) for name in
                ("state_fqns", "num_static_inputs", "num_optimizer_state_inputs", "num_runtime_mesh_inputs", "graph_state") if hasattr(traced, name)}
        specs = getattr(getattr(bundle, "graphs", None), "call_specs", None)
        if isinstance(specs, Mapping):
            entry["metadata"]["call_specs"] = {_enum(k): ex.plain(v) for k, v in specs.items()}
            candidates.update({_enum(k): getattr(v, "module", None) for k, v in specs.items()})
        if not candidates:
            ex.gap(f"stage{index}: unsupported graph bundle API {_class(bundle)}")
        for role, module in candidates.items():
            if module is None:
                continue
            graph_id = f"stage{index}.{role}"
            graph = ex.graph(module, graph_id, int(index), role)
            if graph is not None:
                plan["graphs"][graph_id] = graph
                entry["graphs"][role] = graph_id
        if traced is not None and not isinstance(specs, Mapping) and "joint" in entry["graphs"]:
            # GraphTrainerJointStageGraphs.forward_backward executes traced.gm;
            # native GraphRuntime dispatches FULL_FORWARD_BACKWARD to that method.
            # Accumulation variants require their own call_specs, never this alias.
            entry["graphs"]["FULL_FORWARD_BACKWARD"] = entry["graphs"]["joint"]
        compiled = getattr(bundle, "compiled", None)
        entry["compiled"] = compiled
        if compiled is not False:
            ex.gap(f"stage{index}: compiled execution correspondence is unverified; FX is not a native kernel graph")
        plan["stages"][str(index)] = entry
    if not plan["graphs"]:
        ex.gap("no native FX region graphs extracted")
    if runtime is not None:
        provider = getattr(runtime, "graph_provider", None)
        if provider is not None:
            ex.source(provider)
            plan["graph_provider"] = {"class": _class(provider), "plan": ex.plain(getattr(provider, "plan", None))}
        if getattr(runtime, "overlap_graphs", None):
            ex.gap("native overlap_graphs present; multiplexed pair execution is unsupported")
    ex.gap("optimizer, initialization, checkpoint and allocator/workspace coverage are not established by stage graphs")
    plan["source_files"] = ex.sources
    plan["status"] = "incomplete" if ex.gaps else "extracted"
    plan["capture_status"] = "complete" if plan["graphs"] and rank_actions else "blocked"
    plan["plan_sha256"] = _digest(plan)
    return plan


def validate_plan(plan: Mapping) -> None:
    """Validate the versioned transport envelope without importing a framework."""
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported native plan schema_version")
    for key in ("schedule", "graphs", "stages", "provenance"):
        if not isinstance(plan.get(key), Mapping):
            raise ValueError(f"native plan {key} must be an object")
    if not isinstance(plan.get("unresolved"), list):
        raise ValueError("native plan unresolved must be an array")
    if "plan_sha256" in plan:
        unsigned = dict(plan)
        expected = unsigned.pop("plan_sha256")
        if expected != _digest(unsigned):
            raise ValueError("native plan checksum mismatch")
    for graph_id, graph in plan["graphs"].items():
        if not isinstance(graph, Mapping):
            raise ValueError(f"native graph {graph_id} must be an object")
        if "graph_sha256" in graph:
            unsigned_graph = dict(graph)
            expected_graph = unsigned_graph.pop("graph_sha256")
            if expected_graph != _digest(unsigned_graph):
                raise ValueError(f"native graph {graph_id} checksum mismatch")
    for stage in plan["stages"].values():
        for graph_id in stage.get("graphs", {}).values():
            if graph_id not in plan["graphs"]:
                raise ValueError(f"native stage references missing graph {graph_id}")


def write_plan(path: str | Path, plan: Mapping) -> None:
    validate_plan(plan)
    Path(path).write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text())
    validate_plan(plan)
    return plan


def merge_plans(plans: list[Mapping]) -> dict:
    """Merge rank-local exports from one identical native pipeline schedule.

    Requires identical provenance (apart from its explicit ``rank`` field) and
    native schedule, unique extraction ranks and stage owners. This deliberately
    rejects data/tensor-parallel replica merging without a global-rank contract.
    """
    if not plans:
        raise ValueError("at least one native plan is required")
    for plan in plans:
        validate_plan(plan)
        if plan.get("capture_status") in ("blocked", "incomplete"):
            raise ValueError("cannot merge blocked or incomplete native captures")
    merged = json.loads(json.dumps(plans[0]))
    merged.pop("plan_sha256", None)
    merged["graphs"] = {}
    merged["stages"] = {}
    merged["source_files"] = {}
    merged["unresolved"] = []
    ranks: set[int] = set()
    provenance = {k: v for k, v in plans[0]["provenance"].items() if k != "rank"}
    for plan in plans:
        rank = plan.get("extraction", {}).get("rank")
        if type(rank) is not int or rank in ranks:
            raise ValueError("native plan merge requires unique integer extraction ranks")
        ranks.add(rank)
        if {k: v for k, v in plan["provenance"].items() if k != "rank"} != provenance:
            raise ValueError("native plan provenance differs across ranks")
        if plan["schedule"] != plans[0]["schedule"]:
            raise ValueError("native schedule differs across rank exports")
        def namespace(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: f"rank{rank}:{v}" if k == "storage_id" and v is not None
                        else namespace(v) for k, v in value.items()}
            if isinstance(value, list):
                return [namespace(v) for v in value]
            return value
        for stage, value in plan["stages"].items():
            if stage in merged["stages"]:
                raise ValueError(f"duplicate stage {stage}; replica/global-rank merge is unsupported")
            merged["stages"][stage] = namespace(value)
        for graph_id, graph in plan["graphs"].items():
            if graph_id in merged["graphs"]:
                raise ValueError(f"duplicate graph {graph_id}")
            graph = namespace(graph)
            graph.pop("graph_sha256", None)
            graph["graph_sha256"] = _digest(graph)
            merged["graphs"][graph_id] = graph
        for cls, source in plan.get("source_files", {}).items():
            if cls in merged["source_files"] and merged["source_files"][cls] != source:
                raise ValueError(f"native source differs across ranks for {cls}")
            merged["source_files"][cls] = source
        merged["unresolved"].extend(g for g in plan["unresolved"] if g not in merged["unresolved"])
    missing = set(plans[0]["schedule"].get("stage_to_rank", {})) - set(merged["stages"])
    if missing:
        merged["unresolved"].append("native graph exports missing for stages: " + ", ".join(sorted(missing)))
    merged["provenance"] = provenance
    merged["extraction"] = {"ranks": sorted(ranks), "mode": "native_rank_plan_merge",
                            "executes_graphs": False,
                            "source_plan_sha256": [p.get("plan_sha256") for p in plans]}
    merged["status"] = "incomplete" if merged["unresolved"] else "extracted"
    merged["plan_sha256"] = _digest(merged)
    return merged


def seal_plan(plan: Mapping) -> dict:
    """Copy and checksum a producer-updated plan; this is not semantic proof."""
    result = json.loads(json.dumps(plan, allow_nan=False))
    result.pop("plan_sha256", None)
    result["plan_sha256"] = _digest(result)
    validate_plan(result)
    return result
