"""Declared component lifetimes and capacity scenarios, independent of GPU runs.

Spec: source="independent_component_spec", ranks=[0, ...], ordered phases=[...],
coverage={category: "complete"|"not_applicable"|"unknown"}, components=[...].
Optional coverage_by_rank maps string rank IDs to category overrides. Missing
coverage is unknown. Each component requires name, category, rank, storage_id,
provenance={kind, detail}, and live_range={start: phase, end: phase or null}.
End is exclusive; null means through the final phase. Sizes are integer
size_bytes, or shape plus an explicit storage dtype. size_bytes=null is unknown.
Aliases share a rank/storage_id, size and accounting category; their live ranges
are unioned. device defaults to "cuda"; explicit "cpu" components (including
pinned host buffers) contribute only to separate known_host_bytes totals.
Aliases cannot change devices. Host capacity/complete host coverage is unmodeled.
allocator_cache must declare accounting="unused_reserved": total
reserved bytes cannot be added to allocated storage.

Provenance kinds: upstream_shape, cpu_metadata, analytical_assumption,
declared_assumption, unknown. Unknown sizes require unknown provenance. Numeric
sizes are declarations, not measurements validated by this module. GPU-observed
or fitted-residual components are not accepted as independent predictions.

Hardware requires name and memory_capacity_bytes, or memory_capacity={value,
unit}; B/KB/MB/GB/TB are decimal, KiB/MiB/GiB/TiB binary. No timing fields are
needed. Capacity scenarios reuse exactly the same component bytes. These are
phase-granular budgets, not runtime traces or an upstream scheduling model.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path


CATEGORIES = (
    "parameters", "optimizer", "gradients", "gathered_parameters", "activations",
    "pipeline_buffers", "moe_buffers", "quantization", "workspace", "external",
    "allocator_cache",
)
PROVENANCE_KINDS = frozenset({"upstream_shape", "cpu_metadata", "analytical_assumption",
                              "declared_assumption", "unknown"})
DTYPE_BYTES = {
    "bool": 1, "uint8": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "complex64": 8, "complex128": 16, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "float8_e4m3fnuz": 1, "float8_e5m2fnuz": 1,
}
UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
         "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def capacity_bytes(profile):
    """Normalize explicitly supplied capacity without inventing hardware specs."""
    if not isinstance(profile, dict):
        raise ValueError("hardware must be an object")
    _text(profile.get("name"), "hardware.name")
    if ("memory_capacity_bytes" in profile) == ("memory_capacity" in profile):
        raise ValueError("hardware requires exactly one capacity representation")
    if "memory_capacity_bytes" in profile:
        result = _integer(profile["memory_capacity_bytes"], "memory_capacity_bytes")
    else:
        capacity = profile["memory_capacity"]
        if (not isinstance(capacity, dict) or not isinstance(capacity.get("unit"), str)
                or capacity["unit"] not in UNITS):
            raise ValueError("memory_capacity requires value and an explicit supported unit")
        value = capacity.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError("capacity value must be numeric")
        try:
            amount = Decimal(str(value)) * UNITS[capacity["unit"]]
        except InvalidOperation as error:
            raise ValueError("invalid capacity value") from error
        if not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value():
            raise ValueError("capacity must resolve to a positive integer number of bytes")
        result = int(amount)
    if result <= 0:
        raise ValueError("capacity must be positive")
    return result


def _size(component):
    if "size_bytes" in component:
        if "shape" in component or "dtype" in component:
            raise ValueError("use size_bytes or shape+storage dtype, not both")
        value = component["size_bytes"]
        return None if value is None else _integer(value, "component.size_bytes")
    shape = component.get("shape")
    if not isinstance(shape, list):
        raise ValueError("component requires size_bytes or shape+dtype")
    dtype = component.get("dtype")
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise ValueError("component requires a supported explicit storage dtype; precision labels are insufficient")
    return math.prod(_integer(dim, "shape dimension") for dim in shape) * DTYPE_BYTES[dtype]


def _coverage(mapping):
    if not isinstance(mapping, dict) or set(mapping) - set(CATEGORIES):
        raise ValueError("coverage must map supported categories to coverage states")
    if any(not isinstance(value, str) or value not in {"complete", "not_applicable", "unknown"}
           for value in mapping.values()):
        raise ValueError("coverage states must be complete, not_applicable or unknown")
    return mapping


def component_ledger(spec):
    """Return unique-storage phase budgets without treating unknown bytes as zero."""
    if not isinstance(spec, dict) or spec.get("source") != "independent_component_spec":
        raise ValueError("source must be independent_component_spec; GPU observations are not predicted components")
    ranks, phases = spec.get("ranks"), spec.get("phases")
    if not isinstance(ranks, list) or not ranks:
        raise ValueError("ranks must be a nonempty list")
    for rank in ranks:
        _integer(rank, "rank")
    if len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be unique")
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases must be a nonempty ordered list")
    for phase in phases:
        _text(phase, "phase")
    if len(set(phases)) != len(phases):
        raise ValueError("phases must be unique")
    base_coverage = _coverage(spec.get("coverage", {}))
    overrides = spec.get("coverage_by_rank", {})
    if not isinstance(overrides, dict) or set(overrides) - {str(rank) for rank in ranks}:
        raise ValueError("coverage_by_rank must use declared string rank IDs")
    coverage = {rank: {**base_coverage, **_coverage(overrides.get(str(rank), {}))} for rank in ranks}
    components = spec.get("components")
    if not isinstance(components, list):
        raise ValueError("components must be a list")
    storages, normalized, names = {}, [], set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("component must be an object")
        name = _text(component.get("name"), "component.name")
        rank = _integer(component.get("rank"), "component.rank")
        if rank not in ranks or (rank, name) in names:
            raise ValueError("component requires a declared rank and unique name within that rank")
        names.add((rank, name))
        category = component.get("category")
        if category not in CATEGORIES:
            raise ValueError("unsupported component category")
        if coverage[rank].get(category) == "not_applicable":
            raise ValueError("not_applicable category cannot contain components")
        if component.get("source", "independent_component_spec") != "independent_component_spec":
            raise ValueError("components cannot come from GPU observations or residual fitting")
        device = component.get("device", "cuda")
        if device not in ("cuda", "cpu"):
            raise ValueError("component device must be cuda or cpu")
        if category == "allocator_cache" and component.get("accounting") != "unused_reserved":
            raise ValueError("allocator_cache requires accounting=unused_reserved; allocated+reserved double counts")
        if category != "allocator_cache" and component.get("accounting", "allocated") != "allocated":
            raise ValueError("only allocator_cache can represent unused reserved bytes")
        provenance = component.get("provenance")
        if (not isinstance(provenance, dict) or not isinstance(provenance.get("kind"), str)
                or provenance["kind"] not in PROVENANCE_KINDS):
            raise ValueError("component requires independent provenance; observed or residual-fit bytes are unsupported")
        _text(provenance.get("detail"), "provenance.detail")
        size = _size(component)
        if (size is None) != (provenance["kind"] == "unknown"):
            raise ValueError("unknown size and unknown provenance must be declared together")
        storage_id = _text(component.get("storage_id"), "storage_id")
        lifetime = component.get("live_range")
        if not isinstance(lifetime, dict) or "end" not in lifetime or lifetime.get("start") not in phases:
            raise ValueError("live_range requires start phase and exclusive end phase or null")
        if lifetime["end"] is not None and lifetime["end"] not in phases:
            raise ValueError("live_range end must be a declared phase or null")
        start = phases.index(lifetime["start"])
        end = len(phases) if lifetime["end"] is None else phases.index(lifetime["end"])
        if end <= start:
            raise ValueError("live_range must have positive length")
        key = rank, storage_id
        if key in storages and (storages[key]["size_bytes"], storages[key]["category"], storages[key]["device"]) != (size, category, device):
            raise ValueError("aliases must agree on full backing-storage size, accounting category and device")
        storage = storages.setdefault(key, dict(size_bytes=size, category=category, device=device, live=set(), names=[]))
        storage["live"].update(range(start, end))
        storage["names"].append(name)
        normalized.append({**component, "device": device, "resolved_size_bytes": size})

    per_rank = {}
    for rank in ranks:
        uncovered = [category for category in CATEGORIES if coverage[rank].get(category, "unknown") == "unknown"]
        rows = []
        for index, phase in enumerate(phases):
            by_category = {category: 0 for category in CATEGORIES}
            host_by_category = {category: 0 for category in CATEGORIES}
            unknown, unknown_host = [], []
            live_storage_ids, live_host_storage_ids = [], []
            for (owner, storage_id), storage in storages.items():
                if owner != rank or index not in storage["live"]:
                    continue
                host = storage["device"] == "cpu"
                (live_host_storage_ids if host else live_storage_ids).append(storage_id)
                if storage["size_bytes"] is None:
                    (unknown_host if host else unknown).append({"storage_id": storage_id, "category": storage["category"], "names": storage["names"]})
                else:
                    (host_by_category if host else by_category)[storage["category"]] += storage["size_bytes"]
            known = sum(by_category.values())
            rows.append(dict(phase=phase, known_bytes=known, by_category=by_category,
                             live_storage_ids=live_storage_ids, unknown_components=unknown,
                             known_host_bytes=sum(host_by_category.values()), host_by_category=host_by_category,
                             live_host_storage_ids=live_host_storage_ids, unknown_host_components=unknown_host,
                             uncovered_categories=uncovered,
                             required_memory_bytes=None if uncovered or unknown else known))
        maximum = max(row["known_bytes"] for row in rows)
        complete = all(row["required_memory_bytes"] is not None for row in rows)
        per_rank[str(rank)] = dict(phases=rows, max_known_bytes=maximum,
                                  max_known_host_bytes=max(row["known_host_bytes"] for row in rows),
                                  peak_known_phases=[row["phase"] for row in rows if row["known_bytes"] == maximum],
                                  required_peak_bytes=maximum if complete else None)
    maxima = [dict(phase=phase, max_per_rank_known_bytes=max(per_rank[str(rank)]["phases"][i]["known_bytes"] for rank in ranks),
                   max_per_rank_known_host_bytes=max(per_rank[str(rank)]["phases"][i]["known_host_bytes"] for rank in ranks))
              for i, phase in enumerate(phases)]
    maximum = max(row["max_known_bytes"] for row in per_rank.values())
    complete = all(row["required_peak_bytes"] is not None for row in per_rank.values())
    return dict(name=spec.get("name", "declared component budget"), components=normalized,
                per_rank=per_rank, max_per_rank_by_phase=maxima,
                max_over_ranks_known_bytes=maximum,
                max_over_ranks_known_host_bytes=max(row["max_known_host_bytes"] for row in per_rank.values()),
                required_max_over_ranks_bytes=maximum if complete else None,
                coverage_complete=complete)


def budget_scenarios(spec, hardware_profiles):
    """Compare one immutable byte ledger against explicitly supplied capacities."""
    ledger = component_ledger(spec)
    scenarios = []
    for profile in hardware_profiles:
        capacity = capacity_bytes(profile)
        per_rank = {}
        for rank, row in ledger["per_rank"].items():
            known = row["max_known_bytes"]
            if known > capacity:
                verdict = "declared_known_components_exceed_capacity"
            elif row["required_peak_bytes"] is None:
                verdict = "unknown"
            else:
                verdict = "conditional_fits_capacity"
            per_rank[rank] = dict(known_bytes=known, required_peak_bytes=row["required_peak_bytes"],
                                  known_host_peak_bytes=row["max_known_host_bytes"], host_capacity_evaluated=False,
                                  known_headroom_bytes=capacity-known, capacity_verdict=verdict,
                                  gpu_fit_validated=False)
        scenarios.append(dict(hardware=profile["name"], memory_capacity_bytes=capacity,
                              source="capacity_scenario", per_rank=per_rank,
                              gpu_fit_validated=False))
    return dict(schema_version=1, source="independent_component_budget", ledger=ledger,
                capacity_scenarios=scenarios,
                assumptions=[
                    "Known bytes mean numerically specified, not experimentally validated; declared assumptions remain conditional.",
                    "Missing category coverage or unknown live components keeps required memory unknown.",
                    "A known-component overflow is conditional on the declared sizes and lifetimes; no GPU fit is validated.",
                    "Phase lifetimes are caller declarations, not a CPU runtime trace or an upstream pipeline schedule.",
                    "Pipeline buffers use explicit declared lifetimes; no actual upstream pipeline-parallel execution is claimed.",
                    "CPU and pinned-host buffers are reported separately and never consume GPU capacity; host capacity and full host coverage are not evaluated.",
                    "Aliased backing storage is counted once per rank and phase; phase maxima are not summed.",
                    "Allocator cache means unused reserved bytes only; total reserved and allocated bytes are never added.",
                    "Storage dtype must be explicit; a compute precision or hardware name does not change storage bytes.",
                    "Hardware scenarios reuse the same ledger; different kernels, workspaces or sharding require a separate explicit spec.",
                ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--hardware", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = budget_scenarios(json.loads(args.spec.read_text()),
                                  [json.loads(path.read_text()) for path in args.hardware])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"coverage_complete": result["ledger"]["coverage_complete"],
                      "max_over_ranks_known_bytes": result["ledger"]["max_over_ranks_known_bytes"],
                      "required_max_over_ranks_bytes": result["ledger"]["required_max_over_ranks_bytes"]}))


if __name__ == "__main__":
    main()
