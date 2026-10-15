"""Log-calibrated counter estimates, separate from independent storage simulation.

A fitted runtime bundle is not an activation census. It absorbs unmodeled
storage, baseline error, and the semantics of the counter used for calibration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


METRIC = "training_memory_unspecified"
FORMULA = "state_bytes + runtime_factor * checkpoint_proxy_bytes"
CONTEXT_FIELDS = {
    "revision", "torch_version", "parameter_dtype", "gradient_dtype",
    "optimizer_state_dtype", "optimizer", "attention", "expert_tp",
    "routing_policy", "offload_policy", "allocator_policy", "counter_window",
}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def fit_profile(observation, context, baseline, *, name, assumptions):
    """Fit one coefficient to the largest retained training counter.

    The baseline and context are explicitly reviewed inputs, not inferred from
    a model name. Repeated rankless rows are retained as counts, not rank coverage.
    """
    if observation.get("source") != "training_log_observation":
        raise ValueError("Expected a training_log_observation")
    if not observation.get("sha256") or not observation.get("source_id"):
        raise ValueError("Calibration requires source identity and hash")
    if context.get("memory_metric") != METRIC or not assumptions:
        raise ValueError("Calibration requires matching metric and explicit assumptions")
    state = _positive(baseline["state_bytes"], "state_bytes")
    proxy = _positive(baseline["checkpoint_proxy_bytes"], "checkpoint_proxy_bytes")
    samples = [s for s in observation["samples"] if s["metric"] == METRIC]
    if not samples:
        raise ValueError("No retained training counters")
    by_step = defaultdict(list)
    for s in samples:
        if s.get("unit") != "GiB" or s.get("bytes") is None:
            raise ValueError("Training calibration requires verified GiB conversion")
        value = _positive(s["bytes"], "observed bytes")
        if not math.isclose(value, s["value"] * 2**30, abs_tol=1, rel_tol=0):
            raise ValueError("Inconsistent counter conversion")
        if s.get("source") != observation["source_id"]:
            raise ValueError("Sample source does not match observation")
        by_step[s["step"]].append(s)
    maximum = max(samples, key=lambda s: s["bytes"])
    factor = (maximum["bytes"] - state) / proxy
    _positive(factor, "fitted runtime factor")
    return {
        "schema_version": 1,
        "source": "log_calibrated_memory_profile",
        "name": name,
        "formula": FORMULA,
        "context": dict(context),
        "baseline": dict(baseline),
        "assumptions": list(assumptions),
        "runtime_factor": factor,
        "planning_factor": math.ceil(factor * 10) / 10,
        "planning_factor_policy": "Round fitted coefficient upward to 0.1; heuristic, not a confidence bound",
        "target_metric": METRIC,
        "metric_semantics_verified": False,
        "source_evidence": {
            "source": observation["source_id"], "sha256": observation["sha256"],
            "maximum_sample": maximum,
        },
        "sampled_counter_history": [
            {"step": step, "minimum_bytes": min(s["bytes"] for s in rows),
             "maximum_bytes": max(s["bytes"] for s in rows),
             "sample_count": len(rows),
             "known_ranks": sorted({s["rank"] for s in rows if s["rank"] is not None}),
             "has_unranked_samples": any(s["rank"] is None for s in rows)}
            for step, rows in sorted(by_step.items())
        ],
        "unknown_context_fields": sorted(
            k for k in set(context) | CONTEXT_FIELDS if context.get(k) is None),
        "limitations": [
            "Fit to sampled printed memory; allocated/reserved/peak semantics and reset window unverified.",
            "Runtime factor includes the checkpoint proxy plus unresolved storage and baseline error.",
            "Not an allocation trajectory, all-rank maximum, or independently validated capacity requirement.",
            "Do not add separately modeled workspaces, quantization, communication or allocator cache to this total.",
        ],
    }


def estimate(profile, request, *, planning=False, allow_extrapolation=False):
    """Estimate the same counter; mismatched hardware/configuration is opt-in.

    Even an exact context match is not proof of software equivalence: missing
    revisions and other unknown fields remain visible in the result.
    """
    if profile.get("source") != "log_calibrated_memory_profile" or profile.get("formula") != FORMULA:
        raise ValueError("Expected a supported log-calibrated profile")
    if request.get("source") != "empirical_memory_request":
        raise ValueError("Expected an empirical_memory_request, separate from the component ledger")
    if set(request) - {"source", "context", "baseline"}:
        raise ValueError("Unexpected fields; additional component factors must not be stacked on the fitted total")
    mismatches = {}
    for key in set(profile["context"]) | set(request["context"]):
        old, new = profile["context"].get(key), request["context"].get(key)
        if old != new:
            mismatches[key] = {"calibration": old, "request": new}
    baseline = request["baseline"]
    for key in set(baseline) | set(profile["baseline"]):
        if baseline.get(key) != profile["baseline"].get(key):
            mismatches[f"baseline.{key}"] = {
                "calibration": profile["baseline"].get(key), "request": baseline.get(key)}
    if mismatches and not allow_extrapolation:
        raise ValueError(f"Outside calibration configuration: {', '.join(sorted(mismatches))}")
    state = _positive(baseline["state_bytes"], "state_bytes")
    proxy = _positive(baseline["checkpoint_proxy_bytes"], "checkpoint_proxy_bytes")
    factor = _positive(profile["planning_factor" if planning else "runtime_factor"], "runtime factor")
    total = state + factor * proxy
    return {
        "schema_version": 1, "source": "log_calibrated_memory_estimate",
        "profile": profile["name"], "context": request["context"],
        "target_metric": profile["target_metric"], "estimated_bytes": total,
        "estimated_gib": total / 2**30,
        "decomposition": {"assumed_state_bytes": state, "checkpoint_proxy_bytes": proxy,
                          "runtime_bundle_bytes_including_proxy": factor * proxy},
        "runtime_factor": factor, "rounded_planning_factor_used": planning,
        "extrapolation": bool(mismatches), "configuration_mismatches": mismatches,
        "independent_validation": False, "required_capacity_bytes": None,
        "capacity_verdict": "unknown", "external_allocation_coverage": "unknown",
        "unknown_context_fields": profile["unknown_context_fields"],
        "source_evidence": profile["source_evidence"],
        "assumptions": profile["assumptions"], "limitations": profile["limitations"],
    }


def allocator_ratios(observation):
    """Pair allocator counters only within the same source line, rank and step.

    Ratios cancel a common unknown GB divisor. They do not establish a peak
    reset window or a causal fragmentation fraction. Missing rows are skipped.
    """
    required = ["allocated", "reserved", "max-allocated", "max-reserved"]
    grouped = defaultdict(dict)
    for s in observation["samples"]:
        key = (s["source"], s["line"], s.get("rank"), s.get("step"))
        metric = s["metric"]
        if metric.startswith("allocator-"):
            metric = "mem-" + metric.removeprefix("allocator-") + "-gigabytes"
        # This key is an internal join alias, not a unit conversion. The sample
        # and returned printed_unit retain MB/GB exactly as observed.
        if metric in grouped[key]:
            raise ValueError("Duplicate allocator metric within one row")
        grouped[key][metric] = s
    rows = []
    for (source, line, rank, step), samples in grouped.items():
        keys = [f"mem-{name}-gigabytes" for name in required]
        if rank is None or not all(k in samples for k in keys):
            continue
        if len({samples[k]["unit"] for k in keys}) != 1:
            raise ValueError("Allocator ratios require the same unit within a row")
        values = {name: _positive(samples[k]["value"], k) for name, k in zip(required, keys)}
        allocated = values["allocated"]
        if values["reserved"] < allocated or values["max-allocated"] < allocated or values["max-reserved"] < max(values["reserved"], values["max-allocated"]):
            raise ValueError("Inconsistent allocator row")
        rows.append({
            "source": source, "line": line, "rank": rank, "step": step,
            "printed_values": values, "printed_unit": samples[keys[0]]["unit"],
            "reserved_over_current_allocated": values["reserved"] / allocated,
            "high_water_allocated_over_current": values["max-allocated"] / allocated,
            "high_water_reserved_over_current": values["max-reserved"] / allocated,
            "peak_reset_verified": False,
        })
    return sorted(rows, key=lambda r: (r["step"], r["rank"]))


def project_allocator_envelope(current_allocated_bytes, row):
    """Explicit scenario using one observed allocator row's ratios once.

    Caller selects the appropriate run/rank/window. No implicit hardware or
    workload transfer is claimed. This does not accept a V3 unspecified counter.
    """
    current = _positive(current_allocated_bytes, "current_allocated_bytes")
    peak = _positive(row["high_water_allocated_over_current"], "allocated ratio")
    reserved = _positive(row["high_water_reserved_over_current"], "reserved ratio")
    if peak < 1 or reserved < peak:
        raise ValueError("Inconsistent allocator envelope ratios")
    return {"source": "conditional_allocator_ratio_scenario",
            "input_metric": "current_allocated_bytes", "input_bytes": current,
            "projected_high_water_allocated_bytes": current * peak,
            "projected_high_water_reserved_bytes": current * reserved,
            "ratio_evidence": row, "capacity_verdict": "unknown",
            "required_capacity_bytes": None,
            "limitations": ["Caller must justify configuration, rank and window transfer.",
                            "Pre-graph early-run observations do not bound later graph pools or external allocations."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("--planning", action="store_true")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = estimate(json.loads(args.profile.read_text()), json.loads(args.request.read_text()),
                          planning=args.planning, allow_extrapolation=args.allow_extrapolation)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("estimated_gib", "runtime_factor", "extrapolation", "capacity_verdict")}, indent=2))


if __name__ == "__main__":
    main()
