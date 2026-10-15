"""Bounded interpolation of a verified allocator counter, not tensor simulation.

Fixed intercepts are effective fitted allowances. They do not identify physical
buffers. Context is held fixed; only integer local batch may vary in-range.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from .empirical_memory import _positive

METRIC = "peak_reserved_bytes_between_metric_resets"
SOURCE = "log_calibrated_batch_memory_profile"


def _batch(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("local_batch must be a positive integer")
    return value


def _anchors(anchors):
    if len(anchors) < 2:
        raise ValueError("At least two calibration anchors required")
    batches, hashes = set(), set()
    for anchor in anchors:
        batch = _batch(anchor["local_batch"])
        _positive(anchor["observed_bytes"], "observed_bytes")
        evidence = anchor["evidence"]
        digest = evidence.get("sha256", "")
        if not evidence.get("source") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Each anchor requires source identity and SHA256")
        if batch in batches or digest in hashes:
            raise ValueError("Duplicate calibration batch or source")
        batches.add(batch)
        hashes.add(digest)
    ordered = sorted(anchors, key=lambda a: a["local_batch"])
    if any(b["observed_bytes"] < a["observed_bytes"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Nonmonotonic observations require a different model")
    return ordered


def fit_profile(*, name, context, anchors, assumed_state_bytes, metric_evidence):
    """Fit a continuous affine segment between each pair of observed batches.

    The caller must audit configurations and metric semantics against source.
    Evaluation observations must not be supplied as anchors.
    """
    if any(k in context for k in ("tokens_per_rank", "global_batch", "local_batch")):
        raise ValueError("Batch-dependent fields belong outside fixed context")
    if context.get("memory_metric") != METRIC:
        raise ValueError("Requires a verified peak-reserved counter")
    if not context.get("revision") or metric_evidence.get("revision") != context["revision"]:
        raise ValueError("Metric evidence must match the logged revision")
    if not metric_evidence.get("source_sha256") or not metric_evidence.get("url"):
        raise ValueError("Metric source identity required")
    ordered = _anchors(anchors)
    return {
        "schema_version": 1, "source": SOURCE, "name": name,
        "formula": "piecewise affine interpolation in local batch",
        "context": copy.deepcopy(context), "anchors": copy.deepcopy(ordered),
        "assumed_state_bytes": _positive(assumed_state_bytes, "assumed_state_bytes"),
        "metric_evidence": copy.deepcopy(metric_evidence),
        "target_metric": METRIC,
        "limitations": [
            "Empirical total includes allocator cache and unresolved runtime storage; do not stack other allowances.",
            "Effective intercepts are not measured component allocations or proven execution phases.",
            "Only interpolates local batch under the recorded context; no hardware, sequence, precision or mesh transfer.",
            "Unknown software and allocator fields remain unknown even when context matches.",
            "Not a within-step trajectory, all-rank maximum, independent validation or required-capacity guarantee.",
        ],
    }


def estimate(profile, request):
    if profile.get("source") != SOURCE or profile.get("schema_version") != 1:
        raise ValueError("Expected a supported batch memory profile")
    if profile.get("target_metric") != METRIC:
        raise ValueError("Unsupported target metric")
    if request.get("source") != "batch_memory_request" or set(request) != {"source", "context", "local_batch"}:
        raise ValueError("Expected context and local_batch only; do not stack component allowances")
    if request["context"] != profile["context"]:
        raise ValueError("Outside calibration configuration")
    batch = _batch(request["local_batch"])
    anchors = _anchors(profile["anchors"])
    if not anchors[0]["local_batch"] <= batch <= anchors[-1]["local_batch"]:
        raise ValueError("Outside calibrated batch range; extrapolation is not supported")
    left, right = next((a, b) for a, b in zip(anchors, anchors[1:])
                       if a["local_batch"] <= batch <= b["local_batch"])
    slope = (right["observed_bytes"] - left["observed_bytes"]) / (right["local_batch"] - left["local_batch"])
    intercept = left["observed_bytes"] - slope * left["local_batch"]
    total = intercept + slope * batch
    state = _positive(profile["assumed_state_bytes"], "assumed_state_bytes")
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid fitted result")
    return {
        "source": "log_calibrated_batch_memory_estimate", "profile": profile["name"],
        "target_metric": METRIC, "estimated_bytes": total, "estimated_gib": total / 2**30,
        "local_batch": batch, "segment_batch_range": [left["local_batch"], right["local_batch"]],
        "decomposition": {"assumed_state_bytes": state,
                          "effective_fixed_allowance_bytes": intercept - state,
                          "batch_dependent_allowance_bytes": slope * batch},
        "slope_bytes_per_batch": slope, "source_evidence": [left["evidence"], right["evidence"]],
        "metric_evidence": profile["metric_evidence"],
        "unknown_context_fields": sorted(k for k, v in profile["context"].items() if v is None),
        "independent_validation": False, "required_capacity_bytes": None,
        "capacity_verdict": "unknown", "limitations": profile["limitations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = estimate(json.loads(args.profile.read_text()), json.loads(args.request.read_text()))
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Estimated peak reserved: {result['estimated_gib']:.2f} GiB; required capacity unknown")


if __name__ == "__main__":
    main()
