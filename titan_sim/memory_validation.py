"""Evaluate frozen empirical profiles without fitting the evaluation data."""
import copy
import hashlib
import json

from .empirical_memory import METRIC, estimate


def unique_fact(observation, field):
    values = [record["value"] for record in observation["facts"].get(field, [])]
    if len(values) != 1:
        raise ValueError(f"Expected one unambiguous logged {field}, found {len(values)}")
    return values[0]


def training_signature(observation):
    # Repeated fan-out records do not create independent samples. Excluding
    # source/line permits detection of rewrapped copies of calibration metrics.
    keys = ("step", "value", "unit", "loss", "grad_norm", "tps", "tflops")
    rows = {json.dumps({k: s.get(k) for k in keys}, sort_keys=True)
            for s in observation["samples"] if s["metric"] == METRIC}
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def evaluate_v3(profile, observation, *, compile_policy, calibration_signature=None):
    """Batch-scaling check under known matching hardware/mesh/precision.

    Unknown revisions and counter semantics still prevent a full-engine accuracy
    claim. This routine checks logged configuration before looking at totals.
    """
    if unique_fact(observation, "parameter_elements") != 671026404352:
        raise ValueError("Not the full V3 parameter count")
    if unique_fact(observation, "model") != "deepseek_v3 671B":
        raise ValueError("Not the reviewed full V3 model")
    mesh = unique_fact(observation, "mesh")
    training = unique_fact(observation, "training")
    hardware = unique_fact(observation, "hardware")
    version = unique_fact(observation, "torchtitan_version")
    if version != "0.2.2+gita615ec2":
        raise ValueError("Outside the reviewed TorchTitan package family")
    indicators = observation["indicators"]
    if not indicators.get("full_checkpointing") or not indicators.get("mxfp8_conversion"):
        raise ValueError("Missing FullAC or quantized conversion evidence")
    precision = "mixed NVFP4/MXFP8 compute" if indicators.get("nvfp4_conversion") else "MXFP8 compute"
    context = {"model": "DeepSeek-V3 671B", "framework": "TorchTitan",
               "revision": None, "torch_version": None, "hardware": hardware,
               "world_size": mesh["pp"] * mesh["dp_replicate"] * mesh["dp_shard"] * mesh["cp"] * mesh["tp"],
               **{k: mesh[k] for k in ("tp", "pp", "ep", "cp", "dp_shard", "dp_replicate")},
               "sequence_length": training["sequence_length"],
               "tokens_per_rank": training["local_batch"] * training["sequence_length"],
               "global_batch": training["global_batch"], "microbatches": training["accumulation"],
               "checkpoint_policy": "FullAC", "compile_policy": compile_policy,
               "precision": precision, "memory_metric": METRIC}
    for key, value in profile["context"].items():
        if key not in {"tokens_per_rank", "global_batch"} and context.get(key) != value:
            raise ValueError(f"Configuration mismatch: {key}")
    expected_global = (training["local_batch"] * training["accumulation"]
                       * mesh["dp_shard"] * mesh["dp_replicate"])
    if training["global_batch"] != expected_global:
        raise ValueError("Logged batch and mesh arithmetic disagree")
    baseline = copy.deepcopy(profile["baseline"])
    baseline["checkpoint_proxy_bytes"] = (baseline["layers"] * context["tokens_per_rank"]
                                          * baseline["hidden_size"] * baseline["checkpoint_dtype_bytes"])
    request = {"source": "empirical_memory_request", "context": context, "baseline": baseline}
    fitted = estimate(profile, request, allow_extrapolation=True)
    planning = estimate(profile, request, planning=True, allow_extrapolation=True)
    samples = [s for s in observation["samples"] if s["metric"] == METRIC]
    if not samples or any(s["unit"] != "GiB" or s["bytes"] is None for s in samples):
        raise ValueError("Missing verified-unit training observations")
    maximum = max(samples, key=lambda s: s["bytes"])
    observed = maximum["bytes"] / 2**30
    reused = observation["sha256"] == profile["source_evidence"]["sha256"]
    same_signature = calibration_signature is not None and training_signature(observation) == calibration_signature
    return {
        "source": observation["source_id"], "sha256": observation["sha256"],
        "context": context, "package_version": version, "local_batch": training["local_batch"],
        "profile": profile["name"], "profile_factor_unchanged": profile["runtime_factor"],
        "calibration_source_reused": reused, "same_metric_signature_as_calibration": same_signature,
        "evaluation_kind": "calibration_replay" if reused else "separate_log_counter_comparison",
        "configuration_extrapolation": fitted["extrapolation"],
        "mismatches": fitted["configuration_mismatches"],
        "observed_max_gib": observed, "frozen_estimate_gib": fitted["estimated_gib"],
        "planning_estimate_gib": planning["estimated_gib"],
        "frozen_error_percent": 100 * (fitted["estimated_gib"] / observed - 1),
        "planning_error_percent": 100 * (planning["estimated_gib"] / observed - 1),
        "maximum_evidence": maximum,
        "training_completed": bool(indicators.get("training_completed")),
        "nan_gradient_observed": bool(indicators.get("nan_gradient")),
        "oom_observed": bool(indicators.get("oom")),
        "end_to_end_validation": False,
        "limitations": ["Sparse printed counter, not required device capacity or within-step trajectory.",
                        "Full software identity, logger semantics and complete rank coverage remain unverified.",
                        "A separate file/run is not proof of statistical independence."],
    }
