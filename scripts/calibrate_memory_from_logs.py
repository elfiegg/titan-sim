#!/usr/bin/env python3
"""Reproduce the empirical memory catalog from the reviewed, sanitized logs.

Optional raw paths verify the audit hashes locally; raw logs are never copied.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from titan_sim.empirical_memory import allocator_ratios, estimate, fit_profile

ASSUMPTIONS = [
    "Representative evenly sharded rank: 671026404352 parameter elements / 256 GPUs.",
    "State proxy assumes 4B parameters + 4B gradients + two 4B Adam moments per local element. State dtype/lifetime is not verified by these logs.",
    "Checkpoint proxy assumes 61 BF16 hidden-state boundaries, hidden size 7168, using logged local tokens. This is a normalization basis, not captured FullAC storage.",
    "The fitted runtime bundle includes the checkpoint proxy and all unexplained counter usage, including any baseline error. It does not identify individual components.",
    "Hardware/software/precision/attention differences are confounded; cross-run similarity does not establish a universal scaling law.",
]


def build_catalog(observations, audit):
    profiles, results, requests = {}, {}, {}
    for key, run in audit["runs"].items():
        observation = observations["runs"][key]
        source = run["primary_source"]
        expected_hash = audit["reviewed_sources"][source]["sha256"]
        if observation["source_id"] != source or observation["sha256"] != expected_hash:
            raise ValueError(f"Audited source identity/hash mismatch: {key}")
        if not key.startswith("titan_v3_"):
            continue
        context = run["context"]
        if any(context.get(k) != v for k, v in {
            "model": "DeepSeek-V3 671B", "framework": "TorchTitan",
            "world_size": 256, "pp": 1, "tp": 1, "ep": 64,
            "sequence_length": 4096, "checkpoint_policy": "FullAC",
        }.items()):
            raise ValueError("This reviewed baseline is restricted to the supplied V3 PP1 runs")
        baseline = {
            "state_bytes": 671026404352 // 256 * 16,
            "checkpoint_proxy_bytes": 61 * context["tokens_per_rank"] * 7168 * 2,
            "parameter_elements": 671026404352, "state_bytes_per_local_parameter": 16,
            "layers": 61, "hidden_size": 7168, "checkpoint_dtype_bytes": 2,
        }
        profile = fit_profile(observation, context, baseline, name=key, assumptions=ASSUMPTIONS)
        profile["configuration_evidence"] = run["configuration_evidence"]
        profile["execution_completion"] = run.get("completion")
        profile["context_audit"] = "docs/validation/runtime_logs/run_audit.json"
        profiles[key] = profile
        request = {"source": "empirical_memory_request", "context": context, "baseline": baseline}
        requests[key] = request
        results[key] = {"fitted": estimate(profile, request), "planning": estimate(profile, request, planning=True)}

    stash = {}
    sizes = {"uint8": 1, "bfloat16": 2, "float32": 4}
    for key in ("megatron_v3", "megatron_v4"):
        components = observations["runs"][key]["observed_components"]
        for c in components:
            if math.prod(c["shape"]) * sizes[c["storage_dtype"]] != c["payload_bytes"]:
                raise ValueError("Observed component payload disagrees with geometry")
        stash[key] = {
            "source": "observed_buffer_geometry", "payload_factor": 1.0,
            "gpu_payload_bytes": sum(c["payload_bytes"] for c in components if c["device"] == "cuda"),
            "host_payload_bytes": (sum(c["payload_bytes"] for c in components if c["device"] == "cpu")
                                   if any(c["device"] == "cpu" for c in components) else None),
            "components": components,
            "limitations": "Recorded payload sum only; simultaneous residency, aliases and all-rank coverage unverified. Do not add to a fitted total.",
        }
    rows = allocator_ratios(observations["runs"]["megatron_v4"])
    ranges = {}
    for step in sorted({r["step"] for r in rows}):
        group = [r for r in rows if r["step"] == step]
        ranges[str(step)] = {
            metric: {"min": min(r[metric] for r in group), "max": max(r[metric] for r in group)}
            for metric in ["reserved_over_current_allocated", "high_water_allocated_over_current", "high_water_reserved_over_current"]}
    catalog = {"schema_version": 1, "source": "log_calibrated_memory_catalog",
               "profiles": profiles, "estimates": results, "observed_stash": stash,
               "v4_allocator": {"context": audit["runs"]["megatron_v4"]["context"],
                                "sha256": observations["runs"]["megatron_v4"]["sha256"],
                                "rows": rows, "ranges_by_iteration": ranges,
                                "scope": "Four sampled ranks, iterations 1 and 2, before iteration-4 graph capture; NaN gradients from iteration 2. Ratios assume common logger units."},
               "independent_validation": False}
    return catalog, requests


def plot(catalog, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["titan_v3_expert_mxfp8_batch8", "titan_v3_mxfp8_batch16", "titan_v3_nvfp4_batch16"]
    labels = ["GB200 · batch 8\nExpert MXFP8", "GB300 · batch 16\nMXFP8", "GB300 · batch 16\nMixed NVFP4/MXFP8"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for i, key in enumerate(names):
        p = catalog["profiles"][key]
        s = p["baseline"]["state_bytes"] / 2**30
        m = catalog["estimates"][key]["fitted"]["estimated_gib"]
        planning = catalog["estimates"][key]["planning"]["estimated_gib"]
        ax.bar(i, s, width=.6, color="#48749c", label="Assumed training-state proxy" if i == 0 else None)
        ax.bar(i, m-s, bottom=s, width=.6, color="#d6a350", label="Fitted runtime bundle (includes checkpoints)" if i == 0 else None)
        ax.scatter(i, planning, marker="_", s=700, color="#152e3e", label="Rounded planning factor" if i == 0 else None)
        ax.text(i, planning+3, f"Logged max {m:.2f} GiB\nPlanning {planning:.2f} GiB", ha="center", fontsize=10)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("GiB per represented GPU / printed counter")
    ax.set_ylim(0, 205)
    ax.set_title("DeepSeek-V3: empirical counter calibration")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.08, .025, "Fitted totals reproduce their calibration samples; this is not independent validation or a required-capacity bound.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "memory_factors.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--additional-log", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    source = ROOT / "docs/validation/runtime_logs"
    audit = json.loads((source / "run_audit.json").read_text())
    observations = json.loads((source / "observations.json").read_text())
    verification = {}
    if args.raw_root is not None or args.additional_log is not None:
        if args.raw_root is None or args.additional_log is None:
            parser.error("Supply both --raw-root and --additional-log to verify every reviewed source")
        for name, record in audit["reviewed_sources"].items():
            path = args.additional_log if name.startswith("additional/") else args.raw_root / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Raw source hash mismatch: {name}")
            verification[name] = {"sha256": digest, "verified": True}
    catalog, requests = build_catalog(observations, audit)
    out = ROOT / "docs/validation/memory_factors"
    out.mkdir(parents=True, exist_ok=True)
    profile_dir = ROOT / "profiles/memory"
    profile_dir.mkdir(parents=True, exist_ok=True)
    example_dir = ROOT / "examples/memory/empirical"
    example_dir.mkdir(parents=True, exist_ok=True)
    for key, profile in catalog["profiles"].items():
        (profile_dir / f"{key}.json").write_text(json.dumps(profile, indent=2) + "\n")
        (example_dir / f"{key}.json").write_text(json.dumps(requests[key], indent=2) + "\n")
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    if verification:
        (out / "raw_source_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if args.plot:
        plot(catalog, out)
    print(json.dumps({key: {"factor": p["runtime_factor"], "planning": p["planning_factor"]}
                      for key, p in catalog["profiles"].items()}, indent=2))


if __name__ == "__main__":
    main()
