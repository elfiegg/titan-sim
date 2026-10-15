#!/usr/bin/env python3
"""Audit an expanded archive and evaluate unchanged CPU memory models."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from titan_sim.empirical_memory import allocator_ratios, project_allocator_envelope
from titan_sim.log_observations import ANSI, FACT_PATTERNS, parse_log, summarize
from titan_sim.memory_validation import evaluate_v3, training_signature
from titan_sim.quantization import packed_layout
from titan_sim.runtime import simulate


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def inspect_archive(root):
    """Check every manifest record, resolving declared content duplicates."""
    manifest = json.loads((root / "MANIFEST.json").read_text())
    seen, entries, candidates = {}, [], []
    marker = re.compile(rb"step:\s*\d+[^\n]*memory:|mem-allocated-gigabytes|allocate_stash_buffers cuda:")
    full = re.compile(rb"size:\s*671,026,404,352 total parameters")
    for record in manifest["files"]:
        stored = record.get("duplicate_of", record["archive_path"])
        path = (root / stored).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Archive path escapes input root")
        if stored not in seen:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            seen[stored] = digest
            if marker.search(data):
                text = ANSI.sub("", data.decode(errors="replace"))
                facts = {}
                for field, (pattern, convert) in FACT_PATTERNS.items():
                    # Census only. Detailed selected-run parsing retains source lines.
                    values = {json.dumps(convert(m), sort_keys=True) for m in re.finditer(pattern, text)}
                    facts[field] = [json.loads(v) for v in sorted(values)]
                values = [float(v) for v in re.findall(r"step:\s*\d+[^\n]*?memory:\s*([\d.]+)GiB", text)]
                candidates.append({"source": stored, "sha256": digest,
                                   "full_v3_parameter_count_present": bool(full.search(data)),
                                   "facts_census": facts,
                                   "max_printed_training_memory_gib": max(values) if values else None,
                                   "allocator_counter_present": "mem-allocated-gigabytes" in text,
                                   "stash_geometry_present": "allocate_stash_buffers cuda:" in text})
        if seen[stored] != record["sha256"]:
            raise ValueError(f"Archive hash mismatch: {stored}")
        entries.append({"source": record["archive_path"], "stored_source": stored,
                        "sha256": seen[stored], "verified": True})
    return {"manifest_entries": len(entries), "unique_stored_files": len(seen),
            "duplicate_entries": len(entries) - len(seen), "verified_entries": entries,
            "memory_candidates": candidates,
            "full_v3_count_candidates": sum(c["full_v3_parameter_count_present"] for c in candidates),
            "scope": "All supplied manifest entries hashed. Detailed validation is limited to explicitly selected, configuration-checked runs."}


def compilation_policy(text):
    blocks = "Compiling each TransformerBlock with torch.compile" in text
    loss = "Compiling the loss function with torch.compile" in text
    return "loss and each TransformerBlock" if blocks and loss else None


V4_FIELDS = ("num_layers", "hidden_size", "num_attention_heads", "num_moe_experts",
             "moe_router_topk", "tensor_model_parallel_size", "pipeline_model_parallel_size",
             "virtual_pipeline_model_parallel_size", "expert_model_parallel_size",
             "expert_tensor_parallel_size", "seq_length", "micro_batch_size", "global_batch_size",
             "fp8_recipe", "main_grads_dtype", "params_dtype", "pipeline_model_parallel_layout",
             "recompute_granularity", "recompute_modules", "cuda_graph_impl",
             "moe_router_force_load_balancing", "offload_modules")


def v4_config(text):
    result = {}
    for field in V4_FIELDS:
        matches = [(i, m[1].split("  (Mcore default:")[0].strip())
                   for i, line in enumerate(text.splitlines(), 1)
                   if (m := re.match(r"\s*" + field + r":\s*(.*)", line))]
        values = sorted({value for _, value in matches})
        if len(values) != 1:
            raise ValueError(f"Missing or conflicting V4 config: {field}: {values}")
        result[field] = {"value": values[0], "line": matches[0][0]}
    return result


def plot_results(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for precision, label, color in [("MXFP8 compute", "MXFP8", "#17699a"),
                                    ("mixed NVFP4/MXFP8 compute", "Mixed NVFP4/MXFP8", "#b36a18")]:
        rows = sorted((r for r in results["v3_evaluations"] if r["cohort"] == "aws_expA"
                       and r["context"]["precision"] == precision), key=lambda r: r["local_batch"])
        x = [r["local_batch"] for r in rows]
        axes[0].plot(x, [r["observed_max_gib"] for r in rows], "o-", color=color, label=label + " observed")
        axes[0].plot(x, [r["frozen_estimate_gib"] for r in rows], "--", color=color, label=label + " frozen CPU")
        axes[1].plot(x, [r["frozen_error_percent"] for r in rows], "o-", color=color, label=label)
    axes[0].set_ylabel("Largest printed training-memory counter (GiB)")
    axes[1].set_ylabel("Signed error: (CPU / observed − 1) × 100 (%)")
    for ax in axes:
        ax.set_xlabel("Local batch; sequence 4,096; 256 GB300 GPUs")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    axes[1].axhline(0, color="gray", lw=.8)
    fig.suptitle("Frozen batch-16 memory factors underestimate smaller batches")
    fig.text(.07, .025, "AWS sweep shown; Lyris sweep repeats the same rounded memory maxima. No coefficients refitted.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(output / "batch_scaling.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/expanded_logs")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    census = inspect_archive(args.archive)
    write(args.output / "archive_audit.json", census)
    old = json.loads((ROOT / "docs/validation/runtime_logs/observations.json").read_text())["runs"]
    profiles = {kind: json.loads((ROOT / f"profiles/memory/titan_v3_{kind}_batch16.json").read_text())
                for kind in ("mxfp8", "nvfp4")}
    fingerprints = {kind: hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
                    for kind, profile in profiles.items()}
    evaluations, observations, excluded = [], {}, []
    for candidate in census["memory_candidates"]:
        source = candidate["source"]
        # Paths select a cohort only; evaluate_v3 checks runtime evidence.
        if "dsv3_performance_torchtitan/" not in source or not any(
                f"/{cohort}/" in source for cohort in ("aws_expA", "aws_expB", "lyris_expA", "lyris_expB")):
            continue
        path = args.archive / source
        observation = parse_log(path, source_id=source)
        if observation["sha256"] != candidate["sha256"]:
            raise ValueError("Source changed during validation")
        kind = "nvfp4" if observation["indicators"].get("nvfp4_conversion") else "mxfp8"
        try:
            row = evaluate_v3(profiles[kind], observation, compile_policy=compilation_policy(path.read_text()),
                              calibration_signature=training_signature(old[f"titan_v3_{kind}_batch16"]))
        except ValueError as exc:
            excluded.append({"source": source, "reason": str(exc)})
            continue
        row["cohort"] = next(c for c in ("aws_expA", "aws_expB", "lyris_expA", "lyris_expB") if f"/{c}/" in source)
        evaluations.append(row)
        observations[source] = {**observation, "summary": summarize(observation)}
    v4 = {}
    for candidate in census["memory_candidates"]:
        source = candidate["source"]
        if "rmukundan/" not in source or not any(j in Path(source).name for j in ("2565249_0", "2568526_0")):
            continue
        path = args.archive / source
        obs = parse_log(path, source_id=source)
        if obs["sha256"] != candidate["sha256"]:
            raise ValueError("Source changed during validation")
        text = path.read_text()
        job = "2565249" if "2565249_0" in source else "2568526"
        timings = [(int(m[1]), float(m[2])) for m in re.finditer(
            r"iteration\s+(\d+)/\s*50.*?elapsed time per iteration \(ms\):\s*([\d.]+)", text)]
        v4[job] = {"source": source, "sha256": obs["sha256"], "configuration": v4_config(text),
                   "allocator_rows": allocator_ratios(obs), "stash": obs["observed_components"],
                   "nan_gradient_observed": bool(obs["indicators"].get("nan_gradient")),
                   "observed_iteration_ms": [{"step": s, "ms": ms} for s, ms in timings],
                   "observed_median_iteration_ms_steps_10_50": statistics.median(ms for s, ms in timings if 10 <= s <= 50)}
    if v4["2568526"]["sha256"] != old["megatron_v4"]["sha256"]:
        raise ValueError("V4 ratio source is not the frozen calibration log")
    if v4["2565249"]["sha256"] == old["megatron_v4"]["sha256"]:
        raise ValueError("V4 evaluation source overlaps calibration")
    config_diff = {k: [v4["2568526"]["configuration"][k]["value"], v4["2565249"]["configuration"][k]["value"]]
                   for k in V4_FIELDS if v4["2568526"]["configuration"][k]["value"] != v4["2565249"]["configuration"][k]["value"]}
    if config_diff:
        raise ValueError(f"V4 configuration differences need review: {config_diff}")
    ratio_tests = []
    old_rows = {(r["step"], r["rank"]): r for r in v4["2568526"]["allocator_rows"]}
    for row in v4["2565249"]["allocator_rows"]:
        # Common arbitrary unit cancels in relative errors; do not claim GB bytes.
        scenario = project_allocator_envelope(1.0, old_rows[row["step"], row["rank"]])
        predicted = scenario["projected_high_water_reserved_bytes"] * row["printed_values"]["allocated"]
        measured = row["printed_values"]["max-reserved"]
        ratio_tests.append({"rank": row["rank"], "step": row["step"],
                            "input_current_allocated_printed_units": row["printed_values"]["allocated"],
                            "predicted_reserved_printed_units": predicted, "observed_reserved_printed_units": measured,
                            "error_percent": 100 * (predicted / measured - 1),
                            "conditioning": "Uses held-out current allocated as input, not an end-to-end prediction"})
    components = v4["2565249"]["stash"]
    events = [{"event_id": f"stash_{i}", "kind": "allocate", "storage_id": f"stash_{i}",
               "size_bytes": c["payload_bytes"], "category": "moe_buffers"} for i, c in enumerate(components)]
    replay = simulate(events, {"name": "Storage-only accounting; no hardware timing claim",
                              "peak_flops": {"float32": 1}, "memory_bandwidth_bytes_s": 1,
                              "memory_capacity_bytes": 2**60})
    geometry = packed_layout({"format": "mxfp8", "shape": [3072, 2241344],
                              "orientations": ["rowwise"], "layout": "rowwise_block", "scale_column_alignment": 64})
    payload = components[3]["payload_bytes"]
    scales = components[4]["payload_bytes"]
    results = {"source": "frozen_cpu_model_log_validation", "baseline_commit": "98ff55a",
               "profile_fingerprints": fingerprints, "refitted": False,
               "v3_evaluations": evaluations, "excluded_selected_runs": excluded,
               "v4_runs": v4, "v4_configuration_differences": config_diff,
               "v4_ratio_tests": ratio_tests,
               "component_checks": {
                   "observed_stash_payload_bytes": sum(c["payload_bytes"] for c in components),
                   "cpu_replay_logical_bytes": replay["summary"]["logical_peak_allocated_bytes"],
                   "mxfp8_cpu_layout_bytes": geometry["total_bytes"],
                   "mxfp8_logged_payload_plus_scales_bytes": payload + scales,
                   "mxfp8_layout_assumption": "Transposed logical orientation with explicit scale-column alignment 64; byte-count equivalence only, not proof of backend layout",
                   "independent_allocation_prediction": False,
                   "scope": "Observed buffer geometries replayed as coexisting allocations on one representative rank; not a measured peak."},
               "full_model_execution_validation": False}
    for kind, profile in profiles.items():
        if fingerprints[kind] != hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest():
            raise AssertionError("Validation modified calibration")
    write(args.output / "results.json", results)
    if args.plot:
        plot_results(results, args.output)
    # Retain selected source anchors without whole duplicated run dumps.
    evidence = {source: {k: obs[k] for k in ("sha256", "facts", "indicators", "summary")}
                for source, obs in observations.items()}
    write(args.output / "selected_observations.json", evidence)
    print(json.dumps({"manifest_entries_verified": census["manifest_entries"],
                      "unique_files": census["unique_stored_files"],
                      "memory_candidates": len(census["memory_candidates"]),
                      "v3_evaluations": len(evaluations), "excluded": excluded,
                      "max_v4_reserved_error_percent": max(abs(r["error_percent"]) for r in ratio_tests)}, indent=2))


if __name__ == "__main__":
    main()
