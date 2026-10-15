#!/usr/bin/env python3
"""Audit additional model/configuration evidence; keep fit and transfer separate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from titan_sim.empirical_memory import allocator_ratios
from titan_sim.log_observations import ANSI, parse_log
from titan_sim.memory_regimes import METRIC, estimate, fit_profile
from titan_sim.memory_validation import training_signature, unique_fact

REV16 = "791f2415fa73a0baf07b776bbf6fb8ee6a9e81a9"


def config_dump(text, *, section=None, dotted=False):
    """Keep section ownership: model TP is not dataset/checkpoint TP."""
    current, result = None, {}
    pattern = r"\s*(\w+)\s+\.{3,}\s+(.*)" if dotted else r"\s*(\w+):\s*(.*)"
    for line, raw in enumerate(text.splitlines(), 1):
        if dotted:
            raw = re.sub(r"^\s*\d+:\s*", "", raw)
        header = re.match(r"\[([\w]+)\]", raw)
        if header:
            current = header[1]
        if section is not None and current != section:
            continue
        match = re.fullmatch(pattern, raw)
        if not match:
            continue
        key, value = match[1], match[2].split("  (Mcore default:")[0].strip()
        if key in result and result[key]["value"] != value:
            raise ValueError(f"Conflicting configuration {key}")
        result.setdefault(key, {"value": value, "line": line})
    return result


def frozen_ratio_transfer(source_rows, target_rows):
    """Same rank and reported iteration only; measured target allocation is input."""
    source = {(r["rank"], r["step"]): r for r in source_rows}
    if len(source) != len(source_rows):
        raise ValueError("Repeated rank/step requires explicit window disambiguation")
    out, seen = [], set()
    for row in target_rows:
        key = row["rank"], row["step"]
        if key in seen:
            raise ValueError("Repeated target rank/step")
        seen.add(key)
        if key not in source:
            raise ValueError("No matching calibration rank/step")
        prior = source[key]
        if prior["printed_unit"] != row["printed_unit"]:
            raise ValueError("Unit labels differ; cannot compare")
        a, b = prior["printed_values"], row["printed_values"]
        predicted = b["allocated"] * prior["high_water_reserved_over_current"]
        out.append({"rank": key[0], "step": key[1], "printed_unit": row["printed_unit"],
                    "observed_allocated": b["allocated"], "source_allocated": a["allocated"],
                    "observed_peak_allocated": b["max-allocated"], "source_peak_allocated": a["max-allocated"],
                    "predicted_reserved": predicted, "observed_reserved": b["max-reserved"],
                    "error_percent": 100 * (predicted / b["max-reserved"] - 1),
                    "conditioning": "Target current allocation is measured; not a full CPU prediction",
                    "source_line": prior["line"], "target_line": row["line"]})
    return out


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/other_models")
    parser.add_argument("--refresh-census", action="store_true", help="Rehash the full archive and report newly recognized legacy counters")
    args = parser.parse_args()
    census = json.loads((ROOT / "docs/validation/expanded_logs/archive_audit.json").read_text())
    if args.refresh_census:
        from scripts.validate_expanded_logs import inspect_archive
        refreshed = inspect_archive(args.archive)
        known = {r["source"] for r in census["memory_candidates"]}
        write(args.output / "census_extension.json", {
            "verified_manifest_entries": len(refreshed["verified_entries"]),
            "unique_stored_files": refreshed["unique_stored_files"],
            "prior_candidates": len(known), "current_candidates": len(refreshed["memory_candidates"]),
            "new_candidates": [r for r in refreshed["memory_candidates"] if r["source"] not in known]})
    hashes = {r["stored_source"]: r["sha256"] for r in census["verified_entries"]}

    def read(source):
        data = (args.archive / source).read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes[source]:
            raise ValueError("Archive content changed")
        return ANSI.sub("", data.decode(errors="replace"))

    def git_source(path):
        data = subprocess.check_output(["git", "-C", str(args.torchtitan_root), "show", f"{REV16}:{path}"])
        return data.decode(), {"revision": REV16, "source_sha256": hashlib.sha256(data).hexdigest(),
                               "url": f"https://github.com/rdspring1/torchtitan/blob/{REV16}/{path}"}

    metrics, metric_evidence = git_source("torchtitan/components/metrics.py")
    if 'max_reserved = device_info.get("reserved_bytes.all.peak", -1)' not in metrics or 'memory: {device_mem_stats.max_reserved_gib' not in metrics:
        raise ValueError("Metric definition not reviewed")
    model, model_evidence = git_source("torchtitan/models/deepseek_v3/__init__.py")
    flavor = model.split("def _16b(", 1)[1].split("def _236b(", 1)[0]
    if not all(s in flavor for s in ("dim = 2048", "n_layers = 27", "num_experts = 64")):
        raise ValueError("16B model dimensions differ")
    results = {"source": "additional_model_config_audit", "v3_16b": {}, "v4_long_context": {},
               "other_architecture_candidates": [], "independent_full_model_validation": False}
    sources16 = sorted(p for p in hashes if "/dsv3_16b_m_sweep_performance_aws/run_logs/" in p)
    for kind in ("mxfp8", "nvfp4"):
        rows = []
        for source in sources16:
            if f"16b-{kind}-2n-bs" not in Path(source).name:
                continue
            text = read(source)
            obs = parse_log(args.archive / source, source_id=source)
            facts = {k: unique_fact(obs, k) for k in ("model", "parameter_elements", "hardware", "mesh", "training", "torchtitan_version")}
            expected = {"pp": 1, "dp_replicate": 1, "dp_shard": 8, "cp": 1, "tp": 1, "ep": 8}
            if facts["model"] != "deepseek_v3 16B" or facts["parameter_elements"] != 15706484224 or facts["hardware"] != "NVIDIA GB300" or facts["mesh"] != expected:
                raise ValueError("16B configuration not matched")
            t = facts["training"]
            if t["sequence_length"] != 4096 or t["accumulation"] != 1 or t["global_batch"] != t["local_batch"] * 8 or t["steps"] != 100:
                raise ValueError("16B batch/sequence mismatch")
            revs = set(re.findall(r'"TORCHTITAN_(?:SHA|REVISION)": "([a-f0-9]+)"', text))
            if revs != {REV16} or facts["torchtitan_version"] != "0.2.2+git791f241":
                raise ValueError("16B revision mismatch")
            if not obs["indicators"].get("full_checkpointing") or not obs["indicators"].get("mxfp8_conversion"):
                raise ValueError("Checkpoint/quantization evidence missing")
            if bool(obs["indicators"].get("nvfp4_conversion")) != (kind == "nvfp4"):
                raise ValueError("Precision mismatch")
            if not all(s in text for s in ("Compiling each TransformerBlock", "Compiling the loss function", "Optimizer AdamW")):
                raise ValueError("Compiler/optimizer evidence missing")
            maximum = max((s for s in obs["samples"] if s["metric"] == "training_memory_unspecified"), key=lambda s: s["bytes"])
            rows.append({"local_batch": t["local_batch"], "observed_bytes": maximum["bytes"],
                         "evidence": {"source": source, "sha256": hashes[source], "line": maximum["line"],
                                      "step": maximum["step"], "signature": training_signature(obs)},
                         "facts": obs["facts"], "training_completed": bool(obs["indicators"].get("training_completed"))})
        rows.sort(key=lambda r: r["local_batch"])
        if [r["local_batch"] for r in rows] != [1, 2, 8, 32, 64]:
            raise ValueError("Unexpected 16B sweep")
        context = {"model": "DeepSeek-V3 16B", "parameter_elements": 15706484224, "layers": 27, "hidden_size": 2048,
                   "revision": REV16, "hardware": "NVIDIA GB300", "world_size": 8, **expected,
                   "sequence_length": 4096, "microbatches": 1, "precision": kind,
                   "checkpoint_policy": "FullAC", "compile_policy": "loss and each TransformerBlock",
                   "optimizer": "AdamW", "torch_version": None, "allocator_policy": None,
                   "state_dtype": None, "memory_metric": METRIC,
                   "counter_window": "largest printed peak reserved in 100-step run"}
        anchors = [{k: r[k] for k in ("local_batch", "observed_bytes", "evidence")} for r in rows if r["local_batch"] in (1, 8, 64)]
        profile = fit_profile(name=f"titan_v3_16b_{kind}_batch_regimes", context=context, anchors=anchors,
                              assumed_state_bytes=15706484224 // 8 * 16, metric_evidence=metric_evidence)
        write(args.output / f"{profile['name']}.json", profile)
        old = json.loads((ROOT / f"profiles/memory/titan_v3_{kind}_batch16.json").read_text())
        checks = []
        for row in rows:
            predicted = estimate(profile, {"source": "batch_memory_request", "context": context, "local_batch": row["local_batch"]})
            measured = row["observed_bytes"] / 2**30
            # Deliberately out-of-domain diagnostic: scale shape/state but reuse
            # the old 671B factor. Not an accepted request to the scoped model.
            naive = (profile["assumed_state_bytes"] + old["runtime_factor"] * 27 * 2048 * 4096 * 2 * row["local_batch"]) / 2**30
            role = "fit" if row["local_batch"] in (1, 8, 64) else "evaluation"
            if role == "evaluation" and any(row["evidence"]["signature"] == a["evidence"]["signature"] for a in anchors):
                raise ValueError("Evaluation metric signature overlaps calibration")
            checks.append({**row, "role": role, "observed_gib": measured,
                           "new_fit_gib": predicted["estimated_gib"], "error_percent": 100 * (predicted["estimated_gib"] / measured - 1),
                           "out_of_domain_671b_factor_gib": naive,
                           "out_of_domain_error_percent": 100 * (naive / measured - 1)})
        results["v3_16b"][kind] = {"rows": checks, "model_source": model_evidence,
                                    "metric_source": metric_evidence,
                                    "caveat": "New model-specific calibration; validates the method retrospectively, not transfer of 671B coefficients."}

    v4_fields = "num_layers hidden_size num_experts num_attention_heads moe_router_topk world_size tensor_model_parallel_size pipeline_model_parallel_size context_parallel_size expert_model_parallel_size expert_tensor_parallel_size seq_length micro_batch_size global_batch_size varlen_mock_dataset_config_json experimental_attention_variant fp8_recipe params_dtype recompute_granularity cuda_graph_impl mtp_num_layers".split()
    v4 = {}
    for source in sorted(p for p in hashes if "/dsv4_thd/" in p):
        text = read(source)
        config = config_dump(text, dotted=True)
        selected = {k: config[k] for k in v4_fields}
        if selected["experimental_attention_variant"]["value"] != "dsv4_hybrid":
            raise ValueError("V4 family identity missing")
        obs = parse_log(args.archive / source, source_id=source)
        key = "packed_4x16k" if json.loads(selected["varlen_mock_dataset_config_json"]["value"])["max_seq_len"] == 16384 else "single_64k"
        if key in v4:
            raise ValueError("Duplicate V4 scenario")
        v4[key] = {"source": source, "sha256": hashes[source], "config": selected,
                   "allocator_rows": allocator_ratios(obs), "hardware": None,
                   "maximum_logged_iteration": max(map(int, re.findall(r"iteration\s+(\d+)\s*/", text))),
                   "caveat": "15-layer dsv4_hybrid variant, not full V4 Pro; hardware and memory-unit divisor unverified."}
    differences = {k: {name: run["config"][k]["value"] for name, run in v4.items()}
                   for k in v4_fields if len({run["config"][k]["value"] for run in v4.values()}) > 1}
    comparisons = frozen_ratio_transfer(v4["single_64k"]["allocator_rows"], v4["packed_4x16k"]["allocator_rows"])
    results["v4_long_context"] = {"runs": v4, "reviewed_config_differences": differences,
                                   "ratio_transfer": comparisons, "full_prediction": False,
                                   "max_absolute_error_percent": max(abs(r["error_percent"]) for r in comparisons)}

    source = "lustre/share/coreai_dlalgo_llm/gaod/2203614.log"
    text = read(source)
    config = config_dump(text, dotted=True)
    failed = parse_log(args.archive / source, source_id=source)
    results["v4_five_layer_failed_run"] = {
        "source": source, "sha256": hashes[source],
        "config": {k: config[k] for k in v4_fields if k in config},
        "allocator_rows": allocator_ratios(failed),
        "failure": [{"line": i, "message": m[0]} for i, line in enumerate(text.splitlines(), 1)
                    if (m := re.search(r"ValueError: Attempted to make view with size=.*", line)) and "[rank" in line],
        "eligibility": "One iteration then MXFP8 tensor-view errors; exclude from steady-state validation. Hardware unverified."}

    fields = "num_layers hidden_size num_moe_experts num_attention_heads moe_router_topk tensor_model_parallel_size pipeline_model_parallel_size context_parallel_size expert_model_parallel_size expert_tensor_parallel_size fp8 fp4 fp8_recipe cuda_graph_impl".split()
    for source in sorted(p for p in hashes if "/pretrain_gpt_oss_20b/" in p and p.endswith("_0.out")):
        text = read(source)
        model = config_dump(text, section="model")
        train = config_dump(text, section="train")
        dataset = config_dump(text, section="dataset")
        obs = parse_log(args.archive / source, source_id=source)
        results["other_architecture_candidates"].append({
            "source": source, "sha256": hashes[source],
            "candidate_label_from_path": "GPT-OSS-20B", "model": {k: model.get(k) for k in fields},
            "batch": {k: train.get(k) for k in ("micro_batch_size", "global_batch_size")},
            "sequence_length": dataset.get("sequence_length"), "allocator_rows": allocator_ratios(obs),
            "maximum_logged_iteration": max(map(int, re.findall(r"iteration\s+(\d+)\s*/", text))),
            "eligibility": "Inventory only: model dimensions/precision verified; complete identity, omitted defaults and matched CPU adapter not established."})
    write(args.output / "results.json", results)
    print(json.dumps({"16b_evaluation_max_error": {k: max(abs(r["error_percent"]) for r in v["rows"] if r["role"] == "evaluation") for k, v in results["v3_16b"].items()},
                      "v4_ratio_max_error": results["v4_long_context"]["max_absolute_error_percent"],
                      "other_candidates": len(results["other_architecture_candidates"])}, indent=2))


if __name__ == "__main__":
    main()
