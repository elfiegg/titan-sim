#!/usr/bin/env python3
"""Run a CPU GPT-OSS storage projection and compare independently parsed GPU logs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_other_models import config_dump
from titan_sim.gpt_oss_memory import parameter_inventory, project_state, model_config
from titan_sim.log_observations import ANSI, parse_log
from titan_sim.empirical_memory import allocator_ratios
from titan_sim.recipes import import_torchtitan

BRIDGE = "fcbb6031103d0ca845c1a54d4fee55ecfcca17b6"
MCORE = "d0b3b775498a8ae6870075f1a5fc32bc260aa1b3"
METRIC_SHA256 = "8e495208e49ac5584e06174fc4fc5d0b227c4e26464d48f387e63e8abe6b76d1"


def full_config_fields(text, class_name, expected):
    lines = [(i, line) for i, line in enumerate(text.splitlines(), 1) if class_name + "(" in line]
    if not lines:
        raise ValueError(f"Missing {class_name}")
    out = {}
    for key, value in expected.items():
        values = [(i, re.search(r"\b" + key + r"=([^,)]+)", line)) for i, line in lines]
        found = [(i, match[1]) for i, match in values if match]
        if not found or {v for _, v in found} != {value}:
            raise ValueError(f"Unsupported {class_name}.{key}: {found}")
        out[key] = {"value": value, "line": found[0][0]}
    return out


def verified_config(text):
    sections = {s: config_dump(text, section=s) for s in ("model", "train", "dataset", "tokenizer")}
    def value(section, key, default=None):
        return sections[section].get(key, {"value": default})["value"]
    expected = {"num_layers": "24", "hidden_size": "2880", "ffn_hidden_size": "2880",
                "num_moe_experts": "32", "num_attention_heads": "64", "num_query_groups": "8",
                "kv_channels": "64", "moe_router_topk": "4", "params_dtype": "torch.bfloat16"}
    for key, wanted in expected.items():
        if value("model", key) != wanted:
            raise ValueError(f"GPT-OSS identity mismatch: {key}")
    if value("tokenizer", "vocab_size") != "128256" or value("dataset", "sequence_length") != "8192":
        raise ValueError("Vocabulary or sequence mismatch")
    # These are reviewed defaults in the exact MCore revision, not filename hints.
    for key, default in (("fp8_param", "False"), ("fp4_param", "False"), ("recompute_granularity", "None")):
        if value("model", key, default) != default:
            raise ValueError(f"Unsupported model.{key}")
    if "setting number of microbatches to constant 1" not in text:
        raise ValueError("Gradient accumulation is unverified")
    ddp = full_config_fields(text, "DistributedDataParallelConfig", {
        "use_megatron_fsdp": "False", "grad_reduce_in_fp32": "False",
        "use_distributed_optimizer": "True", "num_distributed_optimizer_instances": "1",
        "fp8_param_gather": "False", "fp4_param_gather": "False"})
    optimizer = full_config_fields(text, "OptimizerConfig", {
        "main_params_dtype": "torch.float32", "exp_avg_dtype": "torch.float32",
        "exp_avg_sq_dtype": "torch.float32", "use_precision_aware_optimizer": "False"})
    mesh = {short: int(value("model", long, "1")) for short, long in (
        ("tp", "tensor_model_parallel_size"), ("pp", "pipeline_model_parallel_size"),
        ("cp", "context_parallel_size"), ("ep", "expert_model_parallel_size"),
        ("etp", "expert_tensor_parallel_size"))}
    micro = int(value("train", "micro_batch_size")); batch = int(value("train", "global_batch_size"))
    if batch % micro:
        raise ValueError("Nonintegral data parallel group")
    world = batch // micro * mesh["tp"] * mesh["cp"] * mesh["pp"]
    precision = "MXFP8" if value("model", "fp8_recipe") == "'mxfp8'" else "NVFP4"
    if precision == "NVFP4" and value("model", "fp4") not in ("'e2m1'", "'nvfp4'"):
        raise ValueError("Unverified FP4 mode")
    selected = set(expected) | {"fp8_recipe", "fp4", "cuda_graph_impl", "cuda_graph_warmup_steps", "first_last_layers_bf16",
        "num_layers_at_end_in_bf16", "num_layers_at_start_in_bf16", "moe_expert_rank_capacity_factor"}
    return {"world_size": world, **mesh, "micro_batch": micro, "global_batch": batch,
            "sequence_length": 8192, "vocab_size": 128256, "precision": precision,
            "model_evidence": {k: v for k, v in sections["model"].items() if k in selected},
            "tokenizer_evidence": {k: v for k, v in sections["tokenizer"].items() if k in ("vocab_size", "pad_vocab_size", "tokenizer_type")},
            "ddp_evidence": ddp, "optimizer_evidence": optimizer,
            "default_evidence_revision": MCORE}


def current_allocation_transfer(runs, basis="end_step_state_bytes"):
    """Retrospective transfer of one 8-GPU calibration per precision.

    This predicts current allocation only, not peak allocated/reserved memory.
    Target GPU counters are accessed only to score predictions, not as inputs.
    """
    results = []
    for precision in ("MXFP8", "NVFP4"):
        cohort = [r for r in runs if r["config"]["precision"] == precision]
        anchors = [r for r in cohort if r["config"]["world_size"] == 8]
        if len(anchors) != 1:
            raise ValueError("Need one 8-GPU calibration per precision")
        anchor = anchors[0]
        reference = next(r for r in anchor["observations"] if r["rank"] == 0 and r["iteration"] == 2)
        factor = reference["allocated_gb"] * 1e9 / anchor["prediction"][basis]
        evaluations = []
        for target in cohort:
            if target is anchor:
                continue
            predicted = factor * target["prediction"][basis] / 1e9
            actual = next(r for r in target["observations"] if r["rank"] == 0 and r["iteration"] == 2)
            evaluations.append({"target_sha256": target["sha256"], "world_size": target["config"]["world_size"],
                "predicted_current_gb": predicted, "observed_current_gb": actual["allocated_gb"],
                "error_percent": 100 * (predicted / actual["allocated_gb"] - 1)})
        results.append({"precision": precision, "calibration_sha256": anchor["sha256"],
            "calibration_world_size": 8, "calibration_rank": 0, "calibration_iteration": 2,
            "factor": factor, "basis": basis, "metric": "current allocated after iteration 2 only",
            "selection": "retrospective; configurations inspected before selecting calibration",
            "evaluations": evaluations})
    return results


def capture_probe():
    """Diagnostic actual dispatch attempt, never used as a completed training trace."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from titan_sim.capture import TensorCapture
    from titan_sim.torchtitan_adapter import post_initialization_scope
    cap = TensorCapture(source="cpu_projection", modeled_device="cuda:0")
    result = {"execution": "unsharded eager BF16 diagnostic, not the Megatron recipe",
              "torch_version": torch.__version__, "used_for_prediction": False}
    try:
        with FakeTensorMode(allow_fallback_kernels=False), torch.device("cpu"), cap:
            old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
            try:
                model = model_config(128256, 8192).build()
            finally:
                torch.set_default_dtype(old)
            with post_initialization_scope():
                model.init_states(buffer_device=torch.device("cpu"))
            cap.register_model(model)
            tokens = torch.zeros(3 * 8192, dtype=torch.long)
            positions = torch.arange(3 * 8192) % 8192
            padding = torch.zeros_like(tokens, dtype=torch.bool)
            masks = model.get_attention_masks(positions, padding_mask=padding, max_context_length=8192)
            output = model(tokens, positions=positions, attention_masks=masks, padding_mask=padding)
            output.float().sum().backward()
        result["status"] = "diagnostic_complete"
    except Exception as exc:
        result.update(status="blocked", error_type=type(exc).__name__, error=str(exc))
    return result


def export_comparison(result, output, plots=False):
    rows = []
    for run in sorted(result["runs"], key=lambda r: (r["config"]["world_size"], r["config"]["precision"])):
        config = run["config"]
        observed = next(r for r in run["observations"] if r["rank"] == 0 and r["iteration"] == 2)
        rows.append({"precision": config["precision"], "gpus": config["world_size"],
                     "tp": config["tp"], "cp": config["cp"], "ep": config["ep"], **observed})
    with (output / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.6))
        x = list(range(len(rows)))
        state = [r["cpu_state_gb"] for r in rows]
        current = [r["allocated_gb"] for r in rows]
        ax.bar(x, state, label="CPU modeled state at reporting point", color="#285f96")
        ax.bar(x, [r["current_minus_state_gb"] for r in rows], bottom=state,
               label="Observed current minus CPU subtotal (unattributed)", color="#dd9655")
        ax.bar(x, [r["peak_above_current_gb"] for r in rows], bottom=current,
               label="Observed peak allocated minus observed current", color="#eed49a")
        ax.scatter(x, [r["peak_reserved_gb"] for r in rows], color="#353535", marker="_", s=450,
                   label="Observed peak reserved (separate high-water mark)", zorder=5)
        ax.set_xticks(x, [f'{r["precision"]} · {r["gpus"]} GPUs\nTP{r["tp"]} CP{r["cp"]} EP{r["ep"]}' for r in rows])
        ax.set_ylabel("Memory per GPU, decimal GB")
        ax.set_title("GPT-OSS-20B: CPU storage baseline versus GPU counters\nRank 0, reported after iteration 2; upper segments are measured gaps")
        ax.legend(loc="upper right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = output / "memory_comparison.svg"
        fig.savefig(path, metadata={"Date": None})
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/gpt_oss")
    parser.add_argument("--attempt-capture", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    revision = import_torchtitan(args.torchtitan_root)
    metric_path = args.bridge_source / "src/megatron/bridge/training/utils/train_utils.py"
    metric_data = metric_path.read_bytes()
    if hashlib.sha256(metric_data).hexdigest() != METRIC_SHA256:
        raise ValueError("Metric divisor not verified")
    inventory = parameter_inventory()
    candidates = json.loads((ROOT / "docs/validation/other_models/results.json").read_text())["other_architecture_candidates"]
    result = {"source": "independent CPU persistent-state projection with GPU comparison",
              "torchtitan_revision": revision, "megatron_bridge_revision": BRIDGE,
              "megatron_core_revision": MCORE, "parameter_elements": sum(r["elements"] for r in inventory),
              "metric_unit": "decimal GB (1e9 bytes)", "metric_source_sha256": hashlib.sha256(metric_data).hexdigest(),
              "full_training_prediction": False, "runs": []}
    for candidate in candidates:
        path = args.archive / candidate["source"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != candidate["sha256"]:
            raise ValueError("Log changed since archive census")
        text = ANSI.sub("", data.decode(errors="replace"))
        if BRIDGE[:7] not in text or MCORE[:7] not in text:
            raise ValueError("Source revisions not verified")
        config = verified_config(text)
        prediction = project_state(inventory, **{k: config[k] for k in ("world_size", "tp", "cp", "pp", "ep", "etp")})
        # Validation data enters only after projection; never fit or pass it to project_state.
        counts = sorted(set(map(int, re.findall(r"number of parameters on .*?: (\d+)", text))))
        if counts != [prediction["local_parameter_elements"]]:
            raise ValueError(f"Per-rank parameter inventory differs: {counts}")
        rows = allocator_ratios(parse_log(path, source_id=candidate["source"]))
        observed = []
        for row in rows:
            v = row["printed_values"]
            p = prediction["end_step_state_bytes"] / 1e9
            observed.append({"rank": row["rank"], "iteration": row["step"], "line": row["line"],
                "allocated_gb": v["allocated"], "peak_allocated_gb": v["max-allocated"],
                "peak_reserved_gb": v["max-reserved"], "cpu_state_gb": p,
                "cpu_core_state_gb": prediction["persistent_state_bytes"] / 1e9,
                "current_minus_state_gb": v["allocated"] - p,
                "current_shortfall_percent": 100 * (1 - p / v["allocated"]),
                "peak_minus_state_gb": v["max-allocated"] - p,
                "peak_above_current_gb": v["max-allocated"] - v["allocated"],
                "reserved_peak_minus_allocated_peak_gb": v["max-reserved"] - v["max-allocated"]})
        graph_lines = [i for i, line in enumerate(text.splitlines(), 1)
                       if "CUDA graph capture done for training" in line]
        result["runs"].append({"source": candidate["source"], "sha256": candidate["sha256"],
            "config": config, "logged_local_parameter_elements": counts[0], "prediction": prediction,
            "observations": observed, "first_full_graph_capture_done_line": min(graph_lines) if graph_lines else None})
    result["current_allocation_factor_transfer"] = current_allocation_transfer(result["runs"])
    result["core_only_factor_diagnostic"] = current_allocation_transfer(result["runs"], "persistent_state_bytes")
    if args.attempt_capture:
        result["dispatch_probe"] = capture_probe()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameter_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    export_comparison(result, args.output, args.plots)
    for run in result["runs"]:
        c = run["config"]; rows = [r for r in run["observations"] if r["iteration"] == 2]
        print(c["precision"], c["world_size"], c["tp"], c["cp"], c["ep"],
              json.dumps(max(rows, key=lambda r: r["peak_allocated_gb"])))
    print(json.dumps(result.get("dispatch_probe")))


if __name__ == "__main__":
    main()
