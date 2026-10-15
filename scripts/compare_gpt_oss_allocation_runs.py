#!/usr/bin/env python3
"""Compare completed GPT-OSS observations with independent CPU state estimates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from titan_sim.gpt_oss_memory import project_state


def load(path):
    return json.loads(path.read_text())


def phase(rows, name, step=None):
    matches = [row for row in rows if row["phase"] == name and (step is None or row["step"] == step)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} at step {step}, got {len(matches)}")
    return matches[0]


def allocation(row):
    return row["counters"]["allocated_bytes.all.current"]


def components(row, cpu):
    evidence = row["attribution"]["0"]
    owned = evidence["exact_storage_categories_bytes"]
    hints = evidence["unattributed_stack_hints_bytes"]
    state = sum(owned.get(key, 0) for key in ("parameters", "model_gradient_buffers",
        "master_weights", "adam_first_moment", "adam_second_moment", "optimizer_gradients"))
    result = {
        "tensor_storage_padding": state - cpu["end_step_state_bytes"],
        "quantization_cache": owned.get("quantization_cache", 0),
        "allocator_block_slack": owned.get("allocator_block_slack", 0),
        "dummy_gradients_from_allocation_stacks": hints.get("te_dummy_weight_gradient_candidate", 0),
        "gemm_workspaces_from_allocation_stacks": hints.get("gemm_workspace_candidate", 0),
    }
    result["other_requested_storage"] = owned.get("unattributed", 0) - result["dummy_gradients_from_allocation_stacks"] - result["gemm_workspaces_from_allocation_stacks"]
    if cpu["end_step_state_bytes"] + sum(result.values()) != allocation(row):
        raise ValueError("Attribution does not reconcile with the sampled allocated counter")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-gpu", type=Path, required=True)
    parser.add_argument("--eight-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = load(ROOT / "docs/validation/gpt_oss/parameter_inventory.json")
    result = {"scope": "GB200, NGC 26.09, GPT-OSS 20B, TP1 CP2 EP4, sequence8192 microbatch1",
              "runs": [], "component_transfer": [], "full_training_peak_prediction": False}
    cases = {}
    for world, directory in ((4, args.four_gpu), (8, args.eight_gpu)):
        cpu = project_state(inventory, world_size=world, tp=1, cp=2, ep=4)
        for precision in ("fp8_mx", "nvfp4"):
            for mode in ("control", "traced"):
                case = directory / f"{precision}-{mode}"
                manifest = load(case / "manifest.json")
                expected = {"training_complete": True, "world_size": world,
                    "tp": 1, "cp": 2, "ep": 4, "pp": 1, "micro_batch": 1,
                    "sequence_length": 8192, "global_batch": world // 2,
                    "local_parameter_elements": cpu["local_parameter_elements"],
                    "gpu": "NVIDIA GB200", "torch": "2.14.0a0+b2c75dd062.nv26.09"}
                for key, value in expected.items():
                    if manifest[key] != value:
                        raise ValueError(f"Unexpected {key} in {case}")
                rows_by_rank = {}
                for path in case.glob("rank*/phases.jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if [r["step"] for r in rows if r["phase"] == "step_end"] != list(range(1, 7)):
                        raise ValueError(f"Incomplete training phase records: {path}")
                    rows_by_rank[rows[0]["rank"]] = rows
                if set(rows_by_rank) != set(range(world)):
                    raise ValueError(f"Missing ranks in {case}")
                metrics = {}
                for rank, rows in rows_by_rank.items():
                    metrics[str(rank)] = {
                        "end_step2_allocated_bytes": allocation(phase(rows, "step_end", 2)),
                        "end_step6_allocated_bytes": allocation(phase(rows, "step_end", 6)),
                        "peak_allocated_bytes": max(r["counters"]["allocated_bytes.all.peak"] for r in rows),
                        "peak_reserved_bytes": max(r["counters"]["reserved_bytes.all.peak"] for r in rows),
                        "max_sampled_device_used_bytes": max(r.get("device_wide_used_bytes", 0) for r in rows)}
                entry = {"world_size": world, "precision": precision, "mode": mode,
                    "cpu_end_step_state_bytes": cpu["end_step_state_bytes"],
                    "metrics_by_rank": metrics, "manifest": manifest}
                if mode == "traced":
                    evidence = load(args.output / f"{precision}-{world}gpu-attribution.json")["rows"]
                    second_end = phase(evidence, "step_end", 2)
                    entry["step2_overhead_components_bytes"] = components(second_end, cpu)
                    before_forward = phase(evidence, "before_model_forward", 2)["sequence"]
                    zeros = [r for r in evidence if r["step"] == 2 and r["sequence"] < before_forward
                             and r["phase"] in ("before_optimizer_zero_grad", "after_optimizer_zero_grad")]
                    first, last = zeros[0], zeros[-1]
                    def gradients(row):
                        return row["attribution"]["0"]["exact_storage_categories_bytes"].get("optimizer_gradients", 0)
                    entry["gradient_release"] = {"before_bytes": gradients(first), "after_bytes": gradients(last),
                        "allocated_counter_drop_bytes": allocation(first) - allocation(last)}
                    entry["hybridep_external_at_step2"] = second_end.get("hybridep_external")
                cases[(world, precision, mode)] = (entry, rows_by_rank[0])
                result["runs"].append(entry)
    for precision in ("fp8_mx", "nvfp4"):
        source = cases[(4, precision, "traced")][0]
        target = cases[(8, precision, "traced")][0]
        frozen = source["step2_overhead_components_bytes"]
        prediction = target["cpu_end_step_state_bytes"] + sum(frozen.values())
        observed = target["metrics_by_rank"]["0"]["end_step2_allocated_bytes"]
        result["component_transfer"].append({"precision": precision,
            "method": "freeze additive per-model-rank overhead from 4 GPUs; change only CPU optimizer sharding",
            "source_world_size": 4, "target_world_size": 8, "phase": "end of step 2",
            "frozen_components_bytes": frozen, "predicted_allocated_bytes": prediction,
            "observed_allocated_bytes": observed, "error_percent": 100 * (prediction / observed - 1),
            "prospective_blind_validation": False,
            "scope": "retrospective check at identical TP/CP/EP, batch, sequence, hardware and software"})
    (args.output / "experiment_comparison.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    labels = ["Setup", "Step 1\nforward", "Step 1\nend", "Step 2\nzero grad", "Step 2\nforward", "Step 2\nend", "Step 6\nend"]
    cpu = project_state(inventory, world_size=8, tp=1, cp=2, ep=4)
    setup = sum(v for k, v in cpu["components_bytes"].items() if k.endswith((".parameters", ".gradients", ".master_weights")))
    base, end = cpu["persistent_state_bytes"], cpu["end_step_state_bytes"]
    for axis, precision in zip(axes, ("fp8_mx", "nvfp4")):
        rows = cases[(8, precision, "control")][1]
        before_forward = phase(rows, "before_model_forward", 2)["sequence"]
        zero = [r for r in rows if r["phase"] == "after_optimizer_zero_grad" and r["step"] == 2 and r["sequence"] < before_forward][-1]
        selected = [phase(rows, "after_model_optimizer_setup"), phase(rows, "after_model_forward", 1),
                    phase(rows, "step_end", 1), zero, phase(rows, "after_model_forward", 2),
                    phase(rows, "step_end", 2), phase(rows, "step_end", 6)]
        axis.plot(range(7), [allocation(r)/1e9 for r in selected], "o-", label="GPU allocated")
        axis.plot(range(7), [r["counters"]["reserved_bytes.all.current"]/1e9 for r in selected], "o-", label="GPU reserved", color="#af742d")
        axis.plot(range(7), [v/1e9 for v in (setup, setup, end, base, base, end, end)], "--", label="CPU training-state subtotal", color="#52646d")
        axis.set(title="MXFP8" if precision == "fp8_mx" else "NVFP4", xticks=range(7), xticklabels=labels, ylim=(0, None))
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Decimal GB per GPU · rank 0")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("GPT-OSS on 8 GB200s: phase samples from control runs\nCPU subtotal excludes activations, caches, workspaces and allocator reserve", fontsize=11)
    fig.savefig(args.output / "phase_memory.svg")
    svg = args.output / "phase_memory.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(args.output / "phase_memory.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result["component_transfer"], indent=2))


if __name__ == "__main__":
    main()
