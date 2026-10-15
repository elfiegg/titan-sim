#!/usr/bin/env python3
"""Reproduce the memory gap, audit its metric, and evaluate bounded batch fits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from titan_sim.log_observations import ANSI, parse_log
from titan_sim.memory_validation import evaluate_v3, training_signature
from titan_sim.memory_regimes import METRIC, estimate, fit_profile

REVISION = "a615ec2db9d66d46e9e8d6f679ed73e890369eea"
TRAIN_BATCHES = (1, 4, 16)
CHECK_BATCHES = (2, 8)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_metric(checkout):
    """Read the immutable git object, without modifying the user's checkout."""
    specs = {
        "torchtitan/components/metrics.py": [
            ('max_reserved = device_info.get("reserved_bytes.all.peak", -1)', "reserved_peak_read"),
            ('f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"', "printed_counter"),
            ("self.device_memory_monitor.reset_peak_stats()", "metric_resets"),
            ("device_module.reset_peak_memory_stats()", "reset_implementation"),
        ],
        "torchtitan/trainer.py": [
            ('f"{device_mem_stats.max_reserved_gib:.2f}GiB"', "initialization_counter"),
            ("self.optimizers.step()", "optimizer_before_logging"),
        ],
        "torchtitan/components/optimizer.py": [
            ("init_optim_state(optim)", "possible_early_state_materialization"),
        ],
        "torchtitan/distributed/deepep/hybridep.py": [
            ("max_tokens_per_rank = num_tokens", "buffer_capacity_depends_on_tokens"),
        ],
    }
    evidence = {}
    for path, patterns in specs.items():
        data = subprocess.check_output(["git", "-C", str(checkout), "show", f"{REVISION}:{path}"])
        lines = data.decode().splitlines()
        anchors = {}
        for pattern, meaning in patterns:
            matches = [i for i, line in enumerate(lines, 1) if pattern in line]
            if not matches:
                raise ValueError(f"Reviewed source no longer matches {meaning}")
            anchors[meaning] = matches
        evidence[path] = {
            "revision": REVISION, "source_sha256": hashlib.sha256(data).hexdigest(),
            "url": f"https://github.com/rdspring1/torchtitan/blob/{REVISION}/{path}",
            "line_anchors": anchors,
        }
    return evidence


def error(predicted, observed):
    return 100 * (predicted / observed - 1)


def plot(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, kind, label in zip(axes, ("mxfp8", "nvfp4"), ("MXFP8", "Mixed NVFP4 / MXFP8")):
        rows = results["precisions"][kind]["aws_rows"]
        x = [r["local_batch"] for r in rows]
        ax.plot(x, [r["frozen_gib"] for r in rows], "--", color="#b45b23", label="Original batch-16 factor")
        ax.plot(x, [r["regime_gib"] for r in rows], "-", color="#17699a", label="New bounded fit")
        for role, marker, desc in (("fit", "o", "Fitting observations"), ("evaluation", "s", "Excluded from fitting")):
            subset = [r for r in rows if r["role"] == role]
            ax.scatter([r["local_batch"] for r in subset], [r["observed_gib"] for r in subset],
                       marker=marker, s=55, color="#222222", label=desc, zorder=3)
        ax.set_title(label)
        ax.set_xlabel("Local batch (sequence 4,096)")
        ax.set_xticks(x)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Largest logged peak-reserved counter (GiB)")
    fig.suptitle("DeepSeek-V3 / 256 GB300: fixed allowances improve small-batch estimates")
    fig.text(.08, .025, "Retrospective split: fit batches 1, 4, 16; check 2, 8. Not a required-capacity or all-rank guarantee.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(output / "memory_gap.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/memory_gap")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    source_evidence = audit_metric(args.torchtitan_root)
    previous = json.loads((ROOT / "docs/validation/expanded_logs/results.json").read_text())
    selected = json.loads((ROOT / "docs/validation/expanded_logs/selected_observations.json").read_text())
    results = {
        "source": "retrospective_memory_gap_experiment", "source_audit": source_evidence,
        "fitting_batches": TRAIN_BATCHES, "evaluation_batches": CHECK_BATCHES,
        "selection_caveat": "All sweep values had already been inspected; excluded from fitting is not blind validation.",
        "experimental_max_error_target_percent": 5, "precisions": {},
    }
    v4_pairs = []
    v4_maps = {job: {(r["rank"], r["step"]): r for r in run["allocator_rows"]}
               for job, run in previous["v4_runs"].items()}
    for key in sorted(v4_maps["2565249"].keys() & v4_maps["2568526"].keys()):
        a, b = (v4_maps[job][key] for job in ("2568526", "2565249"))
        if a["printed_unit"] != b["printed_unit"]:
            raise ValueError("V4 counter units disagree")
        av, bv = a["printed_values"], b["printed_values"]
        v4_pairs.append({
            "rank": key[0], "step": key[1], "printed_unit": a["printed_unit"],
            "calibration_values": av, "evaluation_values": bv,
            "current_allocated_difference": bv["allocated"] - av["allocated"],
            "peak_allocated_difference": bv["max-allocated"] - av["max-allocated"],
            "reserved_difference": bv["reserved"] - av["reserved"],
            "observed_reserved_over_current_range": sorted([a["reserved_over_current_allocated"], b["reserved_over_current_allocated"]]),
            "source_evidence": [{"source": r["source"], "line": r["line"],
                                 "sha256": previous["v4_runs"][job]["sha256"]}
                                for job, r in (("2568526", a), ("2565249", b))],
        })
    results["v4_reservation_variability"] = {
        "paired_rows": v4_pairs,
        "interpretation": "Same allocated counters do not imply the same reservation. Ranges describe these two runs only; not forecast confidence bounds.",
        "metric_caveat": "GB divisor and peak reset window remain unverified for this separate Megatron logger; V3 source audit does not resolve them.",
    }
    for kind, precision in (("mxfp8", "MXFP8 compute"), ("nvfp4", "mixed NVFP4/MXFP8 compute")):
        original = json.loads((ROOT / f"profiles/memory/titan_v3_{kind}_batch16.json").read_text())
        rows = []
        for old in previous["v3_evaluations"]:
            if old["cohort"] != "aws_expA" or old["context"]["precision"] != precision:
                continue
            path = args.archive / old["source"]
            observation = parse_log(path, source_id=old["source"])
            if observation["sha256"] != old["sha256"]:
                raise ValueError("Raw log changed since prior audit")
            text = ANSI.sub("", path.read_text())
            revs = set(re.findall(r'"TORCHTITAN_(?:SHA|REVISION)": "([0-9a-f]+)"', text))
            if revs != {REVISION}:
                raise ValueError("Full logged revision missing or conflicting")
            ao_revs = set(re.findall(r'"TORCHAO_(?:SHA|REVISION)": "([0-9a-f]+)"', text))
            if ao_revs != {"14555766ebf05763588b1bb87df6d2dcede0cb4e"}:
                raise ValueError("TorchAO revision missing or conflicting")
            optimizers = [line for line in text.splitlines() if "Optimizer " in line and " params " in line]
            if not optimizers or any("Optimizer AdamW (model_part=0): 967 params" not in line
                                     or "'fused': True" not in line or "'foreach': False" not in line for line in optimizers):
                raise ValueError("Optimizer configuration outside reviewed scope")
            if not all(msg in text for msg in ("Compiling each TransformerBlock with torch.compile", "Compiling the loss function with torch.compile")):
                raise ValueError("Missing compilation evidence")
            row = evaluate_v3(original, observation, compile_policy="loss and each TransformerBlock")
            row["signature"] = training_signature(observation)
            row["revision_lines"] = [i for i, line in enumerate(text.splitlines(), 1)
                                     if re.search(r'"TORCHTITAN_(?:SHA|REVISION)":', line)]
            summary = selected[old["source"]]["summary"]
            row["initialization_gib"] = max(s["maximum"] for s in summary if s["metric"] == "model_initialization_memory_unspecified")
            row["first_step_gib"] = max(s["maximum"] for s in summary if s["metric"] == "training_memory_unspecified" and s["step"] == 1)
            rows.append(row)
        rows.sort(key=lambda r: r["local_batch"])
        if [r["local_batch"] for r in rows] != [1, 2, 4, 8, 16]:
            raise ValueError("Expected one reviewed run for each batch")
        context = {k: v for k, v in rows[0]["context"].items() if k not in ("tokens_per_rank", "global_batch")}
        context.update(revision=REVISION, memory_metric=METRIC, allocator_policy=None,
                       parameter_dtype=None, gradient_dtype=None, optimizer_state_dtype=None,
                       optimizer="AdamW fused=True foreach=False", attention=None,
                       routing_policy=None, deep_ep_version=None, torchao_version=None,
                       torchao_revision="14555766ebf05763588b1bb87df6d2dcede0cb4e",
                       offload_policy=None, expert_tp=None,
                       counter_window="peak since last metric reset; largest printed value in 50-step run")
        anchors = [{"local_batch": r["local_batch"], "observed_bytes": r["maximum_evidence"]["bytes"],
                    "evidence": {"source": r["source"], "sha256": r["sha256"],
                                 "line": r["maximum_evidence"]["line"], "step": r["maximum_evidence"]["step"],
                                 "training_signature": r["signature"], "revision_lines": r["revision_lines"]}}
                   for r in rows if r["local_batch"] in TRAIN_BATCHES]
        profile = fit_profile(name=f"titan_v3_{kind}_batch_regimes", context=context, anchors=anchors,
                              assumed_state_bytes=original["baseline"]["state_bytes"],
                              metric_evidence=source_evidence["torchtitan/components/metrics.py"])
        write(ROOT / f"profiles/memory/{profile['name']}.json", profile)
        affine = fit_profile(name="endpoint_affine_ablation", context=context, anchors=[anchors[0], anchors[-1]],
                             assumed_state_bytes=original["baseline"]["state_bytes"], metric_evidence=profile["metric_evidence"])
        checks = []
        for row in rows:
            request = {"source": "batch_memory_request", "context": context, "local_batch": row["local_batch"]}
            improved, simple = estimate(profile, request), estimate(affine, request)
            role = "fit" if row["local_batch"] in TRAIN_BATCHES else "evaluation"
            if role == "evaluation" and any(row["sha256"] == a["evidence"]["sha256"] or row["signature"] == a["evidence"]["training_signature"] for a in anchors):
                raise ValueError("Evaluation source overlaps fitting evidence")
            checks.append({"source": row["source"], "sha256": row["sha256"], "local_batch": row["local_batch"],
                           "role": role, "observed_gib": row["observed_max_gib"],
                           "frozen_gib": row["frozen_estimate_gib"], "frozen_error_percent": row["frozen_error_percent"],
                           "affine_gib": simple["estimated_gib"], "affine_error_percent": error(simple["estimated_gib"], row["observed_max_gib"]),
                           "regime_gib": improved["estimated_gib"], "regime_error_percent": error(improved["estimated_gib"], row["observed_max_gib"]),
                           "initialization_gib": row["initialization_gib"], "first_step_gib": row["first_step_gib"],
                           "revision_lines": row["revision_lines"], "maximum_evidence": row["maximum_evidence"],
                           "training_signature": row["signature"]})
            if row["local_batch"] == 8:
                write(ROOT / f"examples/memory/{kind}_batch8_request.json", request)
                write(args.output / f"{kind}_batch8_estimate.json", improved)
        held = [r for r in checks if r["role"] == "evaluation"]
        results["precisions"][kind] = {
            "aws_rows": checks, "max_evaluation_error_percent": max(abs(r["regime_error_percent"]) for r in held),
            "baseline_profile_sha256": hashlib.sha256((ROOT / f"profiles/memory/titan_v3_{kind}_batch16.json").read_bytes()).hexdigest(),
            "lyris_corroboration": [{"source": r["source"], "sha256": r["sha256"], "local_batch": r["local_batch"],
                                      "observed_max_gib": r["observed_max_gib"],
                                      "caveat": "Package suffix only; same rounded maxima do not establish independent runs or complete software identity."}
                                     for r in previous["v3_evaluations"] if r["cohort"] == "lyris_expA" and r["context"]["precision"] == precision],
        }
    write(args.output / "results.json", results)
    if args.plot:
        plot(results, args.output)
    errors = {k: v["max_evaluation_error_percent"] for k, v in results["precisions"].items()}
    print(json.dumps(errors, indent=2))
    if max(errors.values()) > results["experimental_max_error_target_percent"]:
        raise SystemExit("Experimental error target failed")


if __name__ == "__main__":
    main()
