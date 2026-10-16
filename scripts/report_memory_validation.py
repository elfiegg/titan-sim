#!/usr/bin/env python3
"""Recompute the existing evidence comparisons without refitting any profiles.

Raw archive hashes are verified when --archive is supplied. This report keeps
empirical totals, state subtotals and target-conditioned checks separate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from titan_sim.gpt_oss_memory import project_state
from titan_sim.memory_regimes import estimate


def delta(predicted, observed):
    if observed <= 0:
        raise ValueError("Observed memory must be positive")
    if predicted is None:
        return {"delta": None, "delta_percent": None}
    return {"delta": predicted - observed,
            "delta_percent": 100 * (predicted - observed) / observed}


def build_report(archive=None):
    inputs, raw_sources, rows = {}, {}, []

    def load(name):
        path = ROOT / name
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def source(evidence):
        path, digest = evidence["source"], evidence["sha256"]
        if path in raw_sources and raw_sources[path]["sha256"] != digest:
            raise ValueError(f"Conflicting source hashes: {path}")
        if path not in raw_sources:
            verified = False
            if archive is not None:
                actual = hashlib.sha256((archive / path).read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(f"Source changed: {path}")
                verified = True
            raw_sources[path] = {"sha256": digest, "verified_this_run": verified}
        return path

    def add(model, config, method, metric, unit, predicted, observed, **extra):
        rows.append(dict(model=model, config=config, method=method, metric=metric,
                         unit=unit, predicted=predicted, observed=observed,
                         **delta(predicted, observed), **extra))

    gap = load("docs/validation/memory_gap/results.json")
    other = load("docs/validation/other_models/results.json")
    for model, datasets in (("DeepSeek V3 671B", gap["precisions"]),
                            ("DeepSeek V3 16B", other["v3_16b"])):
        for precision, data in datasets.items():
            large = model.endswith("671B")
            profile_path = (f"profiles/memory/titan_v3_{precision}_batch_regimes.json" if large
                            else f"docs/validation/other_models/titan_v3_16b_{precision}_batch_regimes.json")
            profile = load(profile_path)
            for r in data["aws_rows" if large else "rows"]:
                evidence = {**r["maximum_evidence"], "sha256": r["sha256"]} if large else r["evidence"]
                observed = r["maximum_evidence"]["bytes"] if large else r["observed_bytes"]
                prediction = estimate(profile, {"source": "batch_memory_request",
                    "context": profile["context"], "local_batch": r["local_batch"]})
                add(model, f"{precision}, batch {r['local_batch']}", "empirical_interpolation",
                    "peak_reserved_between_resets_max_logged", "GiB",
                    prediction["estimated_bytes"] / 2**30, observed / 2**30,
                    predicted_bytes=prediction["estimated_bytes"], observed_bytes=observed,
                    framework="TorchTitan", role=r["role"], rank=None,
                    step=evidence["step"], source=source(evidence), line=evidence["line"],
                    context=profile["context"], all_rank_max_verified=False,
                    full_allocation_prediction=False, target_measurement_input=False)

    gpt = load("docs/validation/gpt_oss/results.json")
    inventory = load("docs/validation/gpt_oss/parameter_inventory.json")
    for run in gpt["runs"]:
        c = run["config"]
        p = project_state(inventory, **{k: c[k] for k in ("world_size", "tp", "pp", "cp", "ep", "etp")})
        if p["local_parameter_elements"] != run["logged_local_parameter_elements"]:
            raise ValueError("GPT-OSS local parameter census does not match GPU log")
        if p["end_step_state_bytes"] != run["prediction"]["end_step_state_bytes"]:
            raise ValueError("GPT-OSS state model changed; review before updating report")
        path = source(run)
        for o in run["observations"]:
            label = f"{c['precision']}, world {c['world_size']}, TP{c['tp']} CP{c['cp']} EP{c['ep']}"
            common = dict(framework="Megatron-Bridge", rank=o["rank"], step=o["iteration"],
                          source=path, line=o["line"], context=c,
                          full_allocation_prediction=False, target_measurement_input=False)
            add("GPT-OSS 20B archive", label, "state_subtotal", "current_allocated", "GB",
                p["end_step_state_bytes"] / 1e9, o["allocated_gb"], **common)
            for field, metric in (("peak_allocated_gb", "peak_allocated"), ("peak_reserved_gb", "peak_reserved")):
                add("GPT-OSS 20B archive", label, "observation_only", metric, "GB",
                    None, o[field], **common)

    probes = load("docs/validation/gpt_oss_probe/experiment_comparison.json")
    for run in probes["runs"]:
        c = run["manifest"]
        p = project_state(inventory, **{k: c[k] for k in ("world_size", "tp", "cp", "ep", "pp")})
        if p["end_step_state_bytes"] != run["cpu_end_step_state_bytes"]:
            raise ValueError("Probe state model changed")
        label = f"{run['precision']}, world {run['world_size']}, {run['mode']}"
        for rank, m in run["metrics_by_rank"].items():
            common = dict(framework="Megatron-Bridge", rank=int(rank), context=c,
                          mode=run["mode"], full_allocation_prediction=False,
                          target_measurement_input=False,
                          source="docs/validation/gpt_oss_probe/experiment_comparison.json")
            for step in (2, 6):
                add("GPT-OSS 20B GB200", label, "state_subtotal", "current_allocated", "GB",
                    p["end_step_state_bytes"] / 1e9, m[f"end_step{step}_allocated_bytes"] / 1e9,
                    step=step, **common)
            for metric in ("peak_allocated", "peak_reserved", "max_sampled_device_used"):
                add("GPT-OSS 20B GB200", label, "observation_only", metric, "GB",
                    None, m[f"{metric}_bytes"] / 1e9, step="six_step_window", **common)
    for t in probes["component_transfer"]:
        p = project_state(inventory, world_size=t["target_world_size"], tp=1, cp=2, ep=4)
        predicted = p["end_step_state_bytes"] + sum(t["frozen_components_bytes"].values())
        if predicted != t["predicted_allocated_bytes"]:
            raise ValueError("Frozen component transfer no longer reproduces")
        add("GPT-OSS 20B GB200", f"{t['precision']}, world 4 -> 8, traced", "frozen_additive_components",
            "current_allocated", "GB", predicted / 1e9, t["observed_allocated_bytes"] / 1e9,
            framework="Megatron-Bridge", rank=0, step=2, role="retrospective_transfer",
            full_allocation_prediction=False, target_measurement_input=False,
            source="docs/validation/gpt_oss_probe/experiment_comparison.json")

    for r in gap["v4_reservation_variability"]["paired_rows"]:
        a, b = r["calibration_values"], r["evaluation_values"]
        for evidence in r["source_evidence"]:
            source(evidence)
        add("DeepSeek V4 Pro", "MXFP8 world256 TP1 PP4 VPP4 EP64 seq4096", "conditional_reservation_ratio",
            "current_reserved", "printed GB; divisor unverified", a["reserved"] / a["allocated"] * b["allocated"],
            b["reserved"], framework="Megatron-Bridge", rank=r["rank"], step=r["step"],
            source=r["source_evidence"], role="conditional_transfer",
            full_allocation_prediction=False, target_measurement_input=True)

    v4 = other["v4_long_context"]["runs"]
    src, tgt = v4["single_64k"], v4["packed_4x16k"]
    source(src); source(tgt)
    src_rows = {(r["rank"], r["step"]): r for r in src["allocator_rows"]}
    for r in tgt["allocator_rows"]:
        s = src_rows[(r["rank"], r["step"])]
        a, b = s["printed_values"], r["printed_values"]
        add("DeepSeek V4 15-layer hybrid", "world64 TP1 PP1 CP16 EP64; 64K -> packed4x16K",
            "conditional_reservation_ratio", "current_reserved", "printed MB; divisor unverified",
            a["reserved"] / a["allocated"] * b["allocated"], b["reserved"],
            framework="Megatron-Bridge", rank=r["rank"], step=r["step"],
            source=tgt["source"], line=r["line"], role="conditional_transfer",
            full_allocation_prediction=False, target_measurement_input=True)

    return {"schema_version": 1, "delta_convention": "estimator minus GPU; percent divided by GPU",
            "new_gpu_experiments": False, "profiles_refitted": False,
            "all_model_full_allocation_coverage": False, "inputs_sha256": inputs,
            "raw_sources": raw_sources, "rows": rows,
            "unavailable": [
                {"model": "Nemotron Pro / requested approximately 2.3T", "delta": None,
                 "reason": "Architecture and matching GPU memory evidence not established"},
                {"model": "Nemotron Nano / Super", "delta": None,
                 "reason": "Parameter inventories checked; matching GPU runtime evidence absent"},
                {"model": "DeepSeek V4 five-layer reduced", "delta": None,
                 "reason": "Run failed after first iteration; excluded from successful-training validation"}]}


def markdown(report):
    """Readable selected comparisons; JSON/CSV retain every observed rank."""
    rows = report["rows"]
    lines = ["# GPU memory estimates and allocation coverage", "",
        "The existing estimator is useful for several specific comparisons, but does not yet predict the complete device-memory trajectory for every model. DeepSeek V3's calibrated peak-reserved estimates have retrospective evaluation errors up to 2.85% (671B) and 2.20% (16B). GPT-OSS's frozen component transfer is within 0.59% at rank 0, end of step 2. V4 still requires a measured destination allocation counter; Nemotron approximately 2.3T has no established runtime comparison.", "",
        "The implementation work is specified in the [allocation coverage plan](plans/2026-10-15-allocation-coverage.md). This report recomputes existing evidence with the current estimator; it does not claim new GPU experiments or completed allocation integration.", "",
        "## How the comparisons were calculated", "",
        "**Delta = estimator − GPU measurement. Delta % = 100 × delta / GPU measurement.** Negative values mean an underestimate. Calculations use unrounded inputs; table display is rounded. GB means 10^9 bytes and GiB means 2^30 bytes. V4's printed GB/MB retain an unverified divisor and cannot be converted to device capacity here.", "",
        f"The reproduction checked SHA256 for {sum(x['verified_this_run'] for x in report['raw_sources'].values())} of {len(report['raw_sources'])} original archive logs, re-executed the saved V3 profiles and GPT-OSS state estimator, and recomputed the frozen component and V4 ratio transfers. Profiles were not refitted. The [JSON](validation/allocation_coverage/comparisons.json) records source hashes, configuration, rank, step and method; the [CSV](validation/allocation_coverage/comparisons.csv) contains all {len(rows)} comparison/observation rows. Rows are not independent experiments.", "",
        "Three different questions must remain separate: a calibrated total counter, an implementation-derived state subtotal, and a complete allocation trajectory. Calibration anchors are not validation. All existing transfer/evaluation checks here are retrospective; they are not blind prospective tests.", "",
        "## Verified configurations", "",
        "| Model | Framework and hardware | Memory-relevant configuration |",
        "|---|---|---|",
        "| V3 671B | TorchTitan; 256 GB300 | 671,026,404,352 parameters; TP1 PP1 CP1 EP64 FSDP256; sequence 4096; one microbatch/step; FullAC; compiled blocks/loss; AdamW; MXFP8 or mixed NVFP4/MXFP8; local batch 1–16. |",
        "| V3 16B | TorchTitan; 8 GB300 | 15,706,484,224 parameters; 27 layers; hidden 2048; TP1 PP1 CP1 EP8 FSDP8; sequence 4096; FullAC; compiled blocks/loss; AdamW; local batch 1–64. The archive pathname says megatron, but the executing framework is TorchTitan. |",
        "| V4 Pro | Megatron-Bridge/TE; 256 GB300 recipe | 61 layers; hidden 7168; 384 experts, top-6; TP1 PP4 VPP4 EP64 ETP1; sequence 4096; microbatch 1; global batch 4096; selective recompute; attention offload; full-iteration graphs configured. Compared counters precede completed graph capture. |",
        "| V4 hybrid | Megatron-Bridge/TE; 64-device recipe | 15 layers; hidden 7168; 384 experts, top-6; TP1 PP1 CP16 EP64; 64K capacity, single 64K versus packed 4×16K; global batches 256/1024; selective recompute; graph configuration retained. Only 16 ranks have compared counters. |",
        "| GPT-OSS archive | Megatron-Bridge/TE; 8, 72, 512 devices | 20,495,244,864 parameters; 24 layers, hidden 2880; 32 experts, top-4; vocabulary 128256; sequence 8192; PP1 ETP1; meshes shown below. Hardware SKU is not independently established from these logs. |",
        "| GPT-OSS probes | Megatron-Bridge/TE; 4/8 GB200 | Same 128256-vocabulary model; TP1 PP1 CP2 EP4; sequence 8192, microbatch 1, global batch world/2; public NGC 26.09; six steps, paired control/traced runs. |", "",
        "Historical V3 671B uses TorchTitan a615ec2; 16B uses 791f241. Those GPU observations do not validate a changed implementation at the currently pinned 6078523 revision. Exact metadata and unknown version fields remain in the saved profiles. GPT-OSS shape/partition projections match the logged local parameter counts in all five archived runs.", ""]

    def table(selected, label="Case"):
        lines.extend([f"| {label} | Estimator | GPU | Delta | Delta % |", "|---|---:|---:|---:|---:|"])
        for r in selected:
            identity = r["config"]
            if r.get("rank") is not None:
                identity += f"; rank {r['rank']}, step {r['step']}"
            if r.get("role"):
                identity += f"; {r['role']}"
            lines.append(f"| {identity} | {r['predicted']:.3f} | {r['observed']:.3f} | {r['delta']:+.3f} | {r['delta_percent']:+.2f}% |")
        lines.append("")

    for model in ("DeepSeek V3 671B", "DeepSeek V3 16B"):
        lines += [f"## {model}", "", "Metric: largest logged peak-reserved counter between metric resets, in **GiB**. The historical source audit establishes this counter; the raw parser's generic counter label alone does not. The log does not establish an all-rank maximum.", ""]
        table([r for r in rows if r["model"] == model])
        lines += ["Fit rows reproduce their calibration anchors by construction. Evaluation rows were excluded from the numerical fit, but were already available during model selection. The model interpolates only local batch in the recorded context. Its effective fixed and batch-dependent allowances include unresolved runtime and allocator effects; they are not identified physical allocations. Do not add separate cache/workspace factors on top of this total.", ""]
    lines += ["The 16B profile is a separate fit: transferring the 671B coefficients can miss by about 35%. Close agreement within either batch sweep therefore does not establish model-independent or GPU-generation-independent accuracy. These profiles also cannot reconstruct allocations and frees between log samples.", "",
        "## GPT-OSS archived runs", "",
        "The CPU model computes parameter, gradient, master-weight, Adam-state and retained optimizer-gradient storage from the tensor inventory and mesh. FP32 optimizer gradient shards are still live at this reporting point and are included. The following **GB** comparison is a state subtotal versus total current allocated at rank 0 after iteration 2. Its negative delta is the unmodeled remainder, not a full-model prediction error.", ""]
    selected = [r for r in rows if r["model"] == "GPT-OSS 20B archive" and r["method"] == "state_subtotal" and r["rank"] == 0 and r["step"] == 2]
    table(sorted(selected, key=lambda r: (r["context"]["world_size"], r["config"])))
    lines += ["The remainder varies with quantization, topology and batch. The archived runs alone cannot uniquely divide it into caches, workspaces and communication buffers. A single multiplier transferred from the 8-device case underpredicted larger configurations in earlier checks; the component experiments below provide stronger evidence.", "",
        "## GPT-OSS GB200 component experiments", "",
        "Control runs below show rank-0 current allocated after step 2, while peak columns take the maximum over all captured ranks in the six-step observation window. Each peak column may come from a different rank or time. All values are **GB**. No CPU estimate of the full transient peak is available yet.", "",
        "| Configuration | CPU state | GPU current rank 0 | State delta | Delta % | GPU peak allocated | GPU peak reserved | Max sampled device used |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    probes = [r for r in rows if r["model"] == "GPT-OSS 20B GB200" and r.get("mode") == "control"]
    for config in dict.fromkeys(r["config"] for r in probes):
        case = [r for r in probes if r["config"] == config]
        cur = next(r for r in case if r["metric"] == "current_allocated" and r["rank"] == 0 and r["step"] == 2)
        peaks = [max(r["observed"] for r in case if r["metric"] == metric) for metric in ("peak_allocated", "peak_reserved", "max_sampled_device_used")]
        lines.append(f"| {config} | {cur['predicted']:.3f} | {cur['observed']:.3f} | {cur['delta']:+.3f} | {cur['delta_percent']:+.2f}% | " + " | ".join(f"{v:.3f}" for v in peaks) + " |")
    lines += ["", "The 8-GPU NVFP4 control peak is 83.288 GB on rank 3, versus 79.091 GB on rank 0. Profiling rank 0 alone misses the worst observed rank. Sampled device usage is a lower bound on the true continuous device-wide maximum; it includes allocations outside PyTorch and is not interchangeable with reserved memory.", "",
        "The detailed traced rank-0 snapshot at step 2 separates these components. Values are **GB**; ownership is stronger evidence than a stack-based label.", "",
        "| Component | MXFP8 8 GPUs | NVFP4 8 GPUs | Evidence |",
        "|---|---:|---:|---|",
        "| CPU state | 65.620671 | 65.620671 | Tensor inventory and partition policy |",
        "| Storage padding correction | 0.000003 | 0.000003 | Owned storage |",
        "| Quantized caches | 1.314718 | 5.084283 | Owned backend tensors |",
        "| Dummy gradient candidates | 0.841605 | 0.841605 | Allocation stack hints |",
        "| GEMM workspace candidates | 0.201332 | 0.234886 | Allocation stack hints |",
        "| Other requested storage | 0.042470 | 0.035524 | Unattributed remainder |",
        "| Allocator block slack | 0.137172 | 0.412028 | Allocator snapshot |",
        "| Total current allocated | 68.157970 | 72.229000 | Measured traced snapshot |", "",
        "The instrumented HybridEP external allocator separately reports 0.113246 GB at this point. It is outside the PyTorch total above; it is not a complete census of all NCCL, driver or external memory. The table reconciles one measured snapshot and does not independently validate its own component labels or lifetimes.", "",
        "A more useful transfer test freezes the additive overheads from 4 GPUs and changes only CPU optimizer-state sharding for 8 GPUs. The following compares that prediction with the **traced** 8-GPU current allocated counter, in **GB**. Calibration and evaluation use the same TP/CP/EP, local batch, sequence, hardware and software.", ""]
    table([r for r in rows if r["method"] == "frozen_additive_components"])
    lines += ["The remaining 0.302/0.425 GB transfer error is largely a change in allocator block slack; it is not evidence of an equally large unknown parameter tensor. This result does not cover the transient peak, graph pools or later cache growth. In the 8-GPU NVFP4 control, rank-0 current allocated grows from 72.229 GB at step 2 to 73.659 GB at step 6. Lifetimes and warmup therefore matter even when the step-2 estimate is close.", "",
        "## DeepSeek V4", "",
        "V4 currently has a conditional reservation estimate: source reserved/current ratio × destination measured current allocated. Because destination GPU memory is an input, this is an allocator-behavior comparison, not an independent CPU prediction. Units below are **printed GB with an unverified divisor**; compare only within the same logger.", ""]
    table([r for r in rows if r["model"] == "DeepSeek V4 Pro"])
    lines += ["At Pro rank 0, step 2, both runs report current allocated 187.98 and the same peak allocated counter, but reserved differs from 266.13 to 252.87. The 13.26 difference therefore indicates different reservation/cache behavior at the sampled points; it does not show a 13.26 change in live tensors. Reset definitions and post-graph coverage are still needed for a full-run peak interpretation.", "",
        "The 15-layer hybrid comparison has 32 rank/step observations across 16 logged ranks and two iterations. Its largest conditional ratio error is 0.56%, in printed MB. This is a reduced model and a measured-input comparison, not validation of full V4 Pro. All rows are retained in the CSV. The separate five-layer run failed after its first iteration and is excluded from successful-training validation.", "",
        "Existing V4 buffer geometry checks can reproduce explicitly logged layouts under stated coexistence assumptions. They do not recover unlogged lifetimes, prove the inferred buffers overlap, or supply a complete model peak.", "",
        "## Nemotron", "",
        "The approximately 2.3T Pro/Ultra architecture and matching runtime evidence have not been established in this study. The supplied checkpoint path was not present on Ptyche when checked. That does not establish its availability on other filesystems. Nano/Super parameter inventories were reconciled separately, but a matched GPU memory run is still absent. Estimator-versus-GPU deltas are **N/A**, not zero.", "",
        "## What this means for 192 GB", "",
        "![CPU state, GPU current and peak counters versus a 192 GB limit](validation/allocation_coverage/state_vs_peak.png)", "",
        "192 decimal GB is 178.814 GiB. Compare that limit to the simultaneous worst-rank device footprint, with an explicit uncertainty allowance. The V3 GB300 peak-reserved profiles omit unverified external/device-wide costs and do not automatically transfer to a 192 GB GPU or a new topology. GPT-OSS demonstrates why post-step memory alone is insufficient: its 8-device archived MXFP8 case has 125.690 GB current allocated but a logged allocated peak of 267.030 GB at iteration 2. That observed execution cannot fit in 192 GB unchanged, even though its state fits. The archived hardware SKU and full-run rank maximum remain separate verification gaps.", "",
        "Do not issue a fit guarantee from the V4 printed units, an unbounded external-memory remainder or a calibrated state-only estimate. The plan's final gate requires complete phase/rank coverage and a bounded upper estimate. Changing checkpointing, sequence, microbatch, partitioning or precision creates a new scenario with its own runtime tradeoffs.", "",
        "## Reproduction and detailed evidence", "",
        "```sh", "python scripts/report_memory_validation.py \\", "  --archive /path/to/deepseek-v3-v4-training-logs-expanded-20261014 \\", "  --output docs/validation/allocation_coverage", "```", "",
        "The report reads the saved validation evidence and frozen profiles; without `--archive`, it records that raw source hashes were not rechecked. GPT-OSS probe figures are regenerated from the checked-in experiment comparison and manifests; this report does not claim to have reloaded every original snapshot. Source files for the comparison inputs are hashed in the output.", "",
        "- [V3 gap diagnosis](MEMORY_GAP_DIAGNOSIS.md)",
        "- [Other-model evidence](validation/other_models/results.json)",
        "- [GPT-OSS allocation experiment](GPT_OSS_ALLOCATION_EXPERIMENT.md)",
        "- [CPU original-step capture status](CPU_STEP_CAPTURE.md)",
        "- [Implementation work packages and acceptance criteria](plans/2026-10-15-allocation-coverage.md)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparisons.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["model", "config", "method", "metric", "unit", "rank", "step", "role",
              "predicted", "observed", "delta", "delta_percent", "full_allocation_prediction",
              "target_measurement_input", "source", "line"]
    with (args.output / "comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    (ROOT / "docs/ALLOCATION_COVERAGE_REPORT.md").write_text(markdown(report))
    print(json.dumps({"rows": len(report["rows"]), "raw_sources": len(report["raw_sources"]),
                      "raw_hashes_checked": sum(r["verified_this_run"] for r in report["raw_sources"].values())}))


if __name__ == "__main__":
    main()
