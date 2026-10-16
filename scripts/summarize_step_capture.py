#!/usr/bin/env python3
"""Export compact, trace-bound evidence without publishing full capture files."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open() as handle:
        document = json.load(handle)
    metadata = document["metadata"]
    fields = (
        "status", "torchtitan_commit", "torch_version", "scenario", "flavor",
        "seq_len", "tokens", "fake_device", "capture_scope", "attention_contract",
        "activation_checkpoint", "steps", "completed_steps", "step_checks",
        "model_parameter_count", "trainable_parameter_tensors",
        "production_training_capture", "cuda_memory_equivalent",
        "initialization_coverage", "attention_coverage", "error_type", "error",
    )
    events = document["events"]
    checks = metadata.get("step_checks", [])
    expected = metadata.get("trainable_parameter_tensors")
    all_gradients = bool(checks) and expected is not None and all(
        check["parameter_tensors_with_grad"] == expected for check in checks
    )
    return {
        "trace_file": path.name,
        "trace_sha256": digest.hexdigest(),
        "capture": {key: metadata[key] for key in fields if key in metadata},
        "all_trainable_parameter_tensors_have_grad_metadata": all_gradients,
        "numerical_gradients_validated": False,
        "event_count": len(events),
        "storage_count": len(document.get("storages", [])),
        "saved_tensor_record_count": len(document.get("saved_tensors", [])),
        "events_by_execution_phase": dict(Counter(e.get("execution_phase", "unknown") for e in events)),
        "attention_operations_by_phase": dict(Counter(
            f'{e["name"]}:{e.get("execution_phase", "unknown")}'
            for e in events if e["name"] in {"flex_attention", "flex_attention_backward"}
        )),
        "limitations": document.get("limitations", []),
        "interpretation": (
            "Logical unsharded model execution evidence only. Does not establish "
            "production distributed execution, compiled CUDA allocation, GPU "
            "required capacity or runtime. BF16 model parameters retain upstream "
            "operation dtype conversions, including FP32 loss operations."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f'{result["capture"].get("status", "unknown")}: {args.output}')


if __name__ == "__main__":
    main()
