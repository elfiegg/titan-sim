#!/usr/bin/env python3
"""Analyze trusted snapshots produced by titan-sim's GPU allocation recorder."""
import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from titan_sim.allocation_probe import summarize_snapshot, summarize_external_allocations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", help="Optional comma-separated ranks to analyze")
    args = parser.parse_args()
    ranks = {f"rank{int(rank)}" for rank in args.ranks.split(",")} if args.ranks else None
    rows = []
    for path in sorted(args.run_directory.glob("rank*/phases.jsonl")):
        if ranks is not None and path.parent.name not in ranks:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("deepep_log_bytes", 0):
                with (args.run_directory / row["deepep_log"]).open("rb") as handle:
                    events = [json.loads(line) for line in handle.read(row["deepep_log_bytes"]).splitlines()]
                row["hybridep_external"] = summarize_external_allocations(events)
            if "snapshot" not in row:
                rows.append(row)
                continue
            # Pickle is executable: only load this task's own trusted artifacts.
            with (path.parent / row["snapshot"]).open("rb") as handle:
                snapshot = pickle.load(handle)
            inventory = json.loads((path.parent / row["inventory"]).read_text())
            devices = sorted({s["device"] for s in snapshot["segments"]})
            row["attribution"] = {str(device): summarize_snapshot(snapshot, inventory, device) for device in devices}
            row["trace_entries"] = [len(trace) for trace in snapshot.get("device_traces", [])]
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": "GPU allocator snapshot evidence",
        "rows": rows, "timing_validated": False,
        "limitations": ["allocation stack hints are not exact ownership", "history is bounded and may wrap",
            "device usage includes external allocations and is not process-specific",
            "snapshot CPU overhead may change overlap; compare counters-only control",
            "HybridEP direct-allocation probe covers ExtendedMemoryAllocator, not every direct CUDA allocation"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
