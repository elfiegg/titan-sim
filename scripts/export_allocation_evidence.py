#!/usr/bin/env python3
"""Export a small, self-consistent subset of completed allocation observations."""
import argparse
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_directory
    options = {"mode": "w:gz", "compresslevel": 1} if args.output.suffix == ".gz" else {"mode": "w"}
    with tarfile.open(args.output, **options) as archive:
        for case in sorted(root.iterdir()):
            if not case.is_dir() or not (case / "manifest.json").exists():
                continue
            manifest = json.loads((case / "manifest.json").read_text())
            if not manifest.get("training_complete"):
                raise ValueError(f"Refusing to export incomplete case: {case.name}")
            archive.add(case / "manifest.json", arcname=f"{case.name}/manifest.json")
            if (case / "deepep").exists():
                archive.add(case / "deepep", arcname=f"{case.name}/deepep")
            for phases in sorted(case.glob("rank*/phases.jsonl")):
                rows = [json.loads(line) for line in phases.read_text().splitlines()]
                for row in rows:
                    keep = phases.parent.name == "rank0" and (
                        row["phase"] == "after_model_optimizer_setup"
                        or row["phase"] == "step_end" and row["step"] in (2, 6)
                        or row["step"] == 2 and row["phase"] in (
                            "before_optimizer_zero_grad", "after_optimizer_zero_grad",
                            "after_model_forward", "after_fp32_gradient_copy"))
                    if "snapshot" not in row:
                        continue
                    if keep:
                        for key in ("snapshot", "inventory"):
                            path = phases.parent / row[key]
                            archive.add(path, arcname=str(path.relative_to(root)))
                    else:
                        row["snapshot_omitted_from_export"] = True
                        row.pop("snapshot")
                        row.pop("inventory")
                data = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
                info = tarfile.TarInfo(str(phases.relative_to(root)))
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    main()
