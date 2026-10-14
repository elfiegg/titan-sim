#!/usr/bin/env bash
# Bounded checks inside the selected NVIDIA container; no full-model claim.
set -euo pipefail
: "${TITAN_EXTENSION_OUTPUT:?Set output directory}"
mkdir -p "$TITAN_EXTENSION_OUTPUT"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
import torch
files = sorted(path for directory in ('titan_sim', 'tests', 'scripts')
               for path in Path(directory).rglob('*')
               if path.is_file() and path.suffix in ('.py', '.cpp', '.sh'))
record = {'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
          'files_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
(Path(os.environ['TITAN_EXTENSION_OUTPUT']) / 'source-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
status=0
python -m unittest discover -s tests > "$TITAN_EXTENSION_OUTPUT/tests.log" 2>&1 || status=1
python scripts/distributed_probe.py --multiprocess --out "$TITAN_EXTENSION_OUTPUT/distributed" > "$TITAN_EXTENSION_OUTPUT/distributed.log" 2>&1 || status=1
python scripts/calibrate_matmul.py --hardware profiles/gb200.json \
  --torchtitan-commit 607852350b41b4b280d8b8de6340f4847ed71c5e \
  --output "$TITAN_EXTENSION_OUTPUT/calibration.json" > "$TITAN_EXTENSION_OUTPUT/calibration.log" 2>&1 || status=1
python scripts/generate_inductor_example.py --device cuda \
  --output-dir "$TITAN_EXTENSION_OUTPUT/inductor" > "$TITAN_EXTENSION_OUTPUT/inductor.log" 2>&1 || status=1
# This probe is deliberately independent: blocked native support is data, not
# permission to suppress an assertion or alter the selected training recipe.
python -m titan_sim.native_guard --allow-cuda-build > "$TITAN_EXTENSION_OUTPUT/native-guard.json" 2> "$TITAN_EXTENSION_OUTPUT/native-guard.err" || status=1
exit "$status"
