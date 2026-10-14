#!/usr/bin/env bash
# Run inside the pinned NVIDIA container on an allocated four-GPU node.
set -euo pipefail
: "${TITAN_WORKSPACE:?Set TITAN_WORKSPACE to the mounted experiment directory}"
TITAN_CHECKOUT="$TITAN_WORKSPACE/torchtitan"
SIM_CHECKOUT="$TITAN_WORKSPACE/titan-sim"
VALIDATION_OUTPUT="${TITAN_VALIDATION_OUTPUT:-$TITAN_WORKSPACE/results/development-6078523}"
VALIDATION_VENV="$TITAN_WORKSPACE/venv-6078523"
mkdir -p "$VALIDATION_OUTPUT"
printf 'environment\n' > "$VALIDATION_OUTPUT/stage.txt"
trap 'result=$?; printf "%s\n" "$result" > "$VALIDATION_OUTPUT/exit-code.txt"' EXIT
python -m venv --system-site-packages "$VALIDATION_VENV"
python -c 'import torch; print("torch===" + torch.__version__)' > "$VALIDATION_OUTPUT/torch-constraint.txt"
"$VALIDATION_VENV/bin/pip" install -c "$VALIDATION_OUTPUT/torch-constraint.txt" -r "$TITAN_CHECKOUT/requirements.txt" pytest > "$VALIDATION_OUTPUT/dependencies.log" 2>&1
export PATH="$VALIDATION_VENV/bin:$PATH"
export PYTHONPATH="$TITAN_CHECKOUT:$SIM_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHTITAN_ROOT="$TITAN_CHECKOUT"
export TORCHINDUCTOR_CACHE_DIR="$VALIDATION_OUTPUT/inductor"
export TRITON_CACHE_DIR="$VALIDATION_OUTPUT/triton"
export TORCH_COMPILE_DEBUG_DIR="$VALIDATION_OUTPUT/compile-debug"
cd "$SIM_CHECKOUT"
python - "$VALIDATION_OUTPUT/environment.json" <<'PY'
import importlib.metadata as md
import json
import sys
import torch
from titan_sim.recipes import import_torchtitan
import os
result = {'torch_version': torch.__version__, 'cuda': torch.version.cuda,
          'nccl': torch.cuda.nccl.version(), 'torchtitan_commit': import_torchtitan(os.environ['TORCHTITAN_ROOT']),
          'packages': {name: md.version(name) for name in ('spmd_types', 'torch_remat', 'attn-gym')},
          'devices': [{'name': torch.cuda.get_device_name(i), 'capacity_bytes': torch.cuda.get_device_properties(i).total_memory}
                      for i in range(torch.cuda.device_count())]}
assert result['torch_version'] == '2.14.0a0+b2c75dd062.nv26.09', result['torch_version']
assert len(result['devices']) == 4, result['devices']
with open(sys.argv[1], 'w') as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result), flush=True)
PY
printf 'unit-tests\n' > "$VALIDATION_OUTPUT/stage.txt"
python -m unittest discover -s tests > "$VALIDATION_OUTPUT/unit-tests.log" 2>&1
python -m pytest "$TITAN_CHECKOUT/tests/unit_tests/gpu/test_activation_checkpoint.py::TestApplyAC::test_full_ac_does_not_recompute_registered_effects" -q > "$VALIDATION_OUTPUT/fullac-effects.log" 2>&1
printf 'cpu-metadata-probes\n' > "$VALIDATION_OUTPUT/stage.txt"
# Metadata coverage failures are recorded separately and do not masquerade as
# successful projections; the independent real-GPU experiment still proceeds.
set +e
CUDA_VISIBLE_DEVICES= timeout --kill-after=10s 120s python -m titan_sim.launch torchtitan_adapter --torchtitan-root "$TITAN_CHECKOUT" --output "$VALIDATION_OUTPUT/cpu-smoke.json" > "$VALIDATION_OUTPUT/cpu-smoke.log" 2>&1
printf '%s\n' "$?" > "$VALIDATION_OUTPUT/cpu-smoke-exit.txt"
CUDA_VISIBLE_DEVICES= timeout --kill-after=10s 120s python -m titan_sim.launch project_engine --torchtitan-root "$TITAN_CHECKOUT" --flavor debugmodel --world-size 1 --expert-parallel-degree 1 --seq-len 128 --tokens 1024 --output "$VALIDATION_OUTPUT/cpu-engine.json" > "$VALIDATION_OUTPUT/cpu-engine.log" 2>&1
printf '%s\n' "$?" > "$VALIDATION_OUTPUT/cpu-engine-exit.txt"
CUDA_VISIBLE_DEVICES= timeout --kill-after=10s 120s python -m titan_sim.launch project_engine --torchtitan-root "$TITAN_CHECKOUT" --flavor debugmodel --world-size 1 --expert-parallel-degree 1 --seq-len 128 --tokens 1024 --capture-scope post_initialization --output "$VALIDATION_OUTPUT/cpu-engine-postinit.json" > "$VALIDATION_OUTPUT/cpu-engine-postinit.log" 2>&1
printf '%s\n' "$?" > "$VALIDATION_OUTPUT/cpu-engine-postinit-exit.txt"
set -e
printf 'gpu-debug-single\n' > "$VALIDATION_OUTPUT/stage.txt"
python -m torch.distributed.run --standalone --nproc-per-node=1 -m titan_sim.launch gpu_compare --flavor debugmodel --expert-parallel-degree 1 --steps 10 --seq-len 128 --tokens 1024 --output "$VALIDATION_OUTPUT/gpu-debug-single" > "$VALIDATION_OUTPUT/gpu-debug-single.log" 2>&1
printf 'gpu-debug-ep4\n' > "$VALIDATION_OUTPUT/stage.txt"
python -m torch.distributed.run --standalone --nproc-per-node=4 -m titan_sim.launch gpu_compare --flavor debugmodel --expert-parallel-degree 4 --steps 10 --seq-len 128 --tokens 1024 --observe-ranks 0,1,2,3 --output "$VALIDATION_OUTPUT/gpu-debug-ep4" > "$VALIDATION_OUTPUT/gpu-debug-ep4.log" 2>&1
printf 'complete\n' > "$VALIDATION_OUTPUT/stage.txt"
