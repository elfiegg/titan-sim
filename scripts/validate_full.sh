#!/usr/bin/env bash
# Invoke once per node INSIDE the pinned container on an existing allocation.
# This script does not request nodes or submit a scheduler job.
set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
    : "${TITAN_FULL_WORKER_READY:?Internal worker mode requires the validated launcher environment}"
    : "${RANK:?torchrun must supply the global rank}"
    history_step=0
    case ",${TITAN_HISTORY_RANKS}," in
        *",${RANK},"*) history_step="$TITAN_HISTORY_STEP" ;;
    esac
    worker_args=(--flavor "$TITAN_FLAVOR" --expert-parallel-degree "$TITAN_EP_DEGREE"
                 --steps "$TITAN_STEPS" --observe-ranks "$TITAN_OBSERVE_RANKS"
                 --history-step "$history_step" --history-entries "$TITAN_HISTORY_ENTRIES"
                 --output "$TITAN_VALIDATION_OUTPUT/gpu")
    [[ -z "${TITAN_SEQ_LEN:-}" ]] || worker_args+=(--seq-len "$TITAN_SEQ_LEN")
    [[ -z "${TITAN_TOKENS:-}" ]] || worker_args+=(--tokens "$TITAN_TOKENS")
    if [[ "$history_step" != 0 && "$TITAN_DUMP_HISTORY" == 1 ]]; then
        worker_args+=(--dump-history)
    fi
    exec "$TITAN_FULL_PYTHON" -m titan_sim.launch gpu_compare "${worker_args[@]}"
fi
if (( $# != 0 )); then
    printf 'Usage: configure environment, then invoke %s without arguments\n' "$0" >&2
    exit 2
fi

: "${TITAN_WORKSPACE:?Set the mounted directory containing torchtitan and titan-sim}"
: "${TITAN_VALIDATION_OUTPUT:?Set a new shared output directory for this run}"
export TITAN_CONTAINER_IMAGE="${TITAN_CONTAINER_IMAGE:-cached image; digest not independently verified}"
: "${MASTER_ADDR:?Set a reachable rendezvous hostname/IP common to every node}"
: "${MASTER_PORT:?Set a free rendezvous port common to every node}"
: "${NODE_RANK:?Set the unique zero-based node rank}"
export NNODES="${NNODES:-64}" NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TITAN_FLAVOR="${TITAN_FLAVOR:-671B}" TITAN_EP_DEGREE="${TITAN_EP_DEGREE:-64}"
export TITAN_STEPS="${TITAN_STEPS:-10}" TITAN_OBSERVE_RANKS="${TITAN_OBSERVE_RANKS:-all}"
export TITAN_HISTORY_RANKS="${TITAN_HISTORY_RANKS:-0}" TITAN_HISTORY_STEP="${TITAN_HISTORY_STEP:-3}"
export TITAN_HISTORY_ENTRIES="${TITAN_HISTORY_ENTRIES:-250000}" TITAN_DUMP_HISTORY="${TITAN_DUMP_HISTORY:-0}"
export TITAN_WORKSPACE TITAN_VALIDATION_OUTPUT TITAN_CONTAINER_IMAGE MASTER_ADDR MASTER_PORT NODE_RANK
export TITAN_SEQ_LEN="${TITAN_SEQ_LEN:-}" TITAN_TOKENS="${TITAN_TOKENS:-}"
export TITAN_CHECKOUT="${TITAN_CHECKOUT:-$TITAN_WORKSPACE/torchtitan}"
export SIM_CHECKOUT="${SIM_CHECKOUT:-$TITAN_WORKSPACE/titan-sim}"
TITAN_FULL_PYTHON="$(command -v "${TITAN_PYTHON:-python}")"
export TITAN_FULL_PYTHON
export PYTHONPATH="$TITAN_CHECKOUT:$SIM_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHTITAN_ROOT="$TITAN_CHECKOUT"
cd "$SIM_CHECKOUT"

# Cheap argument checks run before creating output directories or importing CUDA.
"$TITAN_FULL_PYTHON" - <<'PY'
import os
for key, minimum in [('NNODES', 1), ('NPROC_PER_NODE', 1), ('NODE_RANK', 0),
                     ('MASTER_PORT', 1), ('TITAN_EP_DEGREE', 1), ('TITAN_STEPS', 10),
                     ('TITAN_HISTORY_STEP', 1), ('TITAN_HISTORY_ENTRIES', 1)]:
    text = os.environ[key]
    if not text.isdecimal() or int(text) < minimum:
        raise ValueError(f'{key} must be an integer >= {minimum}')
nodes, local, rank = (int(os.environ[key]) for key in ('NNODES', 'NPROC_PER_NODE', 'NODE_RANK'))
world = nodes * local
if rank >= nodes or int(os.environ['MASTER_PORT']) > 65535:
    raise ValueError('NODE_RANK or MASTER_PORT is outside its valid range')
flavor, ep = os.environ['TITAN_FLAVOR'], int(os.environ['TITAN_EP_DEGREE'])
if flavor not in ('debugmodel', '671B') or world % ep:
    raise ValueError('flavor must be debugmodel/671B and world size must be divisible by EP')
if flavor == '671B' and (world != 256 or ep != 64):
    raise ValueError('selected full validation requires 256 ranks and EP64; use debugmodel for reduced tests')
for key in ('TITAN_SEQ_LEN', 'TITAN_TOKENS'):
    if os.environ[key] and (not os.environ[key].isdecimal() or int(os.environ[key]) <= 0):
        raise ValueError(f'{key} must be empty or a positive integer')
if int(os.environ['TITAN_HISTORY_STEP']) > int(os.environ['TITAN_STEPS']):
    raise ValueError('history step must occur within the measured steps')
if os.environ['TITAN_DUMP_HISTORY'] not in ('0', '1'):
    raise ValueError('TITAN_DUMP_HISTORY must be 0 or 1')
PY

node_output="$TITAN_VALIDATION_OUTPUT/nodes/$NODE_RANK"
mkdir -p "$node_output"
if [[ -e "$node_output/stage.txt" ]]; then
    printf 'Refusing to overwrite existing node run: %s\n' "$node_output" >&2
    exit 2
fi
printf 'preflight\n' > "$node_output/stage.txt"
trap 'result=$?; printf "%s\n" "$result" > "$node_output/exit-code.txt"' EXIT
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$node_output/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$node_output/triton}"
export TORCH_COMPILE_DEBUG_DIR="${TORCH_COMPILE_DEBUG_DIR:-$node_output/compile-debug}"

"$TITAN_FULL_PYTHON" - "$node_output" <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import torch
from titan_sim.recipes import TORCHTITAN_COMMIT, import_torchtitan

output = Path(sys.argv[1])
profile = json.loads((Path(os.environ['SIM_CHECKOUT']) / 'profiles/versions.json').read_text())
repository = profile['container'].rsplit(':', 1)[0]
expected_image = repository + '@' + profile['container_arm64_digest']
revision = import_torchtitan(os.environ['TITAN_CHECKOUT'])
if revision != profile['torchtitan_commit'] or revision != TORCHTITAN_COMMIT:
    raise ValueError('TorchTitan checkout, shared recipe pin and profile revision differ')
subprocess.run(['git', '-C', os.environ['TITAN_CHECKOUT'], 'diff', '--quiet', 'HEAD', '--',
                'torchtitan', 'torchtitan_recipes'], check=True)
if torch.__version__ != profile['torch_version'] or torch.version.cuda != profile['cuda_version']:
    raise ValueError(f'container runtime mismatch: torch={torch.__version__}, CUDA={torch.version.cuda}')
nccl = torch.cuda.nccl.version()
if '.'.join(map(str, nccl)) != profile['nccl_version']:
    raise ValueError(f'NCCL runtime mismatch: {nccl}')
local_count = int(os.environ['NPROC_PER_NODE'])
if torch.cuda.device_count() != local_count:
    raise ValueError('visible GPU count must equal NPROC_PER_NODE')
devices = [{'name': torch.cuda.get_device_name(i),
            'capacity_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(local_count)]
if any('GB200' not in device['name'] for device in devices):
    raise ValueError(f'the selected validation target requires GB200 GPUs: {devices}')
world = int(os.environ['NNODES']) * local_count

def ranks(value, *, allow_all=False):
    if value == 'all' and allow_all:
        return list(range(world))
    if value == 'none':
        return []
    parts = value.split(',')
    if not parts or any(not part.isdecimal() for part in parts):
        raise ValueError('rank sets must be comma-separated nonnegative integers, all (observations), or none (history)')
    result = [int(part) for part in parts]
    if len(set(result)) != len(result) or any(rank >= world for rank in result):
        raise ValueError('rank sets must be unique and within the distributed world')
    return result

observed = ranks(os.environ['TITAN_OBSERVE_RANKS'], allow_all=True)
history = ranks(os.environ['TITAN_HISTORY_RANKS'])
if not observed or not set(history).issubset(observed):
    raise ValueError('observe at least one rank; history ranks must be a subset of observed ranks')
(output / 'observe-ranks.txt').write_text(','.join(map(str, observed)))
(output / 'history-ranks.txt').write_text(','.join(map(str, history)))
try:
    simulator_commit = subprocess.check_output(['git', '-C', os.environ['SIM_CHECKOUT'], 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
except subprocess.CalledProcessError:
    simulator_commit = None
record = {'torchtitan_commit': revision, 'torch_version': torch.__version__,
          'cuda_version': torch.version.cuda, 'nccl_version': nccl, 'architecture': platform.machine(),
          'expected_container_image': expected_image,
          'declared_container_image': os.environ['TITAN_CONTAINER_IMAGE'],
          'image_digest_verified': False,
          'container_identity_scope': 'cached-image label and expected OCI digest are provenance metadata only; runtime versions independently checked',
          'simulator_commit': simulator_commit,
          'world_size': world, 'node_rank': int(os.environ['NODE_RANK']), 'devices': devices,
          'observed_ranks': observed, 'history_ranks': history,
          'packages': {name: importlib.metadata.version(name) for name in ('spmd_types', 'torch_remat', 'attn-gym')}}
(output / 'environment.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY

TITAN_OBSERVE_RANKS="$(cat "$node_output/observe-ranks.txt")"
TITAN_HISTORY_RANKS="$(cat "$node_output/history-ranks.txt")"
export TITAN_OBSERVE_RANKS TITAN_HISTORY_RANKS TITAN_FULL_WORKER_READY=1
printf 'distributed-training\n' > "$node_output/stage.txt"
"$TITAN_FULL_PYTHON" -m torch.distributed.run \
    --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --rdzv-backend static --max-restarts 0 \
    --no-python "$SIM_CHECKOUT/scripts/validate_full.sh" --worker \
    > "$node_output/training.log" 2>&1
printf 'complete\n' > "$node_output/stage.txt"
