#!/usr/bin/env bash
set -euo pipefail
export PATH=/workspace/venv/bin:/workspace/tools/uv-aarch64-unknown-linux-gnu:$PATH
export VIRTUAL_ENV=/workspace/venv
export PYTHONPATH=/workspace/titan-sim:/workspace/Megatron-Bridge/src:/workspace/Megatron-LM
export UV_CACHE_DIR=/workspace/uv-cache
RANKS=0,1
if [[ "$GPTOSS_MODE" == control ]]; then RANKS=; fi
if [[ "$GPTOSS_MODE" == traced ]]; then
  export TITANSIM_DEEPEP_MEMORY_LOG_DIR="/workspace/results/profile-${SLURM_JOB_ID}/${GPTOSS_PRECISION}-${GPTOSS_MODE}/deepep"
  mkdir -p "$TITANSIM_DEEPEP_MEMORY_LOG_DIR"
fi
export CPATH=/workspace/headers/usr/include:/usr/local/cuda/include/cccl:${CPATH:-}
export LD_LIBRARY_PATH=/workspace/rdma/lib:${LD_LIBRARY_PATH:-}
uv run --no-project --python /workspace/venv/bin/python python -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --nproc-per-node=4 --node-rank="$SLURM_NODEID" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  /workspace/titan-sim/scripts/profile_gpt_oss.py \
  --bridge-root /workspace/Megatron-Bridge --world-size "$((SLURM_NNODES * 4))" --steps 6 \
  --precision "$GPTOSS_PRECISION" --detailed-ranks "$RANKS" \
  --output "/workspace/results/profile-${SLURM_JOB_ID}/${GPTOSS_PRECISION}-${GPTOSS_MODE}"
