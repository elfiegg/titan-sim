#!/usr/bin/env bash
#SBATCH --job-name=core_dlfw_ci-titansim.gptoss-memory
#SBATCH --account=core_dlfw_ci
#SBATCH --partition=backfill
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fsw/core_dlfw_ci/elfieg/gptoss-memory-20261014/profile-%j.out
#SBATCH --error=/lustre/fsw/core_dlfw_ci/elfieg/gptoss-memory-20261014/profile-%j.err
set -euo pipefail
RUN=/lustre/fsw/core_dlfw_ci/elfieg/gptoss-memory-20261014
IMAGE=/lustre/fsw/core_dlfw_ci/elfieg/kimi-k3-fullac-20261001/containers/nvcr-pytorch-26.09-py3-arm64.sqsh
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((18000 + SLURM_JOB_ID % 10000))
export HF_HOME=/workspace/hf-cache
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
failed=0
for precision in fp8_mx nvfp4; do
  for mode in control traced; do
    export GPTOSS_PRECISION=$precision GPTOSS_MODE=$mode
    if srun --kill-on-bad-exit=1 --mpi=none \
      --container-image="$IMAGE" --container-mounts="$RUN:/workspace" --container-workdir=/workspace \
      bash /workspace/profile-node.sh; then
      printf '%s %s completed\n' "$precision" "$mode"
    else
      printf '%s %s failed\n' "$precision" "$mode"
      failed=1
    fi
  done
done
exit "$failed"
