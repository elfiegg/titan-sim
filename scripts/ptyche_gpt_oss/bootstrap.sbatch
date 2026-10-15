#!/usr/bin/env bash
#SBATCH --job-name=core_dlfw_ci-titansim.gptoss-setup
#SBATCH --account=core_dlfw_ci
#SBATCH --partition=backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/core_dlfw_ci/elfieg/gptoss-memory-20261014/setup-%j.out
#SBATCH --error=/lustre/fsw/core_dlfw_ci/elfieg/gptoss-memory-20261014/setup-%j.err
set -euo pipefail
RUN=/lustre/fsw/core_dlfw_ci/elfieg/gptoss-memory-20261014
IMAGE=/lustre/fsw/core_dlfw_ci/elfieg/kimi-k3-fullac-20261001/containers/nvcr-pytorch-26.09-py3-arm64.sqsh
srun --container-image="$IMAGE" --container-mounts="$RUN:/workspace" --container-workdir=/workspace \
  env GPTOSS_OUTPUT=/workspace/results/setup-${SLURM_JOB_ID} bash /workspace/bootstrap.sh
