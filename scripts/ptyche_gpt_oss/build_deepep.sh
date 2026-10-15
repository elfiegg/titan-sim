#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PATH=/workspace/venv/bin:/workspace/tools/uv-aarch64-unknown-linux-gnu:$PATH
export VIRTUAL_ENV=/workspace/venv UV_CACHE_DIR=/workspace/uv-cache
uv run --no-project --python /workspace/venv/bin/python /workspace/install_overlay.py
export CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export HYBRID_EP_MULTINODE=1 TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=16
export RDMA_CORE_HOME=/workspace/rdma
mkdir -p "$RDMA_CORE_HOME/lib" /workspace/headers /workspace/debs
ln -sfn /usr/include "$RDMA_CORE_HOME/include"
for library in ibverbs mlx5; do
  target=$(ldconfig -p | awk -v lib="lib${library}.so.1" '$1==lib {print $NF; exit}')
  test -n "$target"
  ln -sfn "$target" "$RDMA_CORE_HOME/lib/lib${library}.so"
  ln -sfn "$target" "$RDMA_CORE_HOME/lib/lib${library}.so.1"
done
if [[ ! -f /workspace/headers/usr/include/nvml.h && ! -f /usr/include/nvml.h && ! -f /usr/local/cuda/include/nvml.h ]]; then
  cd /workspace/debs
  apt-get download libnvidia-ml-dev
  for deb in libnvidia-ml-dev*.deb; do dpkg-deb -x "$deb" /workspace/headers; done
fi
export CPATH=/workspace/headers/usr/include:/usr/local/cuda/include/cccl:${CPATH:-}
export LIBRARY_PATH=$RDMA_CORE_HOME/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$RDMA_CORE_HOME/lib:${LD_LIBRARY_PATH:-}
cd /workspace/DeepEP
uv run --no-project --python /workspace/venv/bin/python /workspace/patch_deepep.py
uv pip install --python /workspace/venv/bin/python --no-deps --no-build-isolation -v .
export PYTHONPATH=/workspace/titan-sim:/workspace/Megatron-Bridge/src:/workspace/Megatron-LM
uv run --no-project --python /workspace/venv/bin/python /workspace/preflight.py
uv run --no-project --python /workspace/venv/bin/python python -c 'from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP; assert HAVE_HYBRIDEP; from megatron.bridge.training.pretrain import pretrain'
export HF_HOME=/workspace/hf-cache
for precision in fp8_mx nvfp4; do
  uv run --no-project --python /workspace/venv/bin/python /workspace/titan-sim/scripts/profile_gpt_oss.py --bridge-root /workspace/Megatron-Bridge --precision "$precision" --config-only --output "/workspace/results/config-$precision"
done
uv pip freeze --python /workspace/venv/bin/python > /workspace/dependency-freeze.txt
