#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export UV_CACHE_DIR=/workspace/uv-cache
if ! command -v uv >/dev/null; then
  mkdir -p /workspace/tools
  curl -fL --retry 2 https://github.com/astral-sh/uv/releases/download/0.9.7/uv-aarch64-unknown-linux-gnu.tar.gz -o /workspace/tools/uv.tgz
  tar -xzf /workspace/tools/uv.tgz -C /workspace/tools
  export PATH=/workspace/tools/uv-aarch64-unknown-linux-gnu:$PATH
fi
if [[ ! -x /workspace/venv/bin/python ]]; then
  uv venv --system-site-packages --python /usr/bin/python /workspace/venv
fi
# Keep the image's Torch packages; collect their exact versions, not guessed pins.
/usr/bin/python - <<'PY'
import importlib.metadata as m
from pathlib import Path
names=['torch','torchvision','torchaudio','transformer-engine']
pins=[]
for n in names:
 try:pins.append(n+'=='+m.version(n))
 except m.PackageNotFoundError:pass
Path('/workspace/runtime-constraints.txt').write_text('\n'.join(pins)+'\n')
PY
uv run --no-project --python /workspace/venv/bin/python --no-sync /workspace/install_overlay.py
export PATH=/workspace/venv/bin:$PATH
export VIRTUAL_ENV=/workspace/venv
export PYTHONPATH=/workspace/Megatron-Bridge/src:/workspace/Megatron-LM
uv run --no-project --python /workspace/venv/bin/python --no-sync /workspace/preflight.py
uv pip freeze --python /workspace/venv/bin/python > /workspace/dependency-freeze.txt
