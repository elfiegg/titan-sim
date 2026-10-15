"""Install a private Python overlay without resolving/replacing GPU runtime packages."""
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

roots = ['transformers==5.8.1', 'mistral-common>=1.10.0', 'peft>=0.18.1',
         'datasets>=2.20.0', 'accelerate', 'diffusers>=0.36.0', 'einops',
         'omegaconf>=2.3.0', 'tensorboard>=2.19.0', 'rich', 'wandb>=0.25.0',
         'hydra-core==1.3.2', 'qwen-vl-utils', 'timm', 'open-clip-torch>=3.2.0',
         'mlflow>=3.9.0', 'comet-ml>=3.50.0', 'sentencepiece', 'tiktoken', 'ninja',
         'megatron-energon~=7.0', 'nvidia-resiliency-ext==0.6.0',
         'multi-storage-client==0.51.0', 'tensorstore~=0.1,!=0.1.46,!=0.1.72',
         's3fs==2026.2.0', 'fsspec==2026.2.0']
protected = {'torch', 'torchvision', 'torchaudio', 'transformer-engine', 'transformer-engine-torch', 'triton'}
runtime_before = {d.metadata['Name']: d.version for d in md.distributions()
                  if canonicalize_name(d.metadata['Name']) in protected or canonicalize_name(d.metadata['Name']).startswith('nvidia-')}
for iteration in range(12):
    md.FastPath.__new__.cache_clear()
    todo = [Requirement(r) for r in roots]
    visited, missing = set(), set()
    while todo:
        req = todo.pop()
        name = canonicalize_name(req.name)
        try: dist = md.distribution(name)
        except md.PackageNotFoundError: dist = None
        if name in protected or (name.startswith('nvidia-') and name != 'nvidia-resiliency-ext'):
            if dist is None:
                raise RuntimeError(f'Protected runtime dependency missing: {req}')
            if not req.specifier.contains(dist.version, prereleases=True):
                raise RuntimeError(f'Container runtime does not satisfy {req}: {dist.version}')
            continue
        if dist is None or dist.version is None or not req.specifier.contains(dist.version, prereleases=True):
            # The marker was already evaluated in the requesting extra's context.
            # Passing it to uv at top level would drop extra-only requirements.
            missing.add(str(req).split(';', 1)[0].strip()); continue
        key = (name, tuple(sorted(req.extras)))
        if key in visited: continue
        visited.add(key)
        for raw in dist.requires or []:
            child = Requirement(raw)
            if child.marker is None or any(child.marker.evaluate({'extra': extra}) for extra in (req.extras or {''})):
                todo.append(child)
    if not missing:
        break
    print(json.dumps({'dependency_pass': iteration, 'install': sorted(missing)}), flush=True)
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--no-deps', *sorted(missing)], check=True)
else:
    raise RuntimeError('Dependency overlay did not converge')
runtime_after = {name: md.version(name) for name in runtime_before}
assert runtime_before == runtime_after, (runtime_before, runtime_after)
Path('/workspace/overlay-runtime-guard.json').write_text(json.dumps(runtime_after, indent=2))
