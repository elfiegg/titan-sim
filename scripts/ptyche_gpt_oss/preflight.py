import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import torch

out = {"torch": torch.__version__, "cuda": torch.version.cuda,
       "nccl": torch.cuda.nccl.version(), "gpu": torch.cuda.get_device_name(0),
       "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
       "packages": {}, "imports": {}}
for name in ("transformer-engine", "transformers", "megatron-core", "megatron-bridge",
             "flash-attn", "nemo-run", "nvidia-modelopt", "torch-hybrid-ep"):
    try: out["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: out["packages"][name] = None
for name in ("transformer_engine.pytorch", "megatron.core", "megatron.bridge",
             "megatron.bridge.recipes.gpt_oss", "deep_ep"):
    try:
        module = importlib.import_module(name)
        out["imports"][name] = {"ok": True, "file": str(getattr(module, "__file__", None))}
        if name == "deep_ep": out["imports"][name]["has_HybridEPBuffer"] = hasattr(module, "HybridEPBuffer")
    except Exception as exc:
        out["imports"][name] = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
out["record_memory_history_signature"] = str(inspect.signature(torch.cuda.memory._record_memory_history))
torch.cuda.memory._record_memory_history(max_entries=10000)
p = torch.nn.Parameter(torch.zeros(2**20, device="cuda", dtype=torch.float32))
grad_buffer = torch.ones_like(p, dtype=torch.bfloat16)
optimizer = torch.optim.Adam([p], lr=1e-3)
p.grad = grad_buffer.float()
optimizer.step()
torch.cuda.synchronize()
before = torch.cuda.memory_allocated()
grad_bytes = p.grad.numel() * p.grad.element_size()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
after = torch.cuda.memory_allocated()
out["gradient_lifetime_probe"] = {"scope": "PyTorch Adam sanity check, not Megatron training",
    "gradient_bytes": grad_bytes, "before_zero_grad_bytes": before,
    "after_zero_grad_bytes": after, "released_bytes": before - after,
    "grad_is_none": p.grad is None, "matches_expected_release": before - after == grad_bytes}
root = Path(os.environ["GPTOSS_OUTPUT"])
root.mkdir(parents=True, exist_ok=True)
torch.cuda.memory._dump_snapshot(str(root / "preflight-snapshot.pickle"))
torch.cuda.memory._record_memory_history(enabled=None)
(root / "preflight.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2), flush=True)
