"""Pinned upstream entry points; no copied model dimensions or training policies."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

TORCHTITAN_COMMIT = "dbfba1da8ba1cd5688f32d16321a32aaf9132a43"
DEFAULT_RECIPE = "torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b"


def import_torchtitan(checkout: str | Path, *, verify_commit: bool = True) -> str:
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if verify_commit and revision != TORCHTITAN_COMMIT:
        raise ValueError(f"Expected TorchTitan {TORCHTITAN_COMMIT}, found {revision}")
    if not (checkout / "torchtitan" / "__init__.py").is_file():
        raise ValueError(f"No torchtitan package at {checkout}")
    sys.path.insert(0, str(checkout))
    return revision


def load_default_recipe():
    """Return the upstream production recipe with every default intact."""
    module, name = DEFAULT_RECIPE.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)()


def build_smoke_model_config(*, flavor: str = "debugmodel", seq_len: int = 16):
    """Unsharded eager metadata probe, explicitly distinct from production."""
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.deepseek_v3 import build_model_config

    return build_model_config(flavor, seq_len=seq_len, attn_backend="varlen",
                              parallelism=ParallelismConfig())
