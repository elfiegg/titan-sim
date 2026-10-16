"""Observe native selective-checkpoint execution without changing its policy.

FullAC uses PyTorch's selective checkpoint modes even when every operation is
recomputed. Module backward hooks alone cannot distinguish this replay from
gradient computation. Track the original modes' lifetimes; do not replace their
classes, context factory, dispatch, caches, or saved-tensor hooks.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch


_checkpoint_phase = ContextVar("titan_sim_checkpoint_phase", default=None)
_tracking_active = ContextVar("titan_sim_checkpoint_tracking_active", default=False)


def execution_phase(phase: str, *, in_backward: bool = False) -> str:
    """Resolve execution within the caller's step phase, including SAC replay.

    This is an annotation of observed dispatch, not proof that a compiler's tracing
    operations execute on a device. Compiled storage plans need separate evidence.
    """
    checkpoint_phase = _checkpoint_phase.get()
    if checkpoint_phase is not None:
        return checkpoint_phase
    if in_backward:
        return "backward"
    phase = phase.split("_step_", 1)[0]
    return "forward" if phase == "forward_backward" else phase


@contextlib.contextmanager
def checkpoint_phase_tracking():
    """Annotate native SAC forward/recompute contexts for a scoped capture.

    The private PyTorch mode names are the integration boundary. Fail explicitly
    when they are unavailable rather than label all recomputation as backward.
    Like the fake-device runtime, installation is process scoped; capture sessions
    must not enter/exit these patches concurrently from different Python threads.
    ContextVar state itself is isolated to each execution context.
    """
    import torch.utils.checkpoint as checkpoint

    if _tracking_active.get():
        yield
        return
    active_token = _tracking_active.set(True)
    phase_token = _checkpoint_phase.set(None)
    try:
        with contextlib.ExitStack() as stack:
            for name, phase in (("_CachingTorchDispatchMode", "forward"),
                                ("_CachedTorchDispatchMode", "recompute")):
                mode = getattr(checkpoint, name, None)
                if mode is None:
                    raise RuntimeError(f"Checkpoint phase tracking requires torch.utils.checkpoint.{name}")
                _observe_mode(stack, mode, phase)
            yield
    finally:
        _checkpoint_phase.reset(phase_token)
        _tracking_active.reset(active_token)


def _observe_mode(stack, mode, phase):
    original_enter, original_exit = mode.__enter__, mode.__exit__
    tokens = {}

    @wraps(original_enter)
    def enter(self):
        result = original_enter(self)
        # A nested checkpoint may create a new forward context while its outer
        # region is replaying. That work still belongs to recomputation.
        current = _checkpoint_phase.get()
        observed = "recompute" if current == "recompute" else phase
        tokens.setdefault(id(self), []).append(_checkpoint_phase.set(observed))
        return result

    @wraps(original_exit)
    def exit(self, *exc):
        try:
            return original_exit(self, *exc)
        finally:
            pending = tokens.get(id(self))
            if pending:
                _checkpoint_phase.reset(pending.pop())
                if not pending:
                    del tokens[id(self)]

    stack.enter_context(patch.object(mode, "__enter__", enter))
    stack.enter_context(patch.object(mode, "__exit__", exit))
