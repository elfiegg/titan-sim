"""Explicit fail-closed support for importing unused optional TorchTitan paths.

NVIDIA 26.09 predates the pipeline activation liveness exports imported by
TorchTitan's optional Dist-MoE module. The default non-Dist-MoE, non-PP recipe
never invokes these APIs. Missing exports receive raising placeholders, not
implementations. Any execution reaching them fails rather than changing work.
"""
from contextlib import contextmanager
from unittest.mock import patch
from contextlib import ExitStack


@contextmanager
def unused_pipeline_imports():
    import torch.distributed.pipelining as pipeline
    installed = []
    def unavailable(*args, **kwargs):
        raise NotImplementedError('This PyTorch build lacks pipeline activation liveness. Import compatibility cannot execute this optional path.')
    class UnavailableStageInfo:
        def __init__(self, *args, **kwargs):
            unavailable()
    candidates = {'analyze_pipeline_activation_liveness': unavailable,
                  'PipelineStageInfo': UnavailableStageInfo}
    with ExitStack() as stack:
        for name, replacement in candidates.items():
            if not hasattr(pipeline, name):
                stack.enter_context(patch.object(pipeline, name, replacement, create=True))
                installed.append(name)
        yield {'policy': 'raising placeholders for unused optional imports only', 'missing_exports': installed}


@contextmanager
def nvidia2609_compat():
    """Adapt an omitted optional FSDP callback only when its value is None."""
    import inspect
    import torch.distributed.fsdp as fsdp
    with unused_pipeline_imports() as metadata, ExitStack() as stack:
        metadata['none_only_arguments'] = []
        policy = fsdp.MixedPrecisionPolicy
        original = policy.__init__
        if 'param_dtype_override_fn' not in inspect.signature(original).parameters:
            def initialize(self, *args, param_dtype_override_fn=None, **kwargs):
                if param_dtype_override_fn is not None:
                    raise NotImplementedError('This PyTorch cannot implement param_dtype_override_fn; a non-None callback cannot be dropped')
                original(self, *args, **kwargs)
            stack.enter_context(patch.object(policy, '__init__', initialize))
            metadata['none_only_arguments'].append('MixedPrecisionPolicy.param_dtype_override_fn=None')
        yield metadata
