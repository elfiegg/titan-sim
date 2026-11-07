"""Native recipe adapter: use TorchTitan's loader, trainer and original steps.

No model registry or replacement forward lives here. The imported TorchTitan
must expose ConfigLoader.load(args), config.build(), and a trainer with train(),
train_step(), should_continue_training(), engine and close(). Unsupported trainer
contracts fail explicitly. PyTorch imports are lazy to keep the simulator light.
"""
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stdout
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys


@contextmanager
def _replace(instance, name, replacement):
    """Temporarily instrument one instance, never a global trainer class."""
    existed, previous = name in vars(instance), vars(instance).get(name)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(instance, name, previous)
        else:
            delattr(instance, name)


def _register_state(recorder, engine):
    for index, model in enumerate(engine.model_parts):
        recorder.register_model(model, f'model_parts.{index}')
    optim = getattr(engine, 'optim', None)
    optimizers = getattr(optim, 'optimizers', ())
    for optimizer in optimizers:
        recorder.register_optimizer(optimizer)


def capture_trainer(trainer, *, hardware, steps=1, warmup_steps=0, provenance=None,
                    observer='profiler', trace_dir=None, execution_trace=True):
    """Run a bounded native train loop and return a rank-local Workload.

    The caller owns trainer/process-group creation and cleanup. This function
    does not disable compilation, checkpointing, validation or replace data.
    Native warmup/training really executes. It is not a performance benchmark.
    A failed/short trace is returned with capture_status=blocked/incomplete.
    """
    import torch
    from .torch_capture import TorchCapture

    if type(steps) is not int or steps < 1 or type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError('steps must be positive and warmup_steps nonnegative integers')
    for name in ('train', 'train_step', 'should_continue_training'):
        if not callable(getattr(trainer, name, None)):
            raise ValueError('unsupported TorchTitan trainer contract: missing '+name)
    if type(execution_trace) is not bool:
        raise ValueError('execution_trace must be a boolean')
    if observer != 'profiler' and not execution_trace:
        raise ValueError('execution_trace=false applies only to profiler observation')
    engine = trainer.engine
    dist = torch.distributed
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get('RANK', 0))
    world = dist.get_world_size() if dist.is_initialized() else int(os.environ.get('WORLD_SIZE', 1))
    if observer == 'profiler':
        if trace_dir is None:
            raise ValueError('native profiler capture requires trace_dir (a fresh output directory)')
        from .profiler_capture import ProfilerCapture
        recorder = ProfilerCapture(trace_dir=trace_dir, rank=rank, world_size=world,
                                   device=engine.device, provenance=provenance, execution_trace=execution_trace)
    elif observer == 'dispatch':
        recorder = TorchCapture(rank=rank, world_size=world, device=engine.device, provenance=provenance)
    else:
        raise ValueError('observer must be profiler or dispatch')
    original_step, original_continue = trainer.train_step, trainer.should_continue_training
    completed, captured = 0, 0
    manifest = dict(scope='native TorchTitan rank-local train_step operators', observer=observer,
                    capture_status='complete', requested_steps=steps, warmup_steps=warmup_steps,
                    adaptations=['bounded native training loop', observer+' instrumentation'],
                    execution_trace=execution_trace if observer == 'profiler' else None,
                    calibration_scope=('GPU occurrence costs only; reusable implementation matching unavailable'
                                       if observer == 'profiler' and not execution_trace else
                                       'operator signatures are candidates; native kernel matching still required'))

    def observed_step(*args, **kwargs):
        nonlocal completed, captured
        if completed < warmup_steps:
            result = original_step(*args, **kwargs)
        else:
            recorder.step = captured
            _register_state(recorder, engine)
            with recorder:
                result = original_step(*args, **kwargs)
            _register_state(recorder, engine)
            captured += 1
        completed += 1
        return result

    def should_continue():
        return completed < warmup_steps + steps and original_continue()

    with ExitStack() as stack:
        stack.enter_context(_replace(trainer, 'train_step', observed_step))
        stack.enter_context(_replace(trainer, 'should_continue_training', should_continue))
        for name in ('forward_backward', 'optim_step'):
            original = getattr(engine, name, None)
            if callable(original):
                def observed(*args, _original=original, _name=name, **kwargs):
                    with recorder.phase(_name):
                        if _name == 'optim_step' and completed >= warmup_steps:
                            _register_state(recorder, engine)
                        return _original(*args, **kwargs)
                stack.enter_context(_replace(engine, name, observed))
        try:
            trainer.train()
        except Exception as error:
            manifest.update(capture_status='blocked', error_type=type(error).__qualname__, error=str(error))
            recorder.gaps.add('Native training raised an exception; this is a partial trace.')
    if captured != steps:
        if manifest['capture_status'] == 'complete':
            manifest['capture_status'] = 'incomplete'
        recorder.gaps.add('The requested number of training steps was not captured.')
    if not any(n.attrs.get('operator') for n in recorder.nodes):
        manifest['capture_status'] = 'incomplete' if manifest['capture_status'] == 'complete' else manifest['capture_status']
        recorder.gaps.add('No operator dispatch observed; CUDA graph replay/compiled execution may be opaque.')
    manifest.update(completed_steps=completed, captured_steps=captured,
                    trainer_type=type(trainer).__module__+'.'+type(trainer).__qualname__)
    return recorder.finish(hardware=hardware, manifest=manifest)


def _source_manifest(module):
    path = Path(module.__file__)
    return dict(module=module.__name__, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _package_digest(loader_module):
    root = Path(loader_module.__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode()+b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _build_trainer(argv, provenance):
    """Load the unchanged native recipe and retain its resolved identity."""
    loader_module = importlib.import_module('torchtitan.config.loader')
    provenance['loader'] = _source_manifest(loader_module)
    provenance['torchtitan_python_source_sha256'] = _package_digest(loader_module)
    native_config = loader_module.ConfigLoader().load(argv)
    if getattr(native_config, 'create_seed_checkpoint', False):
        raise ValueError('seed-checkpoint recipes do not run training steps')
    from torchtitan.observability.logging import init_logger
    from torchtitan.observability import structured_logger as sl
    init_logger()
    sl.init_structured_logger(source='training', output_dir=native_config.dump_folder,
                              enable=native_config.debug.enable_structured_logging)
    trainer = native_config.build()
    if hasattr(native_config, 'to_dict'):
        from .torch_capture import digest
        provenance['recipe_config'] = native_config.to_dict()
        provenance['resolved_config_sha256'] = digest(provenance['recipe_config'])
    else:
        provenance['recipe_config'] = None
    provenance['trainer'] = _source_manifest(importlib.import_module(type(trainer).__module__))
    return trainer


@contextmanager
def _fake_groups_only():
    """Reject an accidental real backend, including recipe override changes."""
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_backend() != 'fake':
        raise ValueError('fake capture cannot reuse an initialized real process group')
    with ExitStack() as stack:
        for name in ('init_process_group', 'new_group'):
            original = getattr(dist, name)
            def checked(*args, _original=original, _name=name, **kwargs):
                arguments = inspect.signature(_original).bind_partial(*args, **kwargs).arguments
                backend = arguments.get('backend')
                if backend is None and _name == 'new_group' and dist.is_initialized():
                    backend = dist.get_backend()
                if backend != 'fake':
                    raise ValueError('fake capture refuses process-group backend '+str(backend))
                return _original(*args, **kwargs)
            stack.enter_context(_replace(dist, name, checked))
        yield


def capture(config):
    """CLI adapter. Recipe arguments are passed unchanged to ConfigLoader.

    execution must be explicit: native runs the real recipe; fake adds
    FakeTensorMode and requires the recipe's --comm-backend fake argument.
    Fake mode does not emulate CUDA or invent missing custom-op fake kernels.
    Use a dedicated process in the original training environment for either.
    """
    execution = config.get('execution')
    if execution not in ('native', 'fake'):
        raise ValueError('execution must explicitly be native or fake')
    argv = config.get('recipe_args')
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ValueError('recipe_args must be the native TorchTitan argument list')
    if '--print-config' in argv:
        raise ValueError('--print-config exits before capture')
    if execution == 'fake' and not any(
        arg == '--comm-backend=fake' or
        (arg == '--comm-backend' and argv[i+1:i+2] == ['fake']) for i, arg in enumerate(argv)
    ):
        raise ValueError('fake execution requires native recipe_args --comm-backend fake')
    if not isinstance(config.get('hardware'), dict):
        raise ValueError('hardware must explicitly describe the simulation target')
    steps, warmup = config.get('steps', 1), config.get('warmup_steps', 0)
    if type(steps) is not int or steps < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError('steps must be positive and warmup_steps nonnegative integers')
    observer = config.get('observer', 'profiler' if execution == 'native' else 'dispatch')
    if observer not in ('profiler', 'dispatch'):
        raise ValueError('observer must be profiler or dispatch')
    execution_trace = config.get('execution_trace', True)
    if type(execution_trace) is not bool:
        raise ValueError('execution_trace must be a boolean')
    if observer != 'profiler' and not execution_trace:
        raise ValueError('execution_trace=false applies only to profiler observation')
    if observer == 'profiler' and not isinstance(config.get('trace_dir'), str):
        raise ValueError('profiler observer requires trace_dir')
    handoff = None
    if config.get('runtime_manifest'):
        from .runtime import check_runtime
        handoff = check_runtime(config['runtime_manifest'], config.get('capability_report'),
                                config.get('expected_runtime'))
        if handoff['status'] != 'ready':
            raise ValueError('prepare-runtime handoff is blocked: '+'; '.join(handoff['unresolved']))
    import torch
    from .torch_capture import TorchCapture
    from torch._subclasses.fake_tensor import FakeTensorMode

    provenance = dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      execution=execution, observer=observer, execution_trace=execution_trace, recipe_args=argv,
                      runtime=config.get('runtime_provenance', {}),
                      allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      float32_matmul_precision=torch.get_float32_matmul_precision())
    if handoff is not None:
        provenance['runtime_handoff'] = handoff
    trainer, workload = None, None
    owns_group = not torch.distributed.is_initialized()
    context = FakeTensorMode(allow_non_fake_inputs=True) if execution == 'fake' else nullcontext()
    try:
        # Leave stdout for the CLI's JSON result. Native logging stays visible.
        with redirect_stdout(sys.stderr), context, (_fake_groups_only() if execution == 'fake' else nullcontext()):
            trainer = _build_trainer(argv, provenance)
            workload = capture_trainer(trainer, hardware=config['hardware'],
                steps=config.get('steps', 1), warmup_steps=config.get('warmup_steps', 0),
                provenance=provenance, observer=observer, trace_dir=config.get('trace_dir'), execution_trace=execution_trace)
    except Exception as error:
        recorder = TorchCapture(rank=int(os.environ.get('RANK', 0)),
                                world_size=int(os.environ.get('WORLD_SIZE', 1)), provenance=provenance)
        recorder.gaps.add('Native recipe initialization/capture failed; no complete training trace.')
        workload = recorder.finish(hardware=config['hardware'], manifest=dict(
            capture_status='blocked', error_type=type(error).__qualname__, error=str(error)))
    finally:
        if trainer is not None:
            try:
                trainer.close()
            except Exception as error:
                if workload is not None:
                    workload.manifest['capture_status'] = 'blocked'
                    workload.manifest['unresolved'].append('Trainer cleanup failed: '+str(error))
        if owns_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return workload
