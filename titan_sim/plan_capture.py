"""Export native plans without replacing the recipe's execution mechanisms.

Reading prepared objects does not execute them. Observing lazy graph preparation
does run the requested original training steps, including communication, optimizer
updates and configured side effects. It is not an allocation-free planning mode.
"""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import hashlib
import json
import os
import sys

from .torchtitan import _build_trainer, _fake_groups_only, _replace


def export_trainer_plan(trainer, *, provenance=None, rank=0, observe_steps=0):
    """Inspect a prepared native runtime, optionally observing lazy preparation.

    The caller owns the trainer and process groups. The bounded observation never
    stops a rank in the middle of a pipeline step: doing so can strand its peers.
    Unknown eager runtimes need a separate extraction implementation; they are
    never silently converted to GraphTrainer.
    """
    from .native_plan import extract_plan, seal_plan
    if type(observe_steps) is not int or observe_steps < 0:
        raise ValueError('observe_steps must be a nonnegative integer')
    engine = getattr(trainer, 'engine', None)
    runtime = getattr(engine, 'pp_schedule', None)
    if runtime is None:
        raise ValueError('native trainer does not expose engine.pp_schedule')
    provenance = copy.deepcopy(provenance or {})
    if observe_steps == 0:
        plan = extract_plan(runtime, provenance=provenance, rank=rank)
        plan['observation'] = {'mode': 'prepared_objects', 'completed_steps': 0}
        return seal_plan(plan)
    if not callable(getattr(runtime, 'ensure_ready', None)):
        raise ValueError('native runtime has no supported lazy graph preparation hook')
    for method in ('train', 'train_step', 'should_continue_training'):
        if not callable(getattr(trainer, method, None)):
            raise ValueError('native trainer does not expose '+method)

    original_ready = runtime.ensure_ready
    original_step = trainer.train_step
    original_continue = trainer.should_continue_training
    completed = 0
    snapshots = []
    snapshot_hashes = set()
    sampled_steps = set()
    observation_error = None

    def observe_ready(*args, **kwargs):
        nonlocal observation_error
        result = original_ready(*args, **kwargs)
        if observation_error is not None or completed in sampled_steps:
            return result
        try:
            plan = extract_plan(runtime, provenance=provenance, rank=rank)
            sampled_steps.add(completed)
            identity = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'),
                                                 allow_nan=False).encode()).hexdigest()
        except Exception as error:
            # Introspection must not strand peers before the native action runs.
            # Native preparation/execution failures still propagate normally.
            observation_error = error
            return result
        if identity not in snapshot_hashes:
            snapshot_hashes.add(identity)
            snapshots.append(plan)
        return result

    def observe_step(*args, **kwargs):
        nonlocal completed
        result = original_step(*args, **kwargs)
        completed += 1
        return result

    def should_continue():
        return completed < observe_steps and original_continue()

    with ExitStack() as stack:
        stack.enter_context(_replace(runtime, 'ensure_ready', observe_ready))
        stack.enter_context(_replace(trainer, 'train_step', observe_step))
        stack.enter_context(_replace(trainer, 'should_continue_training', should_continue))
        trainer.train()
    if observation_error is not None:
        raise ValueError('native plan observation failed after bounded steps completed: '
                         + str(observation_error)) from observation_error
    if not snapshots:
        raise ValueError('native execution did not expose a prepared graph plan')
    if len(snapshots) != 1:
        raise ValueError('native plan changed during observation; one static plan cannot represent this run')
    plan = snapshots[0]
    plan['observation'] = {'mode': 'native_steps', 'requested_steps': observe_steps,
                           'completed_steps': completed,
                           'snapshot_boundary': 'first ensure_ready return in each step',
                           'training_side_effects': True}
    if completed != observe_steps:
        plan.setdefault('unresolved', []).append('Requested native observation steps did not complete.')
        plan['capture_status'] = 'incomplete'
    return seal_plan(plan)


def export_recipe(config):
    """Trusted CLI exporter using TorchTitan's unchanged loader and trainer."""
    if not isinstance(config, dict):
        raise ValueError('plan export configuration must be an object')
    execution = config.get('execution')
    if execution not in ('native', 'fake'):
        raise ValueError('execution must explicitly be native or fake')
    argv = config.get('recipe_args')
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError('recipe_args must be the native TorchTitan argument list')
    if '--print-config' in argv:
        raise ValueError('--print-config exits before plan export')
    if execution == 'fake' and not any(
        arg == '--comm-backend=fake' or
        (arg == '--comm-backend' and argv[i+1:i+2] == ['fake'])
        for i, arg in enumerate(argv)
    ):
        raise ValueError('fake execution requires native recipe_args --comm-backend fake')
    steps = config.get('observe_steps', 0)
    if type(steps) is not int or steps < 0:
        raise ValueError('observe_steps must be a nonnegative integer')
    handoff = None
    if config.get('runtime_manifest'):
        from .runtime import check_runtime
        handoff = check_runtime(config['runtime_manifest'], config.get('capability_report'),
                                config.get('expected_runtime'))
        if handoff['status'] != 'ready':
            raise ValueError('prepare-runtime handoff is blocked: '+'; '.join(handoff['unresolved']))

    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    provenance = dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      execution=execution, recipe_args=list(argv),
                      runtime=config.get('runtime_provenance', {}),
                      allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      float32_matmul_precision=torch.get_float32_matmul_precision())
    if handoff is not None:
        provenance['runtime_handoff'] = handoff
    trainer = None
    owns_group = not torch.distributed.is_initialized()
    try:
        with redirect_stdout(sys.stderr), (
            FakeTensorMode(allow_non_fake_inputs=True) if execution == 'fake' else nullcontext()
        ), (_fake_groups_only() if execution == 'fake' else nullcontext()):
            trainer = _build_trainer(argv, provenance)
            rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
                    else int(os.environ.get('RANK', 0)))
            plan = export_trainer_plan(trainer, provenance=provenance, rank=rank,
                                       observe_steps=steps)
            plan['execution'] = execution
            from .native_plan import seal_plan
            return seal_plan(plan)
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            if owns_group and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
