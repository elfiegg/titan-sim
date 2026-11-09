"""Freeze reproducible predictions before a separately executed native control.

The control is a trusted adapter. Hashes establish artifact consistency, not
independent proof of its implementation, coverage or measurement semantics.
"""
from copy import deepcopy
from datetime import datetime, timezone
import math
from statistics import median

from .adapters import resolve, validate_catalog
from .core import simulate
from .identity import digest, workload_fingerprint


def _now():
    return datetime.now(timezone.utc).isoformat()


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name+' must be finite and positive')
    return value


def freeze_prediction(workload, catalog, *, context):
    """Freeze model output and all identity hashes before native control execution.

    Context explicitly identifies the recipe, runtime, target, placement, measured
    scope and token count. Full-update comparisons require complete declared
    timing coverage; memory can remain inconclusive independently.
    """
    required = {'recipe', 'runtime', 'hardware', 'placement', 'scope', 'useful_tokens', 'timing_metric'}
    if not isinstance(context, dict) or set(context) != required or any(not context[k] for k in required - {'useful_tokens'}):
        raise ValueError('validation context requires recipe/runtime/hardware/placement/scope/useful_tokens/timing_metric')
    if context['timing_metric'] not in ('completed_scope_wall_seconds', 'completed_scope_device_seconds'):
        raise ValueError('unsupported timing measurement metric')
    if context['scope'] not in ('training_update', 'native_window', 'native_region'):
        raise ValueError('unsupported validation scope')
    if workload.manifest.get('validation_context') != context:
        raise ValueError('validation context differs from producer-recorded source identity')
    if context['hardware'] != workload.hardware or context['placement'] != workload.manifest.get('rank_placement'):
        raise ValueError('validation context differs from workload hardware/placement')
    if context['useful_tokens'] != workload.manifest.get('useful_tokens', 0):
        raise ValueError('validation token count differs from workload')
    if type(context['useful_tokens']) is not int or context['useful_tokens'] < 0:
        raise ValueError('useful tokens must be a nonnegative integer')
    full = context['scope'] == 'training_update'
    if full:
        def check_capture(manifest):
            if manifest.get('capture_status', 'complete') != 'complete':
                raise ValueError('full prediction requires complete capture evidence')
            for source in manifest.get('source_manifests', []):
                check_capture(source)
        check_capture(workload.manifest)
        _positive(context['useful_tokens'], 'training update useful tokens')
        if workload.manifest.get('prediction_scope') != 'training_update':
            raise ValueError('full training update scope must be declared by workload producer')
        world = workload.manifest.get('world_size')
        if type(world) is not int or sorted(workload.manifest['ranks']) != list(range(world)):
            raise ValueError('training update prediction requires complete target ranks')
    validate_catalog(catalog, workload.hardware['id'])
    prediction = simulate(*workload.arguments(), calibration=catalog, strict=full)
    _positive(prediction['seconds'], 'predicted duration')
    result = dict(schema='frozen-native-prediction', version=1, frozen_at=_now(),
        source_workload_sha256=workload_fingerprint(workload), catalog_sha256=digest(catalog),
        context=deepcopy(context), prediction=prediction,
        evidence_class='declared model prediction; independent control has not run')
    result['sha256'] = digest(result)
    return result


def _verify(frozen):
    if frozen.get('schema') != 'frozen-native-prediction' or frozen.get('version') != 1:
        raise ValueError('unsupported frozen prediction')
    if frozen.get('sha256') != digest({k:v for k,v in frozen.items() if k != 'sha256'}):
        raise ValueError('frozen prediction checksum mismatch')


def run_control(frozen, *, runner, config):
    """Invoke a trusted native control after verifying the frozen prediction.

    Runner receives identity context and configuration, never prediction values
    or calibration samples. It must perform a fresh run and attest matching scope.
    No GPU allocation or launch is implicit in the comparator.
    """
    _verify(frozen)
    start = _now()
    control = resolve(runner)(deepcopy(frozen['context']), deepcopy(config))
    result = compare_control(frozen, control)
    result['execution'] = dict(runner=runner, started_at=start, completed_at=_now(),
        order='frozen prediction verified before invoking control adapter',
        limitation='Fresh execution and measurement fidelity remain the trusted adapter contract.')
    return result


def compare_control(frozen, control):
    """Compare matching observations; imported controls alone do not prove holdout order."""
    _verify(frozen)
    if control.get('context') != frozen['context']:
        raise ValueError('control scope/recipe/runtime/target/placement/tokens mismatch')
    if control.get('measurement_kind') != 'fresh_native_control':
        raise ValueError('control must declare fresh native execution')
    if not control.get('run_id') or not isinstance(control.get('evidence'), dict) or not control['evidence']:
        raise ValueError('control requires run identity and retained evidence digests')
    for value in control['evidence'].values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('control evidence requires SHA256 digests')
        if value in (frozen['sha256'], frozen['source_workload_sha256'], frozen['catalog_sha256']):
            raise ValueError('prediction/calibration artifacts cannot serve as independent control evidence')
    samples = control.get('samples_seconds')
    if not isinstance(samples, list) or not samples:
        raise ValueError('control timing samples required')
    for value in samples:
        _positive(value, 'native control sample')
    observed = median(samples)
    predicted = frozen['prediction']['seconds']
    timing = dict(predicted_seconds=predicted, observed_median_seconds=observed,
        relative_error=(predicted-observed)/observed, sample_count=len(samples),
        observed_min_seconds=min(samples), observed_max_seconds=max(samples))
    tokens = frozen['context']['useful_tokens']
    if tokens:
        timing.update(predicted_tokens_per_second=tokens/predicted,
                      observed_tokens_per_second=tokens/observed)
    memory = []
    # Allocator-reserved samples are not interchangeable with physical usage.
    physical = frozen['prediction'].get('physical_memory', {}).get('per_device', {})
    for device, observation in control.get('memory', {}).items():
        if device not in physical:
            raise ValueError('control memory references an unknown physical device')
        metric = observation.get('metric')
        value = observation.get('peak_bytes')
        if type(value) is not int or value < 0:
            raise ValueError('control memory peak must be nonnegative integer bytes')
        row = dict(device=device, metric=metric, observed_peak_bytes=value, comparison='incomparable')
        same_scope = observation.get('scope') == frozen['prediction'].get('physical_memory', {}).get('memory_scope')
        if metric == 'physical_device_used_bytes' and same_scope:
            upper = physical[device].get('peak_upper_bytes')
            row.update(predicted_upper_bytes=upper,
                       comparison='unknown_bound' if upper is None else 'upper_bound_exceeded' if value > upper else 'within_upper_bound')
        elif metric not in ('physical_device_used_bytes', 'process_allocated_bytes', 'process_reserved_bytes'):
            raise ValueError('unsupported control memory metric')
        memory.append(row)
    exceeded = any(row['comparison'] == 'upper_bound_exceeded' and
                   row['observed_peak_bytes'] > physical[row['device']]['budget_bytes'] for row in memory)
    complete_memory = (bool(physical) and len(memory) == len(physical) and
                       all(row['comparison'] in ('upper_bound_exceeded', 'within_upper_bound') for row in memory))
    false_fit = (True if exceeded else False if complete_memory else None) if frozen['prediction']['memory_verdict'] == 'estimated_fit' else None
    return dict(schema='native-prediction-comparison', version=1,
        frozen_prediction_sha256=frozen['sha256'], control_sha256=digest(control),
        scope=frozen['context']['scope'], timing=timing, memory=memory,
        false_fit_observed=false_fit,
        qualification='comparison for the exact declared scope; imported evidence does not prove independent execution order',
        unresolved=deepcopy(frozen['prediction'].get('unresolved', [])))
