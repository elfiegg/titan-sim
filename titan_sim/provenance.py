"""Stable policy fingerprints for conservative comparison eligibility."""
import dataclasses
import enum
import hashlib
import json


def config_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {'__type__': type(value).__module__ + '.' + type(value).__qualname__,
                **{field.name: config_value(getattr(value, field.name)) for field in dataclasses.fields(value)}}
    if isinstance(value, enum.Enum):
        return {'enum': type(value).__module__ + '.' + type(value).__qualname__, 'value': config_value(value.value)}
    if isinstance(value, dict):
        return {str(key): config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_value(item) for item in value]
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if callable(value) and hasattr(value, '__qualname__'):
        return {'callable': value.__module__ + '.' + value.__qualname__}
    raise TypeError(f'Cannot fingerprint configuration value of type {type(value).__qualname__}')


def execution_policy(config, *, routing_scenario):
    training = config.training
    policy = {
        key: config_value(getattr(config, key))
        for key in ('model', 'parallelism', 'activation_remat', 'loss', 'optim')
    }
    policy['training'] = {key: config_value(getattr(training, key)) for key in (
        'dtype', 'mixed_precision_param', 'mixed_precision_reduce', 'disable_cuda_graphs',
        'num_tokens_per_microbatch_per_dp_rank', 'num_tokens_per_train_step', 'max_context_length')}
    policy['routing_scenario'] = routing_scenario
    policy['inputs'] = 'synthetic fixed-length documents; random initialization'
    encoded = json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()
    return {'execution_policy': policy, 'policy_fingerprint': hashlib.sha256(encoded).hexdigest()}


def validate_calibration_context(trace_metadata, hardware, calibrations):
    expected = {key: trace_metadata.get(key) for key in ('torch_version', 'torchtitan_commit')}
    expected['hardware'] = hardware.get('name')
    context = calibrations.get('context', {}) if isinstance(calibrations, dict) else {}
    mismatches = {key: {'expected': value, 'found': context.get(key)}
                  for key, value in expected.items() if value is None or context.get(key) != value}
    if mismatches:
        raise ValueError(f'Calibration environment does not match capture: {mismatches}')
