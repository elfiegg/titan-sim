"""Explicit, trusted Python adapters; no framework/model discovery on import."""
import hashlib
import importlib
import json
import math
from .workload import Workload
from .core import validate


def resolve(reference):
    module, separator, name = reference.partition(':')
    if not separator or not module or not name or ':' in name:
        raise ValueError('adapter reference must be importable.module:function')
    function = getattr(importlib.import_module(module), name, None)
    if not callable(function):
        raise ValueError('adapter reference does not identify a callable: '+reference)
    return function


def capture(reference, config):
    if not isinstance(config, dict):
        raise ValueError('capture configuration must be a JSON object')
    workload = resolve(reference)(config)
    if not isinstance(workload, Workload):
        raise ValueError('capture adapter must return titan_sim.workload.Workload')
    validate(workload.nodes, workload.storage, workload.manifest)
    workload.manifest = dict(workload.manifest, capture_adapter=reference,
                            capture_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())
    return workload


def validate_catalog(catalog, hardware_id):
    if not isinstance(hardware_id, str) or not hardware_id:
        raise ValueError('hardware id must be a nonempty string')
    if not isinstance(catalog, dict):
        raise ValueError('calibration catalog must be a JSON object keyed by cost signature')
    for key, record in catalog.items():
        if not isinstance(key, str) or not key or not isinstance(record, dict):
            raise ValueError('invalid calibration entry')
        if record.get('hardware_id') != hardware_id:
            raise ValueError('calibration hardware mismatch: '+key)
        samples = record.get('samples_seconds')
        duration = record.get('seconds')
        if not isinstance(samples, list) or not samples:
            raise ValueError('calibration samples missing: '+key)
        for value in samples + [duration]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid calibration duration: '+key)
        checksum = hashlib.sha256(json.dumps(samples, separators=(',', ':')).encode()).hexdigest()
        if record.get('evidence_sha256') != checksum:
            raise ValueError('calibration sample checksum mismatch: '+key)
    return len(catalog)


def calibrate(reference, workload, config):
    if not isinstance(config, dict):
        raise ValueError('calibration configuration must be a JSON object')
    catalog = resolve(reference)(workload, config)
    validate_catalog(catalog, workload.hardware['id'])
    return catalog
