"""Canonical identities for local, non-executable evidence artifacts."""
from dataclasses import asdict
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def workload_fingerprint(workload):
    """Bind evidence to the full source, including placement and unresolved gaps."""
    return digest(dict(nodes=[asdict(node) for node in workload.nodes],
                       storage=[asdict(item) for item in workload.storage],
                       manifest=workload.manifest, hardware=workload.hardware))
