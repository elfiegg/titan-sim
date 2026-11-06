"""Versioned JSON persistence for the direct Node/Storage simulator interface.

No executable code, PyTorch, CUDA, or protobuf is loaded when reading a workload.
This experimental schema is distinct from titan_sim.events (which uses µs).
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from .core import Node, Storage, validate


@dataclass
class Workload:
    nodes: list[Node]
    storage: list[Storage]
    manifest: dict
    hardware: dict

    def arguments(self):
        return self.nodes, self.storage, self.manifest, self.hardware


def write_workload(path, workload):
    nodes, storage, manifest, hardware = workload.arguments()
    validate(nodes, storage, manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(schema='recipe-prescreen-workload', version=1, time_unit='seconds',
                nodes=[asdict(n) for n in nodes], storage=[asdict(s) for s in storage],
                manifest=manifest, hardware=hardware)
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')


def write_bundle(path, nodes, storage, manifest, hardware):
    write_workload(Path(path)/'workload.json', Workload(nodes, storage, manifest, hardware))


def read_bundle(path):
    path = Path(path)
    data = json.loads((path/'workload.json' if path.is_dir() else path).read_text())
    if (data.get('schema'), data.get('version'), data.get('time_unit')) != ('recipe-prescreen-workload', 1, 'seconds'):
        raise ValueError('unsupported workload schema, version, or time unit')
    nodes = [Node(**n) for n in data['nodes']]
    storage = [Storage(**s) for s in data['storage']]
    validate(nodes, storage, data['manifest'])
    return nodes, storage, data['manifest'], data['hardware']


def read_workload(path):
    return Workload(*read_bundle(path))
