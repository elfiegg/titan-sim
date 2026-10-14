"""Validated, dependency-aware input events (all times in microseconds)."""
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

KINDS = frozenset({'compute', 'allocate', 'release', 'wait', 'host_sync', 'collective'})


def nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a nonnegative finite number')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{label} must be a nonnegative finite number')
    return result


def identifiers(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f'{label} must be a list of nonempty strings')
    return tuple(value)


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    rank: int = 0
    stream: str = 'default'
    name: str = ''
    deps: tuple[str, ...] = ()
    flops: float = 0
    bytes_read: int = 0
    bytes_write: int = 0
    dtype: str = 'float32'
    storage_id: str | None = None
    size_bytes: int = 0
    category: str = 'unclassified'
    duration_us: float | None = None
    host_time_us: float = 0
    read_storage_ids: tuple[str, ...] = ()
    write_storage_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'Event':
        if not isinstance(raw, Mapping):
            raise ValueError('each event must be an object')
        data = dict(raw)
        for key in ('event_id', 'kind'):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f'{key} must be a nonempty string')
        if data['kind'] not in KINDS:
            raise ValueError(f"unsupported event kind: {data['kind']}")
        rank = data.get('rank', 0)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError('rank must be a nonnegative integer')
        for key in ('stream', 'dtype', 'name', 'category'):
            if key in data and (not isinstance(data[key], str) or not data[key]):
                raise ValueError(f'{key} must be a nonempty string')
        for key in ('flops', 'bytes_read', 'bytes_write', 'size_bytes', 'host_time_us'):
            if key in data:
                nonnegative(data[key], key)
        for key in ('bytes_read', 'bytes_write', 'size_bytes'):
            if key in data and not isinstance(data[key], int):
                raise ValueError(f'{key} must be an integer')
        if data.get('duration_us') is not None:
            nonnegative(data['duration_us'], 'duration_us')
        if data.get('storage_id') is not None and (not isinstance(data['storage_id'], str) or not data['storage_id']):
            raise ValueError('storage_id must be a nonempty string')
        if data['kind'] in ('allocate', 'release') and not data.get('storage_id'):
            raise ValueError('allocate/release requires storage_id')
        data['deps'] = identifiers(data.get('deps', []), 'deps')
        for key, alias in (('read_storage_ids', 'reads'), ('write_storage_ids', 'writes')):
            data[key] = identifiers(data.pop(alias, data.get(key, [])), key)
        metadata = data.pop('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('metadata must be an object')
        known = cls.__dataclass_fields__
        extra = {key: value for key, value in data.items() if key not in known}
        data = {key: value for key, value in data.items() if key in known}
        data['metadata'] = {**metadata, **extra}
        return cls(**data)
