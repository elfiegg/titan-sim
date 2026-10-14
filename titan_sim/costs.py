"""Units-checked analytical lower bounds and exact measured timing lookup."""
import json
from collections.abc import Mapping
from .events import Event, nonnegative

DTYPE_ALIASES = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'fp64': 'float64'}


def validate_hardware(hardware: dict) -> dict:
    if not isinstance(hardware, dict):
        raise ValueError('hardware must be an object')
    for key in ('memory_bandwidth_bytes_s', 'memory_capacity_bytes'):
        if key not in hardware or nonnegative(hardware[key], key) <= 0:
            raise ValueError(f'{key} must be positive')
    rates = hardware.get('peak_flops')
    if not isinstance(rates, dict) or not rates:
        raise ValueError('peak_flops must map execution dtypes to positive FLOP/s')
    for dtype, rate in rates.items():
        if not isinstance(dtype, str) or nonnegative(rate, f'peak_flops[{dtype}]') <= 0:
            raise ValueError('peak FLOP/s must be positive')
    for key in ('network_bandwidth_bytes_s', 'host_device_bandwidth_bytes_s'):
        if key in hardware and nonnegative(hardware[key], key) <= 0:
            raise ValueError(f'{key} must be positive')
    return hardware


def _signature(value) -> str:
    if not isinstance(value, dict):
        raise ValueError('calibration signature must be a JSON object')
    if value.get('status') == 'unsupported':
        raise ValueError('cannot calibrate an unsupported operator signature')
    try:
        return json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('calibration signature must contain finite JSON values') from error


def calibration_key(event: Event | dict) -> str:
    event = Event.from_dict(event) if isinstance(event, dict) else event
    if 'signature' in event.metadata:
        signature = json.loads(_signature(event.metadata['signature']))
        return json.dumps([event.name, {'signature': signature}], separators=(',', ':'), sort_keys=True)
    return json.dumps([event.name, event.metadata.get('shape', event.metadata.get('shapes', [])),
                       event.dtype], separators=(',', ':'), sort_keys=True)


def _calibration(event, calibrations):
    if calibrations is None:
        return None
    if 'signature' in event.metadata:
        _signature(event.metadata['signature'])
    entries = calibrations.get('entries', calibrations) if isinstance(calibrations, dict) else calibrations
    if isinstance(entries, Mapping):
        result = entries.get(calibration_key(event))
        if result is None:
            return None
        if not isinstance(result, dict):
            raise ValueError('calibration entries must include duration_us and provenance')
        return result
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError('calibration entries must be objects')
            if 'signature' in event.metadata:
                if (entry.get('name') == event.name and 'signature' in entry
                        and _signature(entry['signature']) == _signature(event.metadata['signature'])):
                    return entry
                continue
            if (entry.get('name') == event.name and entry.get('dtype') == event.dtype
                    and entry.get('shape', entry.get('shapes', [])) == event.metadata.get('shape', event.metadata.get('shapes', []))):
                return entry
        return None
    raise ValueError('calibrations must be a mapping or list')


def estimate_cost(event: Event | dict, hardware: dict, timing='roofline', calibrations=None) -> dict:
    event = Event.from_dict(event) if isinstance(event, dict) else event
    validate_hardware(hardware)
    if timing not in ('roofline', 'calibrated', 'hybrid'):
        raise ValueError(f'unknown timing mode: {timing}')
    if event.metadata.get('metadata_only') or event.metadata.get('execution_kind') == 'metadata':
        return {'duration_us': 0.0, 'source': 'metadata_only', 'provenance': 'Metadata/view operation; no GPU kernel is launched'}
    if str(event.metadata.get('device', 'cuda')).startswith('meta'):
        return {'duration_us': 0.0, 'source': 'meta_unmodeled', 'provenance': 'Meta tensor construction has no GPU work or allocation'}
    if str(event.metadata.get('device', 'cuda')).startswith('cpu'):
        return {'duration_us': 0.0, 'source': 'host_unmodeled', 'provenance': 'CPU event; host duration not modeled'}
    if event.kind not in ('compute', 'collective'):
        return {'duration_us': 0.0, 'source': 'control', 'provenance': 'zero device work'}
    if event.metadata.get('sparse') or event.metadata.get('sparsity', 0):
        raise ValueError('sparse work requires an explicit supported cost model')
    if event.metadata.get('mixed_precision') or event.metadata.get('work_components'):
        raise ValueError('mixed execution work must be split into separate dtype events')
    # A user-provided event duration is trusted only with a traceable source.
    explicit = event.metadata.get('duration_provenance', event.metadata.get('provenance'))
    if event.duration_us is not None:
        if not explicit:
            raise ValueError(f'{event.event_id}: duration_us requires duration_provenance')
        return {'duration_us': event.duration_us, 'source': 'explicit', 'provenance': explicit}
    if timing in ('calibrated', 'hybrid'):
        measured = _calibration(event, calibrations)
        if measured is not None:
            if not measured.get('provenance') or 'duration_us' not in measured:
                raise ValueError('calibration requires duration_us and provenance')
            duration = nonnegative(measured['duration_us'], 'calibrated duration_us')
            return {'duration_us': duration, 'source': 'calibrated', 'provenance': measured['provenance']}
    host_device = event.metadata.get('traffic_domain') == 'host_device'
    transfer_bytes = nonnegative(event.metadata.get('host_device_transfer_bytes', max(event.bytes_read, event.bytes_write)),
                                 'host_device_transfer_bytes') if host_device else 0
    work = event.flops or event.bytes_read or event.bytes_write or transfer_bytes
    if not work:
        return {'duration_us': 0.0, 'source': 'zero_work', 'provenance': 'no arithmetic or traffic'}
    if timing == 'calibrated':
        raise ValueError(f'{event.event_id}: missing exact calibration for {calibration_key(event)}')
    q = event.bytes_read + event.bytes_write
    if host_device:
        bandwidth = hardware.get('host_device_bandwidth_bytes_s')
        if bandwidth is None:
            raise ValueError('host/device transfer requires exact calibration or host_device_bandwidth_bytes_s; HBM bandwidth cannot model it')
        if event.flops:
            raise ValueError('host/device transfer with arithmetic requires exact calibration or separate work events')
        duration = transfer_bytes / bandwidth * 1e6
        bound = 'host_device'
        compute_us = 0.0
        memory_us = duration
    elif event.kind == 'collective':
        bandwidth = hardware.get('network_bandwidth_bytes_s')
        if bandwidth is None:
            raise ValueError('collective analytical timing requires network_bandwidth_bytes_s')
        if event.flops:
            raise ValueError('collective arithmetic must be represented as separate compute work')
        duration = q / bandwidth * 1e6
        bound = 'network'
        compute_us = 0.0
        memory_us = duration
    else:
        dtype = DTYPE_ALIASES.get(event.dtype, event.dtype)
        rates = {DTYPE_ALIASES.get(key, key): value for key, value in hardware['peak_flops'].items()}
        if event.flops and dtype not in rates:
            raise ValueError(f'no dense peak FLOP/s for execution dtype {event.dtype}')
        compute_us = event.flops / rates[dtype] * 1e6 if event.flops else 0.0
        memory_us = q / hardware['memory_bandwidth_bytes_s'] * 1e6
        duration = max(compute_us, memory_us)
        bound = 'compute' if compute_us >= memory_us else 'memory'
    result = {'duration_us': duration, 'source': 'roofline_fallback' if timing == 'hybrid' else 'roofline',
            'provenance': 'analytical lower bound; no efficiency or launch overhead',
            'bound': bound, 'compute_us': compute_us, 'memory_us': memory_us}

    if host_device:
        result['transfer_bytes'] = transfer_bytes
        result['traffic_basis'] = ('explicit logical host/device payload' if 'host_device_transfer_bytes' in event.metadata
                                   else 'estimated logical host/device payload=max(read,write); conversion/intermediates not modeled')
    return result
