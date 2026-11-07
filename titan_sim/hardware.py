"""Explicit target hardware contracts; published peaks are not measured rates."""
import copy
import math


def _positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError('invalid hardware '+name)


def _capacity(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('capacity_bytes must be a nonnegative integer number of bytes')


def validate_hardware(hardware):
    if not isinstance(hardware, dict) or not isinstance(hardware.get('id'), str) or not hardware['id']:
        raise ValueError('hardware requires a nonempty id')
    devices = hardware.get('devices')
    if devices is not None:
        if not isinstance(devices, dict) or not devices:
            raise ValueError('hardware.devices must map device IDs to capacity records')
        for name, device in devices.items():
            if not isinstance(name, str) or not isinstance(device, dict):
                raise ValueError('invalid device record')
            _capacity(device.get('capacity_bytes'))
            reserve = device.get('reserve_fraction', hardware.get('reserve_fraction', .1))
            _positive(reserve, 'reserve_fraction', zero=True)
            if reserve >= 1:
                raise ValueError('reserve_fraction must be below 1')
    elif 'capacity_bytes' not in hardware:
        raise ValueError('hardware needs capacity_bytes or explicit devices')
    if 'capacity_bytes' in hardware:
        _capacity(hardware['capacity_bytes'])
    if 'reserve_fraction' in hardware:
        _positive(hardware['reserve_fraction'], 'reserve_fraction', zero=True)
        if hardware['reserve_fraction'] >= 1:
            raise ValueError('reserve_fraction must be below 1')
    for key in ('effective_flops_per_second', 'effective_hbm_bytes_per_second',
                'interconnect_bytes_per_second', 'pp_bytes_per_second'):
        if key in hardware:
            _positive(hardware[key], key)
    for key in ('comm_latency_seconds', 'launch_seconds'):
        if key in hardware:
            _positive(hardware[key], key, zero=True)
    return hardware


def analytical_target(specification, config):
    """Construct an explicit analytical scenario from a published specification.

    User must supply usable per-device capacity, placement-independent topology
    assumptions, and effective rates. Rack totals never become device capacity
    or effective throughput implicitly. This deliberately does not create a
    calibrated hardware identity from a product-family name.
    """
    if not isinstance(specification, dict) or not specification.get('source_url'):
        raise ValueError('hardware specification requires a source_url')
    result = copy.deepcopy(config)
    validate_hardware(result)
    if not result.get('assumptions'):
        raise ValueError('analytical target requires explicit assumptions')
    result['specification'] = copy.deepcopy(specification)
    result['evidence_level'] = 'analytical_assumptions'
    return result
