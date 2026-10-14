"""Exact-signature CUDA timing measurements and independent held-out checks."""
from __future__ import annotations

import statistics

from .costs import calibration_key, communication_signature
from .events import Event, nonnegative


def timing_entry(event, samples_us, *, provenance):
    event = Event.from_dict(event) if isinstance(event, dict) else event
    if not isinstance(provenance, dict) or provenance.get('source') != 'cuda_event_measurement':
        raise ValueError('timing provenance must identify CUDA-event measurements')
    if not isinstance(samples_us, list) or len(samples_us) < 3:
        raise ValueError('at least three timing samples are required')
    samples = [nonnegative(value, 'timing sample') for value in samples_us]
    if any(value == 0 for value in samples):
        raise ValueError('zero timing samples cannot calibrate GPU work')
    entry = {'name': event.name, 'dtype': event.dtype, 'duration_us': statistics.median(samples),
             'samples_us': samples, 'minimum_us': min(samples), 'maximum_us': max(samples),
             'provenance': provenance}
    if 'signature' in event.metadata:
        entry['signature'] = event.metadata['signature']
    else:
        entry['shape'] = event.metadata.get('shape', event.metadata.get('shapes', []))
    communication = communication_signature(event)
    if communication is not None:
        entry['communication'] = communication
    return calibration_key(event), entry


def measure_cuda(call, *, warmup=10, samples=20):
    """Measure the current-stream CUDA-event interval around an eager call.

    CPU launch gaps can appear as device idle inside this interval. It is not
    an isolated kernel service-time measurement; do not add the same launch
    overhead again when composing an explicitly timed host schedule.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA timing collection requires an actual GPU')
    if any(isinstance(n, bool) or not isinstance(n, int) for n in (warmup, samples)) or warmup < 1 or samples < 3:
        raise ValueError('warmup >= 1 and samples >= 3 must be integers')
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    durations = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = call()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) * 1000)
        del output
    return durations


def heldout_error(predicted_us, samples_us):
    predicted = nonnegative(predicted_us, 'prediction')
    samples = [nonnegative(value, 'held-out sample') for value in samples_us]
    if len(samples) < 3 or min(samples) <= 0:
        raise ValueError('at least three positive held-out samples are required')
    observed = statistics.median(samples)
    return {'predicted_us': predicted, 'heldout_median_us': observed,
            'absolute_error_us': abs(predicted-observed),
            'relative_error': abs(predicted-observed)/observed,
            'scope': 'same-signature eager operator CUDA-event interval; not full training or memory validation'}
