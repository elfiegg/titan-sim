"""Explicit packed-storage contracts, not a claim of native TE training capture.

Layouts describe independently stored row/column orientations. Padding and scale
geometry are declared; backend-specific layouts must be translated to this
contract rather than inferred from a dtype name.
"""
import json
import math
from .events import Event, nonnegative


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _align(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def packed_layout(spec: dict) -> dict:
    """Return physical payload/scale bytes for one explicit 2-D packed layout."""
    allowed = {'format', 'shape', 'orientations', 'layout', 'data_alignment_elements',
               'scale_row_alignment', 'scale_column_alignment', 'data_row_alignment', 'scale_storage'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError('unknown quantized layout field')
    fmt = spec.get('format')
    formats = {'fp8': (8, 0, 4, 0), 'mxfp8': (8, 32, 1, 0), 'nvfp4': (4, 16, 1, 4)}
    if fmt not in formats:
        raise ValueError(f'unsupported quantized format {fmt!r}')
    bits, block, scale_bytes, tensor_scale_bytes = formats[fmt]
    expected = 'tensorwise' if fmt == 'fp8' else 'rowwise_block'
    if fmt == 'nvfp4' and spec.get('layout') == 'tile_16x16':
        expected = 'tile_16x16'
    if spec.get('layout') != expected:
        raise ValueError(f'{fmt} requires explicit supported layout {expected!r}')
    scale_storage = spec.get('scale_storage', 'per_block')
    if expected == 'tile_16x16':
        if scale_storage not in ('compact_tiles', 'replicated_rows'):
            raise ValueError('tile_16x16 requires explicit scale_storage compact_tiles or replicated_rows')
    elif scale_storage != 'per_block':
        raise ValueError('scale_storage is unsupported for this layout')
    shape = spec.get('shape')
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError('shape must contain two dimensions')
    rows, cols = [_integer(x, 'shape dimension') for x in shape]
    orientations = spec.get('orientations')
    if not isinstance(orientations, list) or not orientations or any(x not in ('rowwise', 'columnwise') for x in orientations) or len(set(orientations)) != len(orientations):
        raise ValueError('orientations must be unique rowwise/columnwise values')
    data_row_align = _integer(spec.get('data_row_alignment', 1), 'data_row_alignment')
    data_align = _integer(spec.get('data_alignment_elements', 1), 'data_alignment_elements')
    scale_ra = _integer(spec.get('scale_row_alignment', 1), 'scale_row_alignment')
    scale_ca = _integer(spec.get('scale_column_alignment', 1), 'scale_column_alignment')
    if fmt == 'fp8' and (scale_ra != 1 or scale_ca != 1):
        raise ValueError('tensorwise fp8 does not support block-scale padding')
    components = []
    for orientation in orientations:
        r, c = (rows, cols) if orientation == 'rowwise' else (cols, rows)
        padded_c = _align(c, math.lcm(data_align, block or 1, 8 // bits))
        block_rows = 16 if expected == 'tile_16x16' else 1
        padded_r = _align(r, math.lcm(data_row_align, block_rows))
        payload = padded_r * padded_c * bits // 8
        stored_scale_rows = padded_r if scale_storage == 'replicated_rows' else padded_r // block_rows
        scale_shape = [_align(stored_scale_rows, scale_ra), _align(padded_c // block, scale_ca)] if block else [1, 1]
        scales = math.prod(scale_shape) * scale_bytes
        components.append({'orientation': orientation, 'logical_shape': [r, c],
                           'padded_shape': [padded_r, padded_c], 'payload_bytes': payload,
                           'scale_shape': scale_shape, 'scale_bytes': scales,
                           'tensor_scale_bytes': tensor_scale_bytes,
                           'total_bytes': payload + scales + tensor_scale_bytes})
    return {'format': fmt, 'layout': expected, 'scale_storage': scale_storage, 'shape': shape, 'bits': bits,
            'scale_encoding': {'fp8': 'float32', 'mxfp8': 'e8m0', 'nvfp4': 'fp8_e4m3'}[fmt],
            'block_elements': (block * (16 if expected == 'tile_16x16' else 1)) or None,
            'block_geometry': [16 if expected == 'tile_16x16' else 1, block] if block else None, 'components': components,
            'total_bytes': sum(c['total_bytes'] for c in components),
            'provenance': 'explicit packed-layout contract; backend allocation equivalence is unverified'}


def lower_quantized_training(spec: dict) -> dict:
    """Allocate independent training states and lower explicitly costed conversions.

    The caller appends consumer events using returned storage IDs. Persistent
    buffers remain live; each conversion scratch allocation is released after
    its last work event. No full forward/backward or optimizer is synthesized.
    """
    allowed = {'schema_version', 'id', 'rank', 'stream', 'provenance', 'tensors', 'calibration_context'}
    if not isinstance(spec, dict) or set(spec) - allowed or spec.get('schema_version') != 1:
        raise ValueError('unsupported quantization schema or field')
    prefix = spec.get('id', 'quantized')
    if not isinstance(prefix, str) or not prefix or not spec.get('provenance'):
        raise ValueError('quantization id and provenance are required')
    context = spec.get('calibration_context', {})
    if not isinstance(context, dict):
        raise ValueError('calibration_context must be an object')
    try:
        json.dumps(context, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('calibration_context must contain finite JSON values') from error
    rank = _integer(spec.get('rank', 0), 'rank', 0)
    stream = spec.get('stream', 'default')
    events, buffers, layouts = [], {}, {}
    def emit(kind, suffix, **kw):
        raw = {'event_id': f'{prefix}:{suffix}', 'kind': kind, 'rank': rank, 'stream': stream,
               'metadata': {'provenance': spec['provenance'], 'model_contract': 'declared_quantized_storage'}, **kw}
        Event.from_dict(raw)
        events.append(raw)
    tensors = spec.get('tensors')
    if not isinstance(tensors, list) or not tensors:
        raise ValueError('tensors must be a nonempty list')
    names = set()
    dtype_bytes = {'float32': 4, 'bfloat16': 2, 'float16': 2}
    for tensor in tensors:
        if not isinstance(tensor, dict) or set(tensor) - {'name', 'layout', 'states', 'conversion'}:
            raise ValueError('unknown quantized tensor field')
        name = tensor.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('tensor names must be unique nonempty strings')
        names.add(name)
        layout = packed_layout(tensor['layout'])
        layouts[name] = layout
        elements = math.prod(layout['shape'])
        states = tensor.get('states')
        required = {'parameter', 'master_parameter', 'gradient', 'first_moment', 'second_moment'}
        if not isinstance(states, dict) or set(states) != required:
            raise ValueError('states must explicitly declare parameter, master_parameter, gradient, first_moment, second_moment (null omits a state)')
        for role, dtype in states.items():
            if dtype is None:
                continue
            if dtype not in dtype_bytes:
                raise ValueError(f'unsupported training-state dtype {dtype!r}')
            sid = f'{prefix}:{name}:{role}'
            size = elements * dtype_bytes[dtype]
            buffers[sid] = {'size_bytes': size, 'category': role, 'dtype': dtype}
            emit('allocate', f'alloc:{name}:{role}', storage_id=sid, size_bytes=size, category=role)
        packed_ids = []
        for component in layout['components']:
            orientation = component['orientation']
            for part in ('payload', 'scale', 'tensor_scale'):
                size = component[f'{part}_bytes']
                if not size:
                    continue
                sid = f'{prefix}:{name}:{orientation}:{part}'
                packed_ids.append(sid)
                buffers[sid] = {'size_bytes': size, 'category': 'quantization', 'component': part, 'orientation': orientation}
                emit('allocate', f'alloc:{name}:{orientation}:{part}', storage_id=sid, size_bytes=size, category='quantization')
        conversion = tensor.get('conversion')
        if not isinstance(conversion, dict) or set(conversion) != {'source_state', 'execution_dtype', 'flops', 'scratch_bytes', 'scratch_read_bytes', 'scratch_write_bytes', 'provenance'}:
            raise ValueError('conversion must explicitly declare source_state, execution_dtype, flops, scratch bytes/traffic and provenance')
        source = f'{prefix}:{name}:{conversion["source_state"]}'
        if source not in buffers or conversion['source_state'] not in required:
            raise ValueError('conversion source_state must refer to an allocated training state')
        if not conversion['provenance']:
            raise ValueError('conversion requires provenance')
        nonnegative(conversion['flops'], 'conversion flops')
        scratch = _integer(conversion['scratch_bytes'], 'scratch_bytes', 0)
        sr = _integer(conversion['scratch_read_bytes'], 'scratch_read_bytes', 0)
        sw = _integer(conversion['scratch_write_bytes'], 'scratch_write_bytes', 0)
        if not scratch and (sr or sw):
            raise ValueError('scratch traffic requires scratch storage')
        scratch_id = f'{prefix}:{name}:conversion_scratch'
        if scratch:
            emit('allocate', f'alloc:{name}:scratch', storage_id=scratch_id, size_bytes=scratch, category='quantization_workspace')
        emit('compute', f'convert:{name}', name=f'quantize_{layout["format"]}', dtype=conversion['execution_dtype'],
             flops=conversion['flops'], bytes_read=buffers[source]['size_bytes'] + sr,
             bytes_write=layout['total_bytes'] + sw,
             read_storage_ids=[source] + ([scratch_id] if sr else []),
             write_storage_ids=packed_ids + ([scratch_id] if scratch else []),
             metadata={'provenance': conversion['provenance'], 'layout': layout,
                       'calibration_context': context,
                       'signature': {'kind': 'quantization_conversion', 'layout': layout,
                                     'conversion': conversion, 'source_dtype': states[conversion['source_state']],
                                     'context': context},
                       'traffic_basis': 'one full source read and one write of all padded packed components, plus declared scratch traffic'})
        if scratch:
            emit('release', f'release:{name}:scratch', storage_id=scratch_id)
    return {'schema_version': 1, 'source': 'declared_quantized_training_storage',
            'metadata': {'provenance': spec['provenance'], 'native_training_capture': False,
                         'calibration_context': context,
                         **{k: context[k] for k in ('torch_version', 'torchtitan_commit') if k in context}},
            'events': events, 'buffers': buffers, 'layouts': layouts,
            'limitations': ['Conversions use declared arithmetic and traffic; kernels and throughput are not measured.',
                            'Training state is explicit and unsharded; optimizer updates and distributed policies require additional events.']}
