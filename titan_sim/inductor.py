"""Static AST importer for a deliberately small Inductor Python-wrapper subset.

Source is parsed, never imported/evaluated. Storage comes from wrapper allocation,
view, assignment and delete statements. Kernel arithmetic/traffic/access modes
must come from a separate exact call-index manifest; no kernel cost is guessed.
"""
from __future__ import annotations

import ast
import hashlib
import json
from copy import deepcopy

from .compiled import artifact_digest, lower_compiled_manifest
from .events import Event

_DTYPE_BYTES = {'float64': 8, 'float32': 4, 'float16': 2, 'bfloat16': 2,
                'float8_e4m3fn': 1, 'float8_e5m2': 1, 'int64': 8, 'int32': 4,
                'int16': 2, 'int8': 1, 'uint8': 1, 'bool': 1}


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + '.' + node.attr
    raise ValueError('unsupported computed callable or attribute')


def _literal(node):
    """Only structural constants; ast.literal_eval does not execute source."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError('dynamic expressions are unsupported in static wrapper geometry') from error


def _geometry(shape, stride, dtype):
    if not isinstance(shape, (list, tuple)) or not isinstance(stride, (list, tuple)) or len(shape) != len(stride):
        raise ValueError('shape and stride must have the same static rank')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (*shape, *stride)):
        raise ValueError('shape/stride must contain nonnegative integer constants')
    if dtype not in _DTYPE_BYTES:
        raise ValueError('unsupported storage dtype: ' + str(dtype))
    elements = 0 if 0 in shape else 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    return {'shape': list(shape), 'stride': list(stride), 'dtype': dtype,
            'size_bytes': elements * _DTYPE_BYTES[dtype]}


def _dtype(node):
    name = _name(node)
    if not name.startswith('torch.') or name.removeprefix('torch.') not in _DTYPE_BYTES:
        raise ValueError('dtype must be an explicitly supported torch dtype')
    return name.removeprefix('torch.')


def _default_inputs(tree):
    """Read only get_args' static rand_strided declarations, not its execution."""
    result = {}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_args']
    if len(functions) != 1:
        return result
    for statement in functions[0].body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            continue
        call = statement.value
        if not isinstance(call, ast.Call) or _name(call.func) != 'rand_strided' or len(call.args) != 2:
            continue
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        if set(kwargs) - {'device', 'dtype', 'extra_size'} or 'dtype' not in kwargs or 'device' not in kwargs:
            raise ValueError('unsupported rand_strided input declaration')
        if 'extra_size' in kwargs and _literal(kwargs['extra_size']) != 0:
            raise ValueError('rand_strided extra_size is unsupported')
        device = _literal(kwargs['device'])
        if device not in ('cpu', 'cuda', 'cuda:0'):
            raise ValueError('only one explicit CPU or CUDA device is supported')
        result[statement.targets[0].id] = {**_geometry(_literal(call.args[0]), _literal(call.args[1]), _dtype(kwargs['dtype'])),
                                         'device': device.split(':')[0]}
    return result


def import_inductor_wrapper(source: str, *, kernel_costs: dict, provenance,
                             region_id='inductor', input_specs=None) -> dict:
    """Return {manifest, artifact, capture, report} from static output_code.py.

    kernel_costs contains source_sha256 and kernels (k0, k1, ... in call order).
    Each kernel entry has target,
    dtype, flops, bytes_read, bytes_write, reads, writes and provenance. Access
    selectors are positional integer indices or keyword names (e.g. 'out').
    Every tensor argument must have declared access. An optional input_specs
    mapping supplies dtype/device/shape/stride when get_args is absent; shape
    and stride must still match the wrapper's guards.
    """
    if not isinstance(source, str) or not source or not provenance:
        raise ValueError('wrapper source and provenance are required')
    if not isinstance(kernel_costs, dict) or set(kernel_costs) - {'source_sha256', 'kernels', 'input_specs_sha256', 'input_alignment'}:
        raise ValueError('kernel_costs requires source_sha256 and an exact kernels mapping')
    cost_document = deepcopy(kernel_costs)
    if kernel_costs.get('source_sha256') != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError('kernel cost manifest source_sha256 does not match wrapper')
    if input_specs is not None and kernel_costs.get('input_specs_sha256') != artifact_digest(input_specs):
        raise ValueError('explicit input_specs require matching input_specs_sha256 in kernel cost manifest')
    kernel_costs = kernel_costs.get('kernels')
    if not isinstance(kernel_costs, dict):
        raise ValueError('kernel_costs.kernels must be an exact call-index mapping')
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'call']
    functions += [node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'Runner'
                  for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'call']
    if len(functions) != 1:
        raise ValueError('wrapper must contain exactly one call(args) or Runner.call(self,args)')
    function = functions[0]
    arguments = [arg.arg for arg in function.args.args]
    if arguments not in (['args'], ['self', 'args']) or function.args.vararg or function.args.kwarg or function.args.kwonlyargs or function.args.defaults:
        raise ValueError('unsupported wrapper call signature')
    wrapper_specs = _default_inputs(tree)
    specs = wrapper_specs if input_specs is None else deepcopy(input_specs)
    if not isinstance(specs, dict):
        raise ValueError('input_specs must map argument names to static metadata')
    for name, spec in specs.items():
        if not isinstance(spec, dict) or set(spec) - {'shape', 'stride', 'dtype', 'device', 'size_bytes'} or not {'shape', 'stride', 'dtype', 'device'} <= set(spec):
            raise ValueError('input metadata requires shape/stride/dtype/device')
        checked = _geometry(spec['shape'], spec['stride'], spec['dtype'])
        if spec['device'] not in ('cpu', 'cuda') or ('size_bytes' in spec and spec['size_bytes'] != checked['size_bytes']):
            raise ValueError('input storage size/device disagrees with static geometry')
        specs[name] = {**checked, 'device': spec['device']}
    if input_specs is not None and wrapper_specs and specs != wrapper_specs:
        raise ValueError('explicit input_specs conflict with compiler get_args dtype/device/geometry')
    variables, slots, views, externals = {}, {}, {}, {}
    kernels, statements, guards, outputs = [], [], set(), []
    seen_unpack, returned, view_counter = False, False, 0
    devices = set()
    stream_variables = set()
    alignment_checks = []
    input_alignment = cost_document.get('input_alignment', {})
    if not isinstance(input_alignment, dict) or set(input_alignment) - set(specs):
        raise ValueError('input_alignment must reference known input names')

    def new_view(name, storage, geometry, offset, *, external=False):
        nonlocal view_counter
        key = f'{name}@{view_counter}'
        view_counter += 1
        view = {'id': key, 'variable': name, 'storage': storage, **geometry,
                'offset_bytes': offset, 'external': external, 'created_before_kernel': len(kernels),
                'deleted_after_kernel': None, 'uses': []}
        views[key] = view
        return view

    def drop(name):
        if name not in variables:
            raise ValueError(f'delete/reuse references unknown or deleted variable {name!r}')
        variables.pop(name)['deleted_after_kernel'] = len(kernels) - 1

    def resolve(node, hint='view'):
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f'use of unknown/deleted tensor {node.id!r}')
            return variables[node.id]
        if isinstance(node, ast.Call) and _name(node.func) == 'reinterpret_tensor':
            if len(node.args) != 4 or node.keywords:
                raise ValueError('reinterpret_tensor requires tensor, static shape/stride and element offset')
            owner = resolve(node.args[0])
            geometry = _geometry(_literal(node.args[1]), _literal(node.args[2]), owner['dtype'])
            offset = _literal(node.args[3])
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError('reinterpret_tensor offset must be a nonnegative integer')
            offset = owner['offset_bytes'] + offset * _DTYPE_BYTES[owner['dtype']]
            size = externals[owner['storage']] if owner['external'] else slots[owner['storage']]['size_bytes']
            if offset + geometry['size_bytes'] > size:
                raise ValueError('reinterpret_tensor view exceeds its backing allocation')
            return new_view(hint, owner['storage'], geometry, offset, external=owner['external'])
        raise ValueError('unsupported tensor expression in wrapper')

    def kernel(call, target):
        key = f'k{len(kernels)}'
        spec = kernel_costs.get(key)
        fields = {'target', 'dtype', 'flops', 'bytes_read', 'bytes_write', 'reads', 'writes', 'provenance'}
        if not isinstance(spec, dict) or set(spec) != fields or spec['target'] != target or not spec['provenance']:
            raise ValueError(f'{key}: missing/mismatched exact kernel cost/access manifest for {target}')
        args = {i: arg for i, arg in enumerate(call.args)}
        for kw in call.keywords:
            if kw.arg is None or kw.arg in args:
                raise ValueError('expanded/duplicate kernel arguments are unsupported')
            args[kw.arg] = kw.value
        tensor_args = {}
        temporary_views = []
        for selector, arg in args.items():
            if isinstance(arg, ast.Name) and arg.id in variables:
                tensor_args[selector] = resolve(arg)
            elif isinstance(arg, ast.Call) and _name(arg.func) == 'reinterpret_tensor':
                tensor_args[selector] = resolve(arg, key + '_view')
                temporary_views.append(tensor_args[selector])
            elif isinstance(arg, ast.Name) and any(view['variable'] == arg.id for view in views.values()):
                resolve(arg)  # report stale/deleted references explicitly
            elif selector in ('grid', 'stream') and target.endswith('.run'):
                # Launch metadata never executes in the importer. Only static
                # grids and a wrapper-owned raw stream are supported.
                if selector == 'stream':
                    if not isinstance(arg, ast.Name) or arg.id not in stream_variables:
                        raise ValueError('kernel stream must be a declared get_raw_stream(0)')
                elif isinstance(arg, ast.Call) and _name(arg.func) == 'grid':
                    if arg.keywords:
                        raise ValueError('dynamic grid keyword arguments unsupported')
                    for value in arg.args:
                        _literal(value)
                else:
                    _literal(arg)
            else:
                _literal(arg)
        accesses = {}
        touched = set()
        for access in ('reads', 'writes'):
            selectors = spec[access]
            if not isinstance(selectors, list) or any(isinstance(s, bool) or not isinstance(s, (int, str)) for s in selectors) or len(set(selectors)) != len(selectors):
                raise ValueError('kernel access selectors must be unique positional indices/keyword names')
            ids = []
            for selector in selectors:
                if selector not in tensor_args:
                    raise ValueError('kernel access selector does not reference a tensor argument')
                view = tensor_args[selector]
                view['uses'].append(len(kernels))
                touched.add(selector)
                if view['external']:
                    ids.append(view['storage'])
                else:
                    ids.extend([view['id'], '__backing_' + view['storage']])
            accesses[access] = sorted(set(ids))
        for view in temporary_views:
            view['deleted_after_kernel'] = len(kernels)
        if touched != set(tensor_args):
            raise ValueError('kernel tensor arguments require explicit read/write access')
        if not touched:
            raise ValueError('kernel with no declared tensor access is unsupported')
        event = {'event_id': key, 'kind': 'compute', **{k: spec[k] for k in ('dtype', 'flops', 'bytes_read', 'bytes_write')}}
        Event.from_dict(event)
        kernels.append({'id': key, 'name': target, **{k: spec[k] for k in ('dtype', 'flops', 'bytes_read', 'bytes_write', 'provenance')},
                        **accesses, 'logical_ops': [key]})

    def parse(body):
        nonlocal seen_unpack, returned
        for statement in body:
            if returned:
                raise ValueError('statements after wrapper return are unsupported')
            statements.append({'line': statement.lineno, 'kind': type(statement).__name__})
            if isinstance(statement, ast.With):
                if len(statement.items) != 1 or statement.items[0].optional_vars is not None:
                    raise ValueError('unsupported context manager in wrapper')
                call = statement.items[0].context_expr
                if not isinstance(call, ast.Call) or _name(call.func) != 'torch.cuda._DeviceGuard' or len(call.args) != 1 or _literal(call.args[0]) != 0 or call.keywords:
                    raise ValueError('only static CUDA device-0 guard is supported')
                devices.add('cuda')
                parse(statement.body)
            elif isinstance(statement, ast.Assign):
                if len(statement.targets) != 1:
                    raise ValueError('chained assignment is unsupported')
                lhs, rhs = statement.targets[0], statement.value
                if isinstance(lhs, (ast.Tuple, ast.List)) and isinstance(rhs, ast.Name) and rhs.id == 'args':
                    if seen_unpack or any(not isinstance(x, ast.Name) for x in lhs.elts):
                        raise ValueError('unsupported argument unpacking')
                    names = [x.id for x in lhs.elts]
                    if len(set(names)) != len(names) or set(names) != set(specs):
                        raise ValueError('all wrapper inputs require exact dtype/device/shape/stride metadata')
                    for name in names:
                        spec = specs[name]
                        sid = f'{region_id}:input:{name}'
                        externals[sid] = spec['size_bytes']
                        variables[name] = new_view(name, sid, spec, 0, external=True)
                        devices.add(spec['device'])
                    seen_unpack = True
                    continue
                if not isinstance(lhs, ast.Name):
                    raise ValueError('only simple tensor assignment is supported')
                name = lhs.id
                if isinstance(rhs, ast.Call) and _name(rhs.func) in ('empty_strided_cuda', 'empty_strided_cpu'):
                    if len(rhs.args) != 3 or rhs.keywords:
                        raise ValueError('allocation requires static shape/stride/dtype')
                    geometry = _geometry(_literal(rhs.args[0]), _literal(rhs.args[1]), _dtype(rhs.args[2]))
                    if not geometry['size_bytes']:
                        raise ValueError('zero-byte wrapper allocations are currently unsupported')
                    device = _name(rhs.func).removeprefix('empty_strided_')
                    devices.add(device)
                    sid = f's{len(slots)}'
                    slots[sid] = {'id': sid, 'size_bytes': geometry['size_bytes'], 'category': 'compiled_storage',
                                  'allocated_before_kernel': len(kernels), 'source_line': statement.lineno}
                    view = new_view(name, sid, geometry, 0)
                elif isinstance(rhs, ast.Call) and _name(rhs.func) == 'copy_if_misaligned':
                    if len(rhs.args) != 1 or rhs.keywords:
                        raise ValueError('unsupported copy_if_misaligned calling convention')
                    owner = resolve(rhs.args[0])
                    input_name = owner['storage'].removeprefix(f'{region_id}:input:')
                    evidence = input_alignment.get(input_name)
                    required = {'alignment_bytes', 'pointer_mod_alignment', 'copy_if_misaligned_returns_same_storage', 'provenance'}
                    if (not owner['external'] or owner['offset_bytes'] != 0 or not isinstance(evidence, dict)
                            or set(evidence) != required or not evidence['provenance']):
                        raise ValueError('copy_if_misaligned requires explicit input alignment and no-copy evidence')
                    alignment = evidence['alignment_bytes']
                    remainder = evidence['pointer_mod_alignment']
                    if (isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0
                            or isinstance(remainder, bool) or not isinstance(remainder, int) or remainder != 0
                            or evidence['copy_if_misaligned_returns_same_storage'] is not True):
                        raise ValueError('unknown/misaligned copy_if_misaligned path requires an explicit copy allocation plan')
                    alignment_checks.append({'input': input_name, 'source_line': statement.lineno, **evidence,
                                             'interpretation': 'conditional aligned no-copy path; not a general allocation prediction'})
                    view = new_view(name, owner['storage'], {k: owner[k] for k in ('shape', 'stride', 'dtype', 'size_bytes')}, 0, external=True)
                elif isinstance(rhs, ast.Call) and _name(rhs.func) == 'get_raw_stream':
                    if len(rhs.args) != 1 or _literal(rhs.args[0]) != 0 or rhs.keywords:
                        raise ValueError('only static get_raw_stream(0) is supported')
                    if name in variables:
                        raise ValueError('stream variable shadows tensor')
                    stream_variables.add(name)
                    continue
                else:
                    previous = resolve(rhs, name)
                    view = new_view(name, previous['storage'], {k: previous[k] for k in ('shape', 'stride', 'dtype', 'size_bytes')}, previous['offset_bytes'], external=previous['external']) if isinstance(rhs, ast.Name) else previous
                if name in variables:
                    drop(name)
                variables[name] = view
            elif isinstance(statement, ast.Delete):
                for target in statement.targets:
                    if not isinstance(target, ast.Name):
                        raise ValueError('only simple tensor delete is supported')
                    drop(target.id)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                target = _name(call.func)
                if target == 'args.clear':
                    if not seen_unpack or call.args or call.keywords:
                        raise ValueError('invalid argument clearing')
                elif target in ('assert_size_stride', 'assert_size_stride_grouped'):
                    if call.keywords or (len(call.args) != 3 and not (target.endswith('grouped') and len(call.args) == 4)):
                        raise ValueError('unsupported size/stride guard')
                    nodes = call.args[0].elts if target.endswith('grouped') and isinstance(call.args[0], (ast.Tuple, ast.List)) else [call.args[0]]
                    shapes, strides = _literal(call.args[1]), _literal(call.args[2])
                    if not target.endswith('grouped'):
                        shapes, strides = [shapes], [strides]
                    if len(nodes) != len(shapes) or len(nodes) != len(strides):
                        raise ValueError('grouped guard length mismatch')
                    for node, shape, stride in zip(nodes, shapes, strides):
                        view = resolve(node)
                        if list(shape) != view['shape'] or list(stride) != view['stride']:
                            raise ValueError('input metadata disagrees with compiled shape/stride guard')
                        if view['external']:
                            guards.add(view['variable'])
                elif target == 'torch.cuda.set_device':
                    if len(call.args) != 1 or _literal(call.args[0]) != 0 or call.keywords:
                        raise ValueError('only CUDA device zero is supported')
                    devices.add('cuda')
                else:
                    kernel(call, target)
            elif isinstance(statement, ast.Return):
                if not isinstance(statement.value, (ast.Tuple, ast.List)):
                    raise ValueError('wrapper return must be an explicit flat tensor tuple/list')
                outputs.extend(resolve(node, 'output') for node in statement.value.elts)
                returned = True
            else:
                raise ValueError(f'unsupported wrapper statement {type(statement).__name__} at line {statement.lineno}')
    parse(function.body)
    if not seen_unpack or not returned or not kernels or guards != set(specs):
        raise ValueError('wrapper requires guarded inputs, explicit tensor return and at least one costed kernel')
    if set(kernel_costs) != {k['id'] for k in kernels}:
        raise ValueError('unused or missing kernel cost entries')
    if len(devices) != 1:
        raise ValueError('mixed devices are unsupported in a static compiled region')
    device = devices.pop()
    last = len(kernels) - 1
    retained_slots = {view['storage'] for view in outputs if not view['external']}
    buffers, retained = [], []
    for sid, slot in slots.items():
        members = [v for v in views.values() if not v['external'] and v['storage'] == sid]
        if not any(v['uses'] for v in members):
            raise ValueError('unused allocation cannot be represented by kernel-bounded lifetimes')
        final = last if any(v['deleted_after_kernel'] is None for v in members) else max(v['deleted_after_kernel'] for v in members)
        first = slot['allocated_before_kernel']
        if first > last or final < first:
            raise ValueError('allocation has no representable kernel lifetime')
        base = '__backing_' + sid
        buffers.append({'id': base, 'slot': sid, 'size_bytes': slot['size_bytes'], 'offset_bytes': 0,
                        'first_kernel': f'k{first}', 'last_kernel': f'k{final}'})
        if sid in retained_slots:
            retained.append(base)
        for view in members:
            if not view['uses']:
                continue
            buffers.append({'id': view['id'], 'slot': sid, 'size_bytes': view['size_bytes'], 'offset_bytes': view['offset_bytes'],
                            'first_kernel': f'k{min(view["uses"])}', 'last_kernel': f'k{max(view["uses"])}', 'alias_of': base})
    manifest = {'schema_version': 1, 'id': region_id, 'device': device, 'plan_source': 'compiler_artifact',
                'provenance': provenance, 'external_storages': externals,
                'slots': [{k: slot[k] for k in ('id', 'size_bytes', 'category')} for slot in slots.values()],
                'buffers': buffers, 'kernels': kernels, 'retained_buffers': retained, 'graph': {'mode': 'none'},
                'calibration_context': {'backend': 'inductor_static_wrapper', 'device': device,
                                        'wrapper_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                        'kernel_cost_manifest_sha256': artifact_digest(cost_document)}}
    bound_keys = ('external_storages', 'slots', 'buffers', 'kernels', 'graph', 'device', 'retained_buffers')
    artifact = {'kind': 'inductor_wrapper_allocation_plan', 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'source_text': source, 'provenance': provenance, 'device': device,
                'input_metadata_source': 'wrapper_get_args' if input_specs is None else 'explicit_input_specs_checked_against_guards',
                'input_specs': specs, 'kernel_costs': cost_document,
                'plan': deepcopy({key: manifest[key] for key in bound_keys})}
    manifest['artifact_sha256'] = artifact_digest(artifact)
    capture = lower_compiled_manifest(manifest, artifact=artifact)
    output_storages = sorted({v['storage'] if v['external'] else f'{region_id}:slot:{v["storage"]}' for v in outputs})
    report = {'source_sha256': artifact['source_sha256'], 'device': device,
              'allocation_count': len(slots), 'physical_allocation_bytes': sum(s['size_bytes'] for s in slots.values()),
              'output_storage_ids': output_storages, 'conditional_alignment_paths': alignment_checks, 'logical_views': list(views.values()), 'parsed_statements': statements,
              'allocation_provenance': 'derived by static AST interpretation of supplied compiler wrapper',
              'cost_provenance': 'explicit exact call-index kernel manifest; not inferred from wrapper or measured',
              'limitations': ['Static one-device Python wrappers only; dynamic shapes/control flow, pools and unknown syntax fail.',
                              'CPU wrapper import proves CPU compiler storage derivation, not CUDA fusion or allocations.',
                              'Source is parsed, never executed; authenticity of supplied wrapper requires external capture provenance.',
                              'Returned physical storage stays allocated until a caller emits its release.',
                              'copy_if_misaligned supports only an explicit aligned/no-copy scenario; unknown or misaligned paths fail.']}
    capture['metadata']['inductor_import'] = report
    return {'manifest': manifest, 'artifact': artifact, 'capture': capture, 'report': report}
