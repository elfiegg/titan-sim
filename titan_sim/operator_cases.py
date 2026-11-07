"""Conservative native ATen dense-matmul case extraction and executable replay.

No model code is replaced. Captured tensor values are unavailable: these cases
measure the captured overload on deterministic finite synthetic inputs, not its
original numerical outputs or a selected compiled kernel. Importing this module
or preparing a catalog does not import torch or execute tensors.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from .core import Node
from .native_runner import native_cost_key
from .workload import Workload


# Explicit semantic allowlist: adding arbitrary schema-compatible operators is
# unsafe (indices, masks, random state and sparse patterns can select work).
_OPERATORS = {
    'aten.mm.default': ('mm', 'default', ('self', 'mat2'), False),
    'aten.bmm.default': ('bmm', 'default', ('self', 'mat2'), False),
    'aten.addmm.default': ('addmm', 'default', ('self', 'mat1', 'mat2'), False),
    'aten.addmm_.default': ('addmm_', 'default', ('self', 'mat1', 'mat2'), True),
}
_DTYPES = {'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8}
_POLICY = 'finite-dense-matmul-v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _canonical(value, aliases):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'storage_id':
                if not isinstance(item, str) or not item:
                    raise ValueError('tensor storage identity is missing')
                result['storage_id'] = aliases.setdefault(item, 's'+str(len(aliases)))
            else:
                result[key] = _canonical(item, aliases)
        return result
    if isinstance(value, list):
        return [_canonical(item, aliases) for item in value]
    return value


def _tensor(spec):
    required = {'shape', 'stride', 'storage_offset', 'dtype', 'device', 'layout', 'storage_id'}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError('tensor requires exact plain strided metadata; distributed/opaque tensors unsupported')
    if spec['dtype'] not in _DTYPES or spec['layout'] != 'torch.strided':
        raise ValueError('only real floating-point strided tensors are supported')
    shape, stride = spec['shape'], spec['stride']
    if (not isinstance(shape, list) or not isinstance(stride, list) or len(shape) != len(stride)
            or any(type(x) is not int or x < 1 for x in shape)
            or any(type(x) is not int or x < 0 for x in stride)
            or type(spec['storage_offset']) is not int or spec['storage_offset'] < 0):
        raise ValueError('empty, symbolic or invalid tensor dimensions/strides/offsets unsupported')
    if not isinstance(spec['device'], str) or not (spec['device'] == 'cpu' or
            spec['device'] == 'cuda' or (spec['device'].startswith('cuda:') and spec['device'][5:].isdigit())):
        raise ValueError('only CPU or CUDA device metadata is supported')
    # Nonoverlapping dense/permuted/padded views only; no expanded or overlapping
    # inputs. This avoids ambiguous writes and hidden input value constraints.
    extent = 1
    for step, size in sorted(zip(stride, shape)):
        if size > 1:
            if step < extent:
                raise ValueError('overlapping/expanded tensor storage unsupported')
            extent += (size - 1) * step
    return spec['storage_offset'] + 1 + sum((s-1)*t for s, t in zip(shape, stride))


def _validate_case(operator, arguments, output):
    if operator not in _OPERATORS:
        raise ValueError('operator is outside the shape-reconstructible native dense-matmul allowlist')
    if not isinstance(arguments, dict) or set(arguments) != {'args', 'kwargs'}:
        raise ValueError('captured args/kwargs are required')
    args, kwargs = arguments['args'], arguments['kwargs']
    names, mutation = _OPERATORS[operator][2:]
    if not isinstance(args, list) or not isinstance(kwargs, dict) or len(args) != len(names):
        raise ValueError('all tensor operands must be captured positionally')
    allowed = {'alpha', 'beta'} if 'addmm' in operator else set()
    if set(kwargs) - allowed:
        raise ValueError('unrecognized scalar keyword')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in kwargs.values()):
        raise ValueError('alpha/beta must be finite real scalars')
    for item in args + [output]:
        _tensor(item)
    if len({v['dtype'] for v in args + [output]}) != 1 or len({v['device'] for v in args + [output]}) != 1:
        raise ValueError('mixed dtype/device operands unsupported')
    if len({v['storage_id'] for v in args}) != len(args):
        raise ValueError('aliased input operands unsupported')
    output_alias = output['storage_id'] in {v['storage_id'] for v in args}
    if mutation:
        if output != args[0]:
            raise ValueError('mutating output must exactly alias the first operand')
    elif output_alias:
        raise ValueError('unexpected output/input alias')
    left, right = args[-2:]
    dims = 3 if operator == 'aten.bmm.default' else 2
    if len(left['shape']) != dims or len(right['shape']) != dims or left['shape'][-1] != right['shape'][-2]:
        raise ValueError('matrix operand dimensions are incompatible')
    expected = left['shape'][:-1] + [right['shape'][-1]]
    if dims == 3 and left['shape'][0] != right['shape'][0]:
        raise ValueError('batch dimensions are incompatible')
    if output['shape'] != expected:
        raise ValueError('captured matrix output shape mismatch')
    if 'addmm' in operator:
        bias = args[0]['shape']
        if len(bias) > 2 or any(x not in (1, y) for x, y in zip(reversed(bias), reversed(expected))):
            raise ValueError('addmm bias is not broadcast-compatible')
    return mutation


def prepare_operator_cases(workload):
    """Return a serializable case catalog and explicit per-node skip reasons."""
    provenance = copy.deepcopy(workload.manifest.get('provenance', {}))
    provenance_sha = _digest(provenance)
    cases, skipped = {}, []
    for node in workload.nodes:
        if node.attrs.get('zero_cost_semantic'):
            continue
        operator = node.attrs.get('operator', node.name)
        try:
            if node.attrs.get('status') != 'returned':
                raise ValueError('dispatch did not return successfully')
            aliases = {}
            arguments = _canonical(node.attrs.get('arguments'), aliases)
            output = _canonical(node.attrs.get('outputs'), aliases)
            mutation = _validate_case(operator, arguments, output)
            signature = dict(operator=operator, arguments=arguments, outputs=output,
                             source_provenance_sha256=provenance_sha, input_policy=_POLICY)
            case_id = 'aten-case:'+_digest(signature)
            occurrence = dict(rank=node.rank, node_id=node.id, source_cost_key=node.attrs.get('cost_key'))
            if case_id in cases:
                cases[case_id]['occurrences'].append(occurrence)
            else:
                cases[case_id] = dict(case_id=case_id, **signature, mutates_inputs=mutation,
                                      occurrences=[occurrence])
        except (ValueError, KeyError, TypeError) as exc:
            skipped.append(dict(rank=node.rank, node_id=node.id, operator=operator, reason=str(exc)))
    return dict(schema='native-operator-cases', version=1, cases=list(cases.values()), skipped=skipped,
                source_provenance_sha256=provenance_sha, source_provenance=provenance,
                limitations=['Finite synthetic tensor values; original numerical outputs are not reproduced.',
                             'Native ATen overload replay; compiled/custom kernels and full training coverage are not inferred.'])


def operator_case_workload(workload, case_ids=None):
    """Create an isolated one-device workload; never silently rewrite training."""
    catalog = prepare_operator_cases(workload)
    selected = set(case_ids) if case_ids is not None else {c['case_id'] for c in catalog['cases']}
    if selected - {c['case_id'] for c in catalog['cases']}:
        raise ValueError('requested case IDs absent from eligible native capture')
    nodes = []
    for case in catalog['cases']:
        if case['case_id'] not in selected:
            continue
        implementation = dict(kind='native_aten_overload', operator=case['operator'],
                              input_policy=_POLICY, source_provenance_sha256=catalog['source_provenance_sha256'])
        context = dict(hardware_id=workload.hardware['id'], cuda_graph=False,
                       case_id=case['case_id'], signature_sha256=_digest({k: case[k] for k in
                           ('operator', 'arguments', 'outputs', 'source_provenance_sha256', 'input_policy')}),
                       device_mapping='captured_cuda_device_to_current_cuda_device')
        binding = dict(kind='native_aten_case', implementation=implementation, context=context)
        nodes.append(Node(len(nodes), case['operator'], 0, attrs=dict(
            cost_key=native_cost_key(implementation, context), native_measurement=binding,
            operator_case=copy.deepcopy(case))))
    if not nodes:
        raise ValueError('no eligible native operator cases selected')
    manifest = dict(ranks=[0], world_size=1, scope='isolated native ATen calibration cases',
                    provenance=catalog['source_provenance'], operator_case_catalog=catalog,
                    unresolved=list(catalog['limitations']))
    return Workload(nodes, [], manifest, copy.deepcopy(workload.hardware))


def execution_context(torch):
    """Capture this alongside recipe provenance in the original runtime."""
    matmul = torch.backends.cuda.matmul
    return dict(torch_git_version=torch.version.git_version,
        build_config_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        allow_fp16_reduced_precision_reduction=matmul.allow_fp16_reduced_precision_reduction,
        allow_bf16_reduced_precision_reduction=matmul.allow_bf16_reduced_precision_reduction,
        allow_fp16_accumulation=getattr(matmul, 'allow_fp16_accumulation', None),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
        environment={k: os.environ.get(k) for k in
            ('CUBLAS_WORKSPACE_CONFIG', 'CUBLASLT_WORKSPACE_SIZE', 'NVIDIA_TF32_OVERRIDE')})


def _runtime_check(torch, provenance):
    required = dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    operator_execution_context=execution_context(torch))
    for field, actual in required.items():
        if field not in provenance or provenance[field] != actual:
            raise ValueError('captured runtime identity mismatch or missing field: '+field)
    if torch.is_autocast_enabled('cuda'):
        raise ValueError('native dispatch replay requires autocast disabled; captured dtypes are already dispatched')
    return required


def _build_region(torch, case, device):
    """Construct native overload on initialized storage (also CPU-testable)."""
    mutation = _validate_case(case['operator'], case['arguments'], case['outputs'])
    name, overload, names, _ = _OPERATORS[case['operator']]
    function = getattr(getattr(torch.ops.aten, name), overload)
    schema = function._schema
    expected_names = list(names) + (['beta', 'alpha'] if 'addmm' in name else [])
    if ([arg.name for arg in schema.arguments] != expected_names or len(schema.returns) != 1
            or bool(schema.is_mutable) != mutation):
        raise ValueError('installed native ATen schema differs from supported contract')
    bases, tensors = [], []
    for index, spec in enumerate(case['arguments']['args']):
        count = _tensor(spec)
        dtype = getattr(torch, spec['dtype'].split('.')[1])
        # Dense nonzero deterministic input distribution. Initialization/reset
        # takes place outside measured regions; allocation is not uninitialized.
        base = torch.full((count,), (index+1)/16., dtype=dtype, device=device)
        tensor = base.as_strided(spec['shape'], spec['stride'], spec['storage_offset'])
        bases.append(base)
        tensors.append(tensor)
    kwargs = copy.deepcopy(case['arguments']['kwargs'])
    def reset():
        if mutation:
            bases[0].fill_(1/16.)
    def run():
        return function(*tensors, **kwargs)
    reset()
    result = run()
    expected = case['outputs']
    actual = dict(shape=list(result.shape), stride=list(result.stride()),
                  storage_offset=result.storage_offset(), dtype=str(result.dtype), layout=str(result.layout))
    if actual != {k: expected[k] for k in actual}:
        raise ValueError('native output signature differs from captured output')
    result_storage = result.untyped_storage()._cdata
    actual_aliases = [i for i, tensor in enumerate(tensors)
                      if tensor.untyped_storage()._cdata == result_storage]
    if actual_aliases != ([0] if mutation else []):
        raise ValueError('native output alias differs from captured contract')
    reset()
    return dict(run=run, before_sample=reset if mutation else None,
                supports_cuda_graph=False, stream_semantics='current_stream',
                schema=str(schema))


def region_provider(workload, config):
    """Trusted provider for native_runner execute mode, one GPU only."""
    import torch
    if config:
        raise ValueError('operator region provider accepts no configuration overrides')
    if workload.manifest.get('scope') != 'isolated native ATen calibration cases':
        raise ValueError('use operator_case_workload to select exact isolated cases')
    if workload.manifest.get('ranks') != [0] or workload.manifest.get('world_size') != 1:
        raise ValueError('native operator cases require one GPU; communication uses a separate provider')
    provenance = workload.manifest.get('provenance', {})
    runtime = _runtime_check(torch, provenance)
    if not torch.cuda.is_available():
        raise ValueError('native operator replay requires CUDA')
    device = torch.device('cuda', torch.cuda.current_device())
    regions, active = [], {}
    for node in workload.nodes:
        case = node.attrs['operator_case']
        signature = {k: case[k] for k in ('operator', 'arguments', 'outputs', 'source_provenance_sha256', 'input_policy')}
        if case['case_id'] != 'aten-case:'+_digest(signature) or case['source_provenance_sha256'] != _digest(provenance):
            raise ValueError('case/source provenance digest mismatch')
        if case['input_policy'] != _POLICY:
            raise ValueError('unsupported synthetic input policy')
        if not all(t['device'].startswith('cuda') for t in case['arguments']['args']):
            raise ValueError('CPU capture cannot be relabeled as CUDA native operator capture')
        binding = node.attrs['native_measurement']
        expected_implementation = dict(kind='native_aten_overload', operator=case['operator'],
            input_policy=_POLICY, source_provenance_sha256=_digest(provenance))
        expected_context = dict(hardware_id=workload.hardware['id'], cuda_graph=False,
            case_id=case['case_id'], signature_sha256=_digest(signature),
            device_mapping='captured_cuda_device_to_current_cuda_device')
        if (binding.get('kind') != 'native_aten_case' or binding.get('implementation') != expected_implementation
                or binding.get('context') != expected_context
                or node.attrs['cost_key'] != native_cost_key(expected_implementation, expected_context)):
            raise ValueError('case/native measurement binding mismatch')
        # Only the current case owns storage. A full training capture can have
        # thousands of cases; retaining every operand would be a false OOM.
        def before_sample(case=case):
            if active.get('case_id') != case['case_id']:
                active.clear()
                active.update(case_id=case['case_id'], region=_build_region(torch, case, device))
            reset = active['region'].get('before_sample')
            if reset is not None:
                reset()
        def run(case_id=case['case_id']):
            if active.get('case_id') != case_id:
                raise ValueError('operator case must be prepared before timing')
            return active['region']['run']()
        regions.append(dict(run=run, before_sample=before_sample, supports_cuda_graph=False,
            stream_semantics='current_stream', cost_key=node.attrs['cost_key'],
            implementation=binding['implementation'], context=binding['context']))
    return dict(regions=regions, provenance=dict(provider_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=runtime, source_provenance_sha256=_digest(provenance), input_policy=_POLICY,
        fidelity='captured native overload and metadata; synthetic values, isolated execution',
        case_ids=[n.attrs['operator_case']['case_id'] for n in workload.nodes]))
