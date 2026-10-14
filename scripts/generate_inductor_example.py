#!/usr/bin/env python3
"""Compile/execute a tiny real CPU or CUDA function and import its storage plan.

Storage is compiler-derived. Arithmetic/traffic is explicitly declared for this
known benchmark, not inferred performance. Source/provenance are saved even if a
new compiler's syntax falls outside the strict importer subset.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _target(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _target(node.value) + '.' + node.attr
    raise ValueError('unsupported generated call target')


def _costs(source, device):
    """Bind the known benchmark's explicit formulas to the generated call sites."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'call']
    functions += [n for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'Runner'
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'call']
    if len(functions) != 1:
        raise ValueError('unknown wrapper entry-point layout')
    calls = [n.value for n in ast.walk(functions[0]) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and (_target(n.value.func) == 'extern_kernels.mm' or 'fused_add_relu' in _target(n.value.func))]
    calls.sort(key=lambda n: n.lineno)
    if len(calls) != 2 or _target(calls[0].func) != 'extern_kernels.mm' or 'fused_add_relu' not in _target(calls[1].func):
        raise ValueError('expected extern GEMM then fused add/ReLU; compiler changed benchmark lowering')
    if len(calls[0].args) != 2 or [kw.arg for kw in calls[0].keywords] != ['out']:
        raise ValueError('unexpected GEMM calling convention')
    target = _target(calls[1].func)
    reads, writes = [0], [0]
    if device == 'cpu':
        if len(calls[1].args) != 1 or calls[1].keywords:
            raise ValueError('unexpected CPU fused-kernel calling convention')
    else:
        if not target.endswith('.run'):
            raise ValueError('expected a generated Triton fused pointwise kernel')
        symbol = target.removesuffix('.run')
        definitions = [n.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == symbol]
        if len(definitions) != 1 or not isinstance(definitions[0], ast.Call) or _target(definitions[0].func) != 'async_compile.triton':
            raise ValueError('unknown Triton source declaration')
        declaration = definitions[0]
        if len(declaration.args) < 2 or not isinstance(declaration.args[1], ast.Constant) or not isinstance(declaration.args[1].value, str):
            raise ValueError('Triton kernel source must be a literal compiler artifact')
        triton_tree = ast.parse(declaration.args[1].value)
        kernels = [n for n in triton_tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol]
        if len(kernels) != 1:
            raise ValueError('cannot identify generated fused Triton function')
        arguments = [arg.arg for arg in kernels[0].args.args]
        pointers = {name for name in arguments if 'ptr' in name}
        accesses = {'tl.load': set(), 'tl.store': set()}
        for node in ast.walk(kernels[0]):
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                operation = _target(node.func)
                if operation in accesses:
                    accesses[operation].update(n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name) and n.id in pointers)
        if not accesses['tl.load'] or not accesses['tl.store'] or (accesses['tl.load'] | accesses['tl.store']) != pointers:
            raise ValueError('unknown pointer-access pattern in generated fused kernel')
        reads = sorted(arguments.index(name) for name in accesses['tl.load'])
        writes = sorted(arguments.index(name) for name in accesses['tl.store'])
    return {'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'kernels': {
        'k0': {'target': 'extern_kernels.mm', 'dtype': 'float32', 'flops': 48,
               'bytes_read': 80, 'bytes_write': 24, 'reads': [0, 1], 'writes': ['out'],
               'provenance': 'Benchmark-declared 2x4 times 4x3 GEMM formula and one logical input/output pass; not measured cache traffic'},
        'k1': {'target': target, 'dtype': 'float32', 'flops': 12,
               'bytes_read': 24, 'bytes_write': 24, 'reads': reads, 'writes': writes,
               'provenance': 'Benchmark-declared six ReLU comparisons plus six adds; one fused read/write pass. CUDA pointer access verified from generated tl.load/tl.store AST.'}}}


def _input_specs(source, device):
    """Fallback metadata for wrappers omitting get_args; bound to known benchmark."""
    from titan_sim.inductor import _default_inputs
    tree = ast.parse(source)
    if _default_inputs(tree):
        return None
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == 'assert_size_stride' and len(node.args) == 3 and isinstance(node.args[0], ast.Name):
            result[node.args[0].id] = {'shape': list(ast.literal_eval(node.args[1])),
                                     'stride': list(ast.literal_eval(node.args[2])), 'dtype': 'float32', 'device': device}
        elif node.func.id == 'assert_size_stride_grouped' and len(node.args) in (3, 4):
            for name, shape, stride in zip(node.args[0].elts, ast.literal_eval(node.args[1]), ast.literal_eval(node.args[2]), strict=True):
                result[name.id] = {'shape': list(shape), 'stride': list(stride), 'dtype': 'float32', 'device': device}
    if sorted(spec['shape'] for spec in result.values()) != [[2, 4], [4, 3]]:
        raise ValueError('wrapper guards do not match known benchmark inputs')
    return result


def main():
    import torch
    from titan_sim.compiled import artifact_digest
    from titan_sim.inductor import import_inductor_wrapper
    from titan_sim.runtime import simulate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'examples/inductor')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda requires an actual CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    def function(x, weight):
        return torch.relu(x @ weight) + 1
    runtime_inputs = (torch.ones(2, 4, device=args.device, dtype=torch.float32),
                      torch.ones(4, 3, device=args.device, dtype=torch.float32))
    with tempfile.TemporaryDirectory(prefix='titan-inductor-debug-') as temporary:
        with torch._inductor.config.patch({'trace.enabled': True, 'trace.debug_dir': temporary,
                                          'force_disable_caches': True}):
            compiled = torch.compile(function, fullgraph=True)
            result = compiled(*runtime_inputs)
        if not torch.equal(result, torch.full((2, 3), 5.0, device=args.device)):
            raise RuntimeError('compiled numerical result differs from expected result')
        files = list(Path(temporary).rglob('output_code.py'))
        if len(files) != 1:
            raise RuntimeError('expected exactly one generated output_code.py; inspect compiler trace')
        source = files[0].read_text()
    provenance = {'source': 'actual_torch_compile_debug_output', 'torch_version': torch.__version__,
                  'device': args.device, 'fullgraph': True, 'numeric_execution_verified': True,
                  'function': 'relu(x @ weight) + 1', 'input_shapes': [[2, 4], [4, 3]],
                  'output_shape': [2, 3], 'output_all_elements': 5.0, 'tf32': False,
                  'cuda_version': torch.version.cuda,
                  'measured_device': torch.cuda.get_device_name() if args.device == 'cuda' else None,
                  'costs': 'explicit mathematical FLOPs and logical boundary bytes; no measured cache traffic/timing'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (args.output_dir / (args.device + '_' + name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    (args.output_dir / (args.device + '_output_code.py')).write_text(source)
    save('provenance', provenance)
    try:
        cost_document = _costs(source, args.device)
        inputs = _input_specs(source, args.device)
        if 'copy_if_misaligned(' in source:
            from titan_sim.inductor import _default_inputs
            from torch._inductor.utils import ALIGNMENT
            declared_inputs = inputs or _default_inputs(ast.parse(source))
            alignment = {}
            for name, spec in declared_inputs.items():
                matches = [tensor for tensor in runtime_inputs
                           if list(tensor.shape) == spec['shape'] and list(tensor.stride()) == spec['stride']
                           and str(tensor.dtype).removeprefix('torch.') == spec['dtype']]
                if len(matches) != 1:
                    raise ValueError('cannot bind runtime alignment observation to compiler input')
                tensor = matches[0]
                checked = torch._C._dynamo.guards.copy_if_misaligned(tensor)
                alignment[name] = {'alignment_bytes': ALIGNMENT, 'pointer_mod_alignment': tensor.data_ptr() % ALIGNMENT,
                                   'copy_if_misaligned_returns_same_storage': (checked.data_ptr() == tensor.data_ptr() and
                                       checked.untyped_storage()._cdata == tensor.untyped_storage()._cdata),
                                   'provenance': {'source': 'actual_runtime_input_observation',
                                                  'torch_version': torch.__version__, 'device': args.device,
                                                  'helper': 'torch._C._dynamo.guards.copy_if_misaligned',
                                                  'alignment_constant': 'torch._inductor.utils.ALIGNMENT',
                                                  'scope': 'same retained input tensors used in verified compiled invocation; pointer remainder only'}}
            cost_document['input_alignment'] = alignment
            provenance['input_alignment'] = alignment
            provenance['allocation_scope'] = 'conditional on recorded aligned no-copy input path'
            save('provenance', provenance)
        if inputs is not None:
            cost_document['input_specs_sha256'] = artifact_digest(inputs)
        save('kernel_costs', cost_document)
        imported = import_inductor_wrapper(source, kernel_costs=cost_document, provenance=provenance, input_specs=inputs)
        capture = imported['capture']
        initial = [{'event_id': 'input_allocate_' + str(i), 'kind': 'allocate', 'storage_id': sid,
                    'size_bytes': size, 'category': 'input', 'metadata': {'device': args.device}}
                   for i, (sid, size) in enumerate(imported['manifest']['external_storages'].items())]
        capture['events'] = initial + capture['events']
        hardware = {'name': 'synthetic-rates-for-wrapper-semantics', 'peak_flops': {'float32': 1e6},
                    'memory_bandwidth_bytes_s': 1e6, 'memory_capacity_bytes': 1000000}
        simulation = simulate(capture['events'], hardware)
        save('import', imported)
        save('capture', capture)
        save('simulation', simulation)
        if args.device == 'cuda':
            # Define expectations from compiler bytes and a fixed generic policy
            # before collecting measurements. Never fit alignment to the result.
            manifest = imported['manifest']
            buffers = {buffer['id']: buffer for buffer in manifest['buffers']}
            retained_slots = {buffers[bid]['slot'] for bid in manifest['retained_buffers']}
            slots = {slot['id']: slot for slot in manifest['slots']}
            rounded = lambda size: ((size + 511) // 512) * 512
            expectation = {
                'scope': 'one tiny compiled forward after warmup, inputs and prior output retained',
                'compiler_new_allocation_payload_bytes': sum(slot['size_bytes'] for slot in slots.values()),
                'compiler_fresh_output_payload_bytes': sum(slots[sid]['size_bytes'] for sid in retained_slots),
                'generic_alignment_bytes': 512,
                'generic_fresh_output_allocated_bytes': sum(rounded(slots[sid]['size_bytes']) for sid in retained_slots),
                'generic_all_new_allocations_bytes': sum(rounded(slot['size_bytes']) for slot in slots.values()),
                'assumptions': ['Fixed 512-byte allocator rounding scenario, declared before measurement.',
                                'Prior output remains alive; fresh output must not alias it.',
                                'Cache reserve and hidden/library workspaces are not predicted.',
                                'This is a single tiny forward, not full-model training validation.']}
            save('memory_expectation', expectation)
            torch.cuda.synchronize()
            baseline = {'allocated_bytes': torch.cuda.memory_allocated(),
                        'reserved_bytes': torch.cuda.memory_reserved()}
            torch.cuda.reset_peak_memory_stats()
            next_result = compiled(*runtime_inputs)
            torch.cuda.synchronize()
            observed = {'baseline': baseline,
                        'current_allocated_bytes': torch.cuda.memory_allocated(),
                        'current_reserved_bytes': torch.cuda.memory_reserved(),
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'prior_and_new_output_same_storage': result.data_ptr() == next_result.data_ptr()}
            observed['allocated_delta_bytes'] = observed['current_allocated_bytes'] - baseline['allocated_bytes']
            observed['peak_allocated_delta_bytes'] = observed['peak_allocated_bytes'] - baseline['allocated_bytes']
            observed['reserved_delta_bytes'] = observed['current_reserved_bytes'] - baseline['reserved_bytes']
            observed['peak_reserved_delta_bytes'] = observed['peak_reserved_bytes'] - baseline['reserved_bytes']
            comparison = {
                'fresh_output_matches_generic_alignment_scenario':
                    not observed['prior_and_new_output_same_storage'] and
                    observed['allocated_delta_bytes'] == expectation['generic_fresh_output_allocated_bytes'],
                'peak_matches_all_new_allocations_scenario':
                    observed['peak_allocated_delta_bytes'] == expectation['generic_all_new_allocations_bytes'],
                'independent_full_training_memory_validation': False}
            save('memory_crosscheck', {'expectation': expectation, 'observed': observed, 'comparison': comparison})
        status = {'status': 'complete', 'actual_compile_and_execution': True, 'device': args.device,
                  'torch_version': torch.__version__, 'derived_allocations': imported['report']['allocation_count'],
                  'derived_physical_bytes': imported['report']['physical_allocation_bytes'],
                  'output_storage_ids': imported['report']['output_storage_ids'],
                  'timing': 'synthetic analytical rates; not GPU timing validation'}
        save('import_status', status)
        print(json.dumps(status, indent=2))
    except Exception as error:
        save('import_status', {'status': 'unsupported_wrapper', 'actual_compile_and_execution': True,
                              'error_type': type(error).__name__, 'error': str(error), 'provenance': provenance})
        raise


if __name__ == '__main__':
    main()
