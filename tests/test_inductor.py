import hashlib
import json
from pathlib import Path
import unittest

from titan_sim.inductor import import_inductor_wrapper
from titan_sim.compiled import lower_compiled_manifest
from titan_sim.runtime import simulate

FIXTURES = Path(__file__).resolve().parents[1] / 'examples/inductor'
HW = {'peak_flops': {'float32': 1e6}, 'memory_bandwidth_bytes_s': 1e6, 'memory_capacity_bytes': 1000000}


def cpu_import(source=None, costs=None):
    source = source or (FIXTURES / 'cpu_output_code.py').read_text()
    cost_document = json.loads((FIXTURES / 'cpu_kernel_costs.json').read_text())
    cost_document['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    if costs is not None:
        cost_document['kernels'] = costs
    return import_inductor_wrapper(source,
                                   kernel_costs=cost_document,
                                   provenance={'source': 'actual_torch_compile_debug_output', 'torch_version': '2.14.1',
                                               'device': 'cpu', 'fixture': 'matmul_relu_add'})


def cuda_import(source=None, costs=None):
    costs = costs or {
        'k0': {'target': 'triton_copy.run', 'dtype': 'float32', 'flops': 0,
               'bytes_read': 64, 'bytes_write': 64, 'reads': [0], 'writes': [1], 'provenance': 'declared fixture'},
        'k1': {'target': 'triton_scale.run', 'dtype': 'float32', 'flops': 8,
               'bytes_read': 32, 'bytes_write': 32, 'reads': [0], 'writes': [0], 'provenance': 'declared fixture'}}
    source = source or (FIXTURES / 'cuda_syntax_fixture.py').read_text()
    return import_inductor_wrapper(source,
                                   kernel_costs={'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'kernels': costs}, provenance='handwritten CUDA syntax fixture, not observed compilation')


def allocations(imported):
    device = imported['report']['device']
    return [{'event_id': 'external_' + str(i), 'kind': 'allocate', 'storage_id': sid,
             'size_bytes': size, 'metadata': {'device': device}}
            for i, (sid, size) in enumerate(imported['manifest']['external_storages'].items())]


class InductorTests(unittest.TestCase):
    def test_actual_cpu_wrapper_reuse_derives_one_allocation(self):
        result = cpu_import()
        self.assertEqual(result['report']['allocation_count'], 1)
        self.assertEqual(result['report']['physical_allocation_bytes'], 24)
        self.assertEqual(result['report']['device'], 'cpu')
        self.assertEqual(result['manifest']['plan_source'], 'compiler_artifact')
        self.assertEqual(result['manifest']['kernels'][1]['name'], 'cpp_fused_add_relu_0')
        self.assertEqual(result['manifest']['kernels'][1]['bytes_read'], 24)
        self.assertEqual(result['capture']['metadata']['retained_storage_ids'], ['inductor:slot:s0'])
        events = result['capture']['events']
        self.assertEqual(len([event for event in events if event['kind'] == 'allocate']), 1)
        self.assertFalse(any(event['kind'] == 'release' for event in events))
        simulation = simulate(allocations(result) + events, HW)
        self.assertEqual(simulation['summary']['peak_allocated_bytes'], 0)
        self.assertTrue(all(row['source'] == 'host_unmodeled' for row in simulation['timeline']))

    def test_cuda_fixture_views_and_retained_backing(self):
        result = cuda_import()
        self.assertEqual(result['report']['physical_allocation_bytes'], 64)
        view = next(v for v in result['report']['logical_views'] if v['variable'] == 'buf1')
        self.assertEqual(view['offset_bytes'], 16)
        self.assertEqual(view['size_bytes'], 32)
        simulation = simulate(allocations(result) + result['capture']['events'], HW)
        self.assertEqual(simulation['summary']['peak_allocated_bytes'], 128)
        self.assertEqual(simulation['summary']['per_rank']['0']['final_allocated_bytes'], 128)
        self.assertEqual(len([e for e in result['capture']['events'] if e['kind'] == 'allocate']), 1)

    def test_missing_cost_and_wrong_target_rejected(self):
        costs = json.loads((FIXTURES / 'cpu_kernel_costs.json').read_text())['kernels']
        del costs['k1']
        with self.assertRaisesRegex(ValueError, 'exact kernel'):
            cpu_import(costs=costs)
        costs = json.loads((FIXTURES / 'cpu_kernel_costs.json').read_text())['kernels']
        costs['k1']['target'] = 'different_kernel'
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            cpu_import(costs=costs)

    def test_unknown_access_and_dynamic_geometry_rejected(self):
        costs = json.loads((FIXTURES / 'cpu_kernel_costs.json').read_text())['kernels']
        costs['k0']['reads'] = [0]
        with self.assertRaisesRegex(ValueError, 'explicit read/write'):
            cpu_import(costs=costs)
        source = (FIXTURES / 'cpu_output_code.py').read_text().replace('empty_strided_cpu((2, 3)', 'empty_strided_cpu((dynamic_size, 3)')
        with self.assertRaisesRegex(ValueError, 'dynamic'):
            cpu_import(source=source)

    def test_use_after_delete_oversized_alias_and_unknown_statement(self):
        source = (FIXTURES / 'cuda_syntax_fixture.py').read_text()
        with self.assertRaisesRegex(ValueError, 'deleted'):
            cuda_import(source.replace('triton_scale.run(buf1', 'triton_scale.run(buf0'))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            cuda_import(source.replace('(2, 4), (4, 1), 4', '(2, 4), (4, 1), 12'))
        with self.assertRaisesRegex(ValueError, 'unsupported wrapper statement'):
            cuda_import(source.replace('        del buf0', '        if True:\n            del buf0'))

    def test_guard_mismatch_rejected(self):
        source = (FIXTURES / 'cpu_output_code.py').read_text().replace("device='cpu', dtype=torch.float32)", "device='cpu', dtype=torch.float16)")
        result = cpu_import(source=source)
        self.assertEqual(sorted(result['manifest']['external_storages'].values()), [16, 24])
        source = (FIXTURES / 'cpu_output_code.py').read_text().replace('rand_strided((2, 4)', 'rand_strided((3, 4)')
        with self.assertRaisesRegex(ValueError, 'guard'):
            cpu_import(source=source)

    def test_artifact_mismatch_and_no_source_execution(self):
        result = cpu_import()
        result['artifact']['source_text'] += '\n# changed\n'
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            lower_compiled_manifest(result['manifest'], artifact=result['artifact'])
        source = "raise RuntimeError('SOURCE MUST NEVER EXECUTE')\n" + (FIXTURES / 'cpu_output_code.py').read_text()
        self.assertEqual(cpu_import(source=source)['report']['allocation_count'], 1)

    def test_exact_source_binding_and_output_release(self):
        source = (FIXTURES / 'cpu_output_code.py').read_text()
        costs = json.loads((FIXTURES / 'cpu_kernel_costs.json').read_text())
        with self.assertRaisesRegex(ValueError, 'source_sha256'):
            import_inductor_wrapper(source + '\n', kernel_costs=costs, provenance='test')
        source = (FIXTURES / 'cuda_syntax_fixture.py').read_text()
        source = source.replace('        del arg0_1\n', '')
        source = source.replace('        return (buf1,)', '        del buf1\n        return (arg0_1,)')
        imported = cuda_import(source)
        self.assertEqual(imported['manifest']['retained_buffers'], [])
        self.assertEqual(imported['report']['output_storage_ids'], ['inductor:input:arg0_1'])
        events = allocations(imported) + imported['capture']['events']
        self.assertEqual(simulate(events, HW)['summary']['per_rank']['0']['final_allocated_bytes'], 64)

    def test_explicit_inputs_bind_signature(self):
        from titan_sim.compiled import artifact_digest
        source = (FIXTURES / 'cpu_output_code.py').read_text().split('def get_args():')[0]
        inputs = {'arg0_1': {'shape': [2, 4], 'stride': [4, 1], 'dtype': 'float32', 'device': 'cpu'},
                  'arg1_1': {'shape': [4, 3], 'stride': [3, 1], 'dtype': 'float32', 'device': 'cpu'}}
        costs = json.loads((FIXTURES / 'cpu_kernel_costs.json').read_text())
        costs['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'input_specs_sha256'):
            import_inductor_wrapper(source, kernel_costs=costs, input_specs=inputs, provenance='test')
        costs['input_specs_sha256'] = artifact_digest(inputs)
        result = import_inductor_wrapper(source, kernel_costs=costs, input_specs=inputs, provenance='test')
        self.assertEqual(result['report']['physical_allocation_bytes'], 24)

    def test_explicit_inputs_cannot_override_compiler_dtype(self):
        from titan_sim.compiled import artifact_digest
        from titan_sim.inductor import _default_inputs
        import ast
        source = (FIXTURES / 'cpu_output_code.py').read_text()
        inputs = _default_inputs(ast.parse(source))
        for spec in inputs.values():
            spec.pop('size_bytes')
            spec['dtype'] = 'float16'
        costs = json.loads((FIXTURES / 'cpu_kernel_costs.json').read_text())
        costs['input_specs_sha256'] = artifact_digest(inputs)
        with self.assertRaisesRegex(ValueError, 'conflict with compiler'):
            import_inductor_wrapper(source, kernel_costs=costs, input_specs=inputs, provenance='test')

    def test_actual_nv2609_cuda_wrapper_alignment_path(self):
        source_dir = FIXTURES
        source = (source_dir / 'nv2609_cuda_output_code.py').read_text()
        costs = json.loads((source_dir / 'nv2609_cuda_kernel_costs.json').read_text())
        with self.assertRaisesRegex(ValueError, 'alignment and no-copy evidence'):
            import_inductor_wrapper(source, kernel_costs=costs, provenance='actual NV26.09 source; no alignment supplied')
        costs['input_alignment'] = {name: {
            'alignment_bytes': 16, 'pointer_mod_alignment': 0,
            'copy_if_misaligned_returns_same_storage': True,
            'provenance': 'declared aligned scenario for CPU parser regression; not an observed GPU pointer'}
            for name in ('arg0_1', 'arg1_1')}
        imported = import_inductor_wrapper(source, kernel_costs=costs,
                                            provenance='actual NV26.09 CUDA compiler source, conditional aligned input scenario')
        self.assertEqual(imported['report']['physical_allocation_bytes'], 24)
        self.assertEqual(len(imported['report']['conditional_alignment_paths']), 2)
        self.assertEqual(imported['manifest']['kernels'][1]['name'], 'triton_poi_fused_add_relu_0.run')
        simulation = simulate(allocations(imported) + imported['capture']['events'], HW)
        self.assertEqual(simulation['summary']['peak_allocated_bytes'], 104)
        self.assertEqual(imported['artifact']['source_sha256'], costs['source_sha256'])
        for change in ({'pointer_mod_alignment': 4}, {'copy_if_misaligned_returns_same_storage': False}):
            original = dict(costs['input_alignment']['arg0_1'])
            costs['input_alignment']['arg0_1'].update(change)
            with self.assertRaisesRegex(ValueError, 'unknown/misaligned'):
                import_inductor_wrapper(source, kernel_costs=costs, provenance='test')
            costs['input_alignment']['arg0_1'] = original

    def test_strided_storage_uses_span_not_numel(self):
        source = (FIXTURES / 'cuda_syntax_fixture.py').read_text().replace('empty_strided_cuda((4, 4), (4, 1)', 'empty_strided_cuda((4, 4), (8, 1)')
        result = cuda_import(source)
        self.assertEqual(result['report']['physical_allocation_bytes'], 28 * 4)
