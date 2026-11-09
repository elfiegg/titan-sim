import copy
import unittest
from unittest.mock import patch

from titan_sim.core import Node
from titan_sim.operator_cases import prepare_operator_cases, operator_case_workload, _build_region, _runtime_check, execution_context, region_provider
from titan_sim.workload import Workload


def tensor(shape, stride, sid, offset=0):
    return dict(shape=shape, stride=stride, storage_offset=offset, dtype='torch.float32',
                layout='torch.strided', device='cuda:0', storage_id=sid)


def fixture(operator='aten.mm.default'):
    args = [tensor([2, 3], [3, 1], 'a'), tensor([3, 4], [4, 1], 'b')]
    output = tensor([2, 4], [4, 1], 'c')
    if 'addmm' in operator:
        args.insert(0, tensor([2, 4], [4, 1], 'bias'))
        if 'addmm_' in operator:
            output = copy.deepcopy(args[0])
    node = Node(10, operator, 2, attrs=dict(operator=operator, status='returned',
        arguments=dict(args=args, kwargs={'alpha': 2., 'beta': .5} if 'addmm' in operator else {}),
        outputs=output, cost_key='torch-dispatch:source'))
    return Workload([node], [], dict(ranks=[2], world_size=4, provenance={'source': 'pinned'}), {'id': 'gpu'})


class CaseExtractionTests(unittest.TestCase):
    def test_extraction_is_pure_and_renames_only_storage_occurrences(self):
        workload = fixture()
        second = copy.deepcopy(workload.nodes[0])
        second.id = 11
        for item in second.attrs['arguments']['args'] + [second.attrs['outputs']]:
            item['storage_id'] += '-new'
        workload.nodes.append(second)
        original = copy.deepcopy(workload)
        with patch.dict('sys.modules', {'torch': None}):
            catalog = prepare_operator_cases(workload)
        self.assertEqual(workload, original)
        self.assertEqual(len(catalog['cases']), 1)
        self.assertEqual(len(catalog['cases'][0]['occurrences']), 2)
        self.assertEqual(catalog['skipped'], [])

    def test_alias_not_erased_and_noncontiguous_offsets_preserved(self):
        workload = fixture()
        a = workload.nodes[0].attrs['arguments']['args'][0]
        a.update(stride=[1, 2], storage_offset=7)
        case = prepare_operator_cases(workload)['cases'][0]
        self.assertEqual(case['arguments']['args'][0]['stride'], [1, 2])
        self.assertEqual(case['arguments']['args'][0]['storage_offset'], 7)
        workload.nodes[0].attrs['arguments']['args'][1]['storage_id'] = a['storage_id']
        catalog = prepare_operator_cases(workload)
        self.assertFalse(catalog['cases'])
        self.assertIn('aliased input', catalog['skipped'][0]['reason'])

    def test_unknown_data_dependent_custom_empty_and_failed_ops_skip(self):
        for operator in ('aten.nonzero.default', 'aten.embedding.default', 'aten.empty.memory_format', 'custom.gemm.default'):
            workload = fixture(operator)
            self.assertFalse(prepare_operator_cases(workload)['cases'])
        workload = fixture()
        workload.nodes[0].attrs['status'] = 'failed'
        self.assertIn('did not return', prepare_operator_cases(workload)['skipped'][0]['reason'])

    def test_overlap_distributed_opaque_nonfinite_and_output_mismatch_skip(self):
        changes = [lambda a: a['arguments']['args'][0].update(stride=[0, 1]),
                   lambda a: a['arguments']['args'][0].update(distributed={'mesh': [0, 1]}),
                   lambda a: a['arguments']['args'][0].update(dtype='torch.complex64'),
                   lambda a: a['outputs'].update(shape=[9, 4]),
                   lambda a: a['arguments']['kwargs'].update(alpha=float('nan'))]
        for change in changes:
            workload = fixture('aten.addmm.default')
            change(workload.nodes[0].attrs)
            self.assertFalse(prepare_operator_cases(workload)['cases'])

    def test_shapes_strides_scalars_source_and_hardware_bind_exact_keys(self):
        source = fixture('aten.addmm.default')
        original = operator_case_workload(source).nodes[0].attrs['cost_key']
        changes = [lambda w: w.hardware.update(id='other'),
                   lambda w: w.manifest['provenance'].update(source='different'),
                   lambda w: w.nodes[0].attrs['arguments']['kwargs'].update(alpha=3),
                   lambda w: w.nodes[0].attrs['arguments']['args'][1].update(storage_offset=2)]
        for change in changes:
            workload = copy.deepcopy(source)
            change(workload)
            self.assertNotEqual(operator_case_workload(workload).nodes[0].attrs['cost_key'], original)

    def test_isolated_case_workload_retains_source_mapping_and_skips(self):
        workload = fixture()
        case = prepare_operator_cases(workload)['cases'][0]
        isolated = operator_case_workload(workload, [case['case_id']])
        self.assertEqual(isolated.manifest['ranks'], [0])
        self.assertEqual(isolated.manifest['world_size'], 1)
        self.assertEqual(isolated.nodes[0].attrs['operator_case']['occurrences'][0]['rank'], 2)
        self.assertTrue(isolated.manifest['unresolved'])
        self.assertEqual(workload.manifest['ranks'], [2])
        with self.assertRaisesRegex(ValueError, 'absent'):
            operator_case_workload(workload, ['missing'])
        with self.assertRaisesRegex(ValueError, 'no eligible'):
            operator_case_workload(workload, [])

    def test_mutation_requires_exact_output_alias(self):
        workload = fixture('aten.addmm_.default')
        self.assertTrue(prepare_operator_cases(workload)['cases'][0]['mutates_inputs'])
        workload.nodes[0].attrs['outputs']['storage_id'] = 'new'
        self.assertIn('exactly alias', prepare_operator_cases(workload)['skipped'][0]['reason'])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional installed torch required for native CPU replay tests')
class NativeReplayTests(unittest.TestCase):
    def case(self, operator):
        workload = fixture(operator)
        for item in workload.nodes[0].attrs['arguments']['args'] + [workload.nodes[0].attrs['outputs']]:
            item['device'] = 'cpu'
        return prepare_operator_cases(workload)['cases'][0]

    def test_real_native_mm_on_transposed_offset_storage(self):
        case = self.case('aten.mm.default')
        case['arguments']['args'][0].update(stride=[1, 2], storage_offset=5)
        region = _build_region(torch, case, 'cpu')
        actual = region['run']()
        expected = torch.full((2, 4), 3/128.)
        torch.testing.assert_close(actual, expected)

    def test_real_native_bmm(self):
        case = self.case('aten.mm.default')
        case['operator'] = 'aten.bmm.default'
        case['arguments']['args'] = [tensor([2, 2, 3], [6, 3, 1], 's0'), tensor([2, 3, 4], [12, 4, 1], 's1')]
        case['outputs'] = tensor([2, 2, 4], [8, 4, 1], 's2')
        region = _build_region(torch, case, 'cpu')
        torch.testing.assert_close(region['run'](), torch.full((2, 2, 4), 3/128.))

    def test_mutation_reset_prevents_drift_and_calls_original_overload(self):
        case = self.case('aten.addmm_.default')
        region = _build_region(torch, case, 'cpu')
        first = region['run']().clone()
        second = region['run']().clone()
        self.assertFalse(torch.equal(first, second))
        region['before_sample']()
        torch.testing.assert_close(region['run'](), first)
        self.assertIn('aten::addmm_', region['schema'])

    def test_actual_torch_capture_extracts_native_metadata(self):
        from titan_sim.torch_capture import TorchCapture
        recorder = TorchCapture(device='cpu', provenance={'source': 'unit-test'})
        a, b = torch.ones(2, 3), torch.ones(3, 4)
        with recorder:
            torch.ops.aten.mm.default(a, b)
        workload = recorder.finish(hardware={'id': 'unit-test'})
        catalog = prepare_operator_cases(workload)
        self.assertEqual(len(catalog['cases']), 1)
        region = _build_region(torch, catalog['cases'][0], 'cpu')
        self.assertEqual(tuple(region['run']().shape), (2, 4))

    def runtime_workload(self):
        workload = fixture()
        workload.manifest['provenance'].update(torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda, allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            operator_execution_context=execution_context(torch))
        return workload

    def test_runtime_context_is_checked_without_changing_native_flags(self):
        workload = self.runtime_workload()
        _runtime_check(torch, workload.manifest['provenance'])
        workload.manifest['provenance']['operator_execution_context']['preferred_blas_library'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'operator_execution_context'):
            _runtime_check(torch, workload.manifest['provenance'])

    def test_provider_validates_binding_before_allocating_any_tensor(self):
        workload = operator_case_workload(self.runtime_workload())
        workload.nodes[0].attrs['native_measurement']['context']['case_id'] = 'wrong'
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'current_device', return_value=0), \
             patch('titan_sim.operator_cases._build_region') as builder:
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                region_provider(workload, {})
            builder.assert_not_called()

    def test_provider_materializes_only_current_case_outside_timing(self):
        source = self.runtime_workload()
        other = copy.deepcopy(source.nodes[0])
        other.id = 11
        other.attrs['arguments']['args'][0]['storage_offset'] = 3
        source.nodes.append(other)
        workload = operator_case_workload(source)
        calls, resets = [], []
        def build(torch, case, device):
            calls.append(case['case_id'])
            return dict(run=lambda: case['case_id'], before_sample=lambda: resets.append(case['case_id']))
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'current_device', return_value=0), \
             patch('titan_sim.operator_cases._build_region', side_effect=build):
            bundle = region_provider(workload, {})
            self.assertEqual(calls, [])
            first, second = bundle['regions']
            with self.assertRaisesRegex(ValueError, 'prepared'):
                first['run']()
            first['before_sample']()
            first['before_sample']()
            self.assertEqual(len(calls), 1)
            self.assertEqual(first['run'](), calls[0])
            second['before_sample']()
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(ValueError, 'prepared'):
                first['run']()
            self.assertEqual(len(resets), 3)

    def test_output_layout_and_runtime_mismatch_rejected(self):
        case = self.case('aten.mm.default')
        case['outputs']['stride'] = [1, 2]
        with self.assertRaisesRegex(ValueError, 'output signature'):
            _build_region(torch, case, 'cpu')
        with self.assertRaisesRegex(ValueError, 'runtime identity'):
            _runtime_check(torch, {})


if __name__ == '__main__':
    unittest.main()
