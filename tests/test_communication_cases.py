import copy
import importlib.util
import unittest
from unittest.mock import patch

from titan_sim.communication_cases import (
    communication_case_workload, describe_communication, prepare_communication_cases, _allocate_tensors, region_provider, execution_context,
)
from titan_sim.core import Node
from titan_sim.native_runner import _descriptor
from titan_sim.workload import Workload


def tensor(shape, storage):
    stride, size = [], 1
    for dim in reversed(shape):
        stride.insert(0, size)
        size *= dim
    return dict(shape=shape, stride=stride, dtype='torch.bfloat16', device='cuda:0',
                layout='torch.strided', storage_offset=0, storage_id=storage)


def fixture():
    comm = dict(operator='c10d._allgather_base_.default', operator_schema='native schema fixture',
        operation='all_gather_into_tensor', participants=[2, 5], group_size=2, backend='nccl',
        group_name='source-group', reduce_op=None, async_op=True, timeout=-1,
        runtime_context={'nccl_version': [2, 28, 0]},
        tensors={'input_tensor': tensor([4], 'input'), 'output_tensor': tensor([8], 'output')},
        evidence='native_dispatch_and_resolved_process_group')
    node = Node(0, comm['operator'], 2, attrs=dict(status='returned', communication=comm,
                                                 cost_key='original-source-dispatch'))
    return Workload([node], [], dict(ranks=[2], world_size=8, provenance={'torch_version':'test'}),
                    {'id':'test-hardware','capacity_bytes':10000})


class CommunicationCasesTests(unittest.TestCase):
    def test_same_size_measurement_group_retains_original_membership(self):
        source = fixture()
        workload = communication_case_workload(source, topology={'fixture': 'two-device'})
        self.assertEqual(workload.manifest['ranks'], [0, 1])
        self.assertEqual(workload.manifest['world_size'], 2)
        self.assertEqual(len(workload.nodes), 2)
        for node in workload.nodes:
            binding = _descriptor(node)
            self.assertEqual(binding['context']['case']['participants'], [2, 5])
            self.assertIn('measurement rank i', binding['context']['rank_mapping'])
        self.assertEqual(source.nodes[0].attrs['cost_key'], 'original-source-dispatch')

    def test_dedup_keeps_source_occurrences_but_not_in_cost_signature(self):
        source = fixture()
        repeated = copy.deepcopy(source.nodes[0])
        repeated.id = 2
        source.nodes.append(repeated)
        result = prepare_communication_cases(source)
        self.assertEqual(len(result['cases']), 1)
        self.assertEqual(len(result['cases'][0]['source_nodes']), 2)
        self.assertEqual(result['skipped'], [])
        single = fixture()
        key1 = communication_case_workload(single, topology={'fixture':'two-device'}).nodes[0].attrs['cost_key']
        key2 = communication_case_workload(source, topology={'fixture':'two-device'}).nodes[0].attrs['cost_key']
        self.assertEqual(key1, key2)
        key3 = communication_case_workload(single, topology={'fixture':'other-topology'}).nodes[0].attrs['cost_key']
        self.assertNotEqual(key1, key3)

    def test_unknown_group_size_is_not_replaced_by_available_gpu_count(self):
        source = fixture()
        source.nodes[0].attrs['communication']['group_size'] = 4
        result = prepare_communication_cases(source)
        self.assertEqual(result['cases'], [])
        self.assertIn('membership', result['skipped'][0]['reason'])
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            communication_case_workload(source, topology={'fixture':'two-device'})

    def test_group_runtime_shapes_and_precision_change_case_identity(self):
        source = fixture()
        original = prepare_communication_cases(source)['cases'][0]['case_id']
        for change in ('members', 'shape', 'dtype', 'runtime'):
            other = copy.deepcopy(source)
            comm = other.nodes[0].attrs['communication']
            if change == 'members': comm['participants'] = [2, 7]
            if change == 'shape':
                comm['tensors']['input_tensor']['shape'] = [8]
                comm['tensors']['output_tensor']['shape'] = [16]
            if change == 'dtype':
                comm['tensors']['input_tensor']['dtype'] = 'torch.float32'
                comm['tensors']['output_tensor']['dtype'] = 'torch.float32'
            if change == 'runtime': other.manifest['provenance']['torch_version'] = 'other'
            with self.subTest(change=change):
                self.assertNotEqual(prepare_communication_cases(other)['cases'][0]['case_id'], original)

    def test_alias_layout_or_failed_native_call_remain_skipped(self):
        for mutation in ('alias', 'stride', 'failed'):
            source = fixture()
            comm = source.nodes[0].attrs['communication']
            if mutation == 'alias': comm['tensors']['output_tensor']['storage_id'] = 'input'
            if mutation == 'stride': comm['tensors']['input_tensor']['stride'] = [2]
            if mutation == 'failed': source.nodes[0].attrs['status'] = 'failed'
            with self.subTest(mutation=mutation):
                result = prepare_communication_cases(source)
                self.assertEqual(result['cases'], [])
                self.assertTrue(result['skipped'])

    def test_collective_mismatched_shape_dtype_and_distributed_input_are_skipped(self):
        for mutation in ('size', 'dtype', 'distributed'):
            source = fixture()
            tensors = source.nodes[0].attrs['communication']['tensors']
            if mutation == 'size': tensors['output_tensor']['shape'] = [16]
            if mutation == 'dtype': tensors['output_tensor']['dtype'] = 'torch.float32'
            if mutation == 'distributed': tensors['input_tensor']['distributed'] = {'mesh':[0,1]}
            with self.subTest(mutation=mutation):
                result = prepare_communication_cases(source)
                self.assertEqual(result['cases'], [])
                self.assertTrue(result['skipped'])

    def alias_peers(self, operation='all_gather_into_tensor'):
        peers = []
        for index, rank in enumerate((2, 5)):
            source = fixture()
            source.nodes[0].rank = rank
            source.manifest['ranks'] = [rank]
            source.manifest['provenance']['rank_specific_config'] = rank
            record = source.nodes[0].attrs['communication']
            if operation == 'reduce_scatter_tensor':
                record.update(operation=operation, operator='c10d._reduce_scatter_base_.default', reduce_op='sum')
                record['tensors'] = {'input_tensor': tensor([8], 'shared'), 'output_tensor': tensor([4], 'shared')}
                view_key = 'output_tensor'
            else:
                for item in record['tensors'].values():
                    item['storage_id'] = 'shared'
                view_key = 'input_tensor'
            record['tensors'][view_key]['storage_offset'] = index * 4
            peers.append(source)
        return peers

    def test_rank_relative_alias_requires_all_peer_evidence(self):
        primary, peer = self.alias_peers()
        self.assertEqual(prepare_communication_cases(primary)['cases'], [])
        catalog = prepare_communication_cases(primary, [peer])
        self.assertEqual(len(catalog['cases']), 1)
        self.assertEqual(catalog['skipped'], [])
        case = catalog['cases'][0]
        self.assertEqual(case['alias']['kind'], 'own_group_rank_slice')
        self.assertEqual(case['tensors']['input_tensor']['storage_offset'], 0)
        self.assertEqual({n['rank']: n['tensor_storage']['input_tensor']['storage_offset']
                          for n in case['source_nodes']}, {2: 0, 5: 4})
        self.assertEqual(set(case['source_provenance_by_rank']), {'2', '5'})
        measurement = communication_case_workload(primary, topology={'test': 'two devices'}, peer_workloads=[peer])
        self.assertEqual(len(measurement.nodes), 2)
        self.assertEqual(measurement.nodes[0].attrs['cost_key'], measurement.nodes[1].attrs['cost_key'])

    def test_peer_capture_job_mismatch_is_rejected_and_missing_id_audited(self):
        primary, peer = self.alias_peers()
        case = prepare_communication_cases(primary, [peer])['cases'][0]
        self.assertIn('caller asserts', case['peer_evidence']['identity_basis'])
        primary.manifest['provenance']['runtime'] = {'slurm_job_id': 'capture-one'}
        peer.manifest['provenance']['runtime'] = {'slurm_job_id': 'capture-two'}
        with self.assertRaisesRegex(ValueError, 'capture run identity'):
            prepare_communication_cases(primary, [peer])
        peer.manifest['provenance']['runtime']['slurm_job_id'] = 'capture-one'
        case = prepare_communication_cases(primary, [peer])['cases'][0]
        self.assertEqual(case['peer_evidence']['capture_identity'], {'slurm_job_id': 'capture-one'})

    def test_rank_zero_offset_cannot_be_copied_to_other_peer(self):
        primary, peer = self.alias_peers()
        peer.nodes[0].attrs['communication']['tensors']['input_tensor']['storage_offset'] = 0
        result = prepare_communication_cases(primary, [peer])
        self.assertEqual(result['cases'], [])
        self.assertTrue(any('owning group rank slice' in item['reason'] for item in result['skipped']))

    def test_inconsistent_peer_shape_or_group_cannot_qualify_alias(self):
        for change in ('shape', 'group', 'base_offset'):
            primary, peer = self.alias_peers()
            record = peer.nodes[0].attrs['communication']
            if change == 'shape':
                record['tensors']['input_tensor'].update(shape=[8], storage_offset=8)
                record['tensors']['output_tensor'].update(shape=[16])
            if change == 'group': record.update(participants=[1, 5])
            if change == 'base_offset': record['tensors']['output_tensor']['storage_offset'] = 3
            self.assertEqual(prepare_communication_cases(primary, [peer])['cases'], [])

    def test_new_schema_config_defaults_preserved_and_nondefaults_rejected(self):
        source = fixture()
        record = source.nodes[0].attrs['communication']
        record['operator_schema'] = ('c10d::_allgather_base_(Tensor output_tensor, Tensor input_tensor, '
            '__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, '
            'int timeout=-1, Dict(str, Any)? config=None) -> (Tensor, Work)')
        source.nodes[0].attrs['arguments'] = {'args': [], 'kwargs': {}}
        case = prepare_communication_cases(source)['cases'][0]
        self.assertEqual(case['extra_arguments'], {'config': None})
        source.nodes[0].attrs['arguments']['kwargs']['config'] = {'changed': True}
        self.assertIn('nondefault', prepare_communication_cases(source)['skipped'][0]['reason'])
        del source.nodes[0].attrs['arguments']
        self.assertIn('evidence is missing', prepare_communication_cases(source)['skipped'][0]['reason'])
        record['extra_arguments'] = {'config': None}
        self.assertEqual(prepare_communication_cases(source)['cases'][0]['extra_arguments'], {'config': None})

    def test_premul_factor_is_required_and_part_of_case_identity(self):
        source = fixture()
        record = source.nodes[0].attrs['communication']
        record.update(operator='c10d._reduce_scatter_base_.default', operation='reduce_scatter_tensor',
                      reduce_op='premul_sum', reduction_factor=.125)
        record['tensors'] = {'input_tensor': tensor([8], 'input'), 'output_tensor': tensor([4], 'output')}
        first = prepare_communication_cases(source)['cases'][0]['case_id']
        record['reduction_factor'] = .25
        self.assertNotEqual(prepare_communication_cases(source)['cases'][0]['case_id'], first)
        for factor in (None, {'opaque_type': 'Tensor'}):
            record['reduction_factor'] = factor
            self.assertEqual(prepare_communication_cases(source)['cases'], [])

        record['reduction_factor'] = float('inf')
        with self.assertRaises(ValueError):
            prepare_communication_cases(source)

    def test_scalar_allreduce_is_one_element_while_zero_length_is_skipped(self):
        source = fixture()
        record = source.nodes[0].attrs['communication']
        record.update(operator='c10d.allreduce_.default', operation='all_reduce', reduce_op='sum')
        record['tensors'] = {'tensors': [tensor([], 'scalar')]}
        case = prepare_communication_cases(source)['cases'][0]
        self.assertEqual(case['logical_payload'], {'input_bytes': 2, 'output_bytes': 2})
        record['tensors'] = {'tensors': [tensor([0], 'empty')]}
        self.assertFalse(prepare_communication_cases(source)['cases'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires native boxed replay')
    def test_provider_lazy_allocation_and_preparation_errors_are_shared_before_run(self):
        import torch
        dist = torch.distributed
        import torch.testing._internal.distributed.fake_pg
        if dist.is_initialized():
            self.skipTest('requires ownership of fake process group')
        dist.init_process_group('fake', store=dist.HashStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        source = fixture()
        source.manifest['provenance'].update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
        record = source.nodes[0].attrs['communication']
        record.update(operator_schema=str(torch.ops.c10d._allgather_base_.default._schema),
                      runtime_context=execution_context(torch))
        second = copy.deepcopy(source.nodes[0])
        second.id = 1
        second.attrs['communication']['tensors']['input_tensor']['shape'] = [6]
        second.attrs['communication']['tensors']['output_tensor']['shape'] = [12]
        source.nodes.append(second)
        topology = {'test': 'fake-native-two-rank'}
        workload = communication_case_workload(source, topology=topology)
        calls, peer_failure = [], []
        def allocate(torch, case, rank, device):
            calls.append(case['case_id'])
            return _allocate_tensors(torch, case, rank, 'cpu')
        def share(output, value):
            output[:] = [value, 'peer allocation failed' if peer_failure else value]
        with patch.object(dist, 'get_backend', return_value='nccl'), \
             patch.object(torch.cuda, 'current_device', return_value=0), \
             patch.object(dist, 'all_gather_object', side_effect=share), \
             patch('titan_sim.communication_cases._allocate_tensors', side_effect=allocate):
            bundle = region_provider(workload, {'topology': topology})
            first, second = bundle['regions']
            self.assertEqual(calls, [])
            with self.assertRaisesRegex(ValueError, 'prepared'):
                first['run']()
            first['before_sample']()
            first['run']()  # Actual boxed native fake collective, no GPU timing.
            first['before_sample']()
            self.assertEqual(len(calls), 1)
            second['before_sample']()
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(ValueError, 'prepared'):
                first['run']()
            peer_failure.append(True)
            with self.assertRaisesRegex(ValueError, 'peer allocation failed'):
                second['before_sample']()
            with self.assertRaisesRegex(ValueError, 'prepared'):
                second['run']()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires native tensor views')
    def test_actual_shared_storage_reconstructed_for_each_group_rank(self):
        import torch
        for operation in ('all_gather_into_tensor', 'reduce_scatter_tensor'):
            primary, peer = self.alias_peers(operation)
            case = prepare_communication_cases(primary, [peer])['cases'][0]
            for rank in (0, 1):
                tensors = _allocate_tensors(torch, case, rank, 'cpu')
                base = tensors[case['alias']['base_tensor']]
                view = tensors[case['alias']['view_tensor']]
                self.assertEqual(base.untyped_storage()._cdata, view.untyped_storage()._cdata)
                self.assertEqual(view.storage_offset(), rank * 4)
                view.fill_(7)
                self.assertEqual(base.flatten()[rank*4:(rank+1)*4].tolist(), [7]*4)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires native reduction boxing')
    def test_native_premul_host_factor_capture_rejects_tensor_factor(self):
        import torch
        dist = torch.distributed
        function = torch.ops.c10d._reduce_scatter_base_.default
        with patch.object(dist, 'get_process_group_ranks', return_value=[0, 1]), \
             patch.object(dist, 'get_backend', return_value='nccl'), \
             patch.object(dist.distributed_c10d, '_get_process_group_name', return_value='0'):
            for factor in (.125, torch.tensor(.125)):
                reduction = dist._make_nccl_premul_sum(factor).boxed()
                args = ('output', 'input', object(), reduction, True, -1)
                if isinstance(factor, torch.Tensor):
                    with self.assertRaisesRegex(ValueError, 'tensor factors unsupported'):
                        describe_communication(function, args, {}, lambda value: value)
                else:
                    observed = describe_communication(function, args, {}, lambda value: value)
                    self.assertEqual(observed['reduce_op'], 'premul_sum')
                    self.assertEqual(observed['reduction_factor'], factor)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires native operator schemas')
    def test_schema_observation_resolves_group_and_default_async_mode(self):
        import torch
        dist = torch.distributed
        sentinel = object()
        with patch.object(dist, 'get_process_group_ranks', return_value=[0, 1]), \
             patch.object(dist, 'get_backend', return_value='nccl'), \
             patch.object(dist.distributed_c10d, '_get_process_group_name', return_value='0'):
            result = describe_communication(torch.ops.c10d._allgather_base_.default,
                ('output', 'input', sentinel), {}, lambda value: value)
        self.assertEqual(result['participants'], [0, 1])
        self.assertEqual(result['async_op'], True)
        self.assertEqual(result['timeout'], -1)
        self.assertEqual(result['tensors'], {'output_tensor':'output', 'input_tensor':'input'})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires native operator schemas')
    def test_functional_schema_uses_resolved_membership(self):
        import torch
        dist = torch.distributed
        with patch.object(dist, 'get_process_group_ranks', return_value=[1, 3]), \
             patch.object(dist, 'get_backend', return_value='nccl'), \
             patch.object(dist.distributed_c10d, '_resolve_process_group', return_value=object()) as resolve, \
             patch.object(dist.distributed_c10d, '_get_process_group_name', return_value='subgroup'):
            result = describe_communication(torch.ops._c10d_functional.reduce_scatter_tensor.default,
                ('input', 'sum', 2, 'subgroup'), {}, lambda value: value)
        resolve.assert_called_once_with('subgroup')
        self.assertEqual(result['reduce_op'], 'sum')
        self.assertEqual(result['participants'], [1, 3])

    def alltoall_peers(self):
        source = fixture()
        record = source.nodes[0].attrs['communication']
        record.update(operator='c10d.alltoall_base_.default', operation='all_to_all_single',
                      split_sizes={'input_split_sizes':[2,2], 'output_split_sizes':[2,2]},
                      tensors={'input':tensor([4,2],'input'), 'output':tensor([4,2],'output')})
        peer = copy.deepcopy(source)
        peer.nodes[0].rank = 5
        peer.manifest['ranks'] = [5]
        return source, peer

    def p2p_peers(self):
        source = fixture()
        record = source.nodes[0].attrs['communication']
        record.update(operator='c10d.send.default', operation='send', async_op=None,
                      p2p={'dst':1, 'tag':0}, tensors={'tensors':[tensor([4], 'input')]})
        peer = copy.deepcopy(source)
        peer.nodes[0].rank = 5
        peer.manifest['ranks'] = [5]
        peer.nodes[0].attrs['communication'].update(operator='c10d.recv_.default',operation='recv',p2p={'src':0,'tag':0})
        return source, peer

    def test_uniform_alltoall_requires_peers_and_preserves_splits(self):
        source, peer = self.alltoall_peers()
        self.assertEqual(prepare_communication_cases(source)['cases'], [])
        case = prepare_communication_cases(source,[peer])['cases'][0]
        self.assertEqual(case['split_sizes']['input_split_sizes'], [2,2])
        self.assertEqual(case['logical_payload'], {'input_bytes':16,'output_bytes':16})
        source.nodes[0].attrs['communication']['split_sizes']['input_split_sizes'] = [1,3]
        result = prepare_communication_cases(source,[peer])
        self.assertFalse(result['cases'])
        self.assertTrue(any('nonuniform' in item['reason'] for item in result['skipped']))

    def test_p2p_exact_two_rank_pair_and_unmatched_failure(self):
        source, peer = self.p2p_peers()
        result = prepare_communication_cases(source,[peer])
        self.assertEqual(result['skipped'], [])
        case = result['cases'][0]
        self.assertEqual(case['operation'], 'p2p')
        self.assertEqual(case['rank_operations']['5']['p2p'], {'src':0,'tag':0})
        self.assertEqual(len(case['source_nodes']), 2)
        peer.nodes[0].attrs['communication']['p2p']['tag'] = 1
        result = prepare_communication_cases(source,[peer])
        self.assertFalse(result['cases'])
        self.assertEqual(len(result['skipped']), 2)

    def test_source_fingerprints_and_hybrid_backend_gaps(self):
        from titan_sim.identity import workload_fingerprint
        source = fixture()
        result = prepare_communication_cases(source)
        self.assertEqual(result['source_workload_sha256_by_rank'], {'2':workload_fingerprint(source)})
        other = copy.deepcopy(source)
        other.nodes[0].name = 'other'
        with self.assertRaisesRegex(ValueError,'duplicate source rank'):
            prepare_communication_cases(source,[other])
        source.nodes[0].name = 'hybrid_ep::dispatch'
        source.nodes[0].attrs.pop('communication')
        self.assertIn('generic NCCL is not a substitute',prepare_communication_cases(source)['skipped'][0]['reason'])

    def test_local_substitution_keeps_fake_discovery_distinct_from_nccl_target(self):
        source = fixture()
        record = source.nodes[0].attrs['communication']
        record.update(backend='fake',execution_backend='fake',target_backend='nccl',
                      evidence='local_cuda_substitution_and_target_group')
        case = prepare_communication_cases(source)['cases'][0]
        self.assertEqual(case['backend'],'nccl')
        self.assertEqual(case['discovery']['execution_backend'],'fake')
        self.assertIn('not native distributed',case['discovery']['limitation'])
        del record['target_backend']
        self.assertFalse(prepare_communication_cases(source)['cases'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires actual native boxed communication schemas')
    def test_actual_native_alltoall_send_recv_abis_and_provider(self):
        import torch
        dist = torch.distributed
        import torch.testing._internal.distributed.fake_pg
        if dist.is_initialized():
            self.skipTest('test owns fake process group')
        dist.init_process_group('fake',store=dist.HashStore(),rank=0,world_size=2)
        self.addCleanup(dist.destroy_process_group)
        for operation in ('alltoall','send','recv'):
            source, peer = self.alltoall_peers() if operation == 'alltoall' else self.p2p_peers()
            if operation == 'recv':
                for item in (source,peer):
                    record = item.nodes[0].attrs['communication']
                    sending = item.nodes[0].rank == 5
                    record.update(operator='c10d.send.default' if sending else 'c10d.recv_.default',
                                  operation='send' if sending else 'recv',
                                  p2p={'dst':0,'tag':0} if sending else {'src':1,'tag':0})
            for item in (source,peer):
                record = item.nodes[0].attrs['communication']
                namespace,name,overload = record['operator'].split('.')
                function = getattr(getattr(getattr(torch.ops,namespace),name),overload)
                record['operator_schema'] = str(function._schema)
                record['extra_arguments'] = {}
                record['runtime_context'] = execution_context(torch)
                item.manifest['provenance'] = dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
            topology = {'fixture':'native-boxed-fake'}
            workload = communication_case_workload(source,topology=topology,peer_workloads=[peer])
            def share(values,value): values[:] = [value,value]
            def allocate(torch,case,rank,device): return _allocate_tensors(torch,case,rank,'cpu')
            with patch.object(dist,'get_backend',return_value='nccl'), \
                 patch.object(dist,'all_gather_object',side_effect=share), \
                 patch.object(torch.cuda,'current_device',return_value=0), \
                 patch('titan_sim.communication_cases._allocate_tensors',side_effect=allocate):
                region = region_provider(workload,{'topology':topology})['regions'][0]
                region['before_sample']()
                region['run']()  # Actual C++ overload/Work.wait; fake transport, no timing claim.

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires native boxed schemas')
    def test_actual_boxed_process_group_reduction_and_native_replay(self):
        import torch
        dist = torch.distributed
        import torch.testing._internal.distributed.fake_pg  # Register the native fake backend.
        if dist.is_initialized():
            self.skipTest('test requires ownership of its fake process group')
        dist.init_process_group('fake', store=dist.HashStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        group = dist.group.WORLD
        reduction = dist.ReduceOp(dist.ReduceOp.SUM).boxed()
        input_tensor = torch.ones(8)
        output = torch.empty(4)
        function = torch.ops.c10d._reduce_scatter_base_.default
        args = (output, input_tensor, group.boxed(), reduction, True, -1)
        with patch.object(dist, 'get_process_group_ranks', return_value=[0, 1]) as ranks, \
             patch.object(dist, 'get_backend', return_value='fake'), \
             patch.object(dist.distributed_c10d, '_get_process_group_name', return_value='fake'):
            result = describe_communication(function, args, {}, lambda value: {'shape': list(value.shape)})
        self.assertIsInstance(ranks.call_args.args[0], dist.ProcessGroup)
        self.assertEqual(result['reduce_op'], 'sum')
        # Fake communication checks the actual native boxing ABI only. It is
        # neither numerical collective validation nor a timing measurement.
        returned = function(**dict(output_tensor=output, input_tensor=input_tensor,
            process_group=group.boxed(), reduce_op=dist.ReduceOp(dist.ReduceOp.SUM).boxed(),
            async_op=True, timeout=-1))
        returned[-1].wait()
        premul = dist._make_nccl_premul_sum(.125)
        returned = function(output, input_tensor, group.boxed(), premul.boxed(), True, -1)
        returned[-1].wait()


if __name__ == '__main__':
    unittest.main()
