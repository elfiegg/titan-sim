import copy
import unittest

from titan_sim.core import Node, Storage
from titan_sim.identity import workload_fingerprint
from titan_sim.schedule_recovery import recover_schedule
from titan_sim.workload import Workload


def tensor(storage, size=4):
    return dict(storage_id=storage, shape=[size], stride=[1], storage_offset=0,
                dtype='torch.float32', layout='torch.strided', device='cuda:0')


def rank_source(rank, *, operation='all_reduce', size=4):
    source = tensor('input', size)
    output = tensor('output', size * 2 if operation == 'all_gather_into_tensor' else
                    size // 2 if operation == 'reduce_scatter_tensor' else size)
    operator = '_c10d_functional.' + operation + '.default'
    record = dict(operator=operator, operator_schema=operator + ' fixture',
                  operation=operation, participants=[0, 1], group_size=2,
                  group_name='group0', backend='nccl', reduce_op='sum',
                  async_op=None, timeout=-1, tensors={'input': source},
                  runtime_context={'nccl_version': [2, 28]},
                  evidence='native_dispatch_and_resolved_process_group')
    # Deliberately nonmonotonic node IDs: only dependencies establish issue order.
    nodes = [Node(20, 'producer', rank, attrs={'cost_key': 'producer'}),
             Node(3, operator, rank, deps=[20], attrs=dict(communication=record,
                  status='returned', outputs=output, step=0, cost_key='host-dispatch')),
             Node(99, 'independent_compute', rank, deps=[3], attrs={'cost_key': 'independent'}),
             Node(7, '_c10d_functional.wait_tensor.default', rank, deps=[99],
                  attrs=dict(status='returned', arguments={'args': [copy.deepcopy(output)]},
                             outputs=copy.deepcopy(output), cost_key='host-wait')),
             Node(2, 'consumer', rank, deps=[7], attrs=dict(arguments={'args': [output]},
                  cost_key='consumer'))]
    return Workload(nodes, [Storage('input', rank, size * 4, producer=20, consumers=[3]),
                           Storage('output', rank, output['shape'][0] * 4, producer=3, consumers=[2])],
                    dict(ranks=[rank], world_size=2, capture_status='complete', observer='dispatch',
                         capture_run_id='same-native-execution', execution='real',
                         provenance={'torch_version': 'fixture'}, unresolved=['CUDA streams unknown']),
                    {'id': 'fixture', 'capacity_bytes': 100000})


class ScheduleRecoveryTests(unittest.TestCase):
    def test_native_issue_completion_and_async_wait_are_separate(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        originals = [workload_fingerprint(source) for source in sources]
        result = recover_schedule(sources)
        report = result.manifest['schedule_recovery']
        self.assertEqual(report['recovered_count'], 1)
        self.assertEqual(report['rejected'], [])
        for rank in (0, 1):
            nodes = {node.attrs.get('source_node_id'): node for node in result.nodes
                     if node.rank == rank and node.kind == 'compute'}
            completion = next(node for node in result.nodes if node.rank == rank and node.kind == 'collective')
            self.assertEqual(completion.deps, [nodes[3].id])
            self.assertEqual(completion.attrs['communication_completion'], 'device')
            self.assertFalse(completion.attrs['completion_measured'])
            self.assertEqual(completion.attrs['source_workload_sha256'], originals[rank])
            self.assertNotEqual(completion.attrs['cost_key'], 'host-dispatch')
            self.assertTrue(nodes[3].attrs['zero_cost_semantic'])
            self.assertNotIn('cost_key', nodes[3].attrs)
            self.assertEqual(nodes[99].deps, [nodes[3].id])
            self.assertNotIn(completion.id, nodes[99].deps)
            self.assertIn(completion.id, nodes[7].deps)
            output = next(item for item in result.storage if item.rank == rank and item.size == 16 and item.producer == nodes[3].id)
            self.assertIn(nodes[2].id, output.consumers)
            self.assertTrue(any(completion.id in item.consumers for item in result.storage if item.rank == rank))
        self.assertEqual([workload_fingerprint(source) for source in sources], originals)
        self.assertIn('CUDA streams unknown', result.manifest['unresolved'])

    def test_missing_rank_never_synthesizes_peer(self):
        result = recover_schedule([rank_source(0)])
        self.assertEqual(result.manifest['ranks'], [0])
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
        self.assertFalse(any(node.kind == 'collective' for node in result.nodes))
        self.assertTrue(any('missing rank' in gap for gap in result.manifest['unresolved']))

    def test_uneven_shards_and_dtype_mismatch_fail_closed(self):
        for change in ('shape', 'dtype'):
            sources = [rank_source(rank, operation='all_gather_into_tensor') for rank in (0, 1)]
            record = sources[1].nodes[1].attrs['communication']
            if change == 'shape':
                record['tensors']['input']['shape'] = [3]
            else:
                record['tensors']['input']['dtype'] = 'torch.bfloat16'
            result = recover_schedule(sources)
            self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
            self.assertTrue(any('uneven shards' in gap for gap in result.manifest['unresolved']))

    def test_fake_wait_is_never_native_completion(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        for source in sources:
            source.manifest['execution'] = 'local_cuda'
            source.nodes[1].attrs['communication'].update(backend='fake', target_backend='nccl',
                evidence='local_cuda_substitution_and_target_group', transport_executed=False,
                substitution='local_representative')
        result = recover_schedule(sources)
        self.assertEqual(result.manifest['schedule_recovery']['candidate_count'], 1)
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
        self.assertTrue(any('substituted' in gap for gap in result.manifest['unresolved']))
        self.assertFalse(any(node.attrs.get('communication_completion') for node in result.nodes))

    def test_missing_or_mismatched_run_identity(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        sources[1].manifest['capture_run_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'run identity'):
            recover_schedule(sources)
        for source in sources:
            source.manifest.pop('capture_run_id')
        result = recover_schedule(sources)
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
        self.assertTrue(any('same-run identity' in gap for gap in result.manifest['unresolved']))

    def test_missing_failed_or_wrong_storage_wait(self):
        for change in ('missing', 'failed', 'storage', 'unordered', 'use_before_wait'):
            sources = [rank_source(rank) for rank in (0, 1)]
            wait = sources[1].nodes[3]
            if change == 'missing':
                wait.name = 'opaque_work_wait'
            elif change == 'failed':
                wait.attrs['status'] = 'failed'
            elif change == 'storage':
                wait.attrs['arguments']['args'][0]['storage_id'] = 'unrelated'
            elif change == 'unordered':
                wait.deps = []
            else:
                sources[1].nodes[2].attrs['arguments'] = {'args': [tensor('output')]}
            with self.subTest(change=change):
                result = recover_schedule(sources)
                self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
                self.assertFalse(any(node.kind == 'collective' for node in result.nodes))

    def test_host_return_is_not_c10d_work_completion(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        for source in sources:
            record = source.nodes[1].attrs['communication']
            record.update(operator='c10d.allreduce_.default', async_op=True,
                          tensors={'tensors': [tensor('input')]})
            source.nodes[1].name = record['operator']
        result = recover_schedule(sources)
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
        self.assertTrue(any('observed Work wait' in gap for gap in result.manifest['unresolved']))

    def test_ambiguous_collective_order_is_not_sorted_by_node_id(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        for source in sources:
            second = copy.deepcopy(source.nodes[1])
            second.id = 500
            second.deps = [20]
            source.nodes.append(second)
        result = recover_schedule(sources)
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
        self.assertTrue(any('ambiguous collective issue order' in gap for gap in result.manifest['unresolved']))

    def test_repeated_collectives_use_dependency_order_not_ids(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        for source in sources:
            second = copy.deepcopy(source.nodes[1])
            second.id = 1
            second.deps = [2]
            second.attrs['outputs']['storage_id'] = 'output_second'
            wait = copy.deepcopy(source.nodes[3])
            wait.id = 0
            wait.deps = [1]
            wait.attrs['arguments']['args'][0]['storage_id'] = 'output_second'
            wait.attrs['outputs']['storage_id'] = 'output_second'
            source.nodes.extend([second, wait])
            source.storage.append(Storage('output_second', source.manifest['ranks'][0], 16, 1, [0]))
        result = recover_schedule(sources)
        report = result.manifest['schedule_recovery']
        self.assertEqual(report['recovered_count'], 2)
        self.assertEqual([record['source_nodes'][0]['id'] for record in report['rendezvous']], [3, 1])

    def test_incomplete_capture_and_unobserved_streams_remain_gaps(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        sources[1].manifest['capture_status'] = 'incomplete'
        result = recover_schedule(sources)
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
        self.assertTrue(any('complete original' in gap for gap in result.manifest['unresolved']))

    def test_supported_uniform_scatter_validates_output_shape(self):
        sources = [rank_source(rank, operation='reduce_scatter_tensor') for rank in (0, 1)]
        self.assertEqual(recover_schedule(sources).manifest['schedule_recovery']['recovered_count'], 1)
        sources[1].nodes[1].attrs['outputs']['shape'] = [3]
        result = recover_schedule(sources)
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
        self.assertTrue(any('output shape' in gap for gap in result.manifest['unresolved']))

    def test_collective_storage_survives_completion_without_original_consumers(self):
        sources = [rank_source(rank) for rank in (0, 1)]
        for source in sources:
            for allocation in source.storage:
                allocation.consumers = []
        result = recover_schedule(sources)
        self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 1)
        for completion in (node for node in result.nodes if node.kind == 'collective'):
            for allocation in result.storage:
                if allocation.rank == completion.rank:
                    self.assertIn(completion.id, allocation.consumers)
            output = next(item for item in result.storage if item.rank == completion.rank and
                          item.producer == completion.attrs['source_issue_node_id'])
            self.assertNotEqual(output.producer, completion.id)

    def test_missing_or_inconsistent_backing_storage_rejects_recovery(self):
        for change in ('missing', 'short_input', 'short_output', 'wrong_producer', 'duplicate_producer'):
            sources = [rank_source(rank) for rank in (0, 1)]
            source = sources[1]
            if change == 'missing':
                source.storage.pop()
            elif change == 'short_input':
                source.storage[0].size = 4
            elif change == 'short_output':
                source.storage[1].size = 4
            elif change == 'wrong_producer':
                source.storage[1].producer = 20
            else:
                second = copy.deepcopy(source.nodes[1])
                second.id = 500
                second.deps = [2]
                source.nodes.append(second)
                other = copy.deepcopy(sources[0].nodes[1])
                other.id = 500
                other.deps = [2]
                sources[0].nodes.append(other)
            with self.subTest(change=change):
                result = recover_schedule(sources)
                self.assertEqual(result.manifest['schedule_recovery']['recovered_count'], 0)
                self.assertTrue(any('storage' in gap for gap in result.manifest['unresolved']))


if __name__ == '__main__':
    unittest.main()
