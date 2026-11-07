"""Native graph storage aliases and schedule-dependent lifetime regressions."""
import unittest

from titan_sim.core import Node
from titan_sim.memory import assess_memory
from titan_sim.plan_memory import derive_plan_memory, lower_graph_memory
from titan_sim.workload import Workload


def leaf(identity, size=16, *, device='cuda:0', path=None):
    return {'storage_id': identity, 'storage_bytes': size, 'logical_bytes': size,
            'device': device, 'shape': [size // 4], 'dtype': 'torch.float32',
            'path': [] if path is None else path}


def node(name, storage=None, inputs=(), op='call_function'):
    return {'name': name, 'op': op, 'inputs': list(inputs),
            'outputs': [] if storage is None else [storage]}


def graph(role, nodes, outputs):
    return {'role': role, 'nodes': nodes, 'outputs': outputs}


def plan(graphs, metadata=None, state=None):
    return {'graphs': graphs, 'stages': {'0': {'metadata': metadata or {},
            'state': state or {}, 'graphs': {v['role']: k for k, v in graphs.items()}}},
            'schedule': {'stage_to_rank': {'0': 0}}}


def invoke(identity, graph_id, mapping, mb=0, **extra):
    return {'id': identity, 'graph_id': graph_id, 'rank': 0, 'stage_index': 0,
            'microbatch_index': mb, 'node_ids': mapping, **extra}


def events(*ids):
    return [Node(i, str(i), 0) for i in ids]


class PlanMemoryTests(unittest.TestCase):
    def test_views_share_backing_extent_and_union_users(self):
        g = graph('custom', [node('x', leaf('x', 128), op='placeholder'),
                            node('view', leaf('x', 128), ['x']),
                            node('sum', leaf('sum', 4), ['view'])], ['sum'])
        result = derive_plan_memory(plan({'g': g}), [invoke('a', 'g', {'x': 0, 'view': 1, 'sum': 2})], events(0, 1, 2))
        self.assertEqual(len(result['storage']), 2)
        backing = next(s for s in result['storage'] if s.size == 128)
        self.assertEqual(backing.consumers, [0, 1, 2])
        self.assertEqual(result['memory']['storage_evidence'], 'unverified')
        self.assertFalse(any(result['memory']['coverage'].values()))

    def test_saved_values_are_separate_per_microbatch_and_live_through_backward(self):
        fw = graph('fw', [node('saved', leaf('trace_saved'))], ['saved'])
        bw = graph('full_bw', [node('saved_input', leaf('bw_trace'), op='placeholder'),
                              node('grad', leaf('gradient'), ['saved_input'])], ['grad'])
        metadata = {'num_user_outputs': 0, 'num_saved_for_backward': 1,
                    'num_param_grad_values': 1,
                    'partition': {'saved_for_backward_names': ['saved_input']}}
        p = plan({'fw': fw, 'bw': bw}, metadata)
        invs = [invoke('f0', 'fw', {'saved': 0}, 0), invoke('f1', 'fw', {'saved': 1}, 1),
                invoke('b0', 'bw', {'saved_input': 2, 'grad': 3}, 0),
                invoke('b1', 'bw', {'saved_input': 4, 'grad': 5}, 1)]
        result = derive_plan_memory(p, invs, events(*range(6)))
        self.assertEqual(len(result['storage']), 4)
        first, second = result['storage'][:2]
        self.assertEqual(first.consumers, [0, 2, 3])
        self.assertEqual(second.consumers, [1, 4, 5])
        self.assertFalse(first.persistent)
        self.assertFalse(second.persistent)
        self.assertNotEqual(first.id, second.id)
        self.assertFalse(any('saved backward values live' in g for g in result['unresolved']))
        hardware = {'capacity_bytes': 100, 'reserve_fraction': 0}
        manifest = {'ranks': [0], 'rank_placement': {'0': 'gpu'}, 'memory': result['memory']}
        overlap = {0: (0, 1), 1: (1, 2), 2: (2, 2), 3: (2, 3), 4: (3, 3), 5: (3, 4)}
        serial = {0: (0, 1), 1: (2, 3), 2: (1, 1), 3: (1, 2), 4: (3, 3), 5: (3, 4)}
        peaks = [assess_memory(events(*range(6)), [first, second], t, hardware, manifest)
                 ['per_device']['gpu']['modeled_peak_lower_bytes'] for t in (overlap, serial)]
        self.assertEqual(peaks, [32, 16])

    def test_async_completion_survives_host_reshard(self):
        unshard = graph('unshard', [node('shard', leaf('shard', 8), op='placeholder'),
                                  node('full', leaf('full', 32), ['shard'])], ['full'])
        fw = graph('fw', [node('parameter', leaf('fake_full', 32), op='placeholder'),
                          node('use', None, ['parameter'])], [])
        metadata = {'unshard_flat_param_indices': [0], 'num_fw_param_inputs': 1,
                    'fwd_input_names': ['parameter'], 'num_user_outputs': 0}
        p = plan({'u': unshard, 'f': fw}, metadata,
                 {'sharded_param_values': [leaf('real_parameter', 8)]})
        invs = [invoke('u0', 'u', {'shard': 0, 'full': 1}),
                invoke('f0', 'f', {'parameter': 2, 'use': 3},
                       completion_node_ids={'use': 6}),
                invoke('reshard', None, {}, action_type='RESHARD', action_node_ids=[4])]
        result = derive_plan_memory(p, invs, events(*range(7)))
        full = next(s for s in result['storage'] if s.size == 32)
        self.assertIn(6, full.consumers)
        self.assertIn(4, full.consumers)
        self.assertFalse(full.persistent)
        timing = {str(i): (i, i + .5) for i in range(7)}
        manifest = {'ranks': [0], 'rank_placement': {'0': 'gpu'}, 'memory': result['memory']}
        assessment = assess_memory(events(*range(7)), result['storage'], timing,
                                   {'capacity_bytes': 100, 'reserve_fraction': 0}, manifest)
        self.assertEqual(assessment['memory_verdict'], 'inconclusive')
        self.assertEqual(assessment['per_device']['gpu']['peak_lower_bytes'], 0)

    def test_persistent_state_aliases_across_groups_and_sparse_slots(self):
        g = graph('fw', [node('p', leaf('trace'), op='placeholder'),
                         node('use', None, ['p'])], [])
        state = {'sharded_param_values': [leaf('actual', path=[1])],
                 'buffer_values': [leaf('actual', path=[0])]}
        metadata = {'num_fw_param_inputs': 0, 'num_sharded_param_values': 2,
                    'fwd_input_names': ['p'], 'fwd_flat_input_indices': [1]}
        result = derive_plan_memory(plan({'g': g}, metadata, state),
                                    [invoke('a', 'g', {'p': 0, 'use': 1})], events(0, 1))
        self.assertEqual(len(result['storage']), 1)
        self.assertTrue(result['storage'][0].persistent)
        self.assertEqual(result['storage'][0].consumers, [0, 1])

    def test_missing_storage_is_gap_and_cpu_offload_not_device_bytes(self):
        g = graph('fw', [node('host', leaf('h', 8192, device='cpu')),
                        node('unknown', leaf(None))], [])
        result = derive_plan_memory(plan({'g': g}),
                                    [invoke('a', 'g', {'host': 0, 'unknown': 1})], events(0, 1))
        self.assertEqual(sum(s.size for s in result['storage']), 16)
        self.assertEqual(result['memory']['host_storage_records_excluded'], 1)
        self.assertTrue(any('lacks backing-storage identity' in g for g in result['unresolved']))

    def test_explicit_cross_graph_binding_preserves_alias(self):
        a = graph('custom', [node('x', leaf('a'))], ['x'])
        b = graph('custom', [node('input', leaf('b'), op='placeholder'),
                             node('view', leaf('b'), ['input'])], ['view'])
        invs = [invoke('a0', 'a', {'x': 0}),
                invoke('b0', 'b', {'input': 1, 'view': 2},
                       input_bindings={'input': {'invocation': 'a0', 'node': 'x'}})]
        result = derive_plan_memory(plan({'a': a, 'b': b}), invs, events(0, 1, 2))
        self.assertEqual(len(result['storage']), 1)
        self.assertEqual(result['storage'][0].consumers, [0, 1, 2])

    def test_scheduled_joint_reuses_native_gradient_accumulators(self):
        first = graph('FIRST', [node('grad', leaf('new_grad'))], [None, 'grad'])
        repeated = graph('FB', [node('acc', leaf('acc'), op='placeholder'),
                               node('updated', leaf('acc'), ['acc'])], [None, 'updated'])
        metadata = {'num_param_grad_values': 1, 'call_specs': {
            'FIRST': {'input_names': [], 'num_param_inputs': 0},
            'FB': {'input_names': [], 'num_param_inputs': 0,
                   'grad_accumulator_input_indices': [0]}}}
        invs = [invoke('first', 'f', {'grad': 0}), invoke('next', 'r', {'acc': 1, 'updated': 2}, 1)]
        result = derive_plan_memory(plan({'f': first, 'r': repeated}, metadata), invs, events(0, 1, 2))
        self.assertEqual(len(result['storage']), 1)
        self.assertTrue(result['storage'][0].persistent)
        self.assertEqual(result['storage'][0].consumers, [0, 1, 2])

    def test_workload_integration_preserves_existing_gaps(self):
        p = plan({'g': graph('fw', [node('x', leaf('x'))], [])})
        n = Node(0, 'x', 0, attrs={'plan_action_id': '0:0', 'plan_graph_id': 'g',
                                  'plan_graph_node': 'x', 'stage_index': 0,
                                  'microbatch_index': 0, 'computation_type': 'F'})
        w = Workload([n], [], {'ranks': [0], 'unresolved': ['existing']}, {})
        self.assertIs(lower_graph_memory(w, p), w)
        self.assertEqual(len(w.storage), 1)
        self.assertEqual(w.manifest['unresolved'][0], 'existing')

    def test_graph_constants_persist_once_across_microbatches(self):
        g = graph('fw', [node('constant', leaf('constant'), op='get_attr'),
                         node('use', None, ['constant'])], [])
        invs = [invoke('f0', 'g', {'constant': 0, 'use': 1}),
                invoke('f1', 'g', {'constant': 2, 'use': 3}, 1)]
        result = derive_plan_memory(plan({'g': g}), invs, events(0, 1, 2, 3))
        self.assertEqual(len(result['storage']), 1)
        self.assertTrue(result['storage'][0].persistent)
        self.assertEqual(result['storage'][0].consumers, [0, 1, 2, 3])

    def test_plain_joint_uses_native_static_prefix(self):
        g = graph('joint', [node('p', leaf('fake_p'), op='placeholder'),
                            node('grad', leaf('grad'), ['p'])], [None, 'grad'])
        metadata = {'traced': {'num_static_inputs': 1, 'graph_state': {'mappings': []}}}
        result = derive_plan_memory(plan({'g': g}, metadata,
                                        {'sharded_param_values': [leaf('actual_p')]}),
                                    [invoke('fb', 'g', {'p': 0, 'grad': 1})], events(0, 1))
        self.assertEqual(len(result['storage']), 2)
        self.assertFalse(any('static-state bindings' in g for g in result['unresolved']))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'native FX metadata tests require installed torch')
class NativeFXMemoryTests(unittest.TestCase):
    def test_real_fake_tensor_views_keep_full_backing_storage(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from titan_sim.native_plan import _Extractor
        with FakeTensorMode():
            x = torch.empty(32, device='cuda')
            gm = make_fx(lambda value: (torch.ops.aten.slice.Tensor(value, 0, 3, 7), value + 1))(x)
        native = _Extractor().graph(gm, 'g', 0, 'fw')
        names = {n['name']: i for i, n in enumerate(native['nodes'])}
        result = derive_plan_memory(plan({'g': native}), [invoke('a', 'g', names)],
                                    events(*names.values()))
        self.assertEqual(len(result['storage']), 2)
        self.assertEqual([s.size for s in result['storage']], [128, 128])
        view_node = next(n for n in native['nodes'] if n['target'] == 'aten.slice.Tensor')
        self.assertEqual(view_node['outputs'][0]['logical_bytes'], 16)
        self.assertEqual(view_node['outputs'][0]['storage_bytes'], 128)


if __name__ == '__main__':
    unittest.main()
