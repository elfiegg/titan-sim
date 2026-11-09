import copy
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from titan_sim.core import simulate
from titan_sim.plan_lowering import lower_plan


HARDWARE = dict(id='analytical-fixture', capacity_bytes=100000, reserve_fraction=0,
                effective_flops_per_second=1, effective_hbm_bytes_per_second=100,
                pp_bytes_per_second=4, interconnect_bytes_per_second=4,
                comm_latency_seconds=0, launch_seconds=0)


@dataclass
class NativeMetaFixture:
    num_user_outputs: int = 1
    num_param_grad_values: int = 0
    num_input_grad_values: int = 1


def graph(stage, role):
    def leaf(name):
        return dict(storage_id=f'{stage}:{role}:{name}', storage_bytes=4,
                    logical_bytes=4, shape=[1], stride=[1], dtype='torch.float32',
                    device='cuda:0', storage_offset=0, path=[])
    return dict(stage_index=stage, role=role, metadata={}, outputs=['op'], nodes=[
        dict(name='x', op='placeholder', target='x', inputs=[], outputs=[leaf('x')]),
        dict(name='op', op='call_function', target='aten.mul.Tensor', inputs=['x'],
             args=[{'node': 'x'}, 2], kwargs={}, outputs=[leaf('op')]),
        dict(name='output', op='output', target='output', inputs=['op'], outputs=[])])


def action(stage, typ, mb=0):
    return dict(stage_index=stage, computation_type=typ, microbatch_index=mb)


def fixture():
    plan = dict(schema_version='titan_sim.native_plan.v1', world_size=2,
                schedule=dict(source_attribute='pipeline_order_with_comms',
                              stage_to_rank={'0': 0, '1': 1}, num_microbatches=1,
                              rank_actions={
                                  '0': [action(0, 'F'), action(0, 'SEND_F'), action(0, 'RECV_B'),
                                        action(0, 'B'), action(0, 'WAIT_SEND_F')],
                                  '1': [action(1, 'RECV_F'), action(1, 'F'), action(1, 'B'),
                                        action(1, 'SEND_B'), action(1, 'WAIT_SEND_B')]}),
                stages={}, graphs={}, provenance={'torch_version': 'fixture'}, unresolved=[])
    for stage in range(2):
        plan['stages'][str(stage)] = dict(graphs={}, metadata=dict(
            num_user_outputs=1, num_param_grad_values=0, num_input_grad_values=1))
        for role in ('fw', 'full_bw'):
            gid = f'{stage}.{role}'
            plan['stages'][str(stage)]['graphs'][role] = gid
            plan['graphs'][gid] = graph(stage, role)
    config = dict(node_attrs={gid: {'op': {'flops': 1}} for gid in plan['graphs']})
    return plan, config


class PlanLoweringTests(unittest.TestCase):
    def test_pipeline_analytical_end_to_end(self):
        plan, config = fixture()
        original = copy.deepcopy(plan)
        workload = lower_plan(plan, HARDWARE, config)
        result = simulate(*workload.arguments())
        self.assertEqual(result['seconds'], 6)
        self.assertEqual(result['cost_coverage']['analytical_network_assumption'], 4)
        self.assertEqual(plan, original)
        self.assertEqual(len(workload.manifest['native_plan']['graph_invocations']), 4)
        transfers = [n for n in workload.nodes if n.kind in ('send', 'recv')]
        self.assertEqual(len({n.attrs['rendezvous'] for n in transfers}), 2)
        self.assertTrue(all(n.attrs['payload_bytes'] == 4 for n in transfers))
        self.assertEqual(result['status'], 'exploratory_uncalibrated')

    def test_stale_plan_checksum_rejected(self):
        from titan_sim.native_plan import seal_plan
        plan, config = fixture()
        plan = seal_plan(plan)
        plan['schedule']['num_microbatches'] = 2
        with self.assertRaisesRegex(ValueError, 'checksum'):
            lower_plan(plan, HARDWARE, config)

    def test_matching_payload_rejects_contradictory_binding(self):
        plan, config = fixture()
        config['action_attrs'] = {'0': {'1': {'payload_bytes': 8}}}
        with self.assertRaisesRegex(ValueError, 'contradicts native'):
            lower_plan(plan, HARDWARE, config)

    def test_missing_payload_is_not_zero(self):
        plan, config = fixture()
        plan['stages']['0']['metadata'] = {}
        with self.assertRaisesRegex(ValueError, 'payload_bytes'):
            lower_plan(plan, HARDWARE, config)

    def test_allrank_ownership_microbatch_and_unsupported(self):
        for edit, pattern in (
            (lambda p: p['schedule']['rank_actions'].pop('1'), 'all-rank'),
            (lambda p: p['schedule']['stage_to_rank'].update({'0': 1}), 'ownership'),
            (lambda p: p['schedule']['rank_actions']['0'][0].update(microbatch_index=1), 'microbatch'),
            (lambda p: p['schedule']['rank_actions']['0'][0].update(computation_type='MAGIC'), 'unsupported'),
            (lambda p: p['schedule'].update(source_attribute='pipeline_order'), 'compute-only')):
            plan, config = fixture()
            edit(plan)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                lower_plan(plan, HARDWARE, config)

    def test_unmatched_and_duplicate_transfer(self):
        plan, config = fixture()
        plan['schedule']['rank_actions']['1'][0] = None
        with self.assertRaisesRegex(ValueError, 'missing required producer/receive|unmatched'):
            lower_plan(plan, HARDWARE, config)
        plan, config = fixture()
        plan['schedule']['rank_actions']['0'].append(action(0, 'SEND_F'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            lower_plan(plan, HARDWARE, config)

    def test_bindings_cannot_override_semantics(self):
        for override in ({'stage_index': 3}, {'zero_cost_semantic': True},
                         {'native_wait': True, 'zero_cost_semantic': True}):
            plan, config = fixture()
            config['node_attrs']['0.fw']['op'] = override
            with self.subTest(override=override), self.assertRaises(ValueError):
                lower_plan(plan, HARDWARE, config)

    def test_cost_identity_binds_args_runtime_and_resources(self):
        plan, config = fixture()
        config['node_attrs']['0.fw']['op']['resources'] = ['gpu:deviceA:compute']
        def key(p):
            workload = lower_plan(p, HARDWARE, config)
            node = next(n for n in workload.nodes if n.name == '0.fw/op')
            self.assertEqual(node.attrs['resources'], ['gpu:deviceA:compute'])
            return node.attrs['cost_key']
        first = key(plan)
        plan['graphs']['0.fw']['nodes'][1]['args'][1] = 3
        self.assertNotEqual(first, key(plan))
        second = key(plan)
        plan['provenance']['torch_version'] = 'other'
        self.assertNotEqual(second, key(plan))

    def test_single_stage_joint_native_action(self):
        plan, config = fixture()
        typ = 'FULL_FORWARD_BACKWARD'
        plan['world_size'] = 1
        plan['schedule'].update(stage_to_rank={'0': 0}, rank_actions={'0': [action(0, typ)]})
        plan['stages'] = {'0': {'graphs': {typ: 'joint'}, 'metadata': {}}}
        plan['graphs'] = {'joint': graph(0, typ)}
        config = {'node_attrs': {'joint': {'op': {'flops': 2}}}}
        workload = lower_plan(plan, HARDWARE, config)
        self.assertEqual(simulate(*workload.arguments())['seconds'], 2)
        self.assertEqual(workload.manifest['native_plan']['graph_invocations'][0]['action_type'], typ)

    def test_collective_rendezvous_occurrences_and_native_wait(self):
        plan, config = fixture()
        typ = 'FULL_FORWARD_BACKWARD'
        plan['world_size'] = 1
        plan['schedule'].update(num_microbatches=2, stage_to_rank={'0': 0},
                                rank_actions={'0': [action(0, typ, 0), action(0, typ, 1)]})
        plan['stages'] = {'0': {'graphs': {'joint': 'joint'}, 'metadata': {}}}
        native_graph = graph(0, 'joint')
        native_graph['nodes'][1]['target'] = '_c10d_functional.all_reduce.default'
        native_graph['nodes'].insert(2, dict(name='wait', op='call_function',
            target='_c10d_functional.wait_tensor.default', inputs=['op'], outputs=[]))
        native_graph['nodes'][-1]['inputs'] = ['wait']
        plan['graphs'] = {'joint': native_graph}
        config = {'node_attrs': {'joint': {
            'op': dict(kind='collective', participants=[0], payload_bytes=4,
                       group='dp', operation='all_reduce', rendezvous='reduce:{microbatch}',
                       resources=['network:device0']),
            'wait': dict(native_wait=True, zero_cost_semantic=True)}}}
        workload = lower_plan(plan, HARDWARE, config)
        self.assertEqual(simulate(*workload.arguments())['seconds'], 2)
        collectives = [n for n in workload.nodes if n.kind == 'collective']
        self.assertEqual({n.attrs['rendezvous'] for n in collectives}, {'reduce:0', 'reduce:1'})
        config['node_attrs']['joint']['op']['rendezvous'] = 'incorrectly-shared'
        with self.assertRaisesRegex(ValueError, 'participant'):
            lower_plan(plan, HARDWARE, config)

    def test_async_prefetch_and_matching_wait(self):
        plan, config = fixture()
        plan['world_size'] = 1
        plan['schedule'].update(stage_to_rank={'0': 0}, rank_actions={'0': [
            action(0, 'UNSHARD', None), action(0, 'F'), action(0, 'B'), action(0, 'RESHARD', None)]})
        plan['stages'] = {'0': plan['stages']['0']}
        plan['graphs'] = {k: v for k, v in plan['graphs'].items() if k.startswith('0.')}
        plan['graphs']['unshard'] = graph(0, 'unshard')
        plan['stages']['0']['graphs']['unshard'] = 'unshard'
        config = {'node_attrs': {gid: {'op': {'flops': 1}} for gid in plan['graphs']}}
        workload = lower_plan(plan, HARDWARE, config)
        result = simulate(*workload.arguments())
        times = {n.name: result['timing'][str(n.id)] for n in workload.nodes}
        self.assertGreaterEqual(times['0.fw/op'][0], times['unshard/op'][1])
        self.assertGreaterEqual(times['0RESHARD'][0], times['0.full_bw/op'][1])

    def test_native_torch_schedule_extraction_and_lowering(self):
        try:
            import torch
            from torch.distributed.pipelining import schedules as native
            from torch.fx.experimental.proxy_tensor import make_fx
        except ImportError:
            self.skipTest('native PyTorch pipelining unavailable')
        if not hasattr(native, '_add_wait_send'):
            self.skipTest('native schedule version does not expose explicit send waits')
        from titan_sim.native_plan import extract_plan
        # Ask the actual upstream native scheduling algorithm for every rank's
        # action list; fake stages provide topology only, not an alternate model.
        schedule = native.ScheduleLoopedBFS.__new__(native.ScheduleLoopedBFS)
        schedule._stages = [SimpleNamespace(stage_index=0, stage_index_to_group_rank={0: 0, 1: 1})]
        schedule._n_microbatches = 2
        schedule.pp_group_size = 2
        schedule.pipeline_order = {rank: schedule._calculate_single_rank_operations(rank) for rank in range(2)}
        schedule.pipeline_order_with_comms = native._add_wait_send(native._add_send_recv(
            schedule.pipeline_order, lambda stage: stage, 2))
        gm = make_fx(lambda x: x * 2)(torch.ones(1))
        bundles = {stage: SimpleNamespace(modules=SimpleNamespace(fw=gm, full_bw=gm), compiled=False,
                                         meta=NativeMetaFixture()) for stage in range(2)}
        plan = extract_plan(schedule, bundles, provenance={'torch_version': torch.__version__})
        config = {'node_attrs': {gid: {n['name']: {'flops': 1} for n in g['nodes'] if n['op'] == 'call_function'}
                                  for gid, g in plan['graphs'].items()}}
        workload = lower_plan(plan, HARDWARE, config)
        result = simulate(*workload.arguments())
        self.assertEqual(len(workload.manifest['native_plan']['graph_invocations']), 8)
        self.assertEqual(len([n for n in workload.nodes if n.kind in ('send', 'recv')]), 8)
        self.assertGreater(result['seconds'], 0)


if __name__ == '__main__':
    unittest.main()
