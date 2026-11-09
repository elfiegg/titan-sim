import copy
import unittest

from titan_sim.case_binding import bind_cases
from titan_sim.core import Node, Storage
from titan_sim.identity import workload_fingerprint
from titan_sim.operator_cases import operator_case_workload
from titan_sim.workload import Workload


def source(rank=0, world=1):
    def tensor(shape, stride, storage):
        return dict(shape=shape, stride=stride, storage_id=storage, storage_offset=0,
                    dtype='torch.bfloat16', layout='torch.strided', device='cuda:0')
    args=[tensor([2,3],[3,1],'a'),tensor([3,4],[4,1],'b')]
    output=tensor([2,4],[4,1],'c')
    nodes=[Node(5,'aten.mm.default',rank,attrs=dict(operator='aten.mm.default',status='returned',
        arguments=dict(args=args,kwargs={}), outputs=output, cost_key='old-dispatch-key')),
        Node(8,'unknown',rank,deps=[5],attrs=dict(status='returned'))]
    return Workload(nodes,[Storage('c',rank,16,5,[8])],dict(ranks=[rank],world_size=world,
        capture_status='complete',provenance={'recipe_args':['unchanged']},
        unresolved=['Unknown CUDA dependencies and physical allocations.']),dict(id='fixture',capacity_bytes=1024))


class CaseBindingTests(unittest.TestCase):
    def live_fixture(self):
        from titan_sim.native_runner import native_cost_key
        original=source()
        contract={'args':[],'kwargs':{}}
        runtime={'grad_enabled':True,'allow_tf32':False}
        original.nodes[0].attrs.update(native_callable_boundary=True,native_callable_index=0,
                                      native_region_member_node_ids=[5,8])
        original.manifest['retained_native_boundaries']=[dict(name='fixture.native',
            reference_source_sha256='b'*64,argument_contract=contract,source_occurrences=[[5,8]],
            source_runtime_identities=[runtime])]
        original.manifest['memory']={'scope':'schedule','storage_evidence':'exact'}
        implementation=dict(name='fixture.native',native_callable_index=0,
            reference_source_sha256='b'*64,argument_contract=contract,registry_token='fixture-registry')
        context=dict(hardware_id='fixture',source_workload_sha256=workload_fingerprint(original),
                     source_node_ids=[5,8],runtime_identity=runtime)
        binding=dict(kind='live_native_callable',implementation=implementation,context=context)
        measurement=Workload([Node(0,'native',0,attrs=dict(cost_key=native_cost_key(implementation,context),
            native_measurement=binding))],[],dict(scope='isolated live native callable measurement',
            ranks=[0],world_size=1,registry_token='fixture-registry'),copy.deepcopy(original.hardware))
        return original,measurement

    def test_observed_native_region_charged_once_and_storage_not_promoted(self):
        original,measurement=self.live_fixture()
        bound=bind_cases([original],[measurement])
        self.assertEqual(len(bound.nodes),1)
        self.assertEqual(bound.nodes[0].attrs['cost_key'],measurement.nodes[0].attrs['cost_key'])
        self.assertEqual(bound.storage[0].producer,bound.nodes[0].id)
        self.assertEqual(bound.storage[0].consumers,[bound.nodes[0].id])
        self.assertEqual(bound.manifest['memory']['storage_evidence'],'unverified')
        self.assertEqual(bound.manifest['case_binding']['bound'][0]['source_node_ids'],[5,8])

    def test_native_region_cannot_hide_intermediate_external_use(self):
        from titan_sim.native_runner import native_cost_key
        original,measurement=self.live_fixture()
        original.nodes.append(Node(9,'external',0,deps=[5]))
        binding=measurement.nodes[0].attrs['native_measurement']
        binding['context']['source_workload_sha256']=workload_fingerprint(original)
        measurement.nodes[0].attrs['cost_key']=native_cost_key(binding['implementation'],binding['context'])
        with self.assertRaisesRegex(ValueError,'intermediate'):
            bind_cases([original],[measurement])

    def test_native_region_preserves_physical_device_resource(self):
        from titan_sim.native_runner import native_cost_key
        original,measurement=self.live_fixture()
        original.manifest['rank_placement']={'0':'shared-device'}
        binding=measurement.nodes[0].attrs['native_measurement']
        binding['context']['source_workload_sha256']=workload_fingerprint(original)
        measurement.nodes[0].attrs['cost_key']=native_cost_key(binding['implementation'],binding['context'])
        bound=bind_cases([original],[measurement])
        self.assertEqual(bound.nodes[0].attrs['resources'],['gpu:shared-device'])

    def test_replay_math_context_must_match_source_observation(self):
        from titan_sim.native_runner import native_cost_key
        original,measurement=self.live_fixture()
        binding=measurement.nodes[0].attrs['native_measurement']
        binding['context']['runtime_identity']={'grad_enabled':False,'allow_tf32':True}
        measurement.nodes[0].attrs['cost_key']=native_cost_key(binding['implementation'],binding['context'])
        with self.assertRaisesRegex(ValueError,'runtime/math'):
            bind_cases([original],[measurement])

    def test_live_registration_without_observed_boundary_is_rejected(self):
        from titan_sim.native_runner import native_cost_key
        original,measurement=self.live_fixture()
        binding=measurement.nodes[0].attrs['native_measurement']
        binding['implementation']['native_callable_index']=1
        measurement.nodes[0].attrs['cost_key']=native_cost_key(binding['implementation'],binding['context'])
        with self.assertRaisesRegex(ValueError,'observed source'):
            bind_cases([original],[measurement])

    def test_custom_dispatch_binding_requires_exact_schema_and_local_contract(self):
        from titan_sim.native_runner import native_cost_key
        original,measurement=self.live_fixture()
        original.nodes=original.nodes[:1]
        original.storage=[]
        node=original.nodes[0]
        node.name='custom.local.default'
        node.attrs.update(operator=node.name,operator_schema='custom::local(Tensor x) -> Tensor',
            native_boundary_kind='custom_dispatch',native_region_member_node_ids=[5])
        record=original.manifest['retained_native_boundaries'][0]
        record.update(name=node.name,boundary_kind='custom_dispatch',operator_schema=node.attrs['operator_schema'],
            source_dispatch_contexts=[{'include':'fixture','exclude':'autograd'}],source_occurrences=[[5]],
            input_output_contract={'args':[],'output':'fixture'})
        binding=measurement.nodes[0].attrs['native_measurement']
        binding['implementation'].update(name=node.name,boundary_kind='custom_dispatch',
            operator_schema=record['operator_schema'],dispatch_context=record['source_dispatch_contexts'][0],
            observed_occurrence=0,state_contract={'communication':'none'})
        binding['context'].update(source_workload_sha256=workload_fingerprint(original),source_node_ids=[5])
        measurement.nodes[0].attrs['cost_key']=native_cost_key(binding['implementation'],binding['context'])
        self.assertEqual(len(bind_cases([original],[measurement]).manifest['case_binding']['bound']),1)
        for field,value in [('operator_schema','different'),('dispatch_context',{}),('state_contract',{})]:
            bad=copy.deepcopy(measurement)
            descriptor=bad.nodes[0].attrs['native_measurement']
            descriptor['implementation'][field]=value
            bad.nodes[0].attrs['cost_key']=native_cost_key(descriptor['implementation'],descriptor['context'])
            with self.assertRaises(ValueError):bind_cases([original],[bad])

    def test_later_native_occurrence_uses_its_own_contract(self):
        from titan_sim.native_runner import native_cost_key
        original,measurement=self.live_fixture()
        later=Node(20,'later native',0,deps=[8],attrs=dict(status='returned',native_callable_boundary=True,
            native_callable_index=0,native_region_member_node_ids=[20]))
        original.nodes.append(later)
        record=original.manifest['retained_native_boundaries'][0]
        contract={'args':['different specialization'],'kwargs':{}}
        record['source_occurrences'].append([20])
        record['source_argument_contracts']=[record['argument_contract'],contract]
        record['source_runtime_identities'].append(record['source_runtime_identities'][0])
        binding=measurement.nodes[0].attrs['native_measurement']
        binding['implementation'].update(observed_occurrence=1,argument_contract=contract)
        binding['context'].update(source_node_ids=[20],source_workload_sha256=workload_fingerprint(original))
        measurement.nodes[0].attrs['cost_key']=native_cost_key(binding['implementation'],binding['context'])
        bound=bind_cases([original],[measurement])
        self.assertEqual(bound.manifest['case_binding']['bound'][0]['source_node_ids'],[20])
        binding['implementation']['argument_contract']=record['argument_contract']
        measurement.nodes[0].attrs['cost_key']=native_cost_key(binding['implementation'],binding['context'])
        with self.assertRaisesRegex(ValueError,'arguments differ'):bind_cases([original],[measurement])

    def communication_sources(self, completion=True):
        def tensor(count, sid):
            return dict(shape=[count],stride=[1],storage_id=sid,storage_offset=0,
                        dtype='torch.float32',layout='torch.strided',device='cuda:0')
        result=[]
        for rank in range(2):
            record=dict(operator='c10d._allgather_base_.default',operator_schema='fixture',
                operation='all_gather_into_tensor',participants=[0,1],group_size=2,
                backend='nccl',group_name='g',reduce_op=None,async_op=True,timeout=-1,
                runtime_context={},evidence='native_dispatch_and_resolved_process_group',
                tensors={'input_tensor':tensor(4,'a'),'output_tensor':tensor(8,'b')})
            attrs=dict(communication=record,status='returned',participants=[0,1],
                       rendezvous='g:0',payload_bytes=16,cost_key='original',
                       communication_completion='device' if completion else 'issue')
            result.append(Workload([Node(0,record['operator'],rank,
                kind='collective' if completion else 'compute',attrs=attrs)],[],
                dict(ranks=[rank],world_size=2,capture_status='complete',provenance={}),
                dict(id='fixture',capacity_bytes=1024)))
        return result

    def test_collective_binding_requires_topology_and_completion_scope(self):
        from titan_sim.communication_cases import communication_case_workload
        topology={'placement':'declared two devices'}
        originals=self.communication_sources()
        cases=communication_case_workload(originals[0],peer_workloads=originals[1:],topology=topology)
        skipped=bind_cases(originals,[cases])
        self.assertEqual(len(skipped.manifest['case_binding']['bound']),0)
        self.assertEqual(len(skipped.manifest['case_binding']['skipped']),2)
        bound=bind_cases(originals,[cases],{'communication_topology':topology})
        self.assertEqual(len(bound.manifest['case_binding']['bound']),2)
        self.assertEqual(bound.nodes[0].attrs['rendezvous'],'g:0')
        self.assertEqual(bound.nodes[0].attrs['cost_key'],bound.nodes[1].attrs['cost_key'])
        originals=self.communication_sources(completion=False)
        cases=communication_case_workload(originals[0],peer_workloads=originals[1:],topology=topology)
        skipped=bind_cases(originals,[cases],{'communication_topology':topology})
        self.assertEqual(len(skipped.manifest['case_binding']['bound']),0)
        self.assertIn('completion',skipped.manifest['case_binding']['skipped'][0]['reason'])

    def test_recovered_cost_attaches_to_completion_not_issue(self):
        from tests.test_schedule_recovery import rank_source
        from titan_sim.communication_cases import communication_case_workload
        originals=[rank_source(rank) for rank in range(2)]
        topology={'fixture':'two devices'}
        measurement=communication_case_workload(originals[0],peer_workloads=originals[1:],topology=topology)
        bound=bind_cases(originals,[measurement],{'recover_schedule':True,'communication_topology':topology})
        self.assertEqual(len(bound.manifest['case_binding']['bound']),2)
        for node in bound.nodes:
            if node.attrs.get('source_node_id') == 3:
                if node.kind == 'collective':
                    self.assertEqual(node.attrs['communication_completion'],'device')
                    self.assertIn('measurement_binding',node.attrs)
                else:
                    self.assertTrue(node.attrs['zero_cost_semantic'])
                    self.assertNotIn('cost_key',node.attrs)

    def test_compute_binding_preserves_source_dependencies_storage_and_gaps(self):
        original=source()
        before=workload_fingerprint(original)
        cases=operator_case_workload(original)
        bound=bind_cases([original],[cases])
        self.assertEqual(workload_fingerprint(original),before)
        self.assertEqual(bound.nodes[0].attrs['cost_key'],cases.nodes[0].attrs['cost_key'])
        self.assertEqual(bound.nodes[1].deps,[bound.nodes[0].id])
        self.assertEqual(bound.storage[0].producer,bound.nodes[0].id)
        self.assertEqual(bound.storage[0].consumers,[bound.nodes[1].id])
        self.assertIn('Unknown CUDA dependencies and physical allocations.',bound.manifest['unresolved'])
        report=bound.manifest['case_binding']
        self.assertEqual(len(report['bound']),1)
        self.assertEqual(report['unbound'][0]['source_node_id'],8)
        self.assertEqual(report['bound'][0]['source_cost_key'],'old-dispatch-key')

    def test_stale_source_even_unrelated_change_is_rejected(self):
        original=source()
        cases=operator_case_workload(original)
        original.nodes[1].name='changed'
        with self.assertRaisesRegex(ValueError,'source.*(checksum|fingerprint)'):
            bind_cases([original],[cases])

    def test_tampered_measurement_descriptor_is_rejected(self):
        original=source()
        cases=operator_case_workload(original)
        cases.nodes[0].attrs['native_measurement']['context']['cuda_graph']=True
        with self.assertRaises(ValueError):
            bind_cases([original],[cases])

    def test_duplicate_measurement_ownership_is_rejected(self):
        original=source()
        cases=operator_case_workload(original)
        with self.assertRaisesRegex(ValueError,'(duplicate|already bound)'):
            bind_cases([original],[cases,cases])

    def test_blocked_source_cannot_be_bound(self):
        original=source()
        original.manifest['capture_status']='blocked'
        cases=operator_case_workload(original)
        with self.assertRaisesRegex(ValueError,'complete'):
            bind_cases([original],[cases])

    def test_two_rank_sources_require_individual_verified_occurrences(self):
        originals=[source(0,2),source(1,2)]
        cases=operator_case_workload(originals[0])
        bound=bind_cases(originals,[cases])
        self.assertEqual(len(bound.manifest['case_binding']['bound']),1)
        self.assertTrue(any(n['rank']==1 for n in bound.manifest['case_binding']['unbound']))

    def test_unselected_occurrence_cannot_be_injected(self):
        original=source()
        cases=operator_case_workload(original)
        cases.nodes[0].attrs['operator_case']['occurrences'].append(dict(rank=0,node_id=8))
        with self.assertRaises(ValueError):
            bind_cases([original],[cases])

    def test_source_fingerprint_is_order_independent_for_json_objects(self):
        a=source()
        b=copy.deepcopy(a)
        b.hardware=dict(reversed(list(b.hardware.items())))
        self.assertEqual(workload_fingerprint(a),workload_fingerprint(b))
