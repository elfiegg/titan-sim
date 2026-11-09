import copy
import unittest

from titan_sim.core import Node, Storage, simulate
from titan_sim.identity import digest
from titan_sim.memory import ALLOCATION_CATEGORIES, COVERAGE_CATEGORIES, assess_memory
from titan_sim.memory_evidence import (build_evidence_bundle, bind_memory_evidence,
    inspect_native_state, observation_from_report)
from titan_sim.workload import Workload


class MemoryEvidenceTests(unittest.TestCase):
    def fixture(self, *, recipe=False, second_device=False, unverified=False):
        workload = Workload([Node(0, 'dispatch', 0), Node(1, 'completion', 0), Node(2, 'other', 0)],
            [Storage('parameter', 0, 60, 0, [0])],
            {'ranks': [0], 'rank_placement': {'0': 'a'}, 'memory': {
                'scope': 'recipe' if recipe else 'schedule', 'storage_evidence': 'unverified' if unverified else 'exact'}},
            {'id': 'synthetic', 'devices': {'a': {'capacity_bytes': 100}}, 'reserve_fraction': .1})
        if second_device:
            workload.manifest['ranks'].append(1)
            workload.manifest['rank_placement']['1'] = 'b'
            workload.hardware['devices']['b'] = {'capacity_bytes': 100}
        phases = ['schedule', 'initialization', 'checkpoint_loading'] if recipe else ['schedule']
        context = dict(source={'commit': 'pinned'}, target={'hardware_id': 'synthetic'},
                       runtime={'torch': 'pinned'}, placement=copy.deepcopy(workload.manifest['rank_placement']),
                       phases=phases, implementation={'id': 'native'})
        return workload, context, {'0': (0, 1), '1': (1, 3), '2': (2, 4)}

    def bundle(self, workload, context, *, allocations=(), coverage=None, observations=(), phase_bounds=()):
        if coverage is None:
            coverage = [dict(device=device, phase=phase, category=category,
                complete=True, contract_id='contract', evidence_ref='evidence', statement='complete synthetic scope')
                for device in set(context['placement'].values()) for phase in context['phases']
                for category in COVERAGE_CATEGORIES]
        return build_evidence_bundle(workload, applicability=context, allocations=allocations,
            coverage=coverage, observations=observations, phase_bounds=phase_bounds,
            evidence_refs=[dict(id='evidence', sha256='a' * 64)],
            producer_contracts=[dict(id='contract', producer='test fixture', statement='synthetic allocation contract',
                                    evidence_ref='evidence', applicability_sha256=digest(context))])

    def allocation(self, **kwargs):
        result = dict(allocation_id='backing', generation=0, device='a', phase='schedule',
            category='storage', kind='exact', lower_bytes=60, upper_bytes=60,
            creation_node_id=0, completion_node_ids=[1], persistent=False,
            storage_ids=['parameter'], contract_id='contract', evidence_ref='evidence')
        result.update(kwargs)
        return result

    def assess(self, workload, context, timing, bundle):
        bound = bind_memory_evidence(workload, bundle, applicability=context)
        return assess_memory(bound.nodes, bound.storage, timing, bound.hardware, bound.manifest)

    def test_checksum_and_stale_workload_rejected(self):
        workload, context, timing = self.fixture()
        bundle = self.bundle(workload, context)
        changed = copy.deepcopy(bundle)
        changed['observations'].append({})
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.assess(workload, context, timing, changed)
        workload.nodes[0].name = 'changed'
        with self.assertRaisesRegex(ValueError, 'workload hash'):
            self.assess(workload, context, timing, bundle)

    def test_revalidates_binding_after_hardware_mutation(self):
        workload, context, timing = self.fixture()
        bound = bind_memory_evidence(workload, self.bundle(workload, context), applicability=context)
        bound.hardware['devices']['a']['capacity_bytes'] = 999
        with self.assertRaisesRegex(ValueError, 'workload hash'):
            assess_memory(bound.nodes, bound.storage, timing, bound.hardware, bound.manifest)

    def test_all_applicability_dimensions_match(self):
        workload, context, timing = self.fixture()
        bundle = self.bundle(workload, context)
        for key in ('source', 'target', 'runtime', 'implementation'):
            changed = copy.deepcopy(context)
            changed[key] = {'changed': True}
            with self.assertRaisesRegex(ValueError, 'applicability mismatch'):
                self.assess(workload, changed, timing, bundle)

    def test_partial_device_coverage_localizes_upper_bound(self):
        workload, context, timing = self.fixture(second_device=True)
        bundle = self.bundle(workload, context)
        bundle['coverage'] = [row for row in bundle['coverage'] if row['device'] == 'a']
        bundle['sha256'] = digest({k: v for k, v in bundle.items() if k != 'sha256'})
        result = self.assess(workload, context, timing, bundle)
        self.assertEqual(result['per_device']['a']['peak_upper_bytes'], 60)
        self.assertIsNone(result['per_device']['b']['peak_upper_bytes'])
        self.assertFalse(result['compatibility_coverage']['storage'])
        self.assertEqual(result['memory_verdict'], 'inconclusive')

    def test_missing_phase_coverage_not_global_boolean(self):
        workload, context, timing = self.fixture(recipe=True)
        bundle = self.bundle(workload, context)
        bundle['coverage'] = [row for row in bundle['coverage'] if row['phase'] != 'checkpoint_loading']
        bundle['sha256'] = digest({k: v for k, v in bundle.items() if k != 'sha256'})
        result = self.assess(workload, context, timing, bundle)
        self.assertFalse(result['coverage_audit']['a']['storage']['checkpoint_loading'])
        self.assertIsNone(result['per_device']['a']['peak_upper_bytes'])

    def test_observation_does_not_promote_projected_storage(self):
        workload, context, timing = self.fixture(unverified=True)
        observation = dict(kind='observation', physical_peak_upper_bytes=None,
                           process_allocator_peak={'allocated_bytes': 9999, 'reserved_bytes': 9999})
        result = self.assess(workload, context, timing, self.bundle(workload, context, observations=[observation]))
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 0)
        self.assertIsNone(result['per_device']['a']['peak_upper_bytes'])

    def test_bound_requires_producer_contract_and_evidence_reference(self):
        workload, context, timing = self.fixture()
        for change in ({'contract_id': 'missing'}, {'evidence_ref': 'missing'}, {'kind': 'observation'}):
            with self.assertRaises(ValueError):
                self.bundle(workload, context, allocations=[self.allocation(**change)])

    def test_aliases_and_native_completion_are_counted_once(self):
        workload, context, timing = self.fixture(unverified=True)
        workload.storage.append(Storage('view', 0, 60, 0, [0]))
        bundle = self.bundle(workload, context, allocations=[self.allocation(storage_ids=['parameter', 'view'])])
        result = self.assess(workload, context, timing, bundle)
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 60)
        self.assertEqual(result['per_device']['a']['peak_upper_bytes'], 60)
        self.assertEqual(workload.storage[0].consumers, [0])

    def test_async_completion_preserves_overlap(self):
        workload, context, timing = self.fixture()
        workload.storage.append(Storage('workspace', 0, 50, 2))
        result = self.assess(workload, context, timing, self.bundle(workload, context, allocations=[self.allocation()]))
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 110)
        self.assertEqual(result['per_device']['a']['physical_capacity_outcome'], 'oom_lower_bound')

    def test_unknown_completion_and_null_size_stay_unknown(self):
        workload, context, timing = self.fixture(unverified=True)
        result = self.assess(workload, context, timing, self.bundle(workload, context,
            allocations=[self.allocation(completion_node_ids=[])]))
        self.assertIsNone(result['per_device']['a']['peak_upper_bytes'])
        unknown = self.allocation(storage_ids=[], kind='conditional', lower_bytes=None, upper_bytes=None,
                                  conditions=['unresolved implementation contract'])
        result = self.assess(workload, context, timing, self.bundle(workload, context, allocations=[unknown]))
        self.assertIsNone(result['per_device']['a']['peak_upper_bytes'])
        self.assertEqual(result['memory_bound_kind'], 'conditional')

    def test_conditional_bounds_cannot_prove_unconditional_fit_or_oom(self):
        workload, context, timing = self.fixture()
        for size, forecast in ((20, 'estimated_fit'), (200, 'estimated_not_fit')):
            row = self.allocation(storage_ids=[], category='workspaces', kind='conditional',
                lower_bytes=size, upper_bytes=size, conditions=['native implementation predicate holds'])
            result = self.assess(workload, context, timing, self.bundle(workload, context, allocations=[row]))
            self.assertEqual(result['memory_verdict'], 'inconclusive')
            self.assertEqual(result['conditional_memory_verdict'], forecast)
            self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 60)
            self.assertIsNone(result['per_device']['a']['peak_upper_bytes'])
            self.assertEqual(result['per_device']['a']['physical_capacity_outcome'], 'unknown')
            self.assertEqual(result['conditional_per_device']['a']['peak_lower_bytes'], 60 + size)

    def test_unknown_completion_does_not_prove_overlap_with_later_allocation(self):
        workload, context, timing = self.fixture()
        workload.storage.append(Storage('later', 0, 50, 2))
        result = self.assess(workload, context, timing, self.bundle(workload, context,
            allocations=[self.allocation(completion_node_ids=[])]))
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 60)
        self.assertIsNone(result['per_device']['a']['peak_upper_bytes'])
        self.assertEqual(result['memory_verdict'], 'inconclusive')

    def test_prior_global_gaps_are_never_removed(self):
        workload, context, timing = self.fixture()
        workload.manifest['unresolved'] = ['schedule uncertainty']
        workload.manifest['memory']['unresolved'] = ['unknown optimizer state']
        result = self.assess(workload, context, timing, self.bundle(workload, context, allocations=[self.allocation()]))
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        self.assertTrue(any('schedule' in issue or 'execution' in issue for issue in result['memory_unresolved']))
        self.assertTrue(any('optimizer' in issue for issue in result['memory_unresolved']))

    def test_budget_overflow_is_not_physical_capacity_oom(self):
        workload, context, timing = self.fixture()
        workload.storage[0].size = 95
        result = self.assess(workload, context, timing, self.bundle(workload, context))
        self.assertEqual(result['memory_verdict'], 'estimated_not_fit')
        self.assertEqual(result['per_device']['a']['budget_outcome'], 'overflow')
        self.assertEqual(result['per_device']['a']['physical_capacity_outcome'], 'within_capacity')

    def test_reused_generations_are_distinct_not_pointer_aliases(self):
        workload, context, timing = self.fixture()
        records = [self.allocation(storage_ids=[], generation=index, creation_node_id=index,
                                   completion_node_ids=[2]) for index in (0, 1)]
        result = self.assess(workload, context, timing, self.bundle(workload, context, allocations=records))
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 120)
        with self.assertRaisesRegex(ValueError, 'generations'):
            self.bundle(workload, context, allocations=[records[0], records[0]])

    def test_lifecycle_bounds_maxed_not_added(self):
        workload, context, timing = self.fixture(recipe=True)
        row = dict(device='a', phase='initialization', scope='whole_device', kind='exact',
                   lower_bytes=75, upper_bytes=80, contract_id='contract', evidence_ref='evidence')
        result = self.assess(workload, context, timing, self.bundle(workload, context, phase_bounds=[row]))
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 75)
        self.assertEqual(result['per_device']['a']['peak_upper_bytes'], 80)

    def test_zero_duration_completion_keeps_boundary_upper_envelope(self):
        workload, context, timing = self.fixture()
        workload.storage.append(Storage('next', 0, 50, 2))
        timing['1'] = (2, 2)
        result = self.assess(workload, context, timing, self.bundle(workload, context,
            allocations=[self.allocation()]))
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 60)
        self.assertEqual(result['per_device']['a']['peak_upper_bytes'], 110)
        self.assertEqual(result['memory_verdict'], 'inconclusive')

    def test_existing_external_allocation_identity_is_not_counted_twice(self):
        workload, context, timing = self.fixture()
        workload.manifest['memory']['allocations'] = [dict(id='workspace', device='a',
            category='workspaces', bytes=10)]
        with self.assertRaisesRegex(ValueError, 'conflicts with existing memory allocation'):
            self.bundle(workload, context, allocations=[self.allocation(allocation_id='workspace',
                storage_ids=[], category='workspaces', lower_bytes=10, upper_bytes=10)])

    def test_existing_alias_cycle_is_rejected_without_looping(self):
        workload, context, timing = self.fixture()
        workload.storage.append(Storage('view', 0, 60, 0))
        workload.manifest['memory']['storage_aliases'] = {'view': 'parameter', 'parameter': 'view'}
        with self.assertRaisesRegex(ValueError, 'alias cycle'):
            self.bundle(workload, context, allocations=[self.allocation()])

    def test_existing_aliases_cannot_claim_separate_generations(self):
        workload, context, timing = self.fixture()
        workload.storage.append(Storage('view', 0, 60, 0))
        workload.manifest['memory']['storage_aliases'] = {'view': 'parameter'}
        with self.assertRaisesRegex(ValueError, 'separate allocation generations'):
            self.bundle(workload, context, allocations=[self.allocation(),
                self.allocation(generation=1, storage_ids=['view'])])
        result = self.assess(workload, context, timing, self.bundle(workload, context,
            allocations=[self.allocation(storage_ids=['view', 'parameter'])]))
        self.assertEqual(result['per_device']['a']['peak_upper_bytes'], 60)

    def test_narrower_proof_does_not_promote_longer_projection(self):
        workload, context, timing = self.fixture(unverified=True)
        workload.storage[0].persistent = True
        result = self.assess(workload, context, timing, self.bundle(workload, context,
            allocations=[self.allocation(completion_node_ids=[0])]))
        self.assertIsNone(result['per_device']['a']['peak_upper_bytes'])
        self.assertEqual(result['per_device']['a']['peak_lower_bytes'], 60)

    def test_existing_simulator_consumes_binding(self):
        workload, context, timing = self.fixture()
        for node in workload.nodes:
            node.attrs['legacy_seconds'] = 1
        workload.nodes[2].deps = [1]
        bound = bind_memory_evidence(workload, self.bundle(workload, context, allocations=[self.allocation()]), applicability=context)
        result = simulate(*bound.arguments())
        self.assertIn('coverage_audit', result['physical_memory'])

    def test_report_adapter_keeps_observation_and_inventory(self):
        report = dict(schema='titan_sim.memory_observation.v1', provenance={'source': 'native'},
            runtime={'torch': 'pinned'}, phases=[{'name': 'capture'}],
            samples=[{'state_inventory': {'category_bytes': {'optimizer_state': 100}}}],
            process_allocator_peak={'reserved_bytes': 1000})
        report['provenance_sha256'] = digest(report['provenance'])
        observation = observation_from_report(report)
        self.assertEqual(observation['samples'], report['samples'])
        self.assertIsNone(observation['physical_peak_upper_bytes'])
        self.assertFalse(observation['supports_unseen_configuration_fit'])


class NativeInventoryTests(unittest.TestCase):
    def test_original_optimizer_state_and_aliases(self):
        try:
            import torch
        except ImportError:
            self.skipTest('native CPU torch unavailable')
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters())
        before = inspect_native_state(models=[model], optimizers=[optimizer], torch_module=torch)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        optimizer.state[next(model.parameters())]['alias'] = next(model.parameters()).detach()
        after = inspect_native_state(models=[model], optimizers=[optimizer], torch_module=torch)
        self.assertGreater(after['unique_storage_bytes'], before['unique_storage_bytes'])
        self.assertTrue(any('optimizer_state' in row['categories'] for row in after['allocations']))
        self.assertTrue(any('parameters' in row['categories'] and 'optimizer_state' in row['categories'] for row in after['allocations']))
        self.assertEqual(after['unresolved'], [])
        self.assertIsNone(after['physical_peak_upper_bytes'])


if __name__ == '__main__':
    unittest.main()
