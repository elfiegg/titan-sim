import copy
import gc
import types
import unittest
import weakref

from titan_sim.allocation_capture import NativeAllocationCapture, SCHEMA, inspect_state
from titan_sim.allocation_contracts import analyze_allocation_capture, produce_allocation_evidence
from titan_sim.core import Node, Storage
from titan_sim.identity import digest
from titan_sim.memory import assess_memory
from titan_sim.memory_evidence import bind_memory_evidence
from titan_sim.workload import Workload


def snapshot(events, live):
    segments = [dict(device=0, address=address, total_size=size, segment_pool_id=[0, 0],
                     blocks=[dict(address=address, size=size, requested_size=size, state=state)])
                for address, size, state in live]
    return dict(segments=segments, device_traces=[copy.deepcopy(events)])


def seal(capture):
    capture['sha256'] = digest({k: v for k, v in capture.items() if k != 'sha256'})
    return capture


class AllocationContractsTests(unittest.TestCase):
    def fixture(self):
        events = [dict(action='snapshot')]
        samples = []
        def sample(node, edge, live, state=()):
            samples.append(dict(label=str(len(samples)), phase='schedule', node_id=node, edge=edge,
                state=dict(allocations=list(state), unresolved=[]), snapshot=snapshot(events, live)))
        state = dict(id='state:0', address=1000, bytes=60, device='cuda:0',
                     aliases=[['model', '0', 'weight'], ['model', '0', 'weight_view']],
                     categories=['parameters', 'buffers'])
        sample(0, 'before', [(1000, 64, 'active_allocated')], [state])
        events.extend([dict(action='segment_alloc', addr=2000, size=80),
                       dict(action='alloc', addr=2000, size=80),
                       dict(action='free_requested', addr=1000, size=64), dict(action='snapshot')])
        sample(0, 'after', [(1000, 64, 'active_awaiting_free'), (2000, 80, 'active_allocated')])
        events.append(dict(action='snapshot'))
        sample(1, 'before', [(1000, 64, 'active_awaiting_free'), (2000, 80, 'active_allocated')])
        events.extend([dict(action='free_completed', addr=1000, size=64),
                       dict(action='segment_free', addr=1000, size=64),
                       dict(action='segment_alloc', addr=1000, size=96),
                       dict(action='alloc', addr=1000, size=96), dict(action='snapshot')])
        sample(1, 'after', [(1000, 96, 'active_allocated'), (2000, 80, 'active_allocated')])
        capture = seal(dict(schema=SCHEMA, rank=0, physical_device='gpu-a', device_index=0,
            history=dict(started_by_capture=True, max_entries=100), samples=samples, unresolved=[]))
        workload = Workload([Node(0, 'native_step', 0), Node(1, 'native_release', 0, deps=[0])], [],
            dict(ranks=[0], rank_placement={'0': 'gpu-a'}, memory=dict(scope='schedule')),
            dict(id='target', devices={'gpu-a': dict(capacity_bytes=10000)}, reserve_fraction=0))
        context = dict(source={'commit': 'pinned'}, target={'id': 'target'}, runtime={'version': 'pinned'},
                       placement={'0': 'gpu-a'}, phases=['schedule'], implementation={'id': 'original'})
        capture['provenance'] = {'applicability': copy.deepcopy(context)}
        seal(capture)
        return capture, workload, context

    def test_actual_free_completion_and_address_reuse(self):
        capture, _, _ = self.fixture()
        result = analyze_allocation_capture(capture)
        self.assertTrue(result['reconstruction_complete'], result['unresolved'])
        first, second, reused = result['allocations']
        self.assertEqual((first['allocation_id'], first['generation']), ('allocator:0', 0))
        self.assertEqual((reused['allocation_id'], reused['generation']), ('allocator:0', 1))
        self.assertEqual(first['free_requested_sample'], 1)
        self.assertEqual(first['free_completed_sample'], 3)
        self.assertEqual(first['completion_node_ids'], [1])
        self.assertEqual(second['creation_node_id'], 0)
        self.assertEqual(result['allocator_boundaries'][1]['active_bytes'], 144)
        self.assertEqual(result['observed_allocator_active_peak_bytes'], 176)
        self.assertIsNone(result['physical_peak_upper_bytes'])

    def test_owned_history_and_endpoint_reconciliation_required(self):
        for mutation, message in (
                (lambda r: r['history'].update(started_by_capture=False), 'settings are unknown'),
                (lambda r: r['history'].update(max_entries=3), 'ring truncation'),
                (lambda r: r['samples'][2]['snapshot']['device_traces'][0].pop(0), 'prefix'),
                (lambda r: r['samples'][1]['snapshot']['segments'].pop(), 'reserved bytes disagree')):
            capture, _, _ = self.fixture()
            mutation(capture)
            result = analyze_allocation_capture(seal(capture))
            self.assertFalse(result['reconstruction_complete'])
            self.assertTrue(any(message in text for text in result['unresolved']), result['unresolved'])
            self.assertIsNone(result['observed_allocator_active_peak_bytes'])

    def test_expandable_mappings_and_native_pending_free_state(self):
        capture, _, _ = self.fixture()
        # Synthetic minimal native schema: virtual reservation is deliberately
        # much larger than mapped physical bytes and must never be charged.
        def segment(expanded=False, pending=False):
            blocks = [dict(address=1000, size=64, state='active_allocated'),
                      dict(address=1064, size=64, state='inactive')]
            if expanded:
                blocks.extend([dict(address=1128, size=32,
                    state='active_pending_free' if pending else 'inactive'),
                    dict(address=1160, size=96, state='inactive')])
            return dict(device=0, address=1000, total_size=256 if expanded else 128,
                        is_expandable=True, expandable_reservation_size=1000000,
                        segment_pool_id=[0, 0], blocks=blocks)
        events = [dict(action='snapshot'), dict(action='segment_map', addr=1128, size=128),
                  dict(action='alloc', addr=1128, size=32),
                  dict(action='free_requested', addr=1128, size=32)]
        capture['samples'] = [dict(label='before', phase='schedule', node_id=0, edge='before',
            state=dict(allocations=[], unresolved=[]),
            snapshot=dict(segments=[segment()], device_traces=[[]])),
            dict(label='after', phase='schedule', node_id=0, edge='after',
            state=dict(allocations=[], unresolved=[]),
            snapshot=dict(segments=[segment(True, True)], device_traces=[events])),
            dict(label='terminal', phase='schedule', node_id=None, edge=None,
            state=dict(allocations=[], unresolved=[]), snapshot=dict(segments=[segment()],
            device_traces=[events + [dict(action='snapshot'),
                dict(action='free_completed', addr=1128, size=32),
                dict(action='segment_unmap', addr=1128, size=128)]]))]
        result = analyze_allocation_capture(seal(capture))
        self.assertTrue(result['reconstruction_complete'], result['unresolved'])
        self.assertEqual(result['observed_allocator_active_peak_bytes'], 96)
        self.assertEqual(result['allocator_boundaries'][1]['reserved_bytes'], 256)
        self.assertEqual(result['allocator_boundaries'][1]['active_bytes'], 96)
        self.assertEqual(result['allocations'][1]['free_requested_sample'], 1)
        self.assertEqual(result['allocations'][1]['free_completed_sample'], 2)
        capture['samples'][1]['snapshot']['device_traces'][0][1]['size'] = 127
        broken = analyze_allocation_capture(seal(capture))
        self.assertFalse(broken['reconstruction_complete'])
        self.assertTrue(any('reserved bytes disagree' in issue for issue in broken['unresolved']))

    def test_free_requested_cannot_permit_reuse(self):
        capture, _, _ = self.fixture()
        # A producer dropping free_completed cannot turn async retention into
        # address reuse even if the final snapshot happens to look plausible.
        for sample in capture['samples']:
            trace = sample['snapshot']['device_traces'][0]
            trace[:] = [event for event in trace if event['action'] != 'free_completed']
        result = analyze_allocation_capture(seal(capture))
        self.assertFalse(result['reconstruction_complete'])
        self.assertTrue(any('reused before free_completed' in text for text in result['unresolved']))

    def test_observed_peak_never_becomes_unconditional_fit(self):
        capture, workload, context = self.fixture()
        bundle = produce_allocation_evidence(workload, capture, applicability=context)
        self.assertTrue(bundle['allocations'])
        self.assertTrue(all(row['kind'] == 'conditional' for row in bundle['allocations']))
        self.assertFalse(any(row['complete'] for row in bundle['coverage']))
        bound = bind_memory_evidence(workload, bundle, applicability=context)
        result = assess_memory(bound.nodes, bound.storage, {'0': (0, 1), '1': (1, 2)}, bound.hardware, bound.manifest)
        self.assertEqual(result['memory_verdict'], 'inconclusive')
        self.assertEqual(result['per_device']['gpu-a']['peak_lower_bytes'], 0)
        self.assertIsNone(result['per_device']['gpu-a']['peak_upper_bytes'])
        self.assertEqual(result['conditional_per_device']['gpu-a']['modeled_peak_lower_bytes'], 0)

    def test_aliases_derive_from_native_state_with_rounding_separate(self):
        capture, workload, context = self.fixture()
        # Persistent native state is the supported equal-byte lifetime case:
        # present at entry, no allocator free, and still retained at exit.
        for sample in capture['samples']:
            trace = sample['snapshot']['device_traces'][0]
            trace[:] = [event for event in trace if not (event.get('addr') == 1000)]
            for segment in sample['snapshot']['segments']:
                if segment['address'] == 1000:
                    segment['total_size'] = 64
                    segment['blocks'][0].update(size=64, requested_size=64, state='active_allocated')
        seal(capture)
        workload.storage = [Storage('weight', 0, 60), Storage('view', 0, 60)]
        binding = dict(sample_index=0, state_id='state:0')
        bundle = produce_allocation_evidence(workload, capture, applicability=context,
                                             storage_bindings={'weight': binding, 'view': binding})
        first, rounding = bundle['allocations'][:2]
        self.assertEqual(first['storage_ids'], ['weight', 'view'])
        self.assertEqual(first['upper_bytes'], 60)
        self.assertEqual(rounding['upper_bytes'], 4)
        self.assertEqual(rounding['category'], 'allocator_slack')
        self.assertTrue(rounding['persistent'])
        self.assertEqual(rounding['lower_bytes'], 4)
        workload.storage[1].size = 61
        with self.assertRaisesRegex(ValueError, 'exact backing bytes'):
            produce_allocation_evidence(workload, capture, applicability=context,
                                        storage_bindings={'weight': binding, 'view': binding})

    def test_transient_mapped_storage_does_not_claim_widened_lower_overlap(self):
        capture, workload, context = self.fixture()
        workload.storage = [Storage('weight', 0, 60)]
        bundle = produce_allocation_evidence(workload, capture, applicability=context,
            storage_bindings={'weight': dict(sample_index=0, state_id='state:0')})
        self.assertEqual(bundle['allocations'], [])
        self.assertTrue(any('finer lifetime' in issue for issue in bundle['observations'][0]['unresolved']))

    def test_unmapped_projected_storage_is_not_added_twice(self):
        capture, workload, context = self.fixture()
        workload.storage = [Storage('projection', 0, 60)]
        bundle = produce_allocation_evidence(workload, capture, applicability=context)
        self.assertEqual(bundle['allocations'], [])
        self.assertTrue(any('identity correspondence' in issue for issue in bundle['observations'][0]['unresolved']))

    def test_preexisting_extra_allocation_is_not_added_twice(self):
        capture, workload, context = self.fixture()
        workload.manifest['memory']['allocations'] = [dict(id='existing', device='gpu-a',
            category='workspaces', lower_bytes=64, upper_bytes=64)]
        bundle = produce_allocation_evidence(workload, capture, applicability=context)
        self.assertEqual(bundle['allocations'], [])
        self.assertTrue(any('existing extra allocations' in issue for issue in bundle['observations'][0]['unresolved']))

    def test_unmapped_completion_stays_unknown(self):
        capture, workload, context = self.fixture()
        capture['samples'][3].update(edge=None, node_id=None)
        bundle = produce_allocation_evidence(workload, seal(capture), applicability=context)
        first = bundle['allocations'][0]
        self.assertFalse(first['persistent'])
        self.assertEqual(first['completion_node_ids'], [])

    def test_missing_transient_events_not_claimed_complete_from_snapshots(self):
        capture, workload, context = self.fixture()
        capture['history']['started_by_capture'] = False
        bundle = produce_allocation_evidence(workload, seal(capture), applicability=context)
        self.assertEqual(bundle['allocations'], [])

    def test_phase_device_and_checksum_fail_closed(self):
        capture, workload, context = self.fixture()
        capture['samples'][0]['phase'] = 'initialization'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            analyze_allocation_capture(capture)
        bundle = produce_allocation_evidence(workload, seal(capture), applicability=context)
        self.assertEqual(bundle['allocations'], [])
        capture['physical_device'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'placement'):
            produce_allocation_evidence(workload, seal(capture), applicability=context)

    def test_invalid_node_correlation_rejected(self):
        capture, workload, context = self.fixture()
        capture['samples'][0]['node_id'] = capture['samples'][1]['node_id'] = 99
        with self.assertRaisesRegex(ValueError, 'workload node'):
            produce_allocation_evidence(workload, seal(capture), applicability=context)

    def test_forecast_requires_exact_capture_applicability(self):
        capture, workload, context = self.fixture()
        for field in ('source', 'target', 'runtime', 'implementation'):
            altered = copy.deepcopy(context)
            altered[field] = {'id': 'other'}
            with self.assertRaisesRegex(ValueError, 'applicability mismatch'):
                produce_allocation_evidence(workload, capture, applicability=altered)


class CaptureTests(unittest.TestCase):
    def fake(self, traces=None):
        calls = []
        snapshots = []
        def get_snapshot():
            if calls:
                snapshots.append(dict(action='snapshot'))
            return dict(segments=[], device_traces=[copy.deepcopy(traces if traces is not None else snapshots)])
        memory = types.SimpleNamespace(_snapshot=get_snapshot,
            _record_memory_history=lambda **kwargs: calls.append(kwargs))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(memory=memory), __version__='test',
                                      version=types.SimpleNamespace(cuda='test'), is_tensor=lambda obj: False)
        return torch, calls

    def test_owned_history_bounded_and_existing_history_preserved(self):
        torch, calls = self.fake([dict(action='snapshot')])
        capture = NativeAllocationCapture(device_index=0, physical_device='device', torch_module=torch)
        with self.assertRaisesRegex(RuntimeError, 'existing'):
            capture.start_history(max_entries=100, dedicated_process=True)
        self.assertEqual(calls, [])
        torch, calls = self.fake()
        capture = NativeAllocationCapture(device_index=0, physical_device='device', torch_module=torch)
        with self.assertRaisesRegex(ValueError, 'dedicated'):
            capture.start_history(max_entries=100)
        with self.assertRaisesRegex(ValueError, 'max_entries'):
            capture.start_history(max_entries=0, dedicated_process=True)
        with capture.record_history(max_entries=100, dedicated_process=True):
            with capture.region(0):
                pass
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['enabled'], 'all')
        self.assertEqual(len(capture.report()['samples']), 4)
        self.assertTrue(analyze_allocation_capture(capture.report())['reconstruction_complete'])

    def test_native_failure_keeps_exit_evidence(self):
        torch, _ = self.fake()
        capture = NativeAllocationCapture(device_index=0, physical_device='device', torch_module=torch)
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            with capture.record_history(max_entries=100, dedicated_process=True):
                with capture.region(0):
                    raise RuntimeError('native failure')
        report = capture.report()
        self.assertEqual(report['samples'][-1]['label'], 'history:exit')
        self.assertFalse(analyze_allocation_capture(report)['reconstruction_complete'])

    def test_real_native_lazy_optimizer_aliases_and_no_retention(self):
        try:
            import torch
        except ImportError:
            self.skipTest('native torch unavailable')
        model = torch.nn.Linear(3, 2)
        model.register_buffer('weight_view', model.weight.view(-1))
        optimizer = torch.optim.Adam(model.parameters())
        before = inspect_state(models=[model], optimizers=[optimizer], torch_module=torch)
        self.assertEqual(len(before['allocations']), 2)
        self.assertFalse(any('optimizer_state' in row['categories'] for row in before['allocations']))
        self.assertEqual(len(before['allocations'][0]['aliases']), 2)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        after = inspect_state(models=[model], optimizers=[optimizer], torch_module=torch)
        self.assertGreater(after['unique_storage_bytes'], before['unique_storage_bytes'])
        self.assertTrue(any('optimizer_state' in row['categories'] for row in after['allocations']))
        self.assertEqual(after['unresolved'], [])
        ref = weakref.ref(model)
        del model, optimizer
        gc.collect()
        self.assertIsNone(ref())


if __name__ == '__main__':
    unittest.main()
