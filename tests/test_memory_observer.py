from copy import deepcopy
from types import SimpleNamespace
import unittest

from titan_sim.memory_observer import MemoryObserver, merge_observations


class Device:
    def __init__(self, kind, index=None):
        if isinstance(kind, Device):
            self.type, self.index = kind.type, kind.index
        else:
            parts = kind.split(':')
            self.type = parts[0]
            self.index = int(parts[1]) if len(parts) > 1 else index

    def __str__(self):
        return self.type+(':'+str(self.index) if self.index is not None else '')

    def __eq__(self, other):
        return str(self) == str(other)


class FakeCuda:
    def __init__(self):
        self.allocated, self.reserved, self.free, self.total = 20, 40, 900, 1000
        self.peak_allocated, self.peak_reserved = 999, 999
        self.memory = self

    def is_available(self): return True
    def current_device(self): return 0
    def synchronize(self, device): pass
    def memory_allocated(self, device): return self.allocated
    def memory_reserved(self, device): return self.reserved
    def max_memory_allocated(self, device): return self.peak_allocated
    def max_memory_reserved(self, device): return self.peak_reserved
    def mem_get_info(self, device): return self.free, self.total
    def get_allocator_backend(self): return 'native'
    def _record_memory_history(self): raise AssertionError('private history must not be called')
    def reset_peak_memory_stats(self, device=None):
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def allocate(self, allocated, reserved):
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

    def memory_snapshot(self):
        return [{'device': 0, 'total_size': 40, 'blocks': [{'state': 'active_allocated', 'size': 20},
                                                       {'state': 'inactive', 'size': 20}]},
                {'device': 1, 'total_size': 900, 'blocks': []}]


class Tensor:
    def __init__(self, pointer, size, device='cuda:0'):
        self.pointer, self.size, self.device, self.grad = pointer, size, Device(device), None

    def untyped_storage(self): return self
    def data_ptr(self): return self.pointer
    def nbytes(self): return self.size


def framework():
    return SimpleNamespace(cuda=FakeCuda(), __version__='pinned-test',
                           version=SimpleNamespace(cuda='test'), device=Device,
                           is_tensor=lambda value: isinstance(value, Tensor))


class MemoryObservationTests(unittest.TestCase):
    def make(self, **kwargs):
        self.torch = framework()
        return MemoryObserver(torch_module=self.torch,
            provenance={'recipe': {'unchanged': True}, 'runtime': {'image': 'pinned'}}, **kwargs)

    def test_peak_sees_transient_allocation_and_keeps_domains_separate(self):
        observer = self.make(snapshot=True)
        with observer:
            with observer.phase('capture'):
                self.torch.cuda.allocate(300, 500)
                self.torch.cuda.allocate(25, 45)
        report = observer.report()
        self.assertEqual(report['process_allocator_peak'], {'allocated_bytes': 300, 'reserved_bytes': 500})
        self.assertEqual(report['device_sampled_used_peak_bytes'], 100)
        self.assertEqual(report['samples'][-1]['process_allocator']['slack_bytes'], 20)
        self.assertIsNone(report['observed_configuration_bound']['physical_peak_upper_bytes'])
        self.assertFalse(report['observed_configuration_bound']['supports_unseen_configuration_fit'])
        self.assertEqual(report['lifecycle'], {'initialization': 'unknown', 'checkpoint_loading': 'unknown'})
        self.assertEqual(report['samples'][0]['allocator_snapshot']['segment_bytes'], 40)
        self.assertFalse(report['capabilities']['history_enabled'])

    def test_phases_reset_local_peaks_but_preserve_overall_max(self):
        observer = self.make()
        with observer:
            with observer.phase('initialization'):
                self.torch.cuda.allocate(100, 120)
            self.torch.cuda.allocate(20, 40)
            with observer.phase('capture'):
                self.torch.cuda.allocate(30, 50)
        report = observer.report()
        self.assertEqual(report['phases'][1]['process_peak_reserved_bytes'], 50)
        self.assertEqual(report['process_allocator_peak']['reserved_bytes'], 120)
        self.assertEqual(report['lifecycle']['initialization'], 'observed_phase')

    def test_exception_preserves_partial_evidence_and_propagates(self):
        observer = self.make()
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            with observer:
                with observer.phase('capture'):
                    self.torch.cuda.allocate(80, 100)
                    raise RuntimeError('native failure')
        report = observer.report()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['phases'][0]['status'], 'failed')
        self.assertEqual(report['process_allocator_peak']['reserved_bytes'], 100)

    def test_inventory_deduplicates_aliases_and_excludes_cpu(self):
        parameter, alias = Tensor(10, 100), Tensor(10, 100)
        parameter.grad = Tensor(20, 100)
        model = SimpleNamespace(parameters=lambda: [parameter, alias], buffers=lambda: [Tensor(30, 4)])
        optimizer = SimpleNamespace(state={'p': {'momentum': [Tensor(40, 100), Tensor(40, 100)],
                                                'step': Tensor(50, 8, device='cpu')}})
        engine = SimpleNamespace(model_parts=[model], optim=SimpleNamespace(optimizers=[optimizer]))
        observer = self.make()
        with observer:
            record = observer.sample('state', engine=engine)['state_inventory']
        self.assertEqual(record['category_bytes'], {'parameters': 100, 'gradients': 100,
                                                  'buffers': 4, 'optimizer_state': 100})
        self.assertEqual(record['unique_storage_bytes'], 304)
        self.assertEqual(record['other_device_tensor_references'], 1)
        self.assertEqual(record['unresolved'], [])

    def test_snapshot_failure_is_explicit_and_does_not_hide_native_work(self):
        observer = self.make(snapshot=True)
        observer.snapshot_function = lambda: (_ for _ in ()).throw(RuntimeError('unsupported allocator'))
        with observer:
            pass
        self.assertEqual(observer.report()['samples'][0]['allocator_snapshot']['status'], 'unavailable')

    def test_single_use_nested_phase_and_device_validation(self):
        observer = self.make()
        with observer:
            with observer.phase('capture'):
                with self.assertRaisesRegex(ValueError, 'cannot nest'):
                    with observer.phase('nested'):
                        pass
        with self.assertRaises(ValueError):
            with observer:
                pass
        with self.assertRaises(ValueError):
            self.make(device='cpu')
        with self.assertRaises(ValueError):
            self.make(rank=True)

    def reports(self):
        reports = []
        for rank in range(2):
            observer = self.make(rank=rank, world_size=2, physical_device='host/device')
            with observer:
                with observer.phase('capture'):
                    self.torch.cuda.allocate(100+rank*10, 150+rank*10)
            reports.append(observer.report())
        return reports

    def test_collocated_rank_peaks_are_maxed_never_summed(self):
        aggregate = merge_observations(self.reports())
        self.assertEqual(aggregate['status'], 'complete')
        self.assertEqual(aggregate['per_device']['host/device']['observed_process_peak_lower_bytes'], 160)
        self.assertIsNone(aggregate['per_device']['host/device']['physical_peak_upper_bytes'])
        self.assertEqual(aggregate['memory_verdict'], 'inconclusive')

    def test_missing_rank_placement_and_partial_world_block_completeness(self):
        reports = self.reports()
        self.assertEqual(merge_observations(reports[:1])['missing_ranks'], [1])
        self.assertEqual(merge_observations(reports[:1], expected_ranks=[0])['status'], 'incomplete')
        reports[1]['physical_device'] = None
        self.assertEqual(merge_observations(reports)['status'], 'incomplete')
        reports[1]['status'] = 'failed'
        self.assertEqual(merge_observations(reports)['status'], 'incomplete')

    def test_duplicates_provenance_tampering_and_mismatched_runtime_rejected(self):
        reports = self.reports()
        with self.assertRaises(ValueError): merge_observations([reports[0], reports[0]])
        changed = deepcopy(reports)
        changed[1]['provenance']['recipe']['unchanged'] = False
        with self.assertRaisesRegex(ValueError, 'digest mismatch'): merge_observations(changed)
        changed = deepcopy(reports)
        changed[1]['runtime']['torch_version'] = 'other'
        with self.assertRaisesRegex(ValueError, 'recipe/runtime'): merge_observations(changed)

    def test_report_does_not_allow_mutation_of_observer(self):
        observer = self.make()
        with observer: pass
        report = observer.report()
        report['samples'][0]['label'] = 'changed'
        self.assertEqual(observer.report()['samples'][0]['label'], 'baseline')

    def test_resolved_provenance_is_copied_and_can_merge(self):
        observers = [self.make(rank=rank, world_size=2, physical_device='host/device') for rank in range(2)]
        provenance = {'resolved_config_sha256': 'native-config', 'runtime': 'pinned'}
        for observer in observers:
            with observer:
                observer.set_provenance(provenance)
        provenance['runtime'] = 'later-mutation'
        self.assertEqual(observers[0].report()['provenance']['runtime'], 'pinned')
        self.assertEqual(merge_observations(o.report() for o in observers)['status'], 'complete')

    def test_failed_baseline_reports_failure(self):
        observer = self.make()
        self.torch.cuda.mem_get_info = lambda device: (_ for _ in ()).throw(RuntimeError('unavailable'))
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            with observer:
                self.fail('body must not run')
        self.assertEqual(observer.report()['status'], 'failed')

    def test_native_reset_retains_transient_peak_without_sync_or_snapshot(self):
        observer = self.make(observe_peak_resets=True, snapshot=True)
        cuda = self.torch.cuda
        with observer:
            with observer.phase('capture'):
                cuda.allocate(300, 500)
                cuda.allocate(25, 45)
                sync = cuda.synchronize
                cuda.synchronize = lambda device: self.fail('reset hook must not synchronize')
                cuda.reset_peak_memory_stats()
                cuda.synchronize = sync
                cuda.allocate(35, 55)
            with observer.phase('next'):
                cuda.allocate(40, 60)
        report = observer.report()
        self.assertEqual(report['process_allocator_peak'], {'allocated_bytes': 300, 'reserved_bytes': 500})
        self.assertEqual(report['phases'][0]['process_peak_reserved_bytes'], 500)
        self.assertEqual(report['phases'][1]['process_peak_reserved_bytes'], 60)
        self.assertEqual(report['phases'][0]['observed_native_reset_count'], 1)
        self.assertEqual(len(report['peak_reset_observation']['pre_reset_counters']), 1)
        self.assertNotIn('reset_peak_memory_stats', vars(cuda))

    def test_reset_hook_scope_aliases_and_original_call_semantics(self):
        observer = self.make(observe_peak_resets=True)
        cuda = self.torch.cuda
        original = cuda.reset_peak_memory_stats
        calls = []
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            original(*args, **kwargs)
            return 'native-result'
        cuda.reset_peak_memory_stats = native
        cuda.memory = SimpleNamespace(reset_peak_memory_stats=native)
        with observer:
            self.assertEqual(cuda.memory.reset_peak_memory_stats(device='cuda:0'), 'native-result')
            cuda.reset_peak_memory_stats(1)
            cuda.reset_peak_memory_stats(0)
        self.assertEqual(calls[-3:], [((), {'device': 'cuda:0'}), ((1,), {}), ((0,), {})])
        self.assertEqual(len(observer.report()['peak_reset_observation']['pre_reset_counters']), 2)
        self.assertIs(cuda.reset_peak_memory_stats, native)
        self.assertIs(cuda.memory.reset_peak_memory_stats, native)

    def test_reset_hook_failure_records_gap_but_delegates(self):
        observer = self.make(observe_peak_resets=True)
        cuda = self.torch.cuda
        with observer:
            cuda.allocate(100, 200)
            cuda.allocate(25, 45)
            getter = cuda.max_memory_allocated
            cuda.max_memory_allocated = lambda device: (_ for _ in ()).throw(RuntimeError('counter unavailable'))
            cuda.reset_peak_memory_stats()
            cuda.max_memory_allocated = getter
            self.assertEqual(cuda.peak_reserved, 45)
        self.assertTrue(any('pre-reset counter observation' in issue for issue in observer.report()['unresolved']))

    def test_reset_hooks_restore_on_failure_and_reject_nested_ownership(self):
        observer = self.make(observe_peak_resets=True)
        cuda = self.torch.cuda
        with self.assertRaisesRegex(RuntimeError, 'body failed'):
            with observer:
                wrapper = cuda.reset_peak_memory_stats
                nested = MemoryObserver(torch_module=self.torch, observe_peak_resets=True)
                with self.assertRaisesRegex(ValueError, 'owns this process'):
                    with nested:
                        self.fail('nested body must not run')
                self.assertIs(cuda.reset_peak_memory_stats, wrapper)
                raise RuntimeError('body failed')
        self.assertNotIn('reset_peak_memory_stats', vars(cuda))
        with MemoryObserver(torch_module=self.torch, observe_peak_resets=True):
            pass

    def test_reset_hooks_restore_when_baseline_fails(self):
        observer = self.make(observe_peak_resets=True)
        cuda = self.torch.cuda
        original = cuda.mem_get_info
        cuda.mem_get_info = lambda device: (_ for _ in ()).throw(RuntimeError('baseline failed'))
        with self.assertRaisesRegex(RuntimeError, 'baseline failed'):
            with observer:
                self.fail('body must not run')
        self.assertNotIn('reset_peak_memory_stats', vars(cuda))
        cuda.mem_get_info = original
        with MemoryObserver(torch_module=self.torch, observe_peak_resets=True):
            pass

    def test_entry_interruptions_restore_aliases_and_release_ownership(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            for failure_point in ('synchronize', 'mem_get_info'):
                with self.subTest(error_type=error_type, failure_point=failure_point):
                    observer = self.make(observe_peak_resets=True)
                    cuda = self.torch.cuda
                    native_reset = cuda.reset_peak_memory_stats
                    cuda.memory = SimpleNamespace(reset_peak_memory_stats=native_reset)
                    original = getattr(cuda, failure_point)
                    interruption = error_type('entry interrupted')
                    def fail(device):
                        raise interruption
                    setattr(cuda, failure_point, fail)
                    with self.assertRaises(error_type) as raised:
                        with observer:
                            self.fail('body must not run')
                    self.assertIs(raised.exception, interruption)
                    self.assertNotIn('reset_peak_memory_stats', vars(cuda))
                    self.assertIs(cuda.memory.reset_peak_memory_stats, native_reset)
                    self.assertEqual(observer.report()['status'], 'failed')
                    self.assertFalse(observer.active)
                    self.assertTrue(observer.finished)
                    setattr(cuda, failure_point, original)
                    with MemoryObserver(torch_module=self.torch, observe_peak_resets=True):
                        pass

    def test_reset_alias_reentrancy_records_once_and_excludes_owned_resets(self):
        observer = self.make(observe_peak_resets=True)
        cuda = self.torch.cuda
        original = cuda.reset_peak_memory_stats
        cuda.memory = SimpleNamespace(reset_peak_memory_stats=original)
        cuda.reset_peak_memory_stats = lambda *args, **kwargs: cuda.memory.reset_peak_memory_stats(*args, **kwargs)
        with observer:
            with observer.phase('capture'):
                cuda.allocate(70, 80)
                cuda.allocate(20, 40)
                cuda.reset_peak_memory_stats()
        self.assertEqual(len(observer.report()['peak_reset_observation']['pre_reset_counters']), 1)

    def test_original_reset_exception_propagates_and_restores_hook(self):
        observer = self.make(observe_peak_resets=True)
        cuda = self.torch.cuda
        original = cuda.reset_peak_memory_stats
        def native(device=None):
            if device == 'invalid':
                raise ValueError('native reset rejected device')
            return original(device)
        cuda.reset_peak_memory_stats = native
        with self.assertRaisesRegex(ValueError, 'native reset rejected device'):
            with observer:
                cuda.reset_peak_memory_stats('invalid')
        self.assertIs(cuda.reset_peak_memory_stats, native)

    def function_repr_reports(self, second_name='factory', second_width=256):
        from titan_sim.prepare import _digest as config_digest
        reports = []
        for rank in range(2):
            observer = self.make(rank=rank, world_size=2, physical_device='host/device')
            config = {'builder': '<function '+('factory' if rank == 0 else second_name)+' at 0x'+str(rank+100)+'>',
                      'width': 256 if rank == 0 else second_width}
            observer.set_provenance({'recipe_config': config,
                                     'resolved_config_sha256': config_digest(config), 'runtime': 'pinned'})
            with observer:
                pass
            reports.append(observer.report())
        return reports

    def test_rank_function_addresses_keep_raw_hashes_and_identity_gap(self):
        reports = self.function_repr_reports()
        aggregate = merge_observations(reports)
        self.assertEqual(aggregate['status'], 'incomplete')
        self.assertIsNone(aggregate['provenance_sha256'])
        comparison = aggregate['recipe_config_comparison']
        self.assertFalse(comparison['callable_identity_verified'])
        self.assertEqual(comparison['normalized_paths'], ['/builder'])
        self.assertEqual(aggregate['rank_provenance_sha256'],
                         {str(r['rank']): r['provenance_sha256'] for r in reports})
        self.assertEqual(len(comparison['sources']), 2)

    def test_function_address_fallback_rejects_semantic_or_other_provenance_changes(self):
        from titan_sim.memory_observer import _digest
        for reports in (self.function_repr_reports(second_name='other_factory'),
                        self.function_repr_reports(second_width=512)):
            with self.assertRaisesRegex(ValueError, 'recipe/runtime'):
                merge_observations(reports)
        reports = self.function_repr_reports()
        reports[1]['provenance']['runtime'] = 'different'
        reports[1]['provenance_sha256'] = _digest(reports[1]['provenance'])
        with self.assertRaisesRegex(ValueError, 'recipe/runtime'):
            merge_observations(reports)
        reports = self.function_repr_reports()
        reports[1]['provenance']['resolved_config_sha256'] = 'invalid'
        reports[1]['provenance_sha256'] = _digest(reports[1]['provenance'])
        with self.assertRaisesRegex(ValueError, 'recipe/runtime'):
            merge_observations(reports)


if __name__ == '__main__':
    unittest.main()
