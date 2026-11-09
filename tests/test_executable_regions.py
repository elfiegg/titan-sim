import unittest

try:
    import torch
    import torch._dynamo  # Keep dispatcher registrations outside scoped fake-module patches.
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class ExecutableRegionTests(unittest.TestCase):
    def test_original_handle_alias_contract_and_native_provider(self):
        from titan_sim.executable_regions import LiveRegionRegistry, region_provider
        registry = LiveRegionRegistry()
        a = torch.ones(2, 2)
        original = torch.mm
        registry.register(original, args=(a, a.t()), implementation={'name': 'aten.mm.default'},
            context={'cuda_graph': False, 'hardware_id':'unit'}, source_workload_sha256='a'*64, source_node_ids=[4],
            state_contract={'mutation': 'read_only'}, replay_safe=True)
        entry = registry.entries[0]
        self.assertIs(entry['native_handle'], original)
        contract = entry['implementation']['argument_contract']['args']
        self.assertEqual(contract[0]['alias'], contract[1]['alias'])
        workload = registry.calibration_workload(hardware={'id': 'unit'})
        bundle = region_provider(workload, {})
        self.assertTrue(torch.equal(bundle['regions'][0]['run'](), torch.full((2, 2), 2.)))
        a.resize_(4)
        with self.assertRaisesRegex(ValueError, 'changed'):
            region_provider(workload, {})
        registry.close()
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            region_provider(workload, {})

    def test_mutable_state_requires_reset(self):
        from titan_sim.executable_regions import LiveRegionRegistry
        registry = LiveRegionRegistry()
        with self.assertRaisesRegex(ValueError, 'reset'):
            registry.register(lambda: None, implementation={'name':'custom'}, context={'mode':'test'},
                source_workload_sha256='a'*64, source_node_ids=[0], state_contract={'mutation':'in_place'},
                replay_safe=True)
        registry.close()

    def test_native_compile_hook_keeps_original_and_observes_boundary(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from titan_sim.executable_regions import observe_native_compile
        from titan_sim.torch_capture import TorchCapture
        module = ModuleType('torchtitan.distributed.local_compile')
        class Config:
            @staticmethod
            def _compile(reference, **options):
                return reference
        module.LocalCompileConfig = Config
        records = []
        recorder = TorchCapture()
        def native(value):
            return value + 1
        with patch.dict(sys.modules, {'torchtitan.distributed.local_compile': module}):
            with observe_native_compile(records, retain_arguments=True, recorder_getter=lambda: recorder):
                observed = Config._compile(native, fullgraph=True)
                observed(torch.ones(2))  # Outside capture: no invented occurrence.
                self.assertEqual(records[0]['source_occurrences'], [])
                with recorder:
                    result = observed(torch.ones(2))
            self.assertIs(Config._compile(native), native)
        self.assertTrue(torch.equal(result, torch.full((2,), 2.)))
        self.assertIs(records[0]['native_handle'], native)
        boundaries = [node for node in recorder.nodes if node.attrs.get('native_callable_boundary')]
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].attrs['status'], 'returned')
        self.assertIn(boundaries[0].id, records[0]['source_occurrences'][0])
        self.assertEqual(records[0]['compiled_correspondence'], 'unverified')

    def test_observation_failure_does_not_replace_native_call(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from titan_sim.executable_regions import observe_native_compile
        module = ModuleType('torchtitan.distributed.local_compile')
        class Config:
            @staticmethod
            def _compile(reference, **options):
                return reference
        module.LocalCompileConfig = Config
        records = []
        with patch.dict(sys.modules, {'torchtitan.distributed.local_compile': module}):
            with observe_native_compile(records), patch('titan_sim.executable_regions._contract', side_effect=RuntimeError('opaque')):
                observed = Config._compile(lambda value: value + 1)
                self.assertEqual(observed(3), 4)
        self.assertIn('opaque', records[0]['observation_error'])

    def test_boundary_only_observation_records_opaque_original_call(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from titan_sim.executable_regions import observe_native_compile, LiveRegionRegistry
        from titan_sim.torch_capture import TorchCapture
        from titan_sim.identity import workload_fingerprint
        module = ModuleType('torchtitan.distributed.local_compile')
        class Config:
            @staticmethod
            def _compile(reference, **options):
                return reference
        module.LocalCompileConfig = Config
        records, recorder = [], TorchCapture()
        with patch.dict(sys.modules, {'torchtitan.distributed.local_compile': module}):
            with observe_native_compile(records, retain_arguments=True,
                    recorder_getter=lambda: recorder, boundary_only=True):
                native = Config._compile(lambda value: value + 1)
                native(torch.ones(2))
        source = recorder.finish(hardware={'id':'unit'})
        self.assertEqual(len([node for node in source.nodes if node.attrs.get('operator')]), 1)
        registry = LiveRegionRegistry()
        registry.register_observed(records[0], native_callable_index=0,
            source_workload_sha256=workload_fingerprint(source), context={'hardware_id':'unit'},
            state_contract={'mutation':'read_only'}, replay_safe=True)
        self.assertIs(registry.entries[0]['native_handle'], records[0]['native_handle'])
        registry.close()

    def test_observed_mutated_metadata_and_runtime_are_rejected(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from titan_sim.executable_regions import observe_native_compile, LiveRegionRegistry
        from titan_sim.torch_capture import TorchCapture
        module = ModuleType('torchtitan.distributed.local_compile')
        class Config:
            @staticmethod
            def _compile(reference, **options):
                return reference
        module.LocalCompileConfig = Config
        records, recorder = [], TorchCapture()
        value = torch.ones(2, 3)
        with patch.dict(sys.modules, {'torchtitan.distributed.local_compile': module}):
            with observe_native_compile(records, retain_arguments=True,
                    recorder_getter=lambda: recorder, boundary_only=True):
                native = Config._compile(lambda value: value.transpose_(0, 1))
                native(value)
        self.assertEqual(records[0]['source_argument_contracts'][0]['args'][0]['shape'], [2, 3])
        registry = LiveRegionRegistry()
        kwargs=dict(native_callable_index=0, source_workload_sha256='a'*64,
                    context={'hardware_id':'unit'}, state_contract={'mutation':'unknown'})
        with self.assertRaisesRegex(ValueError, 'argument contract changed'):
            registry.register_observed(records[0], **kwargs)
        value.transpose_(0, 1)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'runtime/math'):
            registry.register_observed(records[0], **kwargs)
        registry.close()
