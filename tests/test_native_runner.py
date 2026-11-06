"""CPU contract tests. CUDA/collective timing accuracy requires GPU validation."""
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

from titan_sim.core import Node
from titan_sim.native_runner import measure, native_cost_key, _provider_plans
from titan_sim.workload import Workload


class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = [dict(ph='X', cat='cpu_op', name='aten::mm', dur=999),
                       dict(ph='X', cat='kernel', name='native_matmul', dur=12., args={'grid': [8, 1, 1]}),
                       dict(ph='X', cat='kernel', name='native_matmul', dur=18., args={'grid': [8, 1, 1]})]
        self.path = Path(self.temp.name)/'step0.chrome.json'
        self.path.write_text(json.dumps({'traceEvents': self.events}))
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.implementation = dict(name='native_matmul', category='kernel', launch={'grid': [8, 1, 1]})
        self.context = dict(operator_cost_key='exact-native-op-with-source-and-shapes', hardware_id='test')
        self.key = native_cost_key(self.implementation, self.context)
        self.binding = dict(kind='chrome_gpu_event', artifact_sha256=self.sha, event_index=1,
                            event_name='native_matmul', event_category='kernel',
                            implementation=self.implementation, context=self.context)
        self.workload = Workload([Node(0, 'native_matmul', 0,
            attrs=dict(cost_key=self.key, native_measurement=self.binding))], [],
            dict(ranks=[0], world_size=1, native_trace_artifacts=[dict(path=str(self.path), sha256=self.sha)]),
            dict(id='test', capacity_bytes=1024))
        self.config = dict(mode='trace', hardware_id='test', accept_profiler_overhead=True)

    def test_trace_import_uses_gpu_durations_and_aggregates_repeated_exact_keys(self):
        second = copy.deepcopy(self.workload.nodes[0])
        second.id = 1
        second.attrs['native_measurement']['event_index'] = 2
        self.workload.nodes.append(second)
        record = measure(self.workload, self.config)[self.key]
        self.assertEqual(record['samples_seconds'], [12e-6, 18e-6])
        self.assertAlmostEqual(record['seconds'], 15e-6)
        self.assertEqual(record['measurement_kind'], 'profiled_gpu_activity')
        self.assertFalse(record['independent_validation'])
        self.assertEqual(record['sources'][1]['event_index'], 2)

    def test_prepare_key_contract_matches(self):
        from titan_sim.prepare import native_cost_key as prepare_key
        self.assertEqual(self.key, prepare_key(self.implementation, self.context))

    def test_profile_overhead_requires_explicit_acceptance(self):
        del self.config['accept_profiler_overhead']
        with self.assertRaisesRegex(ValueError, 'accept_profiler_overhead'):
            measure(self.workload, self.config)

    def test_projection_cannot_be_presented_as_gpu_cost(self):
        self.binding['event_index'] = 0
        with self.assertRaisesRegex(ValueError, 'completed GPU activity'):
            measure(self.workload, self.config)

    def test_artifact_tampering_and_key_mutation_rejected(self):
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            measure(self.workload, self.config)
        self.workload.nodes[0].attrs['cost_key'] = 'invented-key'
        with self.assertRaisesRegex(ValueError, 'cost key mismatch'):
            measure(self.workload, self.config)

    def test_launch_implementation_must_match_retained_event(self):
        self.binding['implementation']['launch']['grid'] = [99, 1, 1]
        self.workload.nodes[0].attrs['cost_key'] = native_cost_key(self.implementation, self.context)
        with self.assertRaisesRegex(ValueError, 'launch implementation'):
            measure(self.workload, self.config)

    def test_duplicate_activity_is_rejected(self):
        other = copy.deepcopy(self.workload.nodes[0])
        other.id = 1
        self.workload.nodes.append(other)
        with self.assertRaisesRegex(ValueError, 'multiple workload nodes'):
            measure(self.workload, self.config)

    def test_hardware_and_coordinator_contracts(self):
        with self.assertRaisesRegex(ValueError, 'hardware_id'):
            measure(self.workload, dict(self.config, hardware_id='wrong'))
        with patch.dict('os.environ', {'RANK': '1'}):
            with self.assertRaisesRegex(ValueError, 'coordinator'):
                measure(self.workload, self.config)

    def test_invalid_durations_do_not_become_samples(self):
        self.events[1]['dur'] = -1
        self.path.write_text(json.dumps({'traceEvents': self.events}))
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.binding['artifact_sha256'] = sha
        self.workload.manifest['native_trace_artifacts'][0]['sha256'] = sha
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            measure(self.workload, self.config)

    def test_moved_artifacts_can_be_resolved_by_checksum(self):
        self.workload.manifest['native_trace_artifacts'][0]['path'] = '/unavailable/original/location'
        self.config['artifact_paths'] = {self.sha: str(self.path)}
        self.assertIn(self.key, measure(self.workload, self.config))

    def test_trace_processing_releases_each_parsed_artifact_before_loading_next(self):
        second_path = Path(self.temp.name)/'step1.chrome.json'
        second_path.write_text(json.dumps({'traceEvents': self.events, 'fixture': 'second synthetic artifact'}))
        second_sha = hashlib.sha256(second_path.read_bytes()).hexdigest()
        self.workload.manifest['native_trace_artifacts'].append(dict(path=str(second_path), sha256=second_sha))
        second = copy.deepcopy(self.workload.nodes[0])
        second.id = 1
        second.attrs['native_measurement']['artifact_sha256'] = second_sha
        third = copy.deepcopy(self.workload.nodes[0])
        third.id = 2
        third.attrs['native_measurement']['event_index'] = 2
        # Deliberately interleave artifacts. Each is parsed once, not per node.
        self.workload.nodes.extend([second, third])
        class TrackedTrace(dict): pass
        class TrackedEvents(list): pass
        live_refs, load_count = [], []
        original_loads = json.loads
        def tracked_loads(raw):
            self.assertTrue(all(reference() is None for reference in live_refs),
                            'previous parsed Chrome tree survived into the next load')
            trace = TrackedTrace(original_loads(raw))
            trace['traceEvents'] = TrackedEvents(trace['traceEvents'])
            live_refs.extend([weakref.ref(trace), weakref.ref(trace['traceEvents'])])
            load_count.append(True)
            return trace
        with patch('titan_sim.native_runner.json.loads', side_effect=tracked_loads):
            record = measure(self.workload, self.config)[self.key]
        self.assertEqual(len(load_count), 2)
        self.assertTrue(all(reference() is None for reference in live_refs))
        self.assertEqual(sorted(record['samples_seconds']), [12e-6, 12e-6, 18e-6])
        self.assertEqual(len(record['sources']), 3)
        self.assertEqual({item['artifact_sha256'] for item in record['sources']}, {self.sha, second_sha})

    def test_occurrence_only_context_cannot_claim_a_different_rank_or_index(self):
        self.context.update(occurrence_only=True, artifact_sha256=self.sha, event_index=2,
                            rank=0, step=None)
        self.workload.nodes[0].attrs['cost_key'] = native_cost_key(self.implementation, self.context)
        with self.assertRaisesRegex(ValueError, 'occurrence context.*event_index'):
            measure(self.workload, self.config)

    def test_large_source_provenance_is_hash_referenced_not_repeated_in_catalog(self):
        source = dict(torch_version='test-runtime', cuda_version='test-cuda',
                      recipe_config={'large_native_setting': 'distinct-source-value-'*10000})
        self.workload.manifest['provenance'] = source
        second = copy.deepcopy(self.workload.nodes[0])
        second.id = 1
        second.attrs['native_measurement']['event_index'] = 2
        second.attrs['native_measurement']['context']['operator_cost_key'] = 'different-exact-case'
        binding = second.attrs['native_measurement']
        second.attrs['cost_key'] = native_cost_key(binding['implementation'], binding['context'])
        self.workload.nodes.append(second)
        expected_hash = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':'),
                                                  allow_nan=False).encode()).hexdigest()
        catalog = measure(self.workload, self.config)
        self.assertEqual(len(catalog), 2)
        for record in catalog.values():
            reference = record['provenance']
            self.assertEqual(reference['source_provenance_sha256'], expected_hash)
            self.assertEqual(reference['source_provenance_location'], 'input workload /manifest/provenance')
            self.assertEqual(reference['torch_version'], 'test-runtime')
            self.assertNotIn('recipe_config', reference)
        serialized = json.dumps(catalog)
        self.assertNotIn('distinct-source-value-', serialized)
        self.assertLess(len(serialized), len(json.dumps(source)))
        self.assertIs(self.workload.manifest['provenance'], source)
        source['recipe_config']['large_native_setting'] += 'changed'
        changed = measure(self.workload, self.config)[self.key]['provenance']['source_provenance_sha256']
        self.assertNotEqual(changed, expected_hash)

    def test_device_properties_are_auditable_hash_references_not_repeated_raw_objects(self):
        properties = [{'name': 'synthetic GPU', 'large_fixture_field': 'device-evidence-'*10000}]
        self.path.write_text(json.dumps({'traceEvents': self.events, 'deviceProperties': properties}))
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.binding['artifact_sha256'] = sha
        self.workload.manifest['native_trace_artifacts'][0]['sha256'] = sha
        record = measure(self.workload, self.config)[self.key]
        source = record['sources'][0]
        expected_hash = hashlib.sha256(json.dumps(properties, sort_keys=True, separators=(',', ':'),
                                                  allow_nan=False).encode()).hexdigest()
        self.assertEqual(source['artifact_sha256'], sha)
        self.assertEqual(source['device_properties_location'], '/deviceProperties')
        self.assertEqual(source['device_properties_sha256'], expected_hash)
        self.assertNotIn('device_properties', source)
        self.assertNotIn('device-evidence-', json.dumps(record))

    def provider(self, run=lambda: None):
        return dict(provenance={'source_revision': 'pinned-original'}, regions=[
            dict(cost_key=self.key, implementation=self.implementation, context=self.context,
                 stream_semantics='current_stream', run=run)])

    def test_provider_contract_requires_exact_context_and_all_keys(self):
        bundle = self.provider()
        bundle['regions'][0]['context'] = {'wrong': 1}
        with self.assertRaisesRegex(ValueError, 'does not match'):
            _provider_plans(bundle, {self.key: self.binding}, False)
        with self.assertRaisesRegex(ValueError, 'every requested'):
            _provider_plans(dict(provenance={'source': 'x'}, regions=[]), {self.key: self.binding}, False)

    def test_graph_requires_explicit_support_and_stable_inputs(self):
        bundle = self.provider()
        with self.assertRaisesRegex(ValueError, 'graph support'):
            _provider_plans(bundle, {self.key: self.binding}, True)
        bundle['regions'][0]['supports_cuda_graph'] = True
        bundle['regions'][0]['before_sample'] = lambda: None
        with self.assertRaisesRegex(ValueError, 'stable state'):
            _provider_plans(bundle, {self.key: self.binding}, True)

    def fake_torch(self, distributed=False):
        state = types.SimpleNamespace(runs=0, resets=0, barriers=0, records=0, graphs=0, replays=0)
        class Stream:
            def __init__(self, **kwargs): pass
            def wait_stream(self, other): pass
            def synchronize(self): pass
        class Event:
            def __init__(self, **kwargs): pass
            def record(self, stream): state.records += 1
            def synchronize(self): pass
            def elapsed_time(self, end): return 2.5
        class Graph:
            def replay(self): state.replays += 1
        def graph(*args, **kwargs):
            state.graphs += 1
            return nullcontext()
        def barrier(): state.barriers += 1
        def gather(values, value):
            # Emulate the peer's same plan and slower CUDA duration vector.
            values[0] = copy.deepcopy(value)
            values[1] = ([x*2 for x in value] if isinstance(value, list) and value and isinstance(value[0], float)
                         else copy.deepcopy(value))
            if isinstance(values[1], dict) and all(isinstance(record, dict) for record in values[1].values()):
                for record in values[1].values():
                    record['rank'] = 1
        cuda = types.SimpleNamespace(is_available=lambda: True,
            get_device_properties=lambda device: types.SimpleNamespace(name='test GPU', total_memory=1024, major=9, minor=0),
            device=lambda device: nullcontext(), stream=lambda stream: nullcontext(),
            Stream=Stream, Event=Event, CUDAGraph=Graph, graph=graph, current_stream=lambda device: Stream())
        dist = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: distributed,
                                     get_world_size=lambda: 2, get_rank=lambda: 0,
                                     all_gather_object=gather, barrier=barrier)
        torch = types.SimpleNamespace(cuda=cuda, distributed=dist, device=lambda value: types.SimpleNamespace(type='cuda'),
                                      __version__='test', version=types.SimpleNamespace(cuda='test'))
        return torch, state

    def execute_config(self, **kwargs):
        return dict(mode='execute', hardware_id='test', region_provider='trusted:provider',
                    expected_device={'name': 'test GPU', 'total_memory': 1024}, warmup=2, repeats=3, **kwargs)

    def test_execute_warmup_resets_repeats_and_gpu_event_units(self):
        torch, state = self.fake_torch()
        def run(): state.runs += 1
        def reset(): state.resets += 1
        bundle = self.provider(run)
        bundle['regions'][0]['before_sample'] = reset
        with patch.dict('sys.modules', {'torch': torch}), patch('titan_sim.native_runner.resolve', return_value=lambda w,c: bundle):
            record = measure(self.workload, self.execute_config())[self.key]
        self.assertEqual(state.runs, 5)
        self.assertEqual(state.resets, 5)
        self.assertEqual(state.records, 6)
        self.assertEqual(record['samples_seconds'], [.0025]*3)
        self.assertEqual(record['measurement_kind'], 'native_executable_region')

    def test_distributed_rounds_agree_and_cost_uses_slowest_rank(self):
        torch, state = self.fake_torch(distributed=True)
        self.workload.manifest.update(ranks=[0, 1], world_size=2)
        remote = copy.deepcopy(self.workload.nodes[0])
        remote.id, remote.rank = 1, 1
        self.workload.nodes.append(remote)
        bundle = self.provider()
        bundle['regions'][0].update(round_id='case-0', round_context={'participants': [0, 1], 'shape': [8, 8]})
        with patch.dict('sys.modules', {'torch': torch}), patch('titan_sim.native_runner.resolve', return_value=lambda w,c: bundle):
            record = measure(self.workload, self.execute_config())[self.key]
        self.assertEqual(state.barriers, 5)
        self.assertEqual(record['samples_seconds'], [.005]*3)
        self.assertEqual(record['per_rank_samples_seconds'], [[.0025]*3, [.005]*3])
        self.assertEqual(record['contributing_ranks'], [0, 1])

    def test_graph_replay_is_timed_separately_from_capture_and_warmup(self):
        torch, state = self.fake_torch()
        self.context['cuda_graph'] = True
        self.key = native_cost_key(self.implementation, self.context)
        self.workload.nodes[0].attrs['cost_key'] = self.key
        def run(): state.runs += 1
        bundle = self.provider(run)
        bundle['regions'][0]['supports_cuda_graph'] = True
        with patch.dict('sys.modules', {'torch': torch}), patch('titan_sim.native_runner.resolve', return_value=lambda w,c: bundle):
            record = measure(self.workload, self.execute_config(cuda_graph=True))[self.key]
        self.assertEqual(state.runs, 3)  # Two warmups and one graph capture.
        self.assertEqual(state.replays, 3)
        self.assertEqual(state.records, 6)
        self.assertEqual(record['timing_method'], 'cuda_graph_replay_events')

    def test_unmatched_projection_stays_missing_from_partial_trace_catalog(self):
        self.workload.nodes.append(Node(1, 'unmatched CPU projection', 0, attrs={'cost_key': 'old-et-key'}))
        self.assertEqual(set(measure(self.workload, self.config)), {self.key})

    def test_distributed_plan_mismatch_fails_before_any_timed_call(self):
        torch, state = self.fake_torch(distributed=True)
        self.workload.manifest.update(ranks=[0, 1], world_size=2)
        remote = copy.deepcopy(self.workload.nodes[0])
        remote.id, remote.rank = 1, 1
        self.workload.nodes.append(remote)
        original_gather = torch.distributed.all_gather_object
        def gather(values, value):
            original_gather(values, value)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                values[1][0]['round_id'] = 'wrong-order'
        torch.distributed.all_gather_object = gather
        bundle = self.provider()
        bundle['regions'][0].update(round_id='case-0', round_context={'participants': [0, 1], 'shape': [8, 8]})
        with patch.dict('sys.modules', {'torch': torch}), patch('titan_sim.native_runner.resolve', return_value=lambda w,c: bundle):
            with self.assertRaisesRegex(ValueError, 'ordered round'):
                measure(self.workload, self.execute_config())
        self.assertEqual(state.records, 0)

    def test_distributed_hardware_error_is_shared_before_provider_or_barriers(self):
        torch, state = self.fake_torch(distributed=True)
        seen = []
        original_gather = torch.distributed.all_gather_object
        def gather(values, value):
            seen.append(value)
            original_gather(values, value)
        torch.distributed.all_gather_object = gather
        with patch.dict('sys.modules', {'torch': torch}), patch('titan_sim.native_runner.resolve') as provider:
            with self.assertRaisesRegex(ValueError, 'preflight.*hardware'):
                measure(self.workload, dict(self.execute_config(), expected_device={'name': 'wrong', 'total_memory': 1024}))
        self.assertEqual(len(seen), 1)
        self.assertIn('hardware', seen[0])
        self.assertEqual(state.barriers, 0)
        provider.assert_not_called()

    def test_distributed_peer_preflight_error_stops_local_valid_rank(self):
        torch, state = self.fake_torch(distributed=True)
        self.workload.manifest.update(ranks=[0, 1], world_size=2)
        def gather(values, value):
            values[:] = [value, 'ValueError: peer descriptor mismatch']
        torch.distributed.all_gather_object = gather
        provider_calls = []
        def provider(workload, config):
            provider_calls.append(True)
            return self.provider()
        with patch.dict('sys.modules', {'torch': torch}), patch('titan_sim.native_runner.resolve', return_value=provider):
            with self.assertRaisesRegex(ValueError, 'peer descriptor mismatch'):
                measure(self.workload, self.execute_config())
        self.assertFalse(provider_calls)
        self.assertEqual(state.barriers, 0)

    def test_distributed_sample_count_mismatch_rejected_before_provider(self):
        torch, state = self.fake_torch(distributed=True)
        self.workload.manifest.update(ranks=[0, 1], world_size=2)
        original_gather = torch.distributed.all_gather_object
        def gather(values, value):
            original_gather(values, value)
            if isinstance(value, dict) and 'repeats' in value:
                values[1]['repeats'] += 1
        torch.distributed.all_gather_object = gather
        provider_calls = []
        def provider(workload, config):
            provider_calls.append(True)
            return self.provider()
        with patch.dict('sys.modules', {'torch': torch}), patch('titan_sim.native_runner.resolve', return_value=provider):
            with self.assertRaisesRegex(ValueError, 'measurement counts'):
                measure(self.workload, self.execute_config())
        self.assertFalse(provider_calls)
        self.assertEqual(state.barriers, 0)

    def test_execute_rejects_absent_cuda_without_claiming_measurements(self):
        torch, _ = self.fake_torch()
        torch.cuda.is_available = lambda: False
        with patch.dict('sys.modules', {'torch': torch}):
            with self.assertRaisesRegex(ValueError, 'requires CUDA'):
                measure(self.workload, self.execute_config())


if __name__ == '__main__':
    unittest.main()
