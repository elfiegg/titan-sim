import copy
import hashlib
import inspect
from pathlib import Path
import sys
import types
import tempfile
import json
import unittest
from unittest.mock import patch

from titan_sim.core import Node
from titan_sim.identity import digest, workload_fingerprint
from titan_sim.native_runner import native_cost_key
from titan_sim.workload import Workload
from titan_sim.execution_windows import (validate_window_manifest, execution_window_workload,
                                         compose_window, region_provider, bind_execution_window, _agree)


def fixture(world=1):
    nodes = []
    for rank in range(world):
        for ordinal in range(2):
            implementation = {'native_callable': 'fixture', 'ordinal': ordinal}
            context = {'rank':rank, 'hardware_id':'fixture', 'cuda_graph':False}
            nodes.append(Node(rank*2+ordinal, 'native fixture', rank, attrs=dict(
                cost_key=native_cost_key(implementation, context), native_measurement=dict(
                    kind='executable_region', implementation=implementation, context=context))))
    source = Workload(nodes, [], {'ranks':list(range(world)), 'world_size':world}, {'id':'fixture'})
    manifest = dict(schema='native-execution-window', version=1, window_id='fixture-window',
        source_workload_sha256=workload_fingerprint(source), participants=list(range(world)),
        provider=__name__+':provide', provider_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        completion='join_all_streams', physical_topology={'fixture':True}, runtime_context={'fixture':True},
        input_state_policy={'lifetime':'entire_window', 'reset':'before_sample', 'values':'native provider'},
        ordering_evidence={'kind':'caller_assertion', 'description':'controlled test order'},
        qualification='controlled_native_window', communication_contract={'kind':'provider_owned','description':'test fixture has no communication'}, live_in_dependencies=[],
        steps=[dict(occurrence_id=n.id,rank=n.rank,stream=str(n.id%2),deps=[]) for n in nodes])
    return source, manifest


def provide(source, manifest):
    return dict(source_workload_sha256=workload_fingerprint(source),
        input_state_policy=manifest['input_state_policy'], runtime_context=manifest['runtime_context'],
        preparation_is_local=True, communication_contract=manifest['communication_contract'], provenance={'fixture':True}, regions=[dict(
            occurrence_id=n.id, implementation=n.attrs['native_measurement']['implementation'],
            context=n.attrs['native_measurement']['context'], run=lambda:None,
            stream_semantics='current_stream') for n in source.nodes if n.rank == 0])


class FakeCuda:
    def __init__(self):
        self.log, self.index = [], 0
        self.current = self.Stream()
    def Stream(self):
        owner = self
        identifier = self.index
        self.index += 1
        class Stream:
            def wait_stream(self, other): owner.log.append(('wait_stream',identifier,other.identifier))
            def wait_event(self, other): owner.log.append(('wait_event',identifier,other.identifier))
        result = Stream()
        result.identifier = identifier
        return result
    def Event(self):
        owner = self
        identifier = self.index
        self.index += 1
        class Event:
            def record(self, stream): owner.log.append(('record',identifier,stream.identifier))
        result = Event()
        result.identifier = identifier
        return result
    def current_stream(self): return self.current
    def stream(self, stream):
        owner = self
        class Context:
            def __enter__(self):
                self.previous = owner.current
                owner.current = stream
            def __exit__(self, *args): owner.current = self.previous
        return Context()


class WindowTests(unittest.TestCase):
    def test_exact_membership_source_hash_and_whole_window_key(self):
        source, manifest = fixture()
        result = execution_window_workload(source, manifest)
        self.assertEqual(result.nodes[0].attrs['native_measurement']['context']['measurement_scope'], 'execution_window')
        changed = copy.deepcopy(manifest)
        changed['steps'][1]['stream'] = '0'
        self.assertNotEqual(result.nodes[0].attrs['cost_key'], execution_window_workload(source, changed).nodes[0].attrs['cost_key'])
        source.nodes[0].attrs['changed'] = True
        with self.assertRaisesRegex(ValueError,'digest'):
            execution_window_workload(source, manifest)

    def test_invalid_graph_membership_boundaries_and_evidence(self):
        changes = [lambda m:m['steps'].append(copy.deepcopy(m['steps'][0])),
                   lambda m:m['steps'][0].update(deps=[1]),
                   lambda m:m.update(live_in_dependencies=[999]),
                   lambda m:m.update(qualification='source_backed_native_window'),
                   lambda m:m['steps'][0].update(deps=[999]),
                   lambda m:m['steps'][0].update(rank=12),
                   lambda m:m.update(completion='launch_only')]
        for change in changes:
            source, manifest = fixture()
            manifest['steps'][1]['deps'] = [0]
            change(manifest)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_window_manifest(source, manifest)

    def test_internal_source_dependencies_cannot_be_dropped(self):
        source, manifest = fixture()
        source.nodes[1].deps = [0]
        manifest['source_workload_sha256'] = workload_fingerprint(source)
        with self.assertRaisesRegex(ValueError, 'dropped'):
            validate_window_manifest(source, manifest)
        manifest['steps'][1]['deps'] = [0]
        validate_window_manifest(source, manifest)

    def test_stream_start_dependency_and_end_joins(self):
        cuda = FakeCuda()
        source, manifest = fixture()
        steps = manifest['steps']
        steps[1]['deps'] = [0]
        def run(index):
            return lambda:cuda.log.append(('run',index,cuda.current.identifier))
        regions = {i:dict(run=run(i),before_sample=lambda:cuda.log.append(('reset',))) for i in (0,1)}
        region = compose_window(types.SimpleNamespace(cuda=cuda),steps,regions)
        region['before_sample']()
        region['run']()
        self.assertEqual(cuda.log[:2],[('reset',),('reset',)])
        self.assertEqual(cuda.log[2:4],[('wait_stream',1,0),('wait_stream',2,0)])
        self.assertLess(cuda.log.index(('record',3,1)),cuda.log.index(('wait_event',2,3)))
        self.assertEqual(cuda.log[-2:],[('wait_stream',0,1),('wait_stream',0,2)])

    def test_live_outputs_retained_through_window_until_reset(self):
        import weakref
        class Buffer: pass
        refs = []
        def produce():
            value = Buffer()
            refs.append(weakref.ref(value))
            return value
        cuda = FakeCuda()
        _, manifest = fixture()
        region = compose_window(types.SimpleNamespace(cuda=cuda),manifest['steps'],
            {0:{'run':produce},1:{'run':lambda:self.assertIsNotNone(refs[0]())}})
        region['run']()
        self.assertIsNotNone(refs[0]())
        region['before_sample']()
        self.assertIsNone(refs[0]())

    def test_rank_agreement_and_preflight_error_broadcast(self):
        class Dist:
            def all_gather_object(self, values, value): values[:] = [value,'peer differs']
        with self.assertRaisesRegex(ValueError,'agreement'):
            _agree(Dist(),2,'local','manifest agreement')
        with self.assertRaisesRegex(ValueError,'preflight'):
            _agree(Dist(),2,None,'preflight',errors=True)

    def test_provider_exact_contract_and_preparation_failures(self):
        source, manifest = fixture()
        workload = execution_window_workload(source, manifest)
        cuda = FakeCuda()
        dist = types.SimpleNamespace(is_available=lambda:False)
        torch = types.SimpleNamespace(cuda=cuda,distributed=dist)
        with patch.dict(sys.modules,torch=torch):
            bundle = region_provider(workload,{'physical_topology':manifest['physical_topology']})
            bundle['regions'][0]['before_sample']()
            bundle['regions'][0]['run']()
            with self.assertRaisesRegex(ValueError,'topology'):
                region_provider(workload,{'physical_topology':{'wrong':True}})
        original = provide
        def bad(source, manifest):
            bundle = original(source,manifest)
            bundle['regions'][0]['before_sample'] = lambda:(_ for _ in ()).throw(RuntimeError('reset OOM'))
            return bundle
        # Resolve patched callable but retain this same test module source hash.
        with patch.dict(sys.modules,torch=torch), patch(__name__+'.provide',bad):
            bundle = region_provider(workload,{'physical_topology':manifest['physical_topology']})
            with self.assertRaisesRegex(ValueError,'reset OOM'):
                bundle['regions'][0]['before_sample']()

    def test_collective_peer_sequence_mismatch_is_rejected_before_native_calls(self):
        source, manifest = fixture(2)
        for node in source.nodes:
            implementation={'kind':'native_collective_overload'}
            context={'case':{'backend':'nccl','group_size':2,'case_id':str(node.id%2)}}
            node.attrs.update(native_measurement={'implementation':implementation,'context':context},
                              cost_key=native_cost_key(implementation,context))
        manifest['source_workload_sha256']=workload_fingerprint(source)
        validate_window_manifest(source,manifest)
        # Same global manifest on every rank can still describe mismatched calls.
        manifest['steps'][2],manifest['steps'][3]=manifest['steps'][3],manifest['steps'][2]
        with self.assertRaisesRegex(ValueError,'peer signatures'):
            validate_window_manifest(source,manifest)

    def _source_backed(self, source, manifest, directory):
        trace=Path(directory)/'trace.json'
        trace.write_text('{"fixture":"ordering evidence test only"}')
        artifact=dict(schema='native-window-ordering-assertion',version=1,
            source_workload_sha256=workload_fingerprint(source),steps=manifest['steps'],
            producer='unit test fixture',source_artifacts=[dict(path=str(trace),sha256=hashlib.sha256(trace.read_bytes()).hexdigest())],
            input_state_policy_sha256=digest(manifest['input_state_policy']),
            boundary_contract={'entry':'all_live_inputs_ready','exit':'all_streams_joined',
                               'external_consumers_wait_for_window_completion':True})
        path=Path(directory)/'ordering.json'
        path.write_text(json.dumps(artifact))
        manifest.update(qualification='source_backed_native_window',ordering_evidence=dict(
            kind='captured_stream_events',description='unit-test producer assertion',
            artifact_path=str(path),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return path,trace

    def test_bind_one_atomic_event_all_resources_and_preserve_gaps(self):
        source,manifest=fixture(2)
        source.manifest['unresolved']=['prior gap']
        source.nodes[0].attrs['resources']=['network']
        source.nodes.append(Node(10,'outside consumer',0,deps=[0,3],attrs={'zero_cost_semantic':True}))
        manifest['source_workload_sha256']=workload_fingerprint(source)
        with tempfile.TemporaryDirectory() as directory:
            self._source_backed(source,manifest,directory)
            measurement=execution_window_workload(source,manifest)
            bound=bind_execution_window(source,measurement)
            self.assertEqual(len(bound.nodes),2)
            self.assertEqual(bound.nodes[0].attrs['resources'],['gpu:0','gpu:1','network'])
            self.assertEqual(bound.nodes[1].deps,[0])
            self.assertIn('prior gap',bound.manifest['unresolved'])
            self.assertEqual(bound.manifest['memory']['storage_evidence'],'unverified')
            # The aggregate max rank duration appears once, not per rank.
            self.assertEqual(sum('cost_key' in n.attrs for n in bound.nodes),1)

    def test_controlled_window_cannot_bind_and_corrupt_source_artifact_fails(self):
        source,manifest=fixture()
        measurement=execution_window_workload(source,manifest)
        with self.assertRaisesRegex(ValueError,'measurement-only'):
            bind_execution_window(source,measurement)
        with tempfile.TemporaryDirectory() as directory:
            _,trace=self._source_backed(source,manifest,directory)
            measurement=execution_window_workload(source,manifest)
            trace.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source artifact checksum'):
                bind_execution_window(source,measurement)

    def test_binding_rejects_partial_source_rendezvous(self):
        source,manifest=fixture(2)
        for node in source.nodes:
            node.attrs['rendezvous']='round'+str(node.id%2)
        manifest['steps']=[step for step in manifest['steps'] if step['occurrence_id'] in (0,3)]
        manifest['source_workload_sha256']=workload_fingerprint(source)
        with tempfile.TemporaryDirectory() as directory:
            self._source_backed(source,manifest,directory)
            measurement=execution_window_workload(source,manifest)
            with self.assertRaisesRegex(ValueError,'split a source communication rendezvous'):
                bind_execution_window(source,measurement)

    def test_nonconvex_window_contraction_rejects_cycle(self):
        source,manifest=fixture()
        source.nodes.append(Node(10,'outside intermediate',0,deps=[0],attrs={'zero_cost_semantic':True}))
        source.nodes[1].deps=[10]
        manifest['live_in_dependencies']=[10]
        manifest['source_workload_sha256']=workload_fingerprint(source)
        with tempfile.TemporaryDirectory() as directory:
            self._source_backed(source,manifest,directory)
            measurement=execution_window_workload(source,manifest)
            with self.assertRaisesRegex(ValueError,'cycle'):
                bind_execution_window(source,measurement)

    def test_provider_must_cover_every_occurrence(self):
        source, manifest = fixture()
        workload = execution_window_workload(source, manifest)
        original = provide
        def bad(source, manifest):
            bundle = original(source,manifest)
            bundle['regions'].pop()
            return bundle
        torch = types.SimpleNamespace(cuda=FakeCuda(),distributed=types.SimpleNamespace(is_available=lambda:False))
        with patch.dict(sys.modules,torch=torch),patch(__name__+'.provide',bad):
            with self.assertRaisesRegex(ValueError,'exact window membership'):
                region_provider(workload,{'physical_topology':manifest['physical_topology']})


if __name__ == '__main__':
    unittest.main()
