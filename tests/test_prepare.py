import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from titan_sim.core import Node, Storage
from titan_sim.prepare import native_cost_key, prepare
from titan_sim.workload import Workload, write_workload


EVIDENCE = dict(source='test-only explicit assertions', sha256='a' * 64)
HARDWARE = dict(id='test-device', capacity_bytes=1024)


def workload(rank=0, world=1, nodes=None, storage=None):
    return Workload(nodes if nodes is not None else [Node(0, 'op', rank, attrs={'cost_key': 'shape:x'})],
                    storage if storage is not None else [Storage('alias', rank, 16, 0, [0, 0])],
                    dict(ranks=[rank], world_size=world, unresolved=['unknown workspace'],
                         provenance={'recipe_args': ['arbitrary.recipe']}, capture_status='complete'), HARDWARE)


def ref(rank, node=0):
    return dict(rank=rank, id=node)


class PrepareTests(unittest.TestCase):
    def test_allrank_unique_ids_aliases_and_inputs_unchanged(self):
        a, b = workload(0, 2), workload(1, 2)
        result = prepare([a, b])
        self.assertEqual([n.id for n in result.nodes], [0, 1])
        self.assertEqual(len({s.id for s in result.storage}), 2)
        self.assertEqual([s.consumers for s in result.storage], [[0], [1]])
        self.assertEqual([s.producer for s in result.storage], [0, 1])
        self.assertEqual(a.storage[0].consumers, [0, 0])
        self.assertEqual(b.nodes[0].id, 0)
        self.assertEqual(result.manifest['unresolved'], ['unknown workspace'])

    def test_missing_rank_keeps_coverage_gap(self):
        result = prepare([workload(1, 2)])
        self.assertTrue(any('All-rank' in gap for gap in result.manifest['unresolved']))

    def test_duplicate_rank_and_mismatched_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate rank'):
            prepare([workload(), workload()])
        b = workload(1, 2)
        b.manifest['provenance']['recipe_args'] = ['different.recipe']
        with self.assertRaisesRegex(ValueError, 'mismatched provenance'):
            prepare([workload(0, 2), b])

    def test_paths_json_and_global_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'capture.json'
            write_workload(path, workload())
            result = prepare([str(path)], {'useful_tokens': 128})
            self.assertEqual(result.manifest['useful_tokens'], 128)

    def test_exact_region_contraction_and_key_dedup(self):
        nodes = [Node(8, 'a', 0), Node(9, 'b', 0, deps=[8]),
                 Node(10, 'a', 0, deps=[9]), Node(11, 'b', 0, deps=[10])]
        w = workload(nodes=nodes, storage=[Storage('x', 0, 8, 8, [9, 10, 11])])
        binding = dict(implementation={'code_sha256': 'b'*64}, context={'shape': [2, 2], 'dtype': 'f32'})
        result = prepare([w], dict(evidence=EVIDENCE, regions=[
            dict(nodes=[ref(0, 8), ref(0, 9)], **binding),
            dict(nodes=[ref(0, 10), ref(0, 11)], **binding)]))
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(result.nodes[1].deps, [result.nodes[0].id])
        self.assertEqual(len({n.attrs['cost_key'] for n in result.nodes}), 1)
        self.assertEqual(result.storage[0].consumers, [0, 2])
        self.assertIn('unknown workspace', result.manifest['unresolved'])

    def test_region_requires_evidence_and_context(self):
        with self.assertRaisesRegex(ValueError, 'evidence'):
            prepare([workload()], dict(regions=[dict(nodes=[ref(0)])]))
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            native_cost_key({'name': 'kernel'}, {})

    def test_nonconvex_region_rejected_by_cycle_check(self):
        w = workload(nodes=[Node(0, 'a', 0), Node(1, 'b', 0, deps=[0]), Node(2, 'c', 0, deps=[1])], storage=[])
        with self.assertRaisesRegex(ValueError, 'cycle'):
            prepare([w], dict(evidence=EVIDENCE, regions=[dict(nodes=[ref(0, 0), ref(0, 2)],
                implementation={'code': 'x'}, context={'shape': 2})]))

    def test_dependency_assertions_replace_not_append(self):
        w = workload(nodes=[Node(0, 'a', 0), Node(1, 'b', 0, deps=[0])], storage=[])
        result = prepare([w], dict(evidence=EVIDENCE, dependencies=[dict(node=ref(0, 1), deps=[])]))
        self.assertEqual(result.nodes[1].deps, [])
        self.assertEqual(result.manifest['unresolved'], ['unknown workspace'])

    def test_collective_and_point_to_point_reconciliation(self):
        for kind in ('collective', 'send_recv'):
            comm = dict(kind=kind, nodes=[ref(0), ref(1)], group='tp0', sequence=0,
                        operation='all_reduce' if kind == 'collective' else 'transfer',
                        participants=[0, 1], payload_bytes=16)
            result = prepare([workload(0, 2), workload(1, 2)], dict(evidence=EVIDENCE, communications=[comm]))
            self.assertEqual(result.nodes[0].attrs['rendezvous'], result.nodes[1].attrs['rendezvous'])
            if kind == 'send_recv':
                self.assertEqual(result.nodes[0].attrs['peer'], 1)

    def test_communication_signature_mismatch_is_rejected(self):
        a, b = workload(0, 2), workload(1, 2)
        b.nodes[0].attrs['payload_bytes'] = 32
        comm = dict(kind='collective', nodes=[ref(0), ref(1)], group='tp0', sequence=0,
                    operation='all_reduce', participants=[0, 1], payload_bytes=16)
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            prepare([a, b], dict(evidence=EVIDENCE, communications=[comm]))

    def test_backing_allocation_merges_aliases(self):
        w = workload(storage=[Storage('view1', 0, 16, 0, [0]), Storage('view2', 0, 8, 0, [0])])
        result = prepare([w], dict(evidence=EVIDENCE, allocations=[dict(rank=0, id='allocation0',
            source_storage_ids=['view1', 'view2'], size=16, producer=None, consumers=[ref(0)], persistent=True)]))
        self.assertEqual(len(result.storage), 1)
        self.assertEqual(result.storage[0].size, 16)
        self.assertTrue(result.storage[0].persistent)

    def test_chrome_lowering_uses_correlations_and_streams(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'step0.chrome.json'
            events = [dict(name='cpu', cat='cpu_op', args={'External id': 3, 'Record function id': 40})]
            for stream, timestamp in [(1, 10), (2, 12), (1, 15)]:
                events.append(dict(name='compiled_kernel', cat='kernel', ph='X', ts=timestamp, dur=2,
                                   args={'External id': 3, 'stream': stream, 'grid': [1, 1, 1]}))
            path.write_text(json.dumps({'traceEvents': events}))
            node = Node(0, 'compiled', 0, attrs=dict(cost_key='exact:shape2', step=0,
                       execution_trace_id=20, execution_trace_attrs={'rf_id': 40}))
            w = workload(nodes=[node])
            w.manifest['native_trace_artifacts'] = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
            result = prepare([w], dict(chrome_traces=[dict(rank=0, step=0, path=str(path))]))
            self.assertEqual(len(result.nodes), 3)
            self.assertEqual(result.nodes[0].deps, [])
            self.assertEqual(result.nodes[1].deps, [])  # No invented cross-stream edge.
            self.assertEqual(result.nodes[2].deps, [result.nodes[0].id])
            self.assertEqual(len({n.attrs['cost_key'] for n in result.nodes}), 1)
            self.assertIsNone(result.storage[0].producer)
            self.assertEqual(result.storage[0].consumers, [n.id for n in result.nodes])
            self.assertEqual(result.nodes[0].attrs['native_measurement']['event_index'], 1)
            self.assertTrue(any('Cross-stream' in gap for gap in result.manifest['unresolved']))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                prepare([w], dict(chrome_traces=[dict(rank=0, step=0, path=str(path))]))

    def test_unknown_correlation_retains_projection_and_gap(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'step0.chrome.json'
            path.write_text(json.dumps({'traceEvents': [dict(name='opaque', cat='kernel', ph='X', ts=1, dur=1,
                                                           args={'External id': 999, 'stream': 1})]}))
            w = workload()
            w.manifest['native_trace_artifacts'] = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
            result = prepare([w], dict(chrome_traces=[dict(rank=0, step=0, path=str(path))]))
            self.assertEqual(result.nodes[0].name, 'op')
            self.assertTrue(any('correlation' in gap for gap in result.manifest['unresolved']))

    def test_resolution_requires_exact_gap_and_retains_auditable_assertion(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'integration-evidence.json'
            path.write_text('{"scope":"synthetic test only"}')
            evidence = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            resolution = dict(gap='unknown workspace', scope='integration_assertion',
                              reason='Integration supplied a complete backing allocation inventory.', evidence=evidence)
            result = prepare([workload()], dict(resolutions=[resolution]))
            self.assertEqual(result.manifest['unresolved'], [])
            record = result.manifest['preparation']['resolved_gaps'][0]
            self.assertFalse(record['empirical_verification'])
            self.assertEqual(record['validation'], 'artifact_checksum_only')
            with self.assertRaisesRegex(ValueError, 'existing unresolved'):
                prepare([workload()], dict(resolutions=[dict(resolution, gap='unknown other gap')]))
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                prepare([workload()], dict(resolutions=[resolution]))

    def test_memory_contract_can_be_supplied_without_claiming_verification(self):
        result = prepare([workload()], dict(memory={'coverage': 'partial'}, rank_placement={'0': 'device0'}))
        self.assertEqual(result.manifest['memory'], {'coverage': 'partial'})
        self.assertEqual(result.manifest['rank_placement'], {'0': 'device0'})
        self.assertEqual(result.manifest['unresolved'], ['unknown workspace'])

    def test_memory_aliases_follow_namespace_and_backing_reconciliation(self):
        w = workload(storage=[Storage('a', 0, 8, 0, [0]), Storage('b', 0, 8, 0, [0])])
        w.manifest['memory'] = dict(storage_aliases={'b': 'a'}, allocations=[dict(storage_id='a')])
        merged = prepare([w])
        a, b = [s.id for s in merged.storage]
        self.assertEqual(merged.manifest['memory']['storage_aliases'], {b: a})
        self.assertEqual(merged.manifest['memory']['allocations'][0]['storage_id'], a)
        result = prepare([w], dict(evidence=EVIDENCE, allocations=[dict(rank=0, id='backing',
            source_storage_ids=['a', 'b'], size=8, producer=None, consumers=[ref(0)])]))
        self.assertEqual(result.manifest['memory']['storage_aliases'], {})
        self.assertEqual(result.manifest['memory']['allocations'][0]['storage_id'], result.storage[0].id)

    def test_communication_context_is_part_of_native_cost_identity(self):
        binding = dict(implementation={'name': 'nccl'}, context={'dtype': 'f32'})
        comm = dict(kind='collective', nodes=[ref(0), ref(1)], group='tp0', sequence=0,
                    operation='all_reduce', participants=[0, 1], payload_bytes=16)
        config = dict(evidence=EVIDENCE, regions=[dict(nodes=[ref(rank)], **binding) for rank in (0, 1)],
                      communications=[comm])
        result = prepare([workload(0, 2), workload(1, 2)], config)
        keys = [n.attrs['cost_key'] for n in result.nodes]
        for node in result.nodes:
            native = node.attrs['native_measurement']
            self.assertEqual(node.attrs['cost_key'], native_cost_key(native['implementation'], native['context']))
            self.assertEqual(native['context']['communication']['rank'], node.rank)
        comm['payload_bytes'] = 32
        other = prepare([workload(0, 2), workload(1, 2)], config)
        self.assertTrue(all(a != b.attrs['cost_key'] for a, b in zip(keys, other.nodes)))

    def test_rendezvous_contraction_cycle_rejected(self):
        a = workload(0, 2, nodes=[Node(0, 'first', 0), Node(1, 'second', 0, deps=[0])], storage=[])
        b = workload(1, 2, nodes=[Node(0, 'first', 1), Node(1, 'second', 1, deps=[0])], storage=[])
        base = dict(kind='collective', group='tp', operation='all_reduce', participants=[0, 1], payload_bytes=8)
        with self.assertRaisesRegex(ValueError, 'contraction creates'):
            prepare([a, b], dict(evidence=EVIDENCE, communications=[
                dict(base, sequence=0, nodes=[ref(0, 0), ref(1, 1)]),
                dict(base, sequence=1, nodes=[ref(0, 1), ref(1, 0)])]))

    def test_chrome_only_capture_preserves_events_and_reconciles_rank_stream_order(self):
        from titan_sim.native_runner import measure
        with tempfile.TemporaryDirectory() as root:
            inputs = []
            for rank in (0, 1):
                w = workload(rank, 2, nodes=[], storage=[])
                w.manifest.update(observer='pytorch_profiler_chrome_only', execution_trace=False,
                                  native_trace_artifacts=[])
                for step in (0, 1):
                    path = Path(root) / f'rank{rank}' / f'step{step}.chrome.json'
                    path.parent.mkdir(exist_ok=True)
                    event = dict(name='native_fused', cat='kernel', ph='X', ts=100 + step + rank * 10,
                                 dur=2, args={'stream': 7, 'grid': [1, 1, 1]})
                    path.write_text(json.dumps({'traceEvents': [event]}))
                    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                    implementation = dict(name='native_fused', category='kernel', launch={'grid': [1, 1, 1]})
                    context = dict(hardware_id=HARDWARE['id'], occurrence_only=True, artifact_sha256=checksum,
                                   event_index=0, rank=rank, step=step, provenance_sha256='b'*64)
                    binding = dict(kind='chrome_gpu_event', artifact_sha256=checksum, event_index=0,
                                   event_name='native_fused', event_category='kernel',
                                   implementation=implementation, context=context)
                    w.nodes.append(Node(step, 'native_fused', rank, deps=[0] if step else [], attrs=dict(
                        step=step, stream=7, resources=[f'gpu:{rank}:stream:7'], measurement_scope='occurrence_only',
                        native_measurement=binding, cost_key=native_cost_key(implementation, context))))
                    w.manifest['native_trace_artifacts'].append(dict(path=str(path), sha256=checksum))
                inputs.append(w)
            result = prepare(inputs)
            self.assertEqual(len(result.nodes), 4)
            self.assertEqual([n.deps for n in result.nodes], [[], [0], [], [2]])
            self.assertEqual(len({n.attrs['cost_key'] for n in result.nodes}), 4)
            self.assertEqual(result.manifest['preparation']['verified_captured_gpu_events'], 4)
            self.assertEqual(result.manifest['preparation']['lowered_operator_occurrences'], 0)
            self.assertFalse(any('correlation' in gap for gap in result.manifest['unresolved']))
            self.assertTrue(any('Occurrence-only' in gap for gap in result.manifest['unresolved']))
            self.assertEqual(result.storage, [])
            catalog = measure(result, dict(mode='trace', hardware_id=HARDWARE['id'], accept_profiler_overhead=True))
            self.assertEqual(len(catalog), 4)
            inputs[0].nodes[0].attrs['native_measurement']['context'].pop('occurrence_only')
            with self.assertRaisesRegex(ValueError, 'exact occurrence binding'):
                prepare(inputs)

    def test_unverified_storage_evidence_survives_allrank_merge(self):
        a, b = workload(0, 2), workload(1, 2)
        a.manifest['memory'] = {'storage_evidence': 'unverified'}
        result = prepare([a, b])
        self.assertEqual(result.manifest['memory']['storage_evidence'], 'unverified')
        with self.assertRaisesRegex(ValueError, 'evidence source and sha256'):
            prepare([a, b], dict(memory={'storage_evidence': 'exact'}))
        promoted = prepare([a, b], dict(memory={'storage_evidence': 'exact'}, evidence=EVIDENCE))
        self.assertEqual(promoted.manifest['memory']['storage_evidence'], 'exact')
        self.assertEqual(promoted.manifest['preparation']['storage_evidence_promotion']['source_ranks'], [[0]])
        self.assertFalse(promoted.manifest['preparation']['storage_evidence_promotion']['empirical_verification'])

    def test_legacy_native_storage_is_unverified_and_promotion_requires_evidence(self):
        for observer in ('pytorch_execution_trace_and_profiler', 'pytorch_profiler_chrome_only'):
            for nested in (False, True):
                with self.subTest(observer=observer, nested=nested):
                    source = workload()
                    if nested:
                        source.manifest['source_manifests'] = [dict(observer=observer)]
                    else:
                        source.manifest['observer'] = observer
                    result = prepare([source])
                    self.assertEqual(result.manifest['memory']['storage_evidence'], 'unverified')
                    with self.assertRaisesRegex(ValueError, 'evidence source and sha256'):
                        prepare([source], dict(memory={'storage_evidence': 'exact'}))
                    promoted = prepare([source], dict(memory={'storage_evidence': 'exact'}, evidence=EVIDENCE))
                    self.assertEqual(promoted.manifest['memory']['storage_evidence'], 'exact')
                    self.assertEqual(promoted.manifest['preparation']['storage_evidence_promotion']['previous'], 'unverified')


if __name__ == '__main__':
    unittest.main()
