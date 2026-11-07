"""Synthetic fixtures exercise data contracts, never empirical GPU validation.

Every duration below is hand-authored test data. Passing these tests establishes
only preparation/calibration/simulation integration and conservative reporting.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from titan_sim.core import Node, Storage, simulate
from titan_sim.native_runner import measure
from titan_sim.prepare import prepare
from titan_sim.workload import Workload


class SyntheticNativePipelineTests(unittest.TestCase):
    def test_two_rank_multi_launch_and_collective_pipeline_preserves_every_activity(self):
        with tempfile.TemporaryDirectory() as root:
            captures = []
            for rank, durations in enumerate(((8, 10, 4), (12, 14, 6))):
                path = Path(root)/f'rank{rank}'/'step0.chrome.json'
                path.parent.mkdir()
                events = [
                    dict(name='synthetic compiled region', cat='cpu_op', args={'External id': 20, 'Record function id': 30}),
                    dict(name='synthetic collective', cat='cpu_op', args={'External id': 21, 'Record function id': 31}),
                ]
                for index, duration in enumerate(durations):
                    events.append(dict(name='synthetic_matmul' if index < 2 else 'synthetic_nccl',
                        cat='kernel', ph='X', ts=100*rank+20*index, dur=duration,
                        args={'External id': 20 if index < 2 else 21, 'stream': 7, 'grid': [1, 1, 1]}))
                path.write_text(json.dumps(dict(traceEvents=events, fixture='synthetic, not measured')))
                artifact = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                nodes = [Node(11, 'synthetic compiled projection', rank,
                    attrs=dict(cost_key='synthetic:compiled:shape8:source-fixed', step=0,
                               execution_trace_id=100, execution_trace_attrs={'rf_id': 30})),
                    Node(29, 'synthetic collective projection', rank, deps=[11],
                    attrs=dict(cost_key='synthetic:collective:shape8:source-fixed', step=0,
                               execution_trace_id=101, execution_trace_attrs={'rf_id': 31}))]
                captures.append(Workload(nodes, [Storage('view', rank, 64, 11, [11, 29])],
                    dict(ranks=[rank], world_size=2, capture_status='complete',
                         provenance={'recipe_args': ['synthetic.recipe'], 'torch_version': 'fixture'},
                         native_trace_artifacts=[artifact], unresolved=['Synthetic fixture has no physical allocation evidence.']),
                    dict(id='synthetic-device', capacity_bytes=1024)))
            prepared = prepare(captures, dict(evidence=dict(source='synthetic test assertion', sha256='a'*64),
                communications=[dict(kind='collective', nodes=[dict(rank=0, id=29), dict(rank=1, id=29)],
                    group='synthetic-world', sequence=0, operation='all_reduce', participants=[0, 1], payload_bytes=64)]))
            # Two launches from each fused host projection must remain two GPU
            # events, despite sharing the same exact cost key.
            self.assertEqual(len(prepared.nodes), 6)
            self.assertEqual(len({s.id for s in prepared.storage}), 2)
            known_ids = {n.id for n in prepared.nodes}
            self.assertTrue(all(c in known_ids for s in prepared.storage for c in s.consumers))
            catalog = measure(prepared, dict(mode='trace', hardware_id='synthetic-device', accept_profiler_overhead=True))
            self.assertEqual(sorted(len(r['samples_seconds']) for r in catalog.values()), [1, 1, 4])
            self.assertTrue(all(r['measurement_kind'] == 'profiled_gpu_activity' for r in catalog.values()))
            self.assertTrue(all(r['independent_validation'] is False for r in catalog.values()))
            result = simulate(*prepared.arguments(), calibration=catalog)
            self.assertAlmostEqual(result['seconds'], 28e-6)
            self.assertEqual(result['cost_coverage'], {'measured_exact_key': 6})
            self.assertEqual(result['memory_verdict'], 'inconclusive')
            self.assertTrue(result['unresolved'])
            self.assertNotEqual(result['status'], 'calibrated_prediction')
            with self.assertRaisesRegex(ValueError, 'incomplete workload'):
                simulate(*prepared.arguments(), calibration=catalog, strict=True)

    def test_repeated_steps_reuse_rf_ids_without_cross_step_cost_or_storage_aliasing(self):
        with tempfile.TemporaryDirectory() as root:
            artifacts, nodes, storage = [], [], []
            for step, duration in enumerate((5, 9)):
                path = Path(root)/f'step{step}.chrome.json'
                path.write_text(json.dumps({'traceEvents': [
                    dict(cat='cpu_op', name='synthetic host', args={'External id': 1, 'Record function id': 2}),
                    dict(cat='kernel', name='synthetic kernel', ph='X', ts=0, dur=duration,
                         args={'External id': 1, 'stream': 0})]}))
                artifacts.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                nid = 100+10*step
                nodes.append(Node(nid, 'projection', 0, deps=[100] if step else [],
                    attrs=dict(step=step, cost_key=f'synthetic:shape{step+1}', execution_trace_id=12,
                               execution_trace_attrs={'rf_id': 2})))
                storage.append(Storage(f'step{step}:storage', 0, 16*(step+1), nid, [nid]))
            captured = Workload(nodes, storage, dict(ranks=[0], world_size=1, native_trace_artifacts=artifacts),
                                dict(id='synthetic-device', capacity_bytes=1024))
            prepared = prepare([captured])
            self.assertEqual(len(prepared.nodes), 2)
            self.assertEqual(prepared.nodes[1].deps, [prepared.nodes[0].id])
            self.assertNotEqual(prepared.nodes[0].attrs['cost_key'], prepared.nodes[1].attrs['cost_key'])
            for step, allocation in enumerate(prepared.storage):
                self.assertEqual(allocation.consumers, [prepared.nodes[step].id])
            catalog = measure(prepared, dict(mode='trace', hardware_id='synthetic-device', accept_profiler_overhead=True))
            self.assertEqual(sorted(record['seconds'] for record in catalog.values()), [5e-6, 9e-6])
            result = simulate(*prepared.arguments(), calibration=catalog)
            self.assertAlmostEqual(result['seconds'], 14e-6)
            self.assertEqual(result['memory_verdict'], 'inconclusive')

    def test_unknown_gpu_activity_remains_an_explicit_gap(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'step0.chrome.json'
            events = [dict(cat='cpu_op', name='synthetic host', args={'External id': 1, 'Record function id': 2}),
                      dict(cat='kernel', name='known synthetic kernel', ph='X', ts=0, dur=5,
                           args={'External id': 1, 'stream': 0}),
                      dict(cat='kernel', name='unknown synthetic kernel', ph='X', ts=5, dur=1000,
                           args={'External id': 999, 'stream': 1})]
            path.write_text(json.dumps({'traceEvents': events}))
            workload = Workload([Node(4, 'projection', 0, attrs=dict(step=0, cost_key='synthetic:shape',
                execution_trace_id=12, execution_trace_attrs={'rf_id': 2}))], [],
                dict(ranks=[0], world_size=1, native_trace_artifacts=[dict(path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())]),
                dict(id='synthetic-device', capacity_bytes=1024))
            prepared = prepare([workload])
            self.assertTrue(any('correlation' in gap for gap in prepared.manifest['unresolved']))
            catalog = measure(prepared, dict(mode='trace', hardware_id='synthetic-device', accept_profiler_overhead=True))
            result = simulate(*prepared.arguments(), calibration=catalog)
            self.assertAlmostEqual(result['seconds'], 5e-6)
            self.assertEqual(result['memory_verdict'], 'inconclusive')
            self.assertTrue(result['unresolved'])


if __name__ == '__main__':
    unittest.main()
