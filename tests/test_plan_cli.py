import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from titan_sim.cli import main
from titan_sim.native_plan import seal_plan, write_plan


def fixture():
    leaf = {'storage_id': 'x', 'storage_bytes': 16, 'logical_bytes': 16,
            'device': 'cuda:0', 'shape': [4], 'stride': [1], 'dtype': 'torch.float32', 'path': []}
    out = dict(leaf, storage_id='y')
    graph = {'stage_index': 0, 'role': 'joint', 'outputs': ['sin'], 'nodes': [
        {'name': 'x', 'op': 'placeholder', 'target': 'x', 'inputs': [], 'outputs': [leaf]},
        {'name': 'sin', 'op': 'call_function', 'target': 'aten.sin.default',
         'inputs': ['x'], 'outputs': [out], 'args': [{'node': 'x'}], 'kwargs': {}},
        {'name': 'output', 'op': 'output', 'target': 'output', 'inputs': ['sin'], 'outputs': []}]}
    return seal_plan({'schema_version': 'titan_sim.native_plan.v1',
        'extraction': {'rank': 0}, 'status': 'incomplete',
        'provenance': {'fixture': 'synthetic integration only'},
        'schedule': {'source_attribute': 'pipeline_order_with_comms',
                     'rank_namespace': 'pipeline_group', 'stage_to_rank': {'0': 0},
                     'num_microbatches': 1, 'rank_actions': {'0': [
                         {'stage_index': 0, 'computation_type': 'FULL_FORWARD_BACKWARD', 'microbatch_index': 0}]}},
        'graphs': {'joint': graph}, 'stages': {'0': {'graphs': {'joint': 'joint'}, 'metadata': {}, 'state': {}}},
        'unresolved': ['Synthetic integration fixture has no physical allocator coverage.']})


class PlanCLITests(unittest.TestCase):
    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(map(str, args)))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_export_lower_simulate_retains_memory_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'config.json'
            config.write_text('{}')
            plan_path = root/'plan.json'
            with patch('titan_sim.adapters.resolve', return_value=lambda _: fixture()):
                code, _, stderr = self.invoke('export-plan', '--adapter', 'test:export',
                                              '--config', config, '--output', plan_path)
            self.assertEqual(code, 0, stderr)
            (root/'hardware.json').write_text(json.dumps({'id': 'synthetic', 'capacity_bytes': 100000,
                'effective_flops_per_second': 4, 'effective_hbm_bytes_per_second': 16}))
            config.write_text(json.dumps({'rank_placement': {'0': 'gpu0'},
                'node_attrs': {'joint': {'sin': {'flops': 4, 'bytes': 16}}}, 'useful_tokens': 4}))
            code, _, stderr = self.invoke('lower-plan', plan_path, '--hardware', root/'hardware.json',
                '--config', config, '--output', root/'workload.json')
            self.assertEqual(code, 0, stderr)
            workload = json.loads((root/'workload.json').read_text())
            self.assertTrue(workload['storage'])
            code, stdout, stderr = self.invoke('simulate', root/'workload.json',
                '--require-memory-fit', '--output', root/'result.json')
            self.assertEqual(code, 2, stderr)
            result = json.loads(stdout)
            self.assertEqual(result['memory_verdict'], 'inconclusive')
            self.assertEqual(result['seconds'], 1)
            self.assertEqual(result['useful_tokens_per_second'], 4)
            self.assertTrue(result['memory_unresolved'])

    def test_stale_plan_is_rejected_before_lowering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = fixture()
            plan['schedule']['num_microbatches'] = 128
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'hw.json').write_text('{"id":"test","capacity_bytes":1024}')
            code, _, stderr = self.invoke('lower-plan', root/'plan.json', '--hardware', root/'hw.json',
                                          '--output', root/'workload.json')
            self.assertEqual(code, 2)
            self.assertIn('checksum', stderr)
            self.assertFalse((root/'workload.json').exists())

    def test_distributed_export_requires_unique_output_before_loading_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'config.json').write_text('{}')
            with patch.dict(os.environ, WORLD_SIZE='2', RANK='1'):
                with patch('titan_sim.adapters.resolve') as resolve:
                    code, _, stderr = self.invoke('export-plan', '--config', root/'config.json',
                                                  '--output', root/'plan.json')
            self.assertEqual(code, 2)
            self.assertIn('collisions', stderr)
            resolve.assert_not_called()
