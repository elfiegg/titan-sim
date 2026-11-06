import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from titan_sim.cli import main
from titan_sim.core import Node, Storage
from titan_sim.workload import Workload, write_workload


class GenericCLITests(unittest.TestCase):
    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(map(str, args)))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_external_adapter_runner_and_strict_simulation(self):
        module = types.ModuleType('test_external_workload')
        def capture(config):
            return Workload([Node(1, 'user_defined_operation', 0, attrs={'cost_key':'unique-op'})],
                            [Storage('output',0,64,1,[])],
                            {'ranks':[0],'unresolved':[],'scope':'synthetic test fixture only'},
                            {'id':'test-hardware','capacity_bytes':1024})
        def measure(workload, config):
            samples = config['synthetic_test_samples']
            return {'unique-op':dict(hardware_id=workload.hardware['id'], seconds=0.02,
                                     samples_seconds=samples,
                                     evidence_sha256=hashlib.sha256(json.dumps(samples,separators=(',',':')).encode()).hexdigest())}
        module.capture, module.measure = capture, measure
        sys.modules[module.__name__] = module
        try:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp); config=root/'config.json'; workload=root/'workload.json'
                catalog=root/'costs.json'; output=root/'result.json'
                config.write_text(json.dumps({'synthetic_test_samples':[0.01,0.02,0.03]}))
                code, stdout, stderr=self.invoke('capture','--adapter',module.__name__+':capture',
                                                '--config',config,'--output',workload)
                self.assertEqual(code,0,stderr)
                self.assertEqual(json.loads(stdout)['nodes'],1)
                code, stdout, stderr=self.invoke('validate',workload)
                self.assertEqual(code,0,stderr)
                self.assertEqual(json.loads(stdout)['structural_status'],'valid')
                code, stdout, stderr=self.invoke('calibrate',workload,'--runner',module.__name__+':measure',
                                                '--config',config,'--output',catalog)
                self.assertEqual(code,0,stderr)
                self.assertEqual(json.loads(stdout)['missing_cost_keys'],[])
                code, stdout, stderr=self.invoke('simulate',workload,'--calibration',catalog,'--strict','--output',output)
                self.assertEqual(code,0,stderr)
                result=json.loads(output.read_text())
                self.assertEqual(result['seconds'],0.02)
                self.assertEqual(result['modeled_peak_bytes']['0'],64)
                self.assertIn('timing',result)
                self.assertEqual(json.loads(stdout)['status'],'calibrated_prediction')
                bad=json.loads(catalog.read_text());bad['unique-op']['hardware_id']='wrong'
                catalog.write_text(json.dumps(bad))
                code, _, stderr=self.invoke('simulate',workload,'--calibration',catalog,'--strict')
                self.assertEqual(code,2)
                self.assertIn('hardware mismatch',json.loads(stderr)['error'])
        finally:
            del sys.modules[module.__name__]

    def test_structural_validation_does_not_imply_complete_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'workload.json'
            write_workload(file,Workload([Node(1,'unknown',0)],[],
                                        {'ranks':[0],'unresolved':['missing native buffers']},
                                        {'id':'test','capacity_bytes':1024}))
            code, stdout, stderr=self.invoke('validate',file)
            self.assertEqual(code,0,stderr)
            self.assertEqual(json.loads(stdout)['unresolved'],['missing native buffers'])
            code, _, stderr=self.invoke('simulate',file,'--strict')
            self.assertEqual(code,2)
            self.assertIn('incomplete workload',json.loads(stderr)['error'])

    def test_invalid_adapter_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=Path(temp)/'config.json';cfg.write_text('{}')
            code, _, stderr=self.invoke('capture','--adapter','json:loads','--config',cfg,
                                        '--output',Path(temp)/'workload.json')
            self.assertEqual(code,2)
            self.assertTrue(json.loads(stderr)['error'])

    def test_partial_capture_is_written_but_returns_failure(self):
        workload = Workload([], [], {'ranks':[0], 'capture_status':'blocked',
                           'unresolved':['native initialization failed']}, {'id':'test','capacity_bytes':1024})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'config.json'; config.write_text('{}')
            with patch('titan_sim.adapters.capture', return_value=workload):
                code, stdout, stderr = self.invoke('capture', '--adapter', 'test:call', '--config', config,
                                                   '--output', root/'workload.json')
            self.assertEqual(code, 2, stderr)
            self.assertEqual(json.loads(stdout)['capture_status'], 'blocked')
            self.assertTrue((root/'workload.json').is_file())

    def test_distributed_output_requires_rank_placeholder(self):
        workload = Workload([], [], {'ranks':[2], 'unresolved':[]}, {'id':'test','capacity_bytes':1024})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'config.json'; config.write_text('{}')
            with patch.dict('os.environ', WORLD_SIZE='4', RANK='2'):
                with patch('titan_sim.adapters.capture', return_value=workload) as invoke:
                    code, _, stderr = self.invoke('capture', '--adapter', 'test:call', '--config', config,
                                                  '--output', root/'workload.json')
                    self.assertEqual(code, 2)
                    invoke.assert_not_called()
                    self.assertIn('collisions', stderr)
                    code, _, stderr = self.invoke('capture', '--adapter', 'test:call', '--config', config,
                                                  '--output', root/'workload.rank{rank}.json')
                    self.assertEqual(code, 0, stderr)
                    self.assertTrue((root/'workload.rank2.json').is_file())


if __name__ == '__main__':
    unittest.main()
