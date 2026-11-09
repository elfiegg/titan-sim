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

    def test_memory_fit_requirement_keeps_diagnostic_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            workload=Workload([Node(1,'known',0,attrs={'zero_cost_semantic':True})],[],
                              {'ranks':[0],'unresolved':[]},{'id':'test','capacity_bytes':1024})
            write_workload(root/'workload.json',workload)
            code,stdout,stderr=self.invoke('simulate',root/'workload.json','--strict',
                                          '--require-memory-fit','--output',root/'result.json')
            self.assertEqual(code,2,stderr)
            self.assertEqual(json.loads(stdout)['memory_verdict'],'inconclusive')
            self.assertTrue(json.loads((root/'result.json').read_text())['memory_unresolved'])

    def test_prepare_combines_rank_artifacts_with_explicit_placement(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for rank in (0,1):
                write_workload(root/f'rank{rank}.json',Workload([Node(1,'event',rank,
                    attrs={'zero_cost_semantic':True})],[],
                    {'ranks':[rank],'world_size':2,'unresolved':[]},
                    {'id':'test','capacity_bytes':1024}))
            (root/'prepare.json').write_text(json.dumps({'rank_placement':{'0':'a','1':'b'},'useful_tokens':512}))
            code,stdout,stderr=self.invoke('prepare',root/'rank0.json',root/'rank1.json',
                '--config',root/'prepare.json','--output',root/'prepared.json')
            self.assertEqual(code,0,stderr)
            self.assertEqual(json.loads(stdout)['ranks'],2)
            result=json.loads((root/'prepared.json').read_text())
            self.assertEqual(result['manifest']['useful_tokens'],512)
            self.assertEqual(result['manifest']['rank_placement'],{'0':'a','1':'b'})

    def test_runtime_check_blocked_retains_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'manifest.json').write_text(json.dumps({'schema_version':1,'status':'partial',
                'codebase':{},'recipe':{},'target':{},'validation':{}}))
            code,stdout,stderr=self.invoke('runtime-check',root/'manifest.json','--output',root/'check.json')
            self.assertEqual(code,2,stderr)
            self.assertEqual(json.loads(stdout)['status'],'blocked')
            self.assertTrue((root/'check.json').is_file())

    def test_case_binding_roundtrip_and_stale_source_rejection(self):
        from tests.test_case_binding import source
        from titan_sim.workload import read_workload
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original=source()
            write_workload(root/'source.json',original)
            code,_,stderr=self.invoke('calibration-cases',root/'source.json','--kind','compute',
                '--output',root/'cases.json')
            self.assertEqual(code,0,stderr)
            code,stdout,stderr=self.invoke('bind-cases',root/'source.json','--measurement',root/'cases.json',
                '--report',root/'audit.json','--output',root/'bound.json')
            self.assertEqual(code,0,stderr)
            self.assertEqual(json.loads(stdout)['bound'],1)
            bound=read_workload(root/'bound.json')
            cases=read_workload(root/'cases.json')
            self.assertEqual(bound.nodes[0].attrs['cost_key'],cases.nodes[0].attrs['cost_key'])
            self.assertEqual(json.loads((root/'audit.json').read_text())['unbound'][0]['source_node_id'],8)
            original.nodes[1].name='changed source'
            write_workload(root/'source.json',original)
            code,_,stderr=self.invoke('bind-cases',root/'source.json','--measurement',root/'cases.json',
                '--output',root/'invalid.json')
            self.assertEqual(code,2)
            self.assertIn('fingerprint',stderr)
            self.assertFalse((root/'invalid.json').exists())

    def test_conditional_memory_cannot_pass_cli_fit_gate(self):
        from tests.test_memory_evidence import MemoryEvidenceTests
        fixture=MemoryEvidenceTests()
        workload,context,_=fixture.fixture()
        for node in workload.nodes:
            node.attrs['zero_cost_semantic']=True
        allocation=fixture.allocation(storage_ids=[],category='workspaces',lower_bytes=1,upper_bytes=1,
            kind='conditional',persistent=True,conditions=['unverified size bound'])
        bundle=fixture.bundle(workload,context,allocations=[allocation])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_workload(root/'source.json',workload)
            (root/'context.json').write_text(json.dumps(context))
            (root/'bundle.json').write_text(json.dumps(bundle))
            code,_,stderr=self.invoke('bind-memory',root/'source.json','--bundle',root/'bundle.json',
                '--context',root/'context.json','--output',root/'bound.json')
            self.assertEqual(code,0,stderr)
            code,stdout,stderr=self.invoke('simulate',root/'bound.json','--require-memory-fit',
                '--output',root/'prediction.json')
            self.assertEqual(code,2,stderr)
            result=json.loads((root/'prediction.json').read_text())
            self.assertEqual(result['memory_verdict'],'inconclusive')
            self.assertEqual(result['physical_memory']['conditional_memory_verdict'],'estimated_fit')

    def test_controlled_window_can_be_prepared_but_not_bound_to_training(self):
        from tests.test_execution_windows import fixture
        source,manifest=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_workload(root/'source.json',source)
            (root/'manifest.json').write_text(json.dumps(manifest))
            code,_,stderr=self.invoke('window-cases',root/'source.json','--config',root/'manifest.json',
                '--output',root/'window.json')
            self.assertEqual(code,0,stderr)
            code,_,stderr=self.invoke('bind-window',root/'source.json','--measurement',root/'window.json',
                '--output',root/'invalid.json')
            self.assertEqual(code,2)
            self.assertIn('controlled native windows',stderr)
            self.assertFalse((root/'invalid.json').exists())

    def test_freeze_prediction_requires_complete_source_scope(self):
        from tests.test_prediction_validation import fixture
        source,costs,context=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_workload(root/'source.json',source)
            (root/'costs.json').write_text(json.dumps(costs))
            (root/'context.json').write_text(json.dumps(context))
            code,_,stderr=self.invoke('freeze-prediction',root/'source.json','--calibration',root/'costs.json',
                '--context',root/'context.json','--output',root/'frozen.json')
            self.assertEqual(code,0,stderr)
            self.assertEqual(json.loads((root/'frozen.json').read_text())['context'],context)
            source.manifest['capture_status']='blocked'
            write_workload(root/'source.json',source)
            code,_,stderr=self.invoke('freeze-prediction',root/'source.json','--calibration',root/'costs.json',
                '--context',root/'context.json','--output',root/'invalid.json')
            self.assertEqual(code,2)
            self.assertFalse((root/'invalid.json').exists())

    def test_allocation_history_producer_does_not_close_memory_fit(self):
        from tests.test_allocation_contracts import AllocationContractsTests
        capture,source,context=AllocationContractsTests().fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_workload(root/'source.json',source)
            (root/'capture.json').write_text(json.dumps(capture))
            (root/'context.json').write_text(json.dumps(context))
            code,stdout,stderr=self.invoke('allocation-evidence',root/'source.json','--capture',root/'capture.json',
                '--context',root/'context.json','--report',root/'analysis.json','--output',root/'bundle.json')
            self.assertEqual(code,0,stderr)
            self.assertTrue(json.loads(stdout)['reconstruction_complete'])
            bundle=json.loads((root/'bundle.json').read_text())
            self.assertTrue(all(not row['complete'] for row in bundle['coverage']))
            self.assertTrue(all(row['kind']=='conditional' for row in bundle['allocations']))


if __name__ == '__main__':
    unittest.main()
