import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from titan_sim.cli import main
from titan_sim.core import Node
from titan_sim.workload import Workload, read_workload, write_workload


class CalibrationCasesCLITests(unittest.TestCase):
    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(map(str, args)))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_compute_selection_is_separate_from_original_workload(self):
        def tensor(shape, stride, sid):
            return dict(shape=shape,stride=stride,storage_id=sid,storage_offset=0,
                        dtype='torch.bfloat16',layout='torch.strided',device='cuda:2')
        mm = Node(0,'aten.mm.default',2,attrs=dict(status='returned',operator='aten.mm.default',
            arguments={'args':[tensor([2,3],[3,1],'a'),tensor([3,4],[4,1],'b')],'kwargs':{}},
            outputs=tensor([2,4],[4,1],'c')))
        unknown = Node(1,'custom.operator.default',2,attrs={'status':'returned'})
        source = Workload([mm,unknown],[],{'ranks':[2],'world_size':8,
            'capture_status':'complete','provenance':{'fixture':'synthetic'}},{'id':'synthetic'})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_workload(root/'source.json',source)
            before=(root/'source.json').read_bytes()
            code,stdout,stderr=self.invoke('calibration-cases',root/'source.json','--kind','compute',
                '--output',root/'cases.json','--report',root/'inventory.json')
            self.assertEqual(code,0,stderr)
            result=json.loads(stdout)
            self.assertEqual(result['selected_cases'],1)
            self.assertEqual(result['skipped_operations'],1)
            self.assertEqual(result['measurement_world_size'],1)
            self.assertEqual(read_workload(root/'cases.json').manifest['ranks'],[0])
            self.assertEqual(before,(root/'source.json').read_bytes())
            self.assertEqual(len(json.loads((root/'inventory.json').read_text())['skipped']),1)
            code,_,stderr=self.invoke('calibration-cases',root/'source.json','--kind','compute',
                '--world-size','8','--output',root/'invalid.json')
            self.assertEqual(code,2)
            self.assertIn('one GPU',stderr)
            code,_,stderr=self.invoke('calibration-cases',root/'source.json','--kind','compute',
                '--peer-workload',root/'source.json','--output',root/'invalid.json')
            self.assertEqual(code,2)
            self.assertIn('peer-workload',stderr)

    def test_communication_requires_topology_before_selecting_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_workload(root/'source.json',Workload([],[],{'ranks':[0]},{'id':'synthetic'}))
            code,_,stderr=self.invoke('calibration-cases',root/'source.json','--kind','communication',
                '--world-size','2','--output',root/'cases.json')
            self.assertEqual(code,2)
            self.assertIn('--topology',stderr)
            self.assertFalse((root/'cases.json').exists())

    def test_communication_world_size_preserves_explicit_case_subset(self):
        from titan_sim.communication_cases import prepare_communication_cases
        def tensor(size, storage):
            return dict(shape=[size], stride=[1], dtype='torch.bfloat16', device='cuda:0',
                        layout='torch.strided', storage_offset=0, storage_id=storage)
        communication = dict(operator='c10d._allgather_base_.default', operator_schema='fixture',
            operation='all_gather_into_tensor', participants=[0,1], group_size=2,
            backend='nccl', group_name='fixture', reduce_op=None, async_op=True, timeout=-1,
            runtime_context={}, evidence='native_dispatch_and_resolved_process_group')
        nodes=[]
        for size in (4,8):
            record=copy.deepcopy(communication)
            record['tensors']={'input_tensor':tensor(size,'input'), 'output_tensor':tensor(size*2,'output')}
            nodes.append(Node(len(nodes),record['operator'],0,attrs=dict(status='returned',communication=record)))
        source=Workload(nodes,[],{'ranks':[0],'world_size':2,'provenance':{}},{'id':'fixture'})
        inventory=prepare_communication_cases(source)
        self.assertEqual(len(inventory['cases']),2)
        wanted=inventory['cases'][0]['case_id']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_workload(root/'source.json',source)
            (root/'topology.json').write_text(json.dumps({'fixture':'two devices'}))
            code,stdout,stderr=self.invoke('calibration-cases',root/'source.json','--kind','communication',
                '--world-size','2','--case-id',wanted,'--topology',root/'topology.json',
                '--output',root/'selected.json')
            self.assertEqual(code,0,stderr)
            self.assertEqual(json.loads(stdout)['selected_cases'],1)
            selected=read_workload(root/'selected.json')
            self.assertEqual({n.attrs['native_measurement']['context']['case']['case_id']
                              for n in selected.nodes},{wanted})
