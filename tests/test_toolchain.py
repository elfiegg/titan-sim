import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
from titan_sim.core import Node, Storage, simulate
from titan_sim.workload import read_bundle, write_bundle


def hardware():
    return dict(id='synthetic-test', capacity_bytes=2**30, reserve_fraction=0.1,
                effective_flops_per_second=1e6, effective_hbm_bytes_per_second=1e6,
                interconnect_bytes_per_second=1e6, pp_bytes_per_second=1e6,
                comm_latency_seconds=1e-6, launch_seconds=0.)


def checks():
    h=hardware()
    m={'ranks':[0,1],'unresolved':[]}
    # Late rank prevents early allreduce completion and downstream execution.
    ns=[Node(1,'late',1,attrs={'legacy_seconds':10}),
        Node(2,'reduce0',0,'collective',attrs={'rendezvous':'c','participants':[0,1],'payload_bytes':400,'resources':['link']}),
        Node(3,'reduce1',1,'collective',[1],{'rendezvous':'c','participants':[0,1],'payload_bytes':400,'resources':['link']}),
        Node(4,'after',0,deps=[2],attrs={'legacy_seconds':2})]
    ss=[Storage('aliased',0,100,2,[4])]
    r=simulate(ns,ss,m,h)
    assert r['timing']['2'][0]==10 and r['timing']['4'][0]>=r['timing']['3'][1]
    assert r['modeled_peak_bytes'][0]==100
    # Two operations share a link even with no data dependency.
    ns=[Node(1,'a',0,attrs={'legacy_seconds':2,'resources':['link']}),
        Node(2,'b',1,attrs={'legacy_seconds':3,'resources':['link']})]
    assert simulate(ns,[],m,h)['seconds']==5
    ns[1].attrs['resources']=['other_link']
    assert simulate(ns,[],m,h)['seconds']==3
    # A producer allocation survives its consumers; aliases count once.
    ns=[Node(1,'produce',0,attrs={'legacy_seconds':2}),
        Node(2,'consume',0,deps=[1],attrs={'legacy_seconds':3}),
        Node(3,'later',0,deps=[2],attrs={'legacy_seconds':1})]
    ss=[Storage('base',0,100,1,[2]),Storage('later',0,80,3,[])]
    assert simulate(ns,ss,m,h)['modeled_peak_bytes'][0]==100
    ss[0].consumers.append(3)
    assert simulate(ns,ss,m,h)['modeled_peak_bytes'][0]==180
    rejected=[]
    cases=[([Node(1,'unknown',0)],[],m,False),
           ([Node(1,'cycle',0,deps=[1],attrs={'legacy_seconds':1})],[],m,False),
           ([Node(1,'unmatched',0,'send',attrs={'peer':1,'rendezvous':'x','payload_bytes':1})],[],m,False),
           (ns,ss,dict(m,unresolved=['native buffers missing']),True),
           (ns,ss,m,True)]
    for n,s,manifest,strict in cases:
        try:simulate(n,s,manifest,h,strict=strict)
        except ValueError as e:rejected.append(str(e))
        else:raise AssertionError('invalid case unexpectedly accepted')
    return {'collective_waits_for_all_ranks':True,'shared_link_serializes':True,
            'independent_links_overlap':True,'allocation_last_use_and_aliasing':True,
            'rejected_cases':rejected}


class ToolchainTests(unittest.TestCase):
    def test_unknown_event_kind_and_invalid_cost_key_are_rejected(self):
        for node in [Node(1,'arbitrary_name',0,kind='unsupported'),
                     Node(1,'arbitrary_name',0,attrs={'cost_key':[]})]:
            with self.assertRaises(ValueError):
                simulate([node],[],{'ranks':[0],'unresolved':[]},hardware())

    def test_scheduler_and_storage_contracts(self):
        result = checks()
        self.assertTrue(result['collective_waits_for_all_ranks'])
        self.assertTrue(result['allocation_last_use_and_aliasing'])
        self.assertEqual(len(result['rejected_cases']), 5)

    def test_direct_workload_round_trip_and_schema_rejection(self):
        nodes = [Node(1, 'producer', 0, attrs={'legacy_seconds': 2}),
                 Node(2, 'consumer', 0, deps=[1], attrs={'legacy_seconds': 3})]
        storages = [Storage('shared_backing', 0, 128, 1, [2])]
        manifest = {'ranks': [0], 'unresolved': ['synthetic timing scenario']}
        with tempfile.TemporaryDirectory() as temp:
            write_bundle(temp, nodes, storages, manifest, hardware())
            result = simulate(*read_bundle(temp))
            self.assertEqual(result['seconds'], 5)
            self.assertEqual(result['modeled_peak_bytes'][0], 128)
            self.assertEqual(result['memory_verdict'], 'inconclusive')
            with self.assertRaisesRegex(ValueError, 'incomplete workload'):
                simulate(*read_bundle(temp), strict=True)
            path = Path(temp)/'workload.json'
            data = json.loads(path.read_text())
            data['version'] = 999
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'unsupported workload'):
                read_bundle(temp)

    def test_calibration_requires_hardware_and_sample_evidence(self):
        nodes = [Node(1, 'measured_component', 0, attrs={'cost_key': 'case'})]
        manifest = {'ranks': [0], 'unresolved': []}
        h = hardware()
        samples = [0.01, 0.02, 0.03]
        record = dict(hardware_id=h['id'], seconds=0.02, samples_seconds=samples,
                      evidence_sha256=hashlib.sha256(json.dumps(samples, separators=(',', ':')).encode()).hexdigest())
        result = simulate(nodes, [], manifest, h, {'case': record}, strict=True)
        self.assertEqual(result['seconds'], 0.02)
        with self.assertRaisesRegex(ValueError, 'hardware/evidence mismatch'):
            simulate(nodes, [], manifest, h, {'case': dict(record, hardware_id='another-device')}, strict=True)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            simulate(nodes, [], manifest, h, {'case': dict(record, samples_seconds=[1])}, strict=True)

    def test_cpu_entry_point_without_site_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            workload = Path(temp)/'workload.json'
            p = subprocess.run([sys.executable, '-S', '-m', 'titan_sim', 'capture',
                                '--adapter', 'examples.minimal:capture', '--config',
                                'examples/minimal.json', '--output', str(workload)],
                               capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(p.returncode, 0, p.stderr)
            p = subprocess.run([sys.executable, '-S', '-m', 'titan_sim', 'simulate', str(workload)],
                               capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(p.returncode, 0, p.stderr)
            result = json.loads(p.stdout)
            self.assertAlmostEqual(result['seconds'], 0.005)
            self.assertEqual(result['memory_verdict'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()
