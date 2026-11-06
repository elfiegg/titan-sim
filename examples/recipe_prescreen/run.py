#!/usr/bin/env python3
"""Experimental prescreen CLI. CPU simulation does not import PyTorch or protobuf."""
from __future__ import annotations
import argparse
import copy
import json
import statistics
from pathlib import Path
import time
from titan_sim.core import Node,Storage,simulate
from titan_sim.workload import read_bundle,write_bundle
from .kimi import export_kimi,hardware

ROOT=Path(__file__).parent
RUNS=Path('results/titan-sim')


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


def save_result(path,result):
    # Keep detailed event times only in small cases; full-run summary remains readable.
    if len(result['timing'])>500:
        del result['timing']
    (path/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def exercise(name,data,strict=False):
    path=RUNS/name
    write_bundle(path,*data)
    loaded=read_bundle(path)
    assert len(loaded[0])==len(data[0]) and len(loaded[1])==len(data[1])
    # Exercise persistence before simulation; default is the direct JSON workload.
    result=simulate(*loaded,strict=strict)
    save_result(path,result)
    try:simulate(*loaded,strict=True)
    except ValueError as e:strict_result=str(e)
    else:strict_result='accepted'
    worst=max(result['modeled_peak_bytes'].values(),default=0)
    return {'case':name,'bundle_nodes':len(loaded[0]),'ranks':len(loaded[2]['ranks']),
            'seconds':result['seconds'],'tokens_per_second':result['useful_tokens_per_second'],
            'modeled_peak_GiB':worst/2**30,'memory_verdict':result['memory_verdict'],
            'strict_result':strict_result,'cost_coverage':result['cost_coverage']}


def main():
    global RUNS, read_bundle, write_bundle
    ap=argparse.ArgumentParser()
    ap.add_argument('--example',choices=['all','dense','hybridep','kimi','checks','calibration'],default='checks')
    ap.add_argument('--microbatches',type=int,default=128)
    ap.add_argument('--simulate-workload','--simulate-bundle',dest='simulate_bundle',type=Path)
    ap.add_argument('--format',choices=['json','chakra'],default='json')
    ap.add_argument('--output-dir',type=Path,default=RUNS)
    ap.add_argument('--strict',action='store_true')
    ap.add_argument('--cost-multiplier',type=float,default=1.)
    ap.add_argument('--pp-bandwidth-gbps',type=float,default=50.,help='GB/s, decimal bytes')
    ap.add_argument('--extra-memory-gib',type=float,default=0.)
    ap.add_argument('--calibration',type=Path)
    args=ap.parse_args()
    if args.format=='chakra':
        from titan_sim.bundle import read_bundle,write_bundle
    RUNS=args.output_dir
    RUNS.mkdir(parents=True,exist_ok=True)
    if args.simulate_bundle:
        n,s,m,h=read_bundle(args.simulate_bundle)
        h['cost_multiplier']=args.cost_multiplier
        h['pp_bytes_per_second']=args.pp_bandwidth_gbps*1e9
        if args.extra_memory_gib:
            s += [Storage(f'extra:{r}',r,round(args.extra_memory_gib*2**30),persistent=True,category='explicit_unmeasured_allowance') for r in m['ranks']]
        cal=json.loads(args.calibration.read_text()) if args.calibration else None
        result=simulate(n,s,m,h,calibration=cal,strict=args.strict)
        result.pop('timing',None)
        print(json.dumps({'seconds':result['seconds'],'tokens_per_second':result['useful_tokens_per_second'],
                          'peak_GiB':max(result['modeled_peak_bytes'].values(),default=0)/2**30,
                          'memory_verdict':result['memory_verdict'],'status':result['status'],
                          'unresolved':result['unresolved'],'cost_coverage':result['cost_coverage']},indent=2))
        return
    start=time.monotonic()
    summary={'checks':checks(),'cases':[]}
    for case in ['dense','hybridep','kimi']:
        if args.example not in ['all',case]:continue
        if case=='kimi':data=export_kimi(args.microbatches)
        else:
            from .capture import capture_dense,probe_hybridep,probe_native_import
            data=capture_dense() if case=='dense' else probe_hybridep()
        summary['cases'].append(exercise(case,data,strict=args.strict))
        print(json.dumps(summary['cases'][-1],indent=2),flush=True)
    if args.example=='all':
        from .capture import probe_native_import
        summary['native_torchtitan_probe']=probe_native_import()
    if args.example in ['all','calibration']:
        from .calibrate import cpu_matmul_calibration
        data,cal,heldout=cpu_matmul_calibration()
        path=RUNS/'calibrated_cpu'
        write_bundle(path,*data)
        (path/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
        r=simulate(*read_bundle(path),calibration=cal,strict=True)
        save_result(path,r)
        observed=statistics.median(heldout)
        summary['cpu_calibration']={'target':data[3]['id'],'predicted_seconds':r['seconds'],
            'heldout_median_seconds':observed,'relative_error':r['seconds']/observed-1,
            'heldout_samples_seconds':heldout,'strict_accepts':True,
            'scope':'Local CPU calibration plumbing only; NOT GB200/Kimi validation'}
        print(json.dumps(summary['cpu_calibration'],indent=2),flush=True)
    summary['wall_seconds']=time.monotonic()-start
    (RUNS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Summary:',RUNS/'summary.json')


if __name__=='__main__':main()
