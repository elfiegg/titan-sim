"""Exercise real calibration on local CPU only. Never relabel it as GB200 data."""
from __future__ import annotations
import hashlib
import json
import platform
import statistics
import time
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from .capture import Capture
from .kimi import hardware


def cpu_matmul_calibration():
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    h=hardware()
    h['id']='local-CPU-'+platform.machine()+'-torch-'+torch.__version__
    h['provenance']='local CPU only; no GB200 calibration'
    h['capacity_bytes']=2**30
    h['capacity_provenance']='explicit 1 GiB logical-storage budget for this tiny demonstration, not physical CPU RAM'
    h['reserve_fraction']=0.
    with FakeTensorMode():
        a=torch.empty(128,256)
        b=torch.empty(256,128)
        c=torch.empty(128,64)
        capture=Capture(0)
        for t in [a,b,c]:capture.persist(t)
        with capture:
            out=(a@b)@c
        capture.storage[capture.sid(out)].persistent=True
    calibration={}
    for n in capture.nodes:
        shapes=[t['shape'] for t in n.attrs['inputs']]
        x,y=[torch.randn(*s) for s in shapes]
        for _ in range(20):torch.mm(x,y)
        samples=[]
        # Batched wall-time samples include dispatch/allocator overhead. That
        # is appropriate for these eager CPU operations, not CUDA kernel timing.
        for _ in range(15):
            start=time.perf_counter()
            for _ in range(100):torch.mm(x,y)
            samples.append((time.perf_counter()-start)/100)
        evidence=json.dumps(samples,separators=(',',':')).encode()
        calibration[n.attrs['cost_key']]={
            'hardware_id':h['id'],'seconds':statistics.median(samples),
            'samples_seconds':samples,'evidence_sha256':hashlib.sha256(evidence).hexdigest(),
            'method':'15 wall-time batches of 100 eager CPU mm calls, one thread, 20 warmups',
            'op':n.name,'input_shapes':shapes,'torch_version':torch.__version__,
        }
    a,b,c=torch.randn(128,256),torch.randn(256,128),torch.randn(128,64)
    for _ in range(20):out=(a@b)@c
    heldout=[]
    for _ in range(15):
        start=time.perf_counter()
        for _ in range(100):out=(a@b)@c
        heldout.append((time.perf_counter()-start)/100)
    torch.set_num_threads(old_threads)
    m={'case':'calibrated_cpu_matmul','ranks':[0],'unresolved':[],
       'scope':'two eager CPU matmuls; logical tensor storage only, no allocator-reserved-memory claim',
       'export_mode':'actual FakeTensor capture, independent real CPU operation benchmarks',
       'calibration_target':h['id'],'useful_tokens':0}
    return (capture.nodes,list(capture.storage.values()),m,h),calibration,heldout
