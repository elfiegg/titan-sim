"""Actual TorchDispatch/FakeTensor/FakeProcessGroup capture, isolated from simulator."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup
from .core import Node, Storage
from .kimi import hardware

ROOT=Path(__file__).parent
SOURCE=Path(os.environ.get('TORCHTITAN_PROTOTYPE_SOURCE','upstream/torchtitan'))


def verify_source():
    manifest=json.loads((ROOT/'fixtures/torchtitan-provenance.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('source revision mismatch: '+name)


class Capture(TorchDispatchMode):
    def __init__(self,rank,start=1):
        super().__init__()
        self.rank,self.next=rank,start
        self.nodes,self.storage=[],{}
        self.last_writer,self.refs={},[]
        self.previous=None
        self.phase='forward'
        self.collective_index=0

    def tensors(self,value):
        return [x for x in tree_leaves(value) if isinstance(x,torch.Tensor)]

    def sid(self,t,producer=None,persistent=False):
        st=t.untyped_storage()
        # Retain wrappers so tensor/storage identity is never recycled by Python.
        self.refs.extend([t,st])
        key=f'r{self.rank}:s{st._cdata}'
        if key not in self.storage:
            self.storage[key]=Storage(key,self.rank,st.nbytes(),producer,persistent=persistent,
                                      category='persistent_input' if persistent else 'captured_tensor')
        return key

    def persist(self,t):
        self.sid(t,persistent=True)

    def record(self,name,inputs,outputs,attrs=None,kind='compute'):
        nid=self.next
        self.next+=1
        deps=[] if self.previous is None else [self.previous]
        for t in inputs:
            sid=self.sid(t,persistent=True)
            s=self.storage[sid]
            if nid not in s.consumers:s.consumers.append(nid)
            if sid in self.last_writer:deps.append(self.last_writer[sid])
        for t in outputs:
            sid=self.sid(t,producer=nid)
            self.last_writer[sid]=nid
        meta={'phase':self.phase,'evidence':'captured_fake_execution',
              'inputs':[{'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),'storage':self.sid(t)} for t in inputs],
              'outputs':[{'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),'storage':self.sid(t)} for t in outputs]}
        meta.update(attrs or {})
        signature={'op':name,'inputs':[(list(t.shape),str(t.dtype),list(t.stride())) for t in inputs],
                   'torch_version':torch.__version__,
                   'communication':{k:meta[k] for k in ['participants','comm_type','payload_bytes','group','peer'] if k in meta}}
        meta['cost_key']=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        self.nodes.append(Node(nid,name,self.rank,kind,list(dict.fromkeys(deps)),meta))
        self.previous=nid

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        inputs=self.tensors((args,kwargs))
        out=func(*args,**kwargs)
        outputs=self.tensors(out)
        name=str(func)
        byte_count=sum(t.numel()*t.element_size() for t in inputs+outputs)
        attrs={'bytes':byte_count}
        if name.startswith('aten.mm.'):
            attrs['flops']=2*args[0].shape[0]*args[0].shape[1]*args[1].shape[1]
        elif name.startswith('aten.bmm.'):
            attrs['flops']=2*args[0].shape[0]*args[0].shape[1]*args[0].shape[2]*args[1].shape[2]
        elif name.startswith(('aten.view.','aten._unsafe_view.','aten.t.','aten.transpose.','aten.detach.','aten.expand.','aten.empty','aten.as_strided.','aten.alias.')):
            attrs={'zero_cost_semantic':True}
        elif not name.startswith('aten.'):
            attrs['unmodeled_semantics']=True
        self.record(name,inputs,outputs,attrs)
        return out

    def allreduce(self,t,pg,world):
        # Explicit process-group interception: fake work really executes and
        # waits, but its wall time is NOT used as a communication duration.
        pg.allreduce([t]).wait()
        index=self.collective_index
        self.collective_index+=1
        self.record('process_group.allreduce',[t],[t],dict(
            rendezvous=f'allreduce:{index}',participants=list(range(world)),
            comm_type=0,payload_bytes=t.numel()*t.element_size(),
            resources=['allreduce_link'],evidence='intercepted_real_FakeProcessGroup_call'),
            kind='collective')


def capture_dense():
    nodes,storage=[],[]
    start=1
    for rank in range(2):
        pg=FakeProcessGroup._create_internal(rank,2,None)
        with FakeTensorMode():
            # CPU fake tensors permit native autograd on a CPU-only wheel.
            # The exported logical rank is mapped to a modeled GPU separately.
            w=torch.empty(64,128,dtype=torch.bfloat16,requires_grad=True)
            x=torch.empty(32,64,dtype=torch.bfloat16)
            c=Capture(rank,start)
            c.persist(w);c.persist(x)
            with c:
                y=(x@w).relu()
                loss=y.float().square().mean()
                c.phase='backward'
                loss.backward()
            c.phase='gradient_reduction'
            c.allreduce(w.grad,pg,2)
            c.phase='optimizer'
            with torch.no_grad(),c:
                w.add_(w.grad,alpha=-8e-4)
        nodes.extend(c.nodes);storage.extend(c.storage.values());start=c.next
    manifest={'case':'captured_dense_fake_pg','ranks':[0,1],'useful_tokens':64,
              'torch_version':torch.__version__,
              'export_mode':'actual CPU FakeTensor forward/backward/SGD and FakeProcessGroup allreduce',
              'unresolved':['GPU kernel implementation/workspaces are not captured','hardware costs are analytical, not calibrated']}
    return nodes,storage,manifest,hardware()


def probe_hybridep():
    """Execute unmodified native custom-op/fake functions extracted from pinned source.

    Only the dependency-loading/native-buffer wrapper is excluded. This is a
    component capability probe, explicitly NOT a native model execution.
    """
    verify_source()
    p=SOURCE/'torchtitan/distributed/deepep/hybridep.py'
    tree=ast.parse(p.read_text())
    wanted={'_num_permuted_tokens_for_non_blocking','_dispatch_impl','_dispatch_fake',
            '_combine_impl','_combine_fake','_dispatch_bwd_impl','_dispatch_bwd_fake',
            '_combine_bwd_impl','_combine_bwd_fake'}
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    body.extend(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted)
    ns={'torch':torch,'_buffer':None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(p),'exec'),ns)
    with FakeTensorMode():
        x=torch.empty(2048,3584,device='cuda',dtype=torch.bfloat16)
        idx=torch.empty(2048,16,device='cuda',dtype=torch.int64)
        scores=torch.empty(2048,16,device='cuda',dtype=torch.float32)
        c=Capture(0)
        for t in [x,idx,scores]:c.persist(t)
        with c:
            h,s,tpe,handle=torch.ops.hybridep.dispatch(x,idx,scores,896,32,'fake_ep32',True,0.0316162109375,128)
            out=torch.ops.hybridep.combine(h,handle,2048,128)
            c.phase='backward_component_probe'
            dh=torch.ops.hybridep.dispatch_bwd(out,handle,h.shape[0],128)
            dx,ds=torch.ops.hybridep.combine_bwd(dh,s,handle,2048,896)
        shapes={'dispatch':list(h.shape),'combine':list(out.shape),
                'dispatch_backward':list(dh.shape),'combine_backward':[list(dx.shape),list(ds.shape)]}
        assert shapes['dispatch']==[33152,3584]
        assert shapes['combine']==[2048,3584]
    manifest={'case':'native_hybridep_fake_component','ranks':[0],
              'export_mode':'source-extracted native fake implementations; buffers intentionally not initialized',
              'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
              'shapes':shapes,'torch_version':torch.__version__,
              'unresolved':['native CUDA/IPC buffer initialization not executed',
                            'custom communication costs are missing; byte-only cost is NOT predictive',
                            'backward custom ops invoked explicitly, not full-model autograd',
                            'opaque handle and workspace lifetimes are not modeled']}
    return c.nodes,list(c.storage.values()),manifest,hardware()


def probe_native_import():
    sys.path.insert(0,str(SOURCE))
    try:
        verify_source()
        from torchtitan.models.kimi_k3 import build_model_config
        from torchtitan.config.parallelism import ParallelismConfig
        from dataclasses import replace
        parallelism=ParallelismConfig(data_parallel_shard_degree=32,pipeline_parallel_degree=8,
                                      expert_parallel_degree=32,tensor_parallel_degree=1,context_parallel_degree=1)
        config=build_model_config('Kimi-K3',parallelism=parallelism,seq_len=2048)
        config=replace(config,vision_encoder=None)
        result={'import_and_config':'passed','layers':len(config.layers),'width':config.dim,
                'model_executed':False,'build_probes':{}}
        for device in ['cuda','meta']:
            try:
                with FakeTensorMode(),torch.device(device):
                    model=config.build()
                result['build_probes'][device]={'status':'built','parameters':sum(p.numel() for p in model.parameters())}
            except Exception as e:
                result['build_probes'][device]={'status':'blocked','exception':type(e).__name__,
                                               'message':str(e),'traceback':traceback.format_exc()}
        return result
    except Exception as e:
        return {'import_and_config':'blocked','exception':type(e).__name__,
                'message':str(e),'traceback':traceback.format_exc(),'model_executed':False}
