"""Bounded GB200 component calibration. No model training or skill changes.

CUDA events measure completed device work. Timing and memory probes are separate.
Every rank writes raw samples immediately; no result is silently substituted.
"""
from __future__ import annotations
import datetime
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import traceback
import torch
import torch.distributed as dist

RANK=int(os.environ['RANK'])
LOCAL=int(os.environ['LOCAL_RANK'])
WORLD=int(os.environ['WORLD_SIZE'])
ROOT=Path(os.environ.get('CALIBRATION_OUTPUT','results/gb200-calibration'))
ROOT.mkdir(parents=True,exist_ok=True)
torch.cuda.set_device(LOCAL)
torch.set_num_threads(4)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
DEVICE=torch.device('cuda',LOCAL)
OUT=(ROOT/f'rank-{RANK:03d}.jsonl').open('w',buffering=1)


def emit(obj):
    obj.update(rank=RANK,host=socket.gethostname(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    OUT.write(json.dumps(obj,sort_keys=True)+'\n')


def tensor_meta(x):
    return {'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride())}


def memory():
    free,total=torch.cuda.mem_get_info()
    return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved(),
            'device_used':total-free,'total':total}


def measure(name,fn,descriptor,graph=False,samples=7,repeats=5):
    for _ in range(10):fn()
    torch.cuda.synchronize()
    g=None
    if graph:
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        action=g.replay
        for _ in range(10):action()
    else:action=fn
    torch.cuda.synchronize()
    dist.barrier()
    gpu,wall=[],[]
    for _ in range(samples):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        t=time.perf_counter()
        start.record()
        for _ in range(repeats):action()
        end.record();end.synchronize()
        gpu.append(start.elapsed_time(end)/1000/repeats)
        wall.append((time.perf_counter()-t)/repeats)
    # Separate, synchronized allocator observation. Do not add allocated+reserved.
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    before=memory()
    action();torch.cuda.synchronize()
    after=memory()
    descriptor=dict(descriptor,execution='cuda_graph' if graph else 'eager',
                    torch_version=torch.__version__,cuda=torch.version.cuda)
    key=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
    emit({'type':'measurement','name':name,'descriptor':descriptor,'cost_key':key,
          'samples_seconds':gpu,'median_seconds':statistics.median(gpu),'wall_samples_seconds':wall,
          'memory_before':before,'memory_after':after,
          'sample_scope':'rank-local CUDA completed work; collectives must aggregate matched sample indices across ranks',
          'method':{'warmup':10,'samples':samples,'repeats':repeats,'timer':'CUDA events'}})
    del g


def all_ranks_success(ok):
    flag=torch.tensor([int(ok)],device=DEVICE,dtype=torch.int32)
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    return bool(flag.item())


def gemm_cases():
    # Dimensions are full recipe components, not reduced model widths.
    dims=[('latent_down',2048,3584,7168),('latent_up',2048,7168,3584),
          ('mla_q_down',2048,1536,7168),('mla_q_up',2048,18432,1536),
          ('mla_kv_down',2048,576,7168),('mla_kv_up',2048,24576,512),
          ('attention_out',2048,7168,12288),('dense_gate_up',2048,67584,7168),
          ('lm_head',2048,163840,7168)]
    for name,m,n,k in dims:
        a=torch.randn(m,k,device=DEVICE,dtype=torch.bfloat16)
        w=torch.randn(n,k,device=DEVICE,dtype=torch.bfloat16)
        y=torch.empty(m,n,device=DEVICE,dtype=torch.bfloat16)
        pairs=[('forward',a,w.T,y)]
        if name in ['latent_down','latent_up','attention_out']:
            dy=torch.randn_like(y);dx=torch.empty_like(a);dw=torch.empty_like(w)
            pairs += [('dgrad',dy,w,dx),('wgrad',dy.T,a,dw)]
        for role,left,right,out in pairs:
            descriptor={'op':'aten.mm.out','role':role,'inputs':[tensor_meta(left),tensor_meta(right)],'output':tensor_meta(out)}
            fn=lambda left=left,right=right,out=out:torch.mm(left,right,out=out)
            for graph in [False,True]:measure(name+'.'+role,fn,descriptor,graph)
        del pairs,a,w,y
        if name in ['latent_down','latent_up','attention_out']:del dy,dx,dw
        gc.collect();torch.cuda.empty_cache()
    # Batched dense GEMM is explicitly a proxy for grouped expert kernels.
    for tokens in [1152,1184]:
        for name,k,n in [('expert_gate_up',3584,6144),('expert_down',3072,3584)]:
            a=torch.randn(28,tokens,k,device=DEVICE,dtype=torch.bfloat16)
            w=torch.randn(28,n,k,device=DEVICE,dtype=torch.bfloat16)
            y=torch.empty(28,tokens,n,device=DEVICE,dtype=torch.bfloat16)
            fn=lambda:torch.bmm(a,w.transpose(1,2),out=y)
            d={'op':'aten.bmm.out','role':'expert_proxy','inputs':[tensor_meta(a),tensor_meta(w.transpose(1,2))],
               'output':tensor_meta(y),'limitation':'batched regular GEMM; not native grouped/SiTU expert implementation'}
            for graph in [False,True]:measure(name+f'.t{tokens}',fn,d,graph)
            del a,w,y;gc.collect();torch.cuda.empty_cache()


def bandwidth_and_chain():
    x=torch.empty(128*1024*1024,device=DEVICE,dtype=torch.bfloat16)
    y=torch.empty_like(x)
    measure('hbm_copy_256MiB',lambda:y.copy_(x),{'op':'aten.copy_','payload_bytes':x.numel()*x.element_size(),
            'hbm_traffic_bytes':2*x.numel()*x.element_size()},True)
    del x,y;torch.cuda.empty_cache()
    a=torch.randn(2048,7168,device=DEVICE,dtype=torch.bfloat16)
    w1=torch.randn(3584,7168,device=DEVICE,dtype=torch.bfloat16)
    w2=torch.randn(7168,3584,device=DEVICE,dtype=torch.bfloat16)
    y=torch.empty(2048,3584,device=DEVICE,dtype=torch.bfloat16)
    z=torch.empty_like(a)
    def chain():
        torch.mm(a,w1.T,out=y);torch.mm(y,w2.T,out=z)
    for graph in [False,True]:
        measure('heldout_latent_down_up',chain,{'op':'heldout_two_gemm_chain','components':['latent_down.forward','latent_up.forward'],
                'validation_only':True},graph)
    del a,w1,w2,y,z;gc.collect();torch.cuda.empty_cache()


def collective_cases():
    for nbytes in [1024*1024,16*1024*1024,64*1024*1024,224*1024*1024]:
        # Total gathered/reduced tensor is nbytes. Per-rank shard is nbytes/world.
        n=(nbytes//4//WORLD)*WORLD
        x=torch.zeros(n,device=DEVICE,dtype=torch.float32)
        shard=torch.empty(n//WORLD,device=DEVICE,dtype=torch.float32)
        exchanged=torch.empty_like(x)
        for kind,fn in [('all_reduce',lambda:dist.all_reduce(x)),
                        ('reduce_scatter_tensor',lambda:dist.reduce_scatter_tensor(shard,x)),
                        ('all_gather_into_tensor',lambda:dist.all_gather_into_tensor(x,shard)),
                        ('all_to_all_single',lambda:dist.all_to_all_single(exchanged,x))]:
            measure(kind+f'.{nbytes}',fn,{'op':'nccl.'+kind,'world_size':WORLD,'dtype':'float32',
                    'logical_tensor_bytes':n*4,'shard_bytes':n*4//WORLD,
                    'limitation':'NCCL generic collective, not HybridEP'},False)
        del x,shard,exchanged;gc.collect();torch.cuda.empty_cache()
    # Actual traffic path is labelled by placement, never assumed to be IB.
    for offset in [1,4]:
        peer=RANK^offset
        if peer>=WORLD:continue
        x=torch.zeros(2048,7168,device=DEVICE,dtype=torch.bfloat16)
        y=torch.empty_like(x)
        def exchange():
            work=dist.batch_isend_irecv([dist.P2POp(dist.isend,x,peer),dist.P2POp(dist.irecv,y,peer)])
            for w in work:w.wait()
        measure(f'p2p_pair_offset{offset}',exchange,{'op':'nccl.paired_send_recv','world_size':WORLD,
                'peer_offset':offset,'payload_bytes':x.numel()*x.element_size(),'simultaneous_pairs':WORLD//2},False)
        del x,y;gc.collect();torch.cuda.empty_cache()


def hybrid_probe():
    try:
        from deep_ep import HybridEPBuffer
        available=True
    except Exception as e:
        available=False;emit({'type':'unsupported','name':'HybridEP','error':repr(e)})
    if not all_ranks_success(available):return
    # Never rewrite cached extension sources. JIT writes go into a private copy
    # prepared by the launcher; package provenance is recorded in metadata.
    before=memory()
    buffer=HybridEPBuffer(dist.group.WORLD,hidden_dim=3584,max_num_of_tokens_per_rank=2048,
                         num_local_experts=896//WORLD,use_fp8=False,num_sms_dispatch_api=16,
                         num_sms_combine_api=16,load_cached_kernels=True,use_shared_buffer=True,
                         enable_custom_allgather=True)
    torch.cuda.synchronize()
    emit({'type':'native_buffer','name':'HybridEP','before':before,'after':memory(),
          'ranks_per_nvlink_domain':buffer.num_of_hybrid_ep_ranks_per_nvlink_domain})
    x=torch.randn(2048,3584,device=DEVICE,dtype=torch.bfloat16)
    idx=torch.arange(2048*16,device=DEVICE).view(2048,16)%896
    weights=torch.full((2048,16),1/16,device=DEVICE,dtype=torch.float32)
    slots=33152 if WORLD==32 else None
    # Use the source API only after inspecting its signature; absence is explicit.
    signature=inspect.signature(buffer.dispatch_with_permute)
    emit({'type':'api','name':'HybridEP.dispatch_with_permute','signature':str(signature)})
    if slots is None:
        emit({'type':'unsupported','name':'HybridEP exact routing capacity','error':'requires world32'});return
    def dispatch():
        return buffer.dispatch_with_permute(hidden=x,topk_idx=idx,topk_weights=weights,
                    num_of_experts=896,num_of_experts_per_rank=28,num_permuted_tokens=slots,pad_multiple=32,non_blocking=True)
    values=dispatch();torch.cuda.synchronize()
    emit({'type':'api_outputs','name':'HybridEP.dispatch','outputs':[tensor_meta(v) if isinstance(v,torch.Tensor) else type(v).__name__ for v in values]})
    # Native return order verified against the installed implementation.
    hidden,scores,scales,tpe,handle=values
    overflow=handle[-1]
    no_overflow=not (isinstance(overflow,torch.Tensor) and overflow.any().item())
    emit({'type':'routing_validation','name':'HybridEP','pad_multiple':32,'capacity_rows':slots,
          'no_overflow':no_overflow,'tokens_per_expert':tpe.cpu().tolist()})
    if not all_ranks_success(no_overflow):raise RuntimeError('HybridEP overflow')
    descriptor={'op':'hybridep.dispatch_with_permute','world_size':WORLD,'tokens':2048,'hidden':3584,
                'experts':896,'top_k':16,'capacity_rows':slots,'pad_multiple':32,'sms':16,'dtype':'bfloat16'}
    measure('hybridep_dispatch',dispatch,descriptor,False)
    def combine():return buffer.combine_with_unpermute(hidden=hidden,handle=handle)
    measure('hybridep_combine',combine,dict(descriptor,op='hybridep.combine_with_unpermute'),False)
    # Pair capture retains dispatch outputs and consumes the matching handle.
    def pair():
        h,s,sc,t,hnd=dispatch()
        return buffer.combine_with_unpermute(hidden=h,handle=hnd)
    measure('hybridep_dispatch_combine',pair,dict(descriptor,op='hybridep.dispatch_combine_pair'),True)


def main():
    props=torch.cuda.get_device_properties(LOCAL)
    meta={'type':'hardware','job_id':os.environ.get('SLURM_JOB_ID'),'world_size':WORLD,'local_rank':LOCAL,
          'gpu':props.name,'uuid':str(getattr(props,'uuid','unavailable')),'capacity_bytes':props.total_memory,
          'sm_count':props.multi_processor_count,'capability':list(torch.cuda.get_device_capability()),
          'torch':torch.__version__,'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version(),
          'initial_memory':memory(),'precision':'BF16 compute; FP32 NCCL reduction benchmarks',
          'host_platform':os.uname().machine}
    emit(meta)
    if 'GB200' not in props.name:raise RuntimeError('Unexpected hardware: '+props.name)
    phase=os.environ.get('CALIBRATION_PHASES','all')
    if phase not in ['all','compute','communication']:
        raise ValueError('CALIBRATION_PHASES must be all, compute, or communication')
    phases=[]
    if phase in ['all','compute']: phases += [('gemms',gemm_cases),('bandwidth_and_heldout',bandwidth_and_chain)]
    if phase in ['all','communication']: phases += [('collectives',collective_cases),('hybridep',hybrid_probe)]
    for name,fn in phases:
        emit({'type':'phase_start','name':name})
        try:fn();ok=True
        except Exception as e:
            ok=False;emit({'type':'phase_error','name':name,'exception':repr(e),'traceback':traceback.format_exc()})
        if not all_ranks_success(ok):
            emit({'type':'phase_incomplete','name':name})
            # Fail closed after a possible unmatched distributed failure.
            raise RuntimeError('Calibration phase failed on at least one rank: '+name)
        emit({'type':'phase_complete','name':name})
    emit({'type':'complete'})
    dist.destroy_process_group()
    OUT.close()


if __name__=='__main__':main()
