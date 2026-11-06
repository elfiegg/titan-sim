"""Full-shape Kimi *generated* stage graph, not captured TorchTitan execution.

First hypothesis: reuse audited full-shape costs to exercise a 256-rank bundle
and storage lifecycle engine. Opaque stage regions explicitly retain coverage
gaps; strict simulation must reject them. Native capture lives in capture.py.
"""
from __future__ import annotations
import json
from pathlib import Path
from titan_sim.core import Node, Storage

GIB = 2**30
ROOT = Path(__file__).parent


def hardware():
    return {
        'id':'GB200-unmeasured-prototype', 'capacity_bytes':197897617408,
        'capacity_provenance':'historical device-capacity scenario; not a current hardware measurement',
        'reserve_fraction':0.1, 'effective_flops_per_second':1e15,
        'effective_hbm_bytes_per_second':4e12,
        'interconnect_bytes_per_second':4e11, 'pp_bytes_per_second':50e9,
        'comm_latency_seconds':5e-6, 'launch_seconds':5e-6,
        'cost_multiplier':1.,
        'provenance':'ALL rate/latency values are uncalibrated scenario inputs',
    }


def export_kimi(microbatches=128):
    if microbatches < 1:
        raise ValueError('microbatches must be positive')
    old = json.loads((ROOT/'fixtures/kimi-reference.json').read_text())
    pp, dp, seq, width = 8,32,2048,7168
    nodes, storage = [], []
    ids, last = {}, {}
    counter = 0
    def add(name, rank, kind='compute', deps=None, **attrs):
        nonlocal counter
        counter += 1
        n = Node(counter,name,rank,kind,list(deps or []),attrs)
        nodes.append(n)
        return n.id
    # Allocate stable IDs before wiring the communication pairs/dependencies.
    # One opaque compute region per stage/phase is a deliberately declared gap.
    for stage in range(pp):
        ms = old['memory']['stages'][stage]
        for lane in range(dp):
            rank = stage*dp+lane
            for cat, key in [('parameters','weights_gib'),('optimizer','optimizer_states_gib'),
                             ('retained_dense','dense_compute_materialization_gib'),
                             ('gradient_coexistence_scenario','gradient_coexistence_scenario_gib')]:
                storage.append(Storage(f'r{rank}:{cat}',rank,round(ms[key]*GIB),persistent=True,category=cat))
            for m in range(microbatches):
                for phase in ['F','B']:
                    seconds = old['throughput']['stage_forward_seconds' if phase=='F' else 'stage_backward_seconds'][stage]
                    nid = add(f'stage{stage}.{phase}.mb{m}',rank,
                              legacy_seconds=seconds, layer_ids=ms['layers'],
                              evidence='generated_opaque_region',
                              includes='compute+HybridEP historical analytical proxy; not separate calibrated kernels')
                    ids[stage,lane,phase,m] = nid
            ids[stage,lane,'optimizer',0] = add(f'stage{stage}.optimizer',rank,
                       legacy_seconds=old['throughput']['stage_optimizer_seconds'][stage],
                       evidence='historical_analytical_proxy')
    by_id = {n.id:n for n in nodes}
    # Point-to-point completion events join two rank-local Chakra nodes.
    for stage in range(pp-1):
        payload = seq*width*2*((max(old['memory']['stages'][stage]['layers'])//12)+2)
        for lane in range(dp):
            left,right=stage*dp+lane,(stage+1)*dp+lane
            for m in range(microbatches):
                for phase,src,dst,srcstage,dststage in [('F',left,right,stage,stage+1),('B',right,left,stage+1,stage)]:
                    key=f'pp:{stage}:{lane}:{phase}:{m}'
                    attrs=dict(rendezvous=key,payload_bytes=payload,
                               resources=[f'nic:{src}',f'nic:{dst}'],
                               evidence='generated_schedule_transfer')
                    sid=add(key+'.send',src,'send',[ids[srcstage,lane,phase,m]],peer=dst,**attrs)
                    rid=add(key+'.recv',dst,'recv',peer=src,**attrs)
                    by_id[ids[dststage,lane,phase,m]].deps.append(rid)
    # Plain 1F1B local order; full 256 ranks are represented, not just one PP line.
    for stage in range(pp):
        ms=old['memory']['stages'][stage]
        for lane in range(dp):
            rank=stage*dp+lane
            warmup=min(pp-stage-1,microbatches)
            order=[('F',m) for m in range(warmup)]
            for m in range(microbatches-warmup):
                order.extend([('F',m+warmup),('B',m)])
            order.extend(('B',m) for m in range(microbatches-warmup,microbatches))
            previous=None
            for phase,m in order:
                n=by_id[ids[stage,lane,phase,m]]
                if previous is not None:
                    n.deps.append(previous)
                if phase=='B':
                    n.deps.append(ids[stage,lane,'F',m])
                n.deps=list(dict.fromkeys(n.deps))
                previous=n.id
            # Explicit last-gradient-ready barrier before the optimizer. Its
            # duration is the historical exposed FSDP proxy, NOT calibrated comm.
            group=[stage*dp+i for i in range(dp)]
            fsdp=add(f'stage{stage}.grad_reduce',rank,'collective',[previous],
                     rendezvous=f'fsdp:{stage}',participants=group,comm_type=7,
                     payload_bytes=round(ms['dense_compute_materialization_gib']*GIB*2),
                     resources=[f'collective:{stage}',*[f'gpu:{r}' for r in group]],
                     evidence='generated_single_reduction_assumption')
            by_id[ids[stage,lane,'optimizer',0]].deps.append(fsdp)
            per_mb=(ms['checkpoint_inputs_scenario_gib']+ms['residual_blocks_scenario_gib'])*GIB/ms['peak_inflight_microbatches']
            for m in range(microbatches):
                storage.append(Storage(f'r{rank}:mb{m}:saved',rank,round(per_mb),
                           ids[stage,lane,'F',m],[ids[stage,lane,'B',m]],category='saved_activations_scenario'))
    manifest={
        'case':'kimi_k3_full_stage_graph_prototype','ranks':list(range(pp*dp)),
        'useful_tokens':dp*microbatches*seq,
        'recipe':dict(old['configuration'],microbatches=microbatches,global_sequences=dp*microbatches,
                      global_tokens_per_update=dp*microbatches*seq),
        'torchtitan_commit':old['source_commit'],
        'export_mode':'generated stage regions, native TorchTitan model NOT executed',
        'parameter_count':old['parameters']['total'], 'routing':old['routing'],
        'rank_placement':{str(s*dp+l):{'pp':s,'ep':l,'fsdp':l,'nvlink_domain':s} for s in range(pp) for l in range(dp)},
        'unresolved':[
            'KDA/MLA/MoE/checkpoint kernels are opaque historical analytical stage regions, not captured operations',
            'HybridEP communication remains embedded in region costs; no independent contention/overlap calibration',
            'native HybridEP buffers, CUDA graph pools, Muon workspaces and allocator reservation are unbounded',
            'FP32 gradients are a persistent coexistence scenario, not traced gradient lifetimes',
            'checkpoint activation sizes are analytical; only coarse F-to-B lifetimes are simulated',
            'PP send/receive staging-buffer allocations are not yet captured',
            'one end-of-step FSDP reduction and retained dense parameters require runtime verification',
            'initialization/checkpoint loading and actual compiled kernel choices are not represented',
            'all compute and network costs are uncalibrated; topology is an assumed placement',
        ],
    }
    return nodes,storage,manifest,hardware()
