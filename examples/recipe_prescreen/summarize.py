"""Import completed GB200 measurements; never substitute synthetic timings.

Usage: python -m examples.recipe_prescreen.summarize RAW_DIRECTORY OUTPUT_DIRECTORY
RAW_DIRECTORY contains rank-*.jsonl and allocation/hardware metadata.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from titan_sim.core import Node, Storage, simulate
from titan_sim.workload import write_bundle, read_bundle


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def load(raw):
    rows, hashes = [], {}
    for file in sorted(raw.glob('rank-*.jsonl')):
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        for line in file.read_text().splitlines():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError('No actual GPU measurements found; calibration unavailable')
    hardware = [r for r in rows if r['type'] == 'hardware']
    if not hardware:
        raise ValueError('Hardware identity is missing')
    worlds = {r['world_size'] for r in hardware}
    if len(worlds) != 1 or next(iter(worlds)) not in [4,32]:
        raise ValueError(f'Expected a 4- or 32-rank calibration, received {worlds}')
    world = next(iter(worlds))
    if sorted(r['rank'] for r in hardware) != list(range(world)):
        raise ValueError('Incomplete or duplicate hardware metadata')
    identities = {(r['gpu'], r['torch'], r['cuda'], str(r['nccl']), r['job_id']) for r in hardware}
    if len(identities) != 1:
        raise ValueError('Mixed hardware/software/job identities')
    if not all('GB200' in r['gpu'] for r in hardware):
        raise ValueError('Unexpected GPU architecture')
    return rows, hardware, hashes


def summarize(raw, output):
    rows, devices, hashes = load(raw)
    world = len(devices)
    hardware_id = 'GB200-job-' + str(devices[0]['job_id']) + '-' + digest(hashes)[:12]
    groups = defaultdict(list)
    for r in rows:
        if r['type'] == 'measurement':
            expected = hashlib.sha256(json.dumps(r['descriptor'], sort_keys=True).encode()).hexdigest()
            if expected != r['cost_key']:
                raise ValueError('Descriptor checksum mismatch')
            if len(r['samples_seconds']) != 7 or any(not math.isfinite(t) or t <= 0 for t in r['samples_seconds']):
                raise ValueError('Invalid timing samples')
            groups[r['cost_key']].append(r)
    calibration, rank_calibration, table, incomplete = {}, {}, [], []
    indexed = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r['rank'])
        if [r['rank'] for r in records] != list(range(world)):
            incomplete.append({'cost_key': key, 'name': records[0]['name'], 'rank_count': len(records)})
            continue
        first = records[0]
        if any(r['descriptor'] != first['descriptor'] for r in records):
            raise ValueError('Inconsistent operation descriptors')
        # Communication participants must finish; retain the slowest rank at
        # each matched sample. The same aggregation is explicitly conservative
        # for compute. Rank-local records are also retained for the simulator.
        samples = [max(r['samples_seconds'][i] for r in records) for i in range(7)]
        common = {'hardware_id': hardware_id, 'descriptor': first['descriptor'],
                  'source_files_sha256': hashes, 'name': first['name']}
        record = dict(common, seconds=statistics.median(samples), samples_seconds=samples,
                      evidence_sha256=digest(samples), aggregation=f'max across {world} ranks per sample')
        heldout = first['descriptor'].get('validation_only', False)
        if not heldout:
            calibration[key] = record
            for r in records:
                s = r['samples_seconds']
                rank_calibration[key + ':rank:' + str(r['rank'])] = dict(
                    common, seconds=statistics.median(s), samples_seconds=s,
                    evidence_sha256=digest(s), aggregation='rank-local', rank=r['rank'])
        medians = [statistics.median(r['samples_seconds']) for r in records]
        table.append({'name': first['name'], 'execution': first['descriptor']['execution'],
                      'cost_key': key, 'rank_count': world, 'validation_only': heldout,
                      'median_rank_us': statistics.median(medians)*1e6,
                      'min_rank_us': min(medians)*1e6, 'max_rank_us': max(medians)*1e6,
                      'critical_path_us': statistics.median(samples)*1e6,
                      'min_sample_us': min(samples)*1e6, 'max_sample_us': max(samples)*1e6})
        indexed[(first['name'], first['descriptor']['execution'])] = records
    output.mkdir(parents=True, exist_ok=True)
    (output/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
    (output/'rank-calibration.json').write_text(json.dumps(rank_calibration, indent=2)+'\n')
    validations = []
    for mode in ['eager', 'cuda_graph']:
        names = ['latent_down.forward', 'latent_up.forward', 'heldout_latent_down_up']
        if not all((name, mode) in indexed for name in names):
            continue
        down, up, held = [indexed[(name, mode)] for name in names]
        nodes, storage = [], []
        per_rank = []
        for rank in range(world):
            node_id = 2*rank
            for i, r in enumerate([down[rank], up[rank]]):
                nodes.append(Node(node_id+i, r['name'], rank, deps=[] if i == 0 else [node_id],
                                  attrs={'cost_key': r['cost_key']+':rank:'+str(rank)}))
            # Logical tensor bytes for the exact held-out out= GEMM chain.
            # Outputs were allocated before capture in the benchmark, so all
            # five storages persist. No allocator/context/workspace fit claim.
            for name, elements in [('a',2048*7168),('w1',3584*7168),('w2',7168*3584),
                                   ('y',2048*3584),('z',2048*7168)]:
                storage.append(Storage(f'{rank}:{name}', rank, elements*2, persistent=True,
                                       consumers=[node_id,node_id+1], category='logical_tensor'))
            predicted = sum(statistics.median(r['samples_seconds']) for r in [down[rank],up[rank]])
            observed = statistics.median(held[rank]['samples_seconds'])
            per_rank.append({'rank':rank,'predicted_seconds':predicted,'observed_seconds':observed,
                             'relative_error':predicted/observed-1})
        hardware = {'id':hardware_id, 'capacity_bytes':min(r['capacity_bytes'] for r in devices),
                    'reserve_fraction':0.1}
        manifest = {'ranks':list(range(world)), 'unresolved':[], 'useful_tokens':0,
                    'case':'heldout_latent_chain_'+mode,
                    'scope':'Two BF16 GEMMs only; logical tensor memory; excludes CUDA allocator, graph pool and context',
                    'export_mode':'explicit component graph, not full TorchTitan/FPG capture'}
        path = output/('heldout-'+mode)
        write_bundle(path,nodes,storage,manifest,hardware)
        result = simulate(*read_bundle(path),calibration=rank_calibration,strict=True)
        (path/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        observed = max(p['observed_seconds'] for p in per_rank)
        validations.append({'execution':mode,'predicted_seconds':result['seconds'],
                            'observed_seconds':observed,'relative_error':result['seconds']/observed-1,
                            'per_rank':per_rank,'scope':manifest['scope']})
    errors = [r for r in rows if r['type'] in ['unsupported','phase_error','phase_incomplete']]
    summary = {'hardware_id':hardware_id,'hardware':devices,'world_size':world,'source_files_sha256':hashes,
               'completed_ranks':sorted({r['rank'] for r in rows if r['type']=='complete'}),
               'measurements':table,'incomplete_measurements':incomplete,'errors':errors,
               'heldout_validation':validations,
               'native_buffers':[r for r in rows if r['type']=='native_buffer'],
               'routing_validation':[r for r in rows if r['type']=='routing_validation'],
               'full_kimi_calibrated':False,
               'remaining_gaps':['Full native Kimi graph capture and compiled kernel selection',
                 'MLA/KDA, native grouped expert/SiTU kernels and complete backward/optimizer',
                 '256-rank placement, cross-NVLink-domain traffic and contention',
                 'PP8 1F1B overlap validation',
                 'Full-step allocator, CUDA graph pools, optimizer and communication buffer lifetimes']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# GB200 component calibration','',f'Hardware ID: `{hardware_id}`','',
           f'Complete operation configurations: {len(calibration)}. Completed ranks: {len(summary["completed_ranks"])} / {world}.',
           '', 'These are measured component costs, not a calibrated full Kimi K3 throughput or memory prediction.','',
           '| Operation | Mode | Median rank (µs) | Slowest rank (µs) |',
           '|---|---|---:|---:|']
    for r in table:
        lines.append(f'| {r["name"]} | {r["execution"]} | {r["median_rank_us"]:.2f} | {r["max_rank_us"]:.2f} |')
    lines += ['', 'Held-out check: component costs were measured independently; the two-GEMM chain was excluded from the cost catalog.','']
    for v in validations:
        lines.append(f'- {v["execution"]}: predicted {v["predicted_seconds"]*1e6:.2f} µs, measured {v["observed_seconds"]*1e6:.2f} µs; error {v["relative_error"]:+.1%}.')
    lines += ['', 'Remaining gaps:', ''] + ['- '+g for g in summary['remaining_gaps']]
    if errors:
        lines += ['', f'Unsupported/failure records: {len(errors)}; see summary.json for exact errors.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'calibrated_configurations':len(calibration),
                      'completed_ranks':len(summary['completed_ranks']),
                      'heldout_validation':[{k:v for k,v in x.items() if k!='per_rank'} for x in validations]},indent=2))


if __name__ == '__main__':
    summarize(Path(sys.argv[1]),Path(sys.argv[2]))
