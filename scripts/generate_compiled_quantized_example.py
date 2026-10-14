#!/usr/bin/env python3
"""Re-export a real CPU FX graph and lower the explicitly declared example plans."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from titan_sim.compiled import export_cpu_graph, artifact_digest, lower_compiled_manifest
    from titan_sim.quantization import lower_quantized_training
    from titan_sim.runtime import simulate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'examples')
    args = parser.parse_args()
    quant = json.loads((ROOT / 'examples/compiled_quantized_training.json').read_text())
    plan = json.loads((ROOT / 'examples/compiled_quantized_manifest.json').read_text())
    artifact = export_cpu_graph()
    # The executable CPU example is explicitly matmul -> relu. Its graph is real;
    # the saved plans remain declared GPU execution/storage scenarios.
    if [node['target'] for node in artifact['nodes']] != ['aten.matmul.default', 'aten.relu.default']:
        raise ValueError('export graph changed; manually review the declared GPU plan')
    for kernel, node in zip(plan['kernels'], artifact['nodes'], strict=True):
        kernel['logical_ops'] = [node['name']]
    plan['artifact_sha256'] = artifact_digest(artifact)
    for spec in (quant, plan):
        spec['calibration_context']['torch_version'] = artifact['torch_version']
    q = lower_quantized_training(quant)
    plan['external_storages'] = {sid: buffer['size_bytes'] for sid, buffer in q['buffers'].items()}
    plan['external_storages'].update({'input': 16, 'output': 12})
    c = lower_compiled_manifest(plan, artifact=artifact)
    initial = [{'event_id': name + '_allocate', 'kind': 'allocate', 'storage_id': name,
                'size_bytes': size, 'category': 'activation'} for name, size in [('input', 16), ('output', 12)]]
    document = {'schema_version': 1, 'source': 'combined_compiled_quantized_model_contract',
                'metadata': {**c['metadata'], 'quantization': q['metadata'],
                             'scope': 'CPU semantic fixture; actual FX graph, declared GPU plan; no native full training capture'},
                'events': q['events'] + initial + c['events'],
                'limitations': q['limitations'] + c['limitations'],
                'hardware': {'name': 'synthetic_example', 'peak_flops': {'float32': 1e6, 'bfloat16': 2e6, 'mxfp8': 4e6},
                             'memory_bandwidth_bytes_s': 1e6, 'memory_capacity_bytes': 1000000},
                'allocator': {'policy': 'caching', 'alignment_bytes': 1, 'segment_bytes': 1}}
    result = simulate(document['events'], document['hardware'], allocator=document['allocator'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, value in [('fx', artifact), ('training', quant), ('manifest', plan), ('capture', document), ('simulation', result)]:
        (args.output_dir / ('compiled_quantized_' + suffix + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'fx_nodes': [n['name'] for n in artifact['nodes']], 'events': len(document['events']),
                      'logical_peak_bytes': result['summary']['logical_peak_allocated_bytes'],
                      'scheduled_peak_bytes': result['summary']['peak_allocated_bytes'],
                      'allocator_reserved_peak_bytes': result['allocator']['summary']['peak_reserved_bytes'],
                      'flops': sum(row['flops'] for row in result['timeline'])}, indent=2))


if __name__ == '__main__':
    main()
