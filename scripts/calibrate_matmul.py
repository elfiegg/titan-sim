#!/usr/bin/env python3
"""Collect small BF16 GEMM timings and check independent CPU metadata signatures.

Requires one real CUDA GPU. Calibration and held-out measurements are separate
runs of each kernel. This does not validate full training, FP8 kernels or memory.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from titan_sim.capture import TensorCapture
    from titan_sim.calibration import timing_entry, measure_cuda, heldout_error
    from titan_sim.costs import calibration_key, estimate_cost
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--torchtitan-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('requires an actual CUDA GPU')
    hardware = json.loads(args.hardware.read_text())
    if torch.cuda.get_device_name() not in hardware.get('name', ''):
        parser.error('hardware profile name does not identify the actual CUDA device')
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    context = dict(torch_version=torch.__version__, torchtitan_commit=args.torchtitan_commit,
                   hardware=hardware['name'], measured_device=torch.cuda.get_device_name(),
                   cuda=torch.version.cuda, precision='BF16 inputs/output; default GEMM accumulation',
                   compile=False, tf32=False)
    document = {'context': context, 'entries': {}, 'heldout': [],
                'scope': 'three BF16 GEMM shapes; CUDA-event timing, no full-model coverage'}
    for m, k, n in ((1024, 1024, 1024), (4096, 7168, 512), (4096, 2048, 7168)):
        # Prediction signature is generated independently on CPU fake tensors.
        with FakeTensorMode(allow_fallback_kernels=False):
            with TensorCapture(modeled_device='cuda') as capture:
                left = torch.empty((m, k), dtype=torch.bfloat16)
                right = torch.empty((k, n), dtype=torch.bfloat16)
                torch.mm(left, right)
        predicted = next(e for e in capture.events if e['name'] == 'aten.mm.default')
        left = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        right = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
        with TensorCapture(source='gpu_observation') as observed_capture:
            output = torch.mm(left, right)
        observed = next(e for e in observed_capture.events if e['name'] == 'aten.mm.default')
        if calibration_key(predicted) != calibration_key(observed):
            raise RuntimeError('CPU/GPU GEMM metadata signatures do not match')
        operation = lambda: torch.mm(left, right)
        measured = measure_cuda(operation)
        key, entry = timing_entry(predicted, measured, provenance={
            'source': 'cuda_event_measurement', 'context': context, 'warmup': 10,
            'measurement': 'current-stream CUDA events around one eager torch.mm'})
        document['entries'][key] = entry
        heldout = measure_cuda(operation, warmup=3, samples=10)
        prediction = estimate_cost(predicted, hardware, 'calibrated', document)
        document['heldout'].append({'shape_mkn': [m, k, n], 'cpu_gpu_signature_match': True,
                                    **heldout_error(prediction['duration_us'], heldout), 'samples_us': heldout})
        del left, right, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'entries': len(document['entries']), 'heldout': document['heldout']}))


if __name__ == '__main__':
    main()
