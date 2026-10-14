"""Read training logs as observations, never as a simulated allocation trace.

Memory counter names and printed units are preserved. A bare Megatron 'GB'
label is not silently treated as GiB. Unlabelled fan-out lines retain rank=None;
line order cannot establish rank identity or a GPU execution timeline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP = re.compile(r"step:\s*(\d+).*?memory:\s*(" + NUMBER + r")\s*(GiB|GB|MiB|MB)")
ALLOCATOR = re.compile(r"\[Rank (\d+)\]\s*\(after (\d+) iterations\) memory \(([^)]+)\)")
COUNTER = re.compile(r"(mem-[a-z-]+):\s*(" + NUMBER + r")")
STASH = re.compile(r'allocate_stash_buffers cuda:\s*torch.Size\(\[([\d, ]*)\]\)'
                   r'(?:\s+host:\s*torch.Size\(\[([\d, ]*)\]\))?'
                   r'\s+dtype=torch\.(\w+)\s+\(torch\.(\w+)\)')
STORAGE_BYTES = {'uint8': 1, 'int8': 1, 'bfloat16': 2, 'float16': 2,
                 'float32': 4, 'float64': 8, 'int32': 4, 'int64': 8,
                 'float8_e4m3fn': 1, 'float8_e8m0fnu': 1}
FACT_PATTERNS = {
    'model': (r'Building (deepseek_\w+)\s+(\w+)', lambda m: m[1] + ' ' + m[2]),
    'parameter_elements': (r'size:\s*([\d,]+) total parameters', lambda m: int(m[1].replace(',', ''))),
    'hardware': (r'CUDA capacity:\s*(.*?) with [\d.]+GiB', lambda m: m[1]),
    'capacity_gib_printed': (r'CUDA capacity:.*?with ([\d.]+)GiB', lambda m: float(m[1])),
    'torchtitan_version': (r'torchtitan version:\s*([^ ]+)', lambda m: m[1]),
    'mesh': (r'Building device mesh with parallelism:\s*(.*)', lambda m: {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', m[1])}),
    'training': (r'Trainer is initialized with local batch size (\d+), global batch size (\d+), gradient accumulation steps (\d+), sequence length (\d+), total steps (\d+)', lambda m: dict(zip(('local_batch', 'global_batch', 'accumulation', 'sequence_length', 'steps'), map(int, m.groups())))),
}


def _number(value):
    number = float(value.replace(',', ''))
    if not math.isfinite(number):
        raise ValueError('nonfinite numeric value: ' + value)
    return number


def _memory_number(value):
    number = _number(value)
    if number < 0:
        raise ValueError('negative memory counter')
    return number


def parse_log(path, *, source_id=None):
    path = Path(path)
    data = path.read_bytes()
    source_id = source_id or path.name
    facts, samples, indicators, components = {}, [], defaultdict(list), []
    for line_number, raw in enumerate(data.decode(errors='replace').split('\n'), 1):
        clean = ANSI.sub('', raw)
        if clean.lstrip().startswith(('+', '#')):
            continue
        # Concurrent writers can concatenate records onto one physical line.
        # A rank marker starts a new record; it cannot own a later rank's data.
        for line in re.split(r'(?=\[rank\s*\d+\])', clean, flags=re.I):
            # Quotes of shell commands/configs are not operation observations.
            if line.lstrip().startswith(('+', '#')):
                continue
            evidence = {'source': source_id, 'line': line_number}
            for key, (pattern, convert) in FACT_PATTERNS.items():
                match = re.search(pattern, line)
                if match:
                    value = convert(match)
                    records = facts.setdefault(key, [])
                    existing = next((r for r in records if r['value'] == value), None)
                    if existing:
                        existing['occurrences'] += 1
                    else:
                        records.append({'value': value, 'occurrences': 1, 'evidence': evidence})
            rank_match = re.search(r'\[rank\s*(\d+)\]', line, re.I)
            rank = int(rank_match[1]) if rank_match else None
            stash = STASH.search(line)
            if stash:
                # Preserve launcher prefixes as labels: a task ID is not proof of GPU rank.
                prefix = re.match(r'\s*(\d+):', line)
                for device, dimensions in (('cuda', stash[1]), ('cpu', stash[2])):
                    if dimensions is None:
                        continue
                    shape = [int(d.strip()) for d in dimensions.split(',') if d.strip()]
                    width = STORAGE_BYTES.get(stash[3])
                    components.append({**evidence, 'source_kind': 'observed_buffer_geometry',
                                       'name': 'paged_stash', 'device': device, 'rank': rank,
                                       'launcher_label': int(prefix[1]) if prefix else None,
                                       'shape': shape, 'storage_dtype': stash[3], 'logical_dtype': stash[4],
                                       'payload_bytes': math.prod(shape) * width if width is not None else None,
                                       'lifetime': 'unverified', 'storage_identity': None,
                                       'note': 'Payload reconstructed from logged geometry; not an independent prediction or peak. Host omission means unknown, not zero.'})
            match = STEP.search(line)
            if match:
                value, unit = _memory_number(match[2]), match[3]
                extra = {}
                for key in ('loss', 'grad_norm', 'tps', 'tflops'):
                    field = re.search(r'\b' + key + r':\s*(' + NUMBER + r'|[-+]?(?:nan|inf))\b', line, re.I)
                    if field:
                        # Keep NaN as text, not nonstandard JSON NaN.
                        try:
                            extra[key] = _number(field[1])
                        except ValueError:
                            extra[key] = field[1].lower()
                samples.append({**evidence, 'rank': rank, 'step': int(match[1]),
                                'metric': 'training_memory_unspecified', 'value': value, 'unit': unit,
                                'bytes': round(value * {'GiB': 2**30, 'MiB': 2**20, 'GB': 10**9, 'MB': 10**6}[unit]),
                                'metric_semantics_verified': False, 'window': 'unverified', **extra})
            match = re.search(r'CUDA memory usage for model:\s*(' + NUMBER + r')(GiB|GB)', line)
            if match:
                unit, value = match[2], _memory_number(match[1])
                samples.append({**evidence, 'rank': rank, 'step': 0,
                                'metric': 'model_initialization_memory_unspecified', 'value': value, 'unit': unit,
                                'bytes': round(value * (2**30 if unit == 'GiB' else 10**9)),
                                'metric_semantics_verified': False, 'window': 'initialization'})
            allocator_matches = list(ALLOCATOR.finditer(line))
            for index, match in enumerate(allocator_matches):
                end = allocator_matches[index + 1].start() if index + 1 < len(allocator_matches) else len(line)
                for counter, text_value in COUNTER.findall(line[match.end():end]):
                    if 'gigabytes' not in counter:
                        continue
                    value = _memory_number(text_value)
                    samples.append({**evidence, 'rank': int(match[1]), 'step': int(match[2]),
                                    'metric': counter, 'value': value, 'unit': match[3],
                                    'bytes': None, 'metric_semantics_verified': False,
                                    'unit_note': 'Printed label retained; conversion base requires logger source verification',
                                    'window': 'peak_reset_unverified' if '-max-' in counter else 'after_iteration'})
            for key, pattern in {
                'training_completed': r'Training completed',
                'nan_gradient': r'(?:grad.norm|grad_norm)\s*[:=]\s*[-+]?nan',
                'oom': r'OutOfMemoryError|CUDA out of memory',
                'full_checkpointing': r'Applied (?:FullAC|full) activation checkpointing',
                'hybridep': r'Initializing HybridEP buffer|HYBRIDEP MoE:',
                'mxfp8_conversion': r'Converted .*MXFP8|Converted .*mxfp8|Converted .*\(mxfp8\)',
                'nvfp4_conversion': r'Converted .*NVFP4|Converted .*nvfp4',
            }.items():
                if re.search(pattern, line, re.I):
                    if len(indicators[key]) < 8:
                        indicators[key].append(evidence)
    return {'schema_version': 1, 'source': 'training_log_observation',
            'source_id': source_id, 'sha256': hashlib.sha256(data).hexdigest(),
            'facts': facts, 'samples': samples, 'indicators': dict(indicators),
            'observed_components': components,
            'limitations': ['Sparse log samples are not allocation events or a memory timeline.',
                            'Unknown ranks are never inferred from line order or repeated messages.',
                            'Training memory and counter reset semantics need the matching logger source.',
                            'No eligibility for CPU/GPU accuracy comparison is inferred from a model label.']}


def summarize(document):
    groups = defaultdict(list)
    for sample in document['samples']:
        groups[(sample['metric'], sample['step'], sample['unit'])].append(sample)
    rows = []
    for (metric, step, unit), samples in sorted(groups.items()):
        values = [sample['value'] for sample in samples]
        rows.append({'metric': metric, 'step': step, 'unit': unit,
                     'sample_count': len(samples), 'minimum': min(values), 'maximum': max(values),
                     'identified_ranks': sorted({s['rank'] for s in samples if s['rank'] is not None}),
                     'unidentified_rank_samples': sum(s['rank'] is None for s in samples)})
    return rows


MATCH_FIELDS = ('model', 'framework', 'revision', 'torch_version', 'hardware', 'world_size',
                'tp', 'pp', 'ep', 'cp', 'dp_shard', 'dp_replicate', 'expert_tp',
                'sequence_length', 'tokens_per_rank', 'precision', 'parameter_dtype',
                'gradient_dtype', 'optimizer_state_dtype', 'optimizer', 'attention',
                'pipeline_schedule', 'global_batch', 'microbatches', 'offload_policy',
                'checkpoint_policy', 'compile_policy', 'routing_policy', 'memory_metric',
                'rank_scope', 'memory_unit_basis', 'counter_window')


def assess_cross_validation(prediction_context, observation_context):
    """Eligibility only; missing identity/metric evidence is not a match."""
    missing, mismatches = [], {}
    for field in MATCH_FIELDS:
        a, b = prediction_context.get(field), observation_context.get(field)
        def unknown(value):
            return (value is None or value == '' or value == [] or value == {}
                    or isinstance(value, str) and any(term in value.lower() for term in ('unknown', 'unverified', 'unspecified'))
                    or isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value))
        if unknown(a) or unknown(b):
            missing.append(field)
        elif a != b:
            mismatches[field] = {'prediction': a, 'observation': b}
    return {'eligible': not missing and not mismatches, 'missing': missing, 'mismatches': mismatches,
            'note': 'Eligibility does not itself establish complete execution, aligned phases or prediction accuracy.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    document = parse_log(args.log)
    document['summary'] = summarize(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'samples': len(document['samples']), 'source': document['source_id']}))


if __name__ == '__main__':
    main()
