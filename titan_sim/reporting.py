"""Portable reports with explicit separation of predictions and observations."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

GIB = 1024 ** 3


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def memory_svg(rows, *, coordinate, title, value='allocated_bytes', capacity=None):
    """Generate a standalone vector plot without optional plotting dependencies."""
    points = [(float(row[coordinate]), float(row[value]) / GIB) for row in rows]
    width, height, left, top, bottom = 1000, 480, 90, 65, 75
    plot_width, plot_height = width-left-30, height-top-bottom
    xmax = max((x for x, _ in points), default=1) or 1
    ymax = max([y for _, y in points] + ([capacity/GIB] if capacity else []) + [1]) * 1.08
    def xy(x, y):
        return left+x/xmax*plot_width, top+plot_height-y/ymax*plot_height
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<g font-family="sans-serif" font-size="14" fill="#152536">',
                f'<text x="{left}" y="30" font-size="20">{html.escape(title)}</text>',
                f'<text x="20" y="55">GiB</text>']
    for tick in range(6):
        y = ymax*tick/5
        _, py = xy(0,y)
        elements += [f'<path d="M {left} {py} H {width-30}" stroke="#dce3ea"/>',
                     f'<text x="{left-10}" y="{py+5}" text-anchor="end">{y:.1f}</text>']
        x = xmax*tick/5
        px, _ = xy(x,0)
        elements.append(f'<text x="{px}" y="{height-bottom+25}" text-anchor="middle">{x:.0f}</text>')
    if capacity:
        _, py = xy(0, capacity/GIB)
        elements.append(f'<path d="M {left} {py} H {width-30}" stroke="#bd3c35" stroke-dasharray="6 4"/>')
    path = []
    for index, (x, y) in enumerate(points):
        px, py = xy(x,y)
        path.append(f'{"M" if index == 0 else "H"} {px:.2f}' + (f' {py:.2f}' if index == 0 else f' V {py:.2f}'))
    elements += [f'<path d="{" ".join(path)}" fill="none" stroke="#176b98" stroke-width="2"/>',
                 f'<text x="{width/2}" y="{height-18}" text-anchor="middle">{html.escape(coordinate)}</text>', '</g></svg>']
    return '\n'.join(elements)


def export_result(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'simulation.json').write_text(json.dumps(result, indent=2)+'\n')
    for name in ('timeline', 'memory', 'logical_memory'):
        write_csv(output/f'{name}.csv', result[name])
    for rank in result['summary']['per_rank']:
        rows = [row for row in result['logical_memory'] if row['rank'] == int(rank)]
        (output/f'logical-memory-rank{rank}.svg').write_text(memory_svg(
            rows, coordinate='event_index', title=f'Rank {rank}: logical tensor ownership (no allocator reserve)'))
    return output


def comparison_summary(projection, observed):
    """Refuse accuracy claims for unmatched configurations or incomplete runs."""
    predicted_meta, actual_meta = projection['metadata'], observed['metadata']
    mismatches = {}
    for key in ('torchtitan_commit', 'torch_version', 'world_size', 'flavor',
                'sequence_length', 'tokens_per_microbatch_per_dp_rank', 'rank'):
        if key not in predicted_meta or key not in actual_meta or predicted_meta[key] != actual_meta[key]:
            mismatches[key] = [predicted_meta.get(key), actual_meta.get(key)]
    for key in ('policy_fingerprint',):
        if not predicted_meta.get(key) or predicted_meta.get(key) != actual_meta.get(key):
            mismatches[key] = [predicted_meta.get(key), actual_meta.get(key)]
    complete = predicted_meta.get('status') == 'complete' and observed.get('status') == 'complete'
    coverage_limitations = []
    for side, metadata in (('prediction', predicted_meta), ('observation', actual_meta)):
        scope = metadata.get('capture_scope', 'strict')
        if scope != 'strict':
            coverage_limitations.append(f'{side} capture_scope={scope} excludes full execution coverage')
        initialization = metadata.get('initialization_coverage', {})
        for field in ('initialization_time_included', 'initialization_peak_included'):
            if initialization.get(field) is False:
                coverage_limitations.append(f'{side} {field}=false')
    # Metadata equality alone does not certify matched schedules or allocations.
    return {'config_mismatches': mismatches, 'both_complete': complete,
            'coverage_limitations': coverage_limitations,
            'eligible_for_phase_comparison': complete and not mismatches and not coverage_limitations,
            'note': 'Logical tensor ownership excludes allocator reserve, kernel scratch and deferred CUDA use. Compare phase boundaries only after matching execution policies and routing.'}
