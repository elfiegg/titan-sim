"""Compare independently generated storage projections and GPU observations.

No time/event-axis interpolation is performed. Named phases must match exactly,
including rank, step and module. Otherwise supply explicit correspondences, e.g.::

    [{"label": "step 1 forward/backward boundary", "mode": "boundary",
      "prediction": {"event_id": "r0:e42"},
      "observation": {"sample_index": 17}}]

A boundary selects exactly one row on each side. ``sampled_peak`` selects a
specified group (phase, step, or inclusive event/sample index range) and compares
its sampled maxima. This does not turn allocator bytes into logical storage or
certify that observed samples captured the actual within-phase peak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .events import nonnegative
from .reporting import comparison_summary, write_csv

PREDICTED_METRIC = 'logical_gpu_tensor_storage_bytes'
OBSERVED_METRIC = 'pytorch_cuda_allocator_allocated_bytes'


def _simulation(projection):
    simulation = projection.get('simulation', projection)
    if not isinstance(simulation, dict):
        raise ValueError('projection.simulation must be an object')
    return simulation


def _rows(document, key, rank):
    raw = document.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'{key} must be a nonempty list of independently collected samples')
    rows = []
    for index, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise ValueError(f'{key}[{index}] must be an object')
        if sample.get('rank', rank) != rank:
            continue
        if 'allocated_bytes' not in sample:
            raise ValueError(f'{key}[{index}] has no allocated_bytes')
        value = nonnegative(sample['allocated_bytes'], f'{key}[{index}].allocated_bytes')
        if not value.is_integer():
            raise ValueError('storage byte counters must be integers')
        rows.append({**sample, 'rank': rank, '_index': index, 'allocated_bytes': int(value)})
    if not rows:
        raise ValueError(f'{key} has no samples for matched rank {rank}')
    return rows


def _phase_key(row):
    return (row['rank'], row.get('phase'), row.get('step'), row.get('module', ''))


def _select(rows, selector, side):
    if not isinstance(selector, dict) or not selector:
        raise ValueError(f'{side} selector must be a nonempty object')
    allowed = {'phase', 'step', 'module', 'rank'}
    allowed |= {'event_id', 'event_index', 'event_index_start', 'event_index_end'} if side == 'prediction' else {'sample_index', 'sample_index_start', 'sample_index_end'}
    for key, value in selector.items():
        if 'index' in key or key in ('rank', 'step'):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{side} selector {key} must be a nonnegative integer')
    for coordinate in ('event_index', 'sample_index'):
        lower, upper = selector.get(coordinate + '_start'), selector.get(coordinate + '_end')
        if lower is not None and upper is not None and lower > upper:
            raise ValueError('selector range start must not exceed end')
    unknown = set(selector) - allowed
    if unknown:
        raise ValueError(f'unsupported {side} selector fields: {sorted(unknown)}')
    def matches(row):
        for key, value in selector.items():
            if key.endswith('_start') or key.endswith('_end'):
                coordinate = key.rsplit('_', 1)[0]
                actual = row['_index'] if coordinate == 'sample_index' else row.get(coordinate)
                if actual is None or (key.endswith('_start') and actual < value) or (key.endswith('_end') and actual > value):
                    return False
            elif key == 'sample_index':
                if row['_index'] != value:
                    return False
            elif row.get(key, '' if key == 'module' else None) != value:
                return False
        return True
    selected = [row for row in rows if matches(row)]
    if not selected:
        raise ValueError(f'{side} selector matches no rows: {selector}')
    return selected


def compare_documents(projection: dict, observed: dict, correspondence: list[dict] | None = None) -> dict:
    """Return labeled differences after the mandatory configuration/status gate.

    The currently supported metrics have different definitions, so neither
    signed differences nor sampled maxima are labeled prediction errors.
    """
    if not isinstance(projection, dict) or not isinstance(observed, dict):
        raise ValueError('projection and observation must be JSON objects')
    try:
        gate = comparison_summary(projection, observed)
    except (KeyError, TypeError) as error:
        raise ValueError('both documents require complete comparison metadata') from error
    if not gate['eligible_for_phase_comparison']:
        raise ValueError('comparison rejected by reporting.comparison_summary: ' + json.dumps(gate, sort_keys=True))
    simulation = _simulation(projection)
    if simulation is not projection and 'metadata' in simulation:
        nested_gate = comparison_summary(simulation, observed)
        if not nested_gate['eligible_for_phase_comparison']:
            raise ValueError('nested simulation metadata fails comparison_summary: ' + json.dumps(nested_gate, sort_keys=True))
    if projection.get('partial_trace') or simulation.get('partial_trace'):
        raise ValueError('partial projections cannot be compared')
    for document in (projection, simulation, observed):
        if 'status' in document and document['status'] != 'complete':
            raise ValueError('blocked or incomplete documents cannot be compared')
        if 'status' in document.get('metadata', {}) and document['metadata']['status'] != 'complete':
            raise ValueError('blocked or incomplete metadata cannot be compared')
    if projection.get('source') not in (None, 'cpu_projection') or observed.get('source') != 'gpu_observation':
        raise ValueError('inputs must be an independent CPU projection and GPU observation')
    rank = projection['metadata']['rank']
    predicted = _rows(simulation, 'logical_memory', rank)
    actual = _rows(observed, 'samples', rank)
    alignments = []
    if correspondence is None:
        groups = []
        for rows in (predicted, actual):
            grouped = {}
            for row in rows:
                if isinstance(row.get('phase'), str) and row['phase']:
                    grouped.setdefault(_phase_key(row), []).append(row)
            groups.append(grouped)
        for key, pred_rows in groups[0].items():
            if key in groups[1]:
                label = f'rank {key[0]} / {key[1]}'
                if key[2] is not None:
                    label += f' / step {key[2]}'
                if key[3]:
                    label += f' / {key[3]}'
                alignments.append((label, 'sampled_peak', pred_rows, groups[1][key], 'exact_named_phase'))
    else:
        if not isinstance(correspondence, list):
            raise ValueError('correspondence must be a list')
        labels = set()
        for entry in correspondence:
            if not isinstance(entry, dict) or not isinstance(entry.get('label'), str) or not entry['label']:
                raise ValueError('each correspondence requires a nonempty label')
            if entry['label'] in labels:
                raise ValueError('correspondence labels must be unique')
            labels.add(entry['label'])
            mode = entry.get('mode', 'boundary')
            if mode not in ('boundary', 'sampled_peak'):
                raise ValueError('correspondence mode must be boundary or sampled_peak')
            p = _select(predicted, entry.get('prediction'), 'prediction')
            a = _select(actual, entry.get('observation'), 'observation')
            if mode == 'boundary' and (len(p) != 1 or len(a) != 1):
                raise ValueError('boundary correspondence must select exactly one sample on each side')
            alignments.append((entry['label'], mode, p, a, 'explicit_correspondence'))
    if not alignments:
        raise ValueError('no exact named-phase matches; supply explicit correspondence instead of interpolating axes')
    rows = []
    for label, mode, p, a, basis in alignments:
        predicted_value = max(row['allocated_bytes'] for row in p)
        observed_value = max(row['allocated_bytes'] for row in a)
        difference = predicted_value - observed_value
        rows.append({'label': label, 'rank': rank, 'mode': mode, 'alignment': basis,
                     'predicted_metric': PREDICTED_METRIC, 'observed_metric': OBSERVED_METRIC,
                     'predicted_bytes': predicted_value, 'observed_bytes': observed_value,
                     'logical_minus_allocator_bytes': difference,
                     'difference_over_observed': difference / observed_value if observed_value else None,
                     'metric_definitions_match': False, 'peak_error_bytes': None,
                     'prediction_event_indices': [row.get('event_index') for row in p],
                     'prediction_event_ids': [row.get('event_id') for row in p],
                     'observation_sample_indices': [row['_index'] for row in a],
                     'prediction_sample_count': len(p), 'observation_sample_count': len(a),
                     'peak_error_unavailable_reason': 'Logical tensor ownership and CUDA allocator allocation are different metrics.'})
    return {'schema_version': 1, 'status': 'complete', 'source': 'independent_comparison',
            'gate': gate, 'metadata': dict(projection['metadata']), 'comparisons': rows,
            'metric_definitions': {
                PREDICTED_METRIC: 'Reference-owned GPU tensor storage in capture program order; excludes reserve, hidden workspaces and deferred CUDA usage.',
                OBSERVED_METRIC: 'Independent PyTorch CUDA memory_allocated sample; includes allocator-backed live allocations, excludes reserve and non-PyTorch allocations.'},
            'assumptions': [
                'Phase alignment is exact named phase identity or caller-supplied correspondence; routing/schedule equivalence is not inferred beyond the metadata gate.',
                'sampled_peak is a maximum over selected samples only, not a measured continuous within-phase peak.',
                'GPU peak_allocated_bytes is not used: its reset interval is not generally the selected phase.',
                'No interpolation or equivalence is assumed between GPU wall time and CPU event indices.',
                'Signed metric differences are not prediction accuracy errors; no allocator observation is copied into the projection.']}


def comparison_plot(report: dict, output: str | Path) -> None:
    """Save a standalone categorical comparison plot using optional matplotlib."""
    if not report.get('gate', {}).get('eligible_for_phase_comparison') or report.get('status') != 'complete':
        raise ValueError('plot requires a completed, gated comparison report')
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as error:
        raise ValueError('plotting requires the optional matplotlib dependency') from error
    rows = report['comparisons']
    figure = Figure(figsize=(max(8, min(24, len(rows) * 1.2)), 5.8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    positions = list(range(len(rows)))
    axes.bar([x - .2 for x in positions], [row['predicted_bytes'] / 1024**3 for row in rows], width=.4,
             label='Projected logical tensor storage', color='#0072B2')
    axes.bar([x + .2 for x in positions], [row['observed_bytes'] / 1024**3 for row in rows], width=.4,
             label='Observed allocator allocated (sampled)', color='#D55E00')
    axes.set_xticks(positions, [row['label'] for row in rows], rotation=35, ha='right')
    axes.set_ylabel('GiB')
    axes.set_xlabel('Explicitly aligned boundary or named phase; no shared time axis')
    axes.set_title('Independent memory metrics; differences are not accuracy errors')
    axes.legend()
    axes.grid(axis='y', alpha=.2)
    figure.savefig(output)


def export_comparison(projection, observed, output, *, correspondence=None, plot=False):
    report = compare_documents(projection, observed, correspondence)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    write_csv(output / 'comparison.csv', report['comparisons'])
    if plot:
        comparison_plot(report, output / 'comparison.svg')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('projection', type=Path)
    parser.add_argument('observation', type=Path)
    parser.add_argument('--correspondence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Save comparison.svg with optional matplotlib')
    args = parser.parse_args(argv)
    try:
        report = export_comparison(json.loads(args.projection.read_text()), json.loads(args.observation.read_text()),
                                   args.output, correspondence=json.loads(args.correspondence.read_text()) if args.correspondence else None,
                                   plot=args.plot)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({'status': report['status'], 'aligned_comparisons': len(report['comparisons'])}))


if __name__ == '__main__':
    main()
