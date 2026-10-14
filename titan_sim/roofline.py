"""Standalone SVG roofline plots of event work and explicitly sourced timing.

Logical tensor traffic is not DRAM traffic. Analytical event markers are ceilings,
never achieved throughput. Only calibrated timings produce measured markers.
"""
from html import escape
import math

from .costs import DTYPE_ALIASES, validate_hardware
from .events import nonnegative

_COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')


def _positive(value, field):
    number = nonnegative(value, field)
    return number if number > 0 else None


def roofline_svg(result: dict, hardware: dict) -> str:
    """Return a self-contained log-log SVG; no plotting dependencies required."""
    validate_hardware(hardware)
    rates = {DTYPE_ALIASES.get(dtype, dtype): float(rate)
             for dtype, rate in hardware['peak_flops'].items()}
    rows = [row for row in result.get('timeline', [])
            if row.get('kind') == 'compute' and row.get('source') not in ('host_unmodeled', 'meta_unmodeled', 'metadata_only')]
    points = []
    zero_flops, zero_traffic, missing_peak = 0, 0, 0
    for row in rows:
        flops = _positive(row.get('flops', 0), 'flops')
        read = nonnegative(row.get('bytes_read', 0), 'bytes_read')
        write = nonnegative(row.get('bytes_write', 0), 'bytes_write')
        traffic = read + write
        if flops is None:
            zero_flops += 1
        if traffic == 0:
            zero_traffic += 1
        if flops is None or traffic == 0:
            continue
        dtype = DTYPE_ALIASES.get(row.get('dtype', 'float32'), row.get('dtype', 'float32'))
        if dtype not in rates:
            missing_peak += 1
            continue
        intensity = flops / traffic
        ceiling = min(rates[dtype], hardware['memory_bandwidth_bytes_s'] * intensity)
        measured = None
        if row.get('source') == 'calibrated':
            duration = nonnegative(row.get('duration_us', 0), 'duration_us')
            if duration > 0:
                measured = flops / (duration * 1e-6)
        points.append({'event_id': str(row.get('event_id', '')), 'name': str(row.get('name', '')),
                       'dtype': dtype, 'intensity': intensity, 'ceiling': ceiling,
                       'measured': measured, 'provenance': str(row.get('provenance', ''))})
    dtypes = sorted({point['dtype'] for point in points})
    if not dtypes:
        preferred = ['bfloat16', 'float16', 'float32']
        dtypes = [dtype for dtype in preferred if dtype in rates][:1] or [sorted(rates)[0]]
    ridges = [rates[dtype] / hardware['memory_bandwidth_bytes_s'] for dtype in dtypes]
    intensities = [point['intensity'] for point in points] + ridges
    xmin = math.floor(math.log10(min(intensities))) - 1
    xmax = math.ceil(math.log10(max(intensities))) + 1
    # Vertical units are TFLOP/s, with all bounds/measurements visible.
    bandwidth = hardware['memory_bandwidth_bytes_s']
    values = [min(rates[dtype], bandwidth * 10**xmin) / 1e12 for dtype in dtypes]
    values += [rates[dtype] / 1e12 for dtype in dtypes]
    values += [point['measured'] / 1e12 for point in points if point['measured'] is not None]
    ymin = math.floor(math.log10(min(values))) - 1
    ymax = math.ceil(math.log10(max(values))) + 1
    width, height = 1050, 770 + 20 * max(0, len(dtypes) - 3)
    left, top, plot_width, plot_height = 105, 115, 860, 410

    def x(value):
        return left + (math.log10(value) - xmin) / (xmax - xmin) * plot_width

    def y(value):
        return top + plot_height - (math.log10(value / 1e12) - ymin) / (ymax - ymin) * plot_height

    def text(value, px, py, **attrs):
        extra = ''.join(f' {key.replace("_", "-")}="{escape(str(val), quote=True)}"' for key, val in attrs.items())
        return f'<text x="{px:.2f}" y="{py:.2f}"{extra}>{escape(str(value))}</text>'

    title = str(hardware.get('name', 'Hardware'))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
           '<title id="title">Logical-traffic roofline: analytical ceilings and calibrated measurements</title>',
           '<desc id="description">Logarithmic axes. Open circles show analytical ceilings, not achieved throughput. Filled diamonds show throughput derived only from calibrated durations. Logical traffic is not measured DRAM traffic.</desc>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, sans-serif" font-size="13" fill="#20242a">',
           text('Roofline using logical tensor traffic', left, 35, font_size=24, font_weight='bold'),
           text(title, left, 60, font_size=15),
           text('Analytical markers are ceilings; measured throughput requires calibrated duration.', left, 83, font_size=14)]
    # Limit tick density for unusually broad trace ranges.
    xstep = max(1, math.ceil((xmax - xmin) / 9))
    ystep = max(1, math.ceil((ymax - ymin) / 7))
    for exponent in range(xmin, xmax + 1, xstep):
        px = x(10**exponent)
        svg.append(f'<path d="M {px:.2f} {top} V {top + plot_height}" stroke="#e1e5e9"/>')
        svg.append(text(f'10^{exponent}', px, top + plot_height + 23, text_anchor='middle'))
    for exponent in range(ymin, ymax + 1, ystep):
        py = y(10**exponent * 1e12)
        svg.append(f'<path d="M {left} {py:.2f} H {left + plot_width}" stroke="#e1e5e9"/>')
        svg.append(text(f'10^{exponent}', left - 12, py + 4, text_anchor='end'))
    svg += [f'<path d="M {left} {top} V {top + plot_height} H {left + plot_width}" fill="none" stroke="#30363d"/>',
            text('Arithmetic intensity (FLOP / logical byte)', left + plot_width / 2, top + plot_height + 55, text_anchor='middle', font_size=15),
            f'<text transform="translate(24 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-size="15">Throughput (TFLOP/s)</text>']
    colors = {dtype: _COLORS[index % len(_COLORS)] for index, dtype in enumerate(dtypes)}
    for index, dtype in enumerate(dtypes):
        ridge = rates[dtype] / bandwidth
        coords = [(x(10**xmin), y(min(rates[dtype], bandwidth * 10**xmin))),
                  (x(ridge), y(rates[dtype])), (x(10**xmax), y(rates[dtype]))]
        path = ' '.join(f'{"M" if i == 0 else "L"} {px:.2f} {py:.2f}' for i, (px, py) in enumerate(coords))
        svg.append(f'<path class="roofline-ceiling" data-dtype="{escape(dtype, quote=True)}" d="{path}" fill="none" stroke="{colors[dtype]}" stroke-width="2"/>')
        legend_y = 607 + index * 20
        svg.append(f'<path d="M {left} {legend_y - 4} h 25" stroke="{colors[dtype]}" stroke-width="2"/>')
        svg.append(text(f'{dtype}: dense theoretical peak {rates[dtype] / 1e12:g} TFLOP/s', left + 35, legend_y))
    for point in points:
        px, py = x(point['intensity']), y(point['ceiling'])
        description = f"{point['event_id']} {point['name']}; {point['dtype']}; intensity={point['intensity']:.6g} FLOP/logical byte; analytical ceiling={point['ceiling'] / 1e12:.6g} TFLOP/s (not achieved)"
        svg.append(f'<circle class="analytical-ceiling" data-event-id="{escape(point["event_id"], quote=True)}" cx="{px:.2f}" cy="{py:.2f}" r="4" fill="white" stroke="{colors[point["dtype"]]}"><title>{escape(description)}</title></circle>')
        if point['measured'] is not None:
            my = y(point['measured'])
            label = f"{point['event_id']}: calibrated throughput={point['measured'] / 1e12:.6g} TFLOP/s; provenance={point['provenance']}"
            svg.append(f'<path class="calibrated-measurement" data-event-id="{escape(point["event_id"], quote=True)}" d="M {px:.2f} {my - 5:.2f} l 5 5 l -5 5 l -5 -5 Z" fill="{colors[point["dtype"]]}"><title>{escape(label)}</title></path>')
    bottom = max(675, 625 + len(dtypes) * 20)
    svg.append(text('○ Analytical ceiling    ◆ Calibrated throughput (when available)', left, bottom, font_size=14))
    svg.append(text(f'GPU compute events: {len(rows)}; plotted: {len(points)}; zero/unreported FLOPs: {zero_flops}; zero traffic: {zero_traffic}; missing peak: {missing_peak}.', left, bottom + 23))
    unknown = result.get('unknown_flop_ops', result.get('summary', {}).get('unknown_flop_ops'))
    if isinstance(unknown, dict):
        unknown_count = sum(nonnegative(count, 'unknown FLOP count') for count in unknown.values())
        coverage = f'Capture reports {unknown_count:g} events across {len(unknown)} operators without a FLOP formula.'
    else:
        coverage = 'Operator FLOP coverage unavailable; zero FLOPs do not prove an operation is modeled.'
    svg.append(text(coverage, left, bottom + 44))
    svg.append(text('HBM roofline uses logical bytes, not observed DRAM traffic; collective and CPU events are excluded.', left, bottom + 65))
    svg.append('</g></svg>')
    return '\n'.join(svg)
