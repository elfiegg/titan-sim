"""Run event simulations without importing PyTorch or initializing CUDA."""
import argparse
import json
from pathlib import Path
from .runtime import simulate
from .reporting import export_result
from .roofline import roofline_svg
from .provenance import validate_calibration_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--timing', choices=['roofline', 'calibrated', 'hybrid'], default='roofline')
    parser.add_argument('--calibrations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true', help='Explicitly export a blocked partial trace; not a complete-model estimate')
    args = parser.parse_args()
    document = json.loads(args.trace.read_text())
    status = document.get('metadata', {}).get('status')
    if status not in (None, 'complete') and not args.allow_partial:
        parser.error(f'Trace status is {status}; use --allow-partial to inspect incomplete coverage')
    hardware = json.loads(args.hardware.read_text())
    calibrations = json.loads(args.calibrations.read_text()) if args.calibrations else None
    if args.timing in ('calibrated', 'hybrid') and document.get('source') == 'cpu_projection':
        validate_calibration_context(document.get('metadata', {}), hardware, calibrations)
    result = simulate(document['events'], hardware, args.timing, calibrations)
    result['metadata'] = document.get('metadata', {})
    result['capture_coverage'] = document.get('coverage', {})
    result['unknown_flop_ops'] = document.get('unknown_flop_ops', {})
    result['partial_trace'] = status not in (None, 'complete')
    export_result(result, args.output)
    (args.output / "roofline.svg").write_text(roofline_svg(result, hardware))
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
