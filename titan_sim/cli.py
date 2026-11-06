"""Capture through explicit adapters, validate workloads, calibrate, or simulate."""
import argparse
import json
from pathlib import Path
import sys
from .core import simulate, validate
from .workload import read_workload, write_workload


def read_json(path):
    return json.loads(Path(path).read_text())


def emit(document, path=None):
    text = json.dumps(document, indent=2, allow_nan=False)+'\n'
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture', help='invoke a trusted framework adapter to produce a workload')
    capture.add_argument('--adapter', required=True, help='importable.module:function; executes Python code')
    capture.add_argument('--config', type=Path, required=True, help='adapter-owned JSON configuration')
    capture.add_argument('--output', type=Path, required=True, help='workload JSON file')
    check = commands.add_parser('validate', help='check workload structure, not runtime fidelity or capacity')
    check.add_argument('workload', type=Path)
    calibrate = commands.add_parser('calibrate', help='invoke a trusted measurement runner for a workload')
    calibrate.add_argument('workload', type=Path)
    calibrate.add_argument('--runner', required=True, help='importable.module:function; may execute GPU work')
    calibrate.add_argument('--config', type=Path, required=True, help='runner-owned JSON configuration')
    calibrate.add_argument('--output', type=Path, required=True, help='calibration catalog JSON file')
    run = commands.add_parser('simulate', help='simulate a supplied workload with explicit costs')
    run.add_argument('workload', type=Path)
    run.add_argument('--calibration', type=Path)
    run.add_argument('--hardware', type=Path, help='explicit replacement hardware JSON; no implicit defaults')
    run.add_argument('--strict', action='store_true', help='reject declared gaps and uncalibrated operation costs')
    run.add_argument('--output', type=Path, help='full result JSON file; summary is also printed')
    args = parser.parse_args(argv)
    try:
        if args.command == 'capture':
            from .adapters import capture as invoke
            workload = invoke(args.adapter, read_json(args.config))
            write_workload(args.output, workload)
            result = dict(command='capture', output=str(args.output),
                          **validate(workload.nodes, workload.storage, workload.manifest),
                          unresolved=workload.manifest.get('unresolved', []))
        else:
            workload = read_workload(args.workload)
            if args.command == 'validate':
                result = dict(command='validate', structural_status='valid',
                              **validate(workload.nodes, workload.storage, workload.manifest),
                              unresolved=workload.manifest.get('unresolved', []),
                              scope='DAG and storage references only; not execution, calibration or memory-fit validation')
            elif args.command == 'calibrate':
                from .adapters import calibrate as invoke
                catalog = invoke(args.runner, workload, read_json(args.config))
                emit(catalog, args.output)
                required = {n.attrs['cost_key'] for n in workload.nodes if 'cost_key' in n.attrs}
                result = dict(command='calibrate', output=str(args.output), records=len(catalog),
                              missing_cost_keys=sorted(required-set(catalog)),
                              unkeyed_nonsemantic_nodes=sum(not n.attrs.get('cost_key') and not n.attrs.get('zero_cost_semantic') for n in workload.nodes))
            else:
                from .adapters import validate_catalog
                if args.hardware:
                    workload.hardware = read_json(args.hardware)
                catalog = read_json(args.calibration) if args.calibration else None
                if catalog is not None:
                    validate_catalog(catalog, workload.hardware['id'])
                result = simulate(*workload.arguments(), calibration=catalog, strict=args.strict)
                if args.output:
                    emit(result, args.output)
                result = {k:v for k,v in result.items() if k != 'timing'}
        print(emit(result), end='')
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, ArithmeticError) as error:
        print(emit({'error':str(error), 'command':args.command}), end='', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
