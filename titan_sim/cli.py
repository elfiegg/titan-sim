"""Capture through explicit adapters, validate workloads, calibrate, or simulate."""
import argparse
import json
import os
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
    runtime = commands.add_parser('runtime-check', help='check prepare-runtime handoff artifacts without launching work')
    runtime.add_argument('manifest', type=Path)
    runtime.add_argument('--capability-report', type=Path)
    runtime.add_argument('--expected', type=Path, help='expected recipe, source, image, target and required check IDs')
    runtime.add_argument('--output', type=Path)
    prepare = commands.add_parser('prepare', help='reconcile captured ranks and explicit execution/storage evidence')
    prepare.add_argument('workloads', type=Path, nargs='+')
    prepare.add_argument('--config', type=Path, help='preparation evidence JSON')
    prepare.add_argument('--output', type=Path, required=True)
    export = commands.add_parser('export-plan', help='export prepared native schedules and graphs through a trusted adapter')
    export.add_argument('--adapter', default='titan_sim.plan_capture:export_recipe',
                        help='importable.module:function; defaults to native TorchTitan exporter')
    export.add_argument('--config', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    lower = commands.add_parser('lower-plan', help='lower native plan actions and graph storage into a CPU workload')
    lower.add_argument('plans', type=Path, nargs='+', help='rank-local exports from one native pipeline group')
    lower.add_argument('--hardware', type=Path, required=True)
    lower.add_argument('--config', type=Path, help='explicit placement and cost/communication bindings')
    lower.add_argument('--output', type=Path, required=True)
    calibrate = commands.add_parser('calibrate', help='invoke a trusted measurement runner for a workload')
    calibrate.add_argument('workload', type=Path)
    calibrate.add_argument('--runner', default='titan_sim.native_runner:measure', help='importable.module:function; defaults to bundled native runner')
    calibrate.add_argument('--config', type=Path, required=True, help='runner-owned JSON configuration')
    calibrate.add_argument('--output', type=Path, required=True, help='calibration catalog JSON file')
    run = commands.add_parser('simulate', help='simulate a supplied workload with explicit costs')
    run.add_argument('workload', type=Path)
    run.add_argument('--calibration', type=Path)
    run.add_argument('--hardware', type=Path, help='explicit replacement hardware JSON; no implicit defaults')
    run.add_argument('--strict', action='store_true', help='reject declared gaps and uncalibrated operation costs')
    run.add_argument('--require-memory-fit', action='store_true', help='return failure unless physical memory has an estimated_fit verdict')
    run.add_argument('--output', type=Path, help='full result JSON file; summary is also printed')
    args = parser.parse_args(argv)
    try:
        if args.command == 'runtime-check':
            from .runtime import check_runtime
            result = check_runtime(args.manifest, args.capability_report,
                                   read_json(args.expected) if args.expected else None)
            result['command'] = args.command
            if args.output:
                emit(result, args.output)
        elif args.command == 'export-plan':
            from .adapters import resolve
            from .native_plan import validate_plan, write_plan
            output = str(args.output)
            if int(os.environ.get('WORLD_SIZE', 1)) > 1 and '{rank}' not in output:
                raise ValueError('distributed plan export requires {rank} in --output to prevent rank file collisions')
            plan = resolve(args.adapter)(read_json(args.config))
            validate_plan(plan)
            rank = plan.get('extraction', {}).get('rank', int(os.environ.get('RANK', 0)))
            output = Path(output.replace('{rank}', str(rank)))
            output.parent.mkdir(parents=True, exist_ok=True)
            write_plan(output, plan)
            result = dict(command=args.command, output=str(output),
                          status=plan.get('status', 'unspecified'),
                          capture_status=plan.get('capture_status', 'complete'),
                          graphs=len(plan['graphs']), stages=len(plan['stages']),
                          unresolved=plan['unresolved'])
        elif args.command == 'lower-plan':
            from .native_plan import read_plan, merge_plans
            from .plan_lowering import lower_plan
            from .plan_memory import lower_graph_memory
            plans = [read_plan(path) for path in args.plans]
            plan = merge_plans(plans) if len(plans) > 1 else plans[0]
            workload = lower_plan(plan, read_json(args.hardware),
                                  read_json(args.config) if args.config else None)
            workload = lower_graph_memory(workload, plan)
            write_workload(args.output, workload)
            result = dict(command=args.command, output=str(args.output),
                          **validate(workload.nodes, workload.storage, workload.manifest),
                          unresolved=workload.manifest.get('unresolved', []))
        elif args.command == 'prepare':
            from .prepare import prepare as prepare_workload
            workload = prepare_workload(args.workloads, read_json(args.config) if args.config else None)
            write_workload(args.output, workload)
            result = dict(command=args.command, output=str(args.output),
                          **validate(workload.nodes, workload.storage, workload.manifest),
                          unresolved=workload.manifest.get('unresolved', []))
        elif args.command == 'capture':
            from .adapters import capture as invoke
            output = str(args.output)
            if int(os.environ.get('WORLD_SIZE', 1)) > 1 and '{rank}' not in output:
                raise ValueError('distributed capture requires {rank} in --output to prevent rank file collisions')
            workload = invoke(args.adapter, read_json(args.config))
            ranks = workload.manifest.get('ranks', [])
            rank = str(ranks[0]) if len(ranks) == 1 else os.environ.get('RANK', '0')
            output = Path(output.replace('{rank}', rank))
            write_workload(output, workload)
            result = dict(command='capture', output=str(output),
                          capture_status=workload.manifest.get('capture_status', 'unspecified'),
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
                config = read_json(args.config)
                distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
                bundled = args.runner == 'titan_sim.native_runner:measure'
                if distributed and not bundled:
                    raise ValueError('custom distributed calibration must be coordinated outside this CLI')
                catalog = invoke(args.runner, workload, config)
                coordinator = not distributed or int(os.environ.get('RANK', 0)) == config.get('coordinator_rank', 0)
                if coordinator:
                    emit(catalog, args.output)
                required = {n.attrs['cost_key'] for n in workload.nodes if 'cost_key' in n.attrs}
                result = dict(command='calibrate', output=str(args.output), records=len(catalog),
                              coordinator=coordinator,
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
        if args.command in ('capture', 'export-plan') and result['capture_status'] in ('blocked', 'incomplete'):
            return 2
        if args.command == 'runtime-check' and result['status'] != 'ready':
            return 2
        if args.command == 'simulate' and args.require_memory_fit and result['memory_verdict'] != 'estimated_fit':
            return 2
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, ArithmeticError) as error:
        print(emit({'error':str(error), 'command':args.command}), end='', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
