"""Native GPU cost collection without model/operator reimplementations.

``measure(workload, config)`` supports two explicit modes:

* ``trace`` reads checksum-verified Chrome GPU events selected by prepare-runtime.
  Requires hardware_id and accept_profiler_overhead=True. These are instrumented
  observations of the captured run, NOT an independent held-out benchmark.
* ``execute`` invokes a trusted ``region_provider(workload, provider_config)``.
  The provider returns {"regions": [...], "provenance": {...}}. Each region has
  cost_key, implementation, context, run (a zero-argument callable), and
  stream_semantics="current_stream". ``run`` must join any auxiliary work to the
  current stream before returning. The provider owns representative inputs,
  native implementation selection, state reset, compilation and tensor lifetime.
  Optional ``before_sample`` resets inputs outside timing (also before warmup).
  Graph replay requires supports_cuda_graph=True and no before_sample callback;
  providers must prepare stable storage and graph-safe state themselves.

Distributed execution requires an already initialized process group. Every rank
provides the same ordered round_id/round_context sequence, with all ranks listed
in round_context.participants. Each rank supplies its own exact-key callable;
barriers occur outside timing and costs use the maximum duration across ranks.
Subgroup-only rounds are unsupported. Only the coordinator returns a catalog;
other ranks return an empty dictionary and must not write an output file.
No process groups, allocations, jobs, model code or dependencies are changed.
"""
from collections import defaultdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics

from .adapters import resolve, validate_catalog


GPU_CATEGORIES = frozenset(('kernel', 'gpu_memcpy', 'gpu_memset'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def native_cost_key(implementation, context):
    """Exact implementation/context identity, shared with runtime preparation."""
    return 'native-runtime:' + _digest(dict(implementation=implementation, context=context))


def _descriptor(node):
    binding = node.attrs.get('native_measurement')
    if not isinstance(binding, dict):
        raise ValueError('native measurement binding missing for node '+str(node.id))
    for field in ('implementation', 'context'):
        if not isinstance(binding.get(field), dict) or not binding[field]:
            raise ValueError('nonempty native '+field+' required')
    expected = native_cost_key(binding['implementation'], binding['context'])
    if node.attrs.get('cost_key') != expected:
        raise ValueError('native implementation/context cost key mismatch')
    return binding


def _record(workload, samples, **metadata):
    if not samples or any(isinstance(s, bool) or not isinstance(s, (int, float))
                          or not math.isfinite(s) or s < 0 for s in samples):
        raise ValueError('GPU samples must be finite nonnegative seconds')
    return dict(hardware_id=workload.hardware['id'], seconds=statistics.median(samples),
                samples_seconds=samples,
                evidence_sha256=hashlib.sha256(json.dumps(samples, separators=(',', ':')).encode()).hexdigest(),
                statistic='median', independent_validation=False, **metadata)


def _coordinator(config):
    coordinator = config.get('coordinator_rank', 0)
    if type(coordinator) is not int or coordinator < 0:
        raise ValueError('coordinator_rank must be a nonnegative integer')
    if int(os.environ.get('RANK', '0')) != coordinator:
        raise ValueError('native runner must run only in the catalog coordinator process')


def _trace(workload, config):
    if config.get('accept_profiler_overhead') is not True:
        raise ValueError('trace costs require accept_profiler_overhead=true; they are instrumented observations')
    artifacts = {}
    for item in workload.manifest.get('native_trace_artifacts', []):
        if isinstance(item, dict) and item.get('sha256') and item.get('path'):
            artifacts[item['sha256']] = item['path']
    grouped = defaultdict(list)
    samples, sources, identities = defaultdict(list), defaultdict(list), {}
    used = set()
    for node in workload.nodes:
        if node.attrs.get('zero_cost_semantic') or 'native_measurement' not in node.attrs:
            continue
        binding = _descriptor(node)
        if binding.get('kind') != 'chrome_gpu_event':
            raise ValueError('trace mode requires prepared Chrome GPU event bindings')
        sha, index = binding.get('artifact_sha256'), binding.get('event_index')
        if type(index) is not int or index < 0 or sha not in artifacts:
            raise ValueError('GPU event binding has no retained artifact/index')
        if (sha, index) in used:
            raise ValueError('same GPU event charged by multiple workload nodes')
        used.add((sha, index))
        if binding['context'].get('occurrence_only') is True:
            for field, value in dict(artifact_sha256=sha, event_index=index, rank=node.rank,
                                     step=node.attrs.get('step')).items():
                if binding['context'].get(field) != value:
                    raise ValueError('GPU occurrence context differs from bound '+field)
        grouped[sha].append((node, binding, index))
    # Parsed Chrome JSON can be much larger than the retained file. Process one
    # artifact at a time even when workload nodes interleave ranks/steps. Never
    # retain all-rank trace trees in a cache; only compact sample audits survive.
    for sha, bindings in grouped.items():
        path = Path(config.get('artifact_paths', {}).get(sha, artifacts[sha]))
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('native trace artifact checksum mismatch')
        trace = json.loads(raw)
        del raw
        events = trace.get('traceEvents', [])
        device_metadata = (dict(device_properties_sha256=_digest(trace['deviceProperties']),
                                device_properties_location='/deviceProperties')
                           if 'deviceProperties' in trace else {})
        for node, binding, index in bindings:
            if index >= len(events):
                raise ValueError('GPU event index outside retained trace')
            event = events[index]
            if event.get('ph') != 'X' or event.get('cat') not in GPU_CATEGORIES:
                raise ValueError('bound trace event is not a completed GPU activity')
            if (event.get('name') != binding.get('event_name') or
                    event.get('cat') != binding.get('event_category')):
                raise ValueError('GPU event implementation identity mismatch')
            args = event.get('args', {})
            implementation = dict(name=event['name'], category=event['cat'],
                                  launch={k: args[k] for k in ('grid', 'block', 'registers per thread',
                                                              'shared memory', 'bytes') if k in args})
            if binding['implementation'] != implementation:
                raise ValueError('GPU launch implementation differs from exact cost descriptor')
            if binding['context'].get('hardware_id') != workload.hardware['id']:
                raise ValueError('trace context hardware does not match workload')
            # Chrome/Kineto complete event timestamps and durations are microseconds;
            # displayTimeUnit only controls viewer presentation, not stored units.
            duration = event.get('dur')
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise ValueError('GPU event duration missing or invalid')
            key = node.attrs['cost_key']
            samples[key].append(duration / 1e6)
            identities[key] = binding
            sources[key].append(dict(artifact_sha256=sha, event_index=index, rank=node.rank,
                                     step=node.attrs.get('step'), event_name=event['name'],
                                     event_category=event['cat'], **device_metadata))
        del events, trace
    if not samples:
        raise ValueError('no matched native GPU events are available for trace costs')
    # The source recipe/runtime manifest is retained in the input workload.
    # Repeating it for every occurrence can inflate a catalog by gigabytes.
    # Bind it once and keep a compact immutable-source reference in each record.
    source_provenance = workload.manifest.get('provenance', {})
    provenance_reference = dict(source_provenance_location='input workload /manifest/provenance',
                                source_provenance_sha256=_digest(source_provenance))
    for field in ('torch_version', 'cuda_version'):
        value = source_provenance.get(field)
        if isinstance(value, str) and len(value) <= 128:
            provenance_reference[field] = value
    return {key: _record(workload, values, measurement_kind='profiled_gpu_activity',
                        timing_method='kineto_chrome_gpu_duration',
                        implementation=identities[key]['implementation'],
                        context=identities[key]['context'], sources=sources[key],
                        provenance=provenance_reference,
                        profiling_overhead_accepted=True,
                        hardware_identity_basis='captured_workload_and_user_attestation',
                        limitations=['Profiler instrumentation may perturb durations.',
                                     'Same captured execution supplies workload and costs; not held-out validation.'])
            for key, values in samples.items()}


def _positive_int(config, key, default, minimum):
    value = config.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(key+' must be an integer >= '+str(minimum))
    return value


def _provider_plans(bundle, expected, graph_mode):
    if not isinstance(bundle, dict) or not isinstance(bundle.get('provenance'), dict) or not bundle['provenance']:
        raise ValueError('region provider must return regions and nonempty provenance')
    regions = bundle.get('regions')
    if not isinstance(regions, list):
        raise ValueError('region provider regions must be a list')
    plans, seen = [], set()
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError('region descriptor must be a dictionary')
        key = region.get('cost_key')
        if key not in expected or key in seen:
            raise ValueError('provider region key is unknown or duplicated')
        seen.add(key)
        binding = expected[key]
        if any(region.get(k) != binding[k] for k in ('implementation', 'context')):
            raise ValueError('provider implementation/context does not match workload')
        if region.get('stream_semantics') != 'current_stream' or not callable(region.get('run')):
            raise ValueError('provider must supply run and current_stream completion contract')
        reset = region.get('before_sample')
        if reset is not None and not callable(reset):
            raise ValueError('before_sample must be callable')
        if graph_mode and (region.get('supports_cuda_graph') is not True or reset is not None):
            raise ValueError('CUDA graph timing requires explicit graph support and stable state without before_sample')
        plans.append((key, region))
    if set(expected) != seen:
        raise ValueError('provider must supply every requested exact native cost key')
    return plans


def _execution_preflight(workload, config, torch, rank, world):
    if config.get('hardware_id') != workload.hardware.get('id'):
        raise ValueError('explicit measurement hardware_id must match workload')
    warmup = _positive_int(config, 'warmup', 5, 1)
    repeats = _positive_int(config, 'repeats', 20, 2)
    graph_mode = config.get('cuda_graph', False)
    if type(graph_mode) is not bool:
        raise ValueError('cuda_graph must be a boolean')
    expected = {}
    for node in workload.nodes:
        if not node.attrs.get('zero_cost_semantic'):
            binding = _descriptor(node)
            expected[node.attrs['cost_key']] = binding
            if binding['context'].get('cuda_graph', False) != graph_mode:
                raise ValueError('CUDA graph timing mode must match the exact workload context key')
    if not torch.cuda.is_available():
        raise ValueError('native executable measurement requires CUDA; no GPU samples were collected')
    device = torch.device(config.get('device', 'cuda:'+os.environ.get('LOCAL_RANK', '0')))
    if device.type != 'cuda':
        raise ValueError('native measurement device must be CUDA')
    required_device = config.get('expected_device')
    if not isinstance(required_device, dict) or not {'name', 'total_memory'} <= required_device.keys():
        raise ValueError('execute mode requires expected_device with name and total_memory in bytes')
    props = torch.cuda.get_device_properties(device)
    actual = dict(name=props.name, total_memory=props.total_memory,
                  compute_capability=[props.major, props.minor])
    if any(actual.get(k) != v for k, v in required_device.items()):
        raise ValueError('actual CUDA hardware does not match expected_device')
    declared_world = workload.manifest.get('world_size', len(workload.manifest.get('ranks', [0])))
    if declared_world != world or int(os.environ.get('WORLD_SIZE', str(world))) != world:
        raise ValueError('distributed execution requires the existing matching process group')
    if sorted(workload.manifest.get('ranks', [])) != list(range(world)):
        raise ValueError('execute mode requires one aggregate workload containing every process-group rank')
    coordinator = config.get('coordinator_rank', 0)
    if type(coordinator) is not int or not 0 <= coordinator < world:
        raise ValueError('invalid catalog coordinator rank')
    expected = {n.attrs['cost_key']: expected[n.attrs['cost_key']] for n in workload.nodes
                if n.rank == rank and not n.attrs.get('zero_cost_semantic')}
    provider = resolve(config.get('region_provider', ''))
    try:
        provider_path = inspect.getsourcefile(provider)
    except (TypeError, OSError):
        provider_path = None
    source_sha = hashlib.sha256(Path(provider_path).read_bytes()).hexdigest() if provider_path else None
    if world > 1 and graph_mode:
        raise ValueError('distributed CUDA graph execution is not supported by this runner')
    return dict(warmup=warmup, repeats=repeats, graph_mode=graph_mode, device=device,
                actual=actual, coordinator=coordinator, expected=expected,
                provider=provider, source_sha=source_sha)


def _execute(workload, config):
    # Existing process-group transport and device assignment belong to the
    # launcher/application. Once entered, validation errors are shared before
    # any rank starts provider execution or waits in a measurement barrier.
    import torch
    dist = torch.distributed
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else int(os.environ.get('RANK', '0'))
    world = dist.get_world_size() if distributed else 1
    def gather(value):
        if not distributed:
            return [value]
        values = [None] * world
        dist.all_gather_object(values, value)
        return values
    def barrier():
        if distributed:
            dist.barrier()
    preflight, error = None, None
    try:
        preflight = _execution_preflight(workload, config, torch, rank, world)
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    errors = gather(error)
    if any(errors):
        raise ValueError('native execution preflight failed on rank(s): '+str(errors))
    warmup, repeats, graph_mode = (preflight[k] for k in ('warmup', 'repeats', 'graph_mode'))
    device, actual, coordinator = (preflight[k] for k in ('device', 'actual', 'coordinator'))
    expected, provider, source_sha = (preflight[k] for k in ('expected', 'provider', 'source_sha'))
    schedule = dict(warmup=warmup, repeats=repeats, graph_mode=graph_mode, coordinator=coordinator,
                    provider=config.get('region_provider'), hardware_id=config.get('hardware_id'))
    if any(peer != schedule for peer in gather(schedule)):
        raise ValueError('all ranks must agree on measurement counts, mode, provider and coordinator')
    with torch.cuda.device(device):
        bundle, plans, error = None, None, None
        try:
            bundle = provider(workload, config.get('provider_config', {}))
            plans = _provider_plans(bundle, expected, graph_mode)
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
        errors = gather(error)
        if any(errors):
            raise ValueError('native provider contract failed on rank(s): '+str(errors))
        if distributed:
            rounds = [dict(round_id=r.get('round_id'), round_context=r.get('round_context')) for _, r in plans]
            all_rounds = gather(rounds)
            if any(plan != rounds for plan in all_rounds):
                raise ValueError('all ranks must agree on ordered round IDs and contexts before measurement')
            if any(not isinstance(r['round_id'], str) or not r['round_id'] or
                   not isinstance(r['round_context'], dict) or
                   r['round_context'].get('participants') != list(range(world)) for r in rounds):
                raise ValueError('distributed rounds require IDs and all-rank participants; subgroup-only rounds are unsupported')
            if len({r['round_id'] for r in rounds}) != len(rounds):
                raise ValueError('duplicate distributed round ID')
        catalog = {}
        for key, region in plans:
            run, reset = region['run'], region.get('before_sample')
            # Side-stream warmup also permits subsequent graph capture. Callback
            # work on additional streams must rejoin this stream by contract.
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    barrier()
                    if reset is not None:
                        reset()
                    run()
            stream.synchronize()
            replay = run
            graph = None
            if graph_mode:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run()
                replay = graph.replay
            values = []
            with torch.cuda.stream(stream):
                for _ in range(repeats):
                    barrier()
                    if reset is not None:
                        reset()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    replay()
                    end.record(stream)
                    end.synchronize()
                    values.append(start.elapsed_time(end) / 1000.)
            torch.cuda.current_stream(device).wait_stream(stream)
            all_samples = gather(values)
            # All ranks use the same round representative so communication
            # rendezvous completion reflects its slowest participant.
            representative = [max(v[i] for v in all_samples) for i in range(repeats)]
            catalog[key] = _record(workload, representative,
                measurement_kind='native_executable_region',
                timing_method='cuda_graph_replay_events' if graph_mode else 'cuda_events',
                implementation=region['implementation'], context=region['context'],
                warmup=warmup, repeats=repeats, cuda_graph=graph_mode,
                hardware=actual, rank=rank, per_rank_samples_seconds=all_samples,
                distributed_round=region.get('round_id'), round_context=region.get('round_context'),
                hardware_identity_basis='actual_device_properties_checked',
                provenance=dict(provider=config['region_provider'],
                                provider_provenance=bundle['provenance'], provider_source_sha256=source_sha,
                                torch_version=str(torch.__version__), cuda_version=torch.version.cuda),
                limitations=['Trusted provider owns exact native implementation and input fidelity.',
                             ('Window timing qualifies only the supplied occurrences, streams and inputs; external training contention remains unqualified.'
                              if region['context'].get('measurement_scope') == 'execution_window' else
                              'Isolated region timing does not establish overlap or contention in training.')])
        shards = gather(catalog)
        if rank != coordinator:
            return {}
        merged = {}
        for shard in shards:
            for key, record in shard.items():
                if key in merged and merged[key]['samples_seconds'] != record['samples_seconds']:
                    raise ValueError('same native cost key has conflicting distributed round measurements')
                if key in merged:
                    merged[key].setdefault('contributing_ranks', [merged[key]['rank']]).append(record['rank'])
                else:
                    merged[key] = record
        return merged


def measure(workload, config):
    """Collect native costs; this function never writes competing rank outputs."""
    if not isinstance(config, dict):
        raise ValueError('native runner configuration must be an object')
    mode = config.get('mode')
    if mode not in ('trace', 'execute'):
        raise ValueError('native runner mode must explicitly be trace or execute')
    if mode == 'trace':
        if config.get('hardware_id') != workload.hardware.get('id'):
            raise ValueError('explicit measurement hardware_id must match workload')
        _coordinator(config)
    catalog = _trace(workload, config) if mode == 'trace' else _execute(workload, config)
    validate_catalog(catalog, workload.hardware['id'])
    return catalog
