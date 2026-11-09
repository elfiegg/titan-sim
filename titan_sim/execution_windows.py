"""Exact, bounded native callable windows with explicit CUDA stream completion.

A trusted provider retains native tensors/model state and creates callables. This
module schedules them; it does not reconstruct models or infer stream ordering.
All-rank validation happens before providers or sample preparation can proceed.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
from pathlib import Path

from .adapters import resolve
from .core import Node, validate
from .identity import digest, workload_fingerprint
from .native_runner import native_cost_key, _descriptor
from .workload import Workload


def validate_window_manifest(source, manifest):
    """Validate exact membership, local DAG, boundaries and evidence scope."""
    if (manifest.get('schema'), manifest.get('version')) != ('native-execution-window', 1):
        raise ValueError('unsupported execution window schema/version')
    if manifest.get('source_workload_sha256') != workload_fingerprint(source):
        raise ValueError('window source workload digest mismatch')
    participants = manifest.get('participants')
    if participants != list(range(source.manifest.get('world_size', 0))) or participants != sorted(source.manifest.get('ranks', [])):
        raise ValueError('window requires exact existing all-rank membership')
    for key in ('window_id', 'provider', 'provider_source_sha256', 'completion'):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError('window requires '+key)
    if manifest['completion'] != 'join_all_streams':
        raise ValueError('window must join all streams before timing completion')
    for key in ('physical_topology', 'runtime_context', 'input_state_policy'):
        if not isinstance(manifest.get(key), dict) or not manifest[key]:
            raise ValueError('window requires nonempty '+key)
    policy = manifest['input_state_policy']
    if policy.get('lifetime') != 'entire_window' or policy.get('reset') != 'before_sample':
        raise ValueError('window inputs/state must remain live for entire_window and reset before_sample')
    evidence = manifest.get('ordering_evidence', {})
    if evidence.get('kind') not in ('caller_assertion', 'captured_stream_events') or not evidence.get('description'):
        raise ValueError('window ordering evidence must distinguish caller assertion from capture')
    if evidence['kind'] == 'captured_stream_events' and not evidence.get('artifact_sha256'):
        raise ValueError('captured ordering requires retained artifact hash')
    if manifest.get('qualification') not in ('controlled_native_window', 'source_backed_native_window'):
        raise ValueError('invalid window qualification')
    if manifest['qualification'] == 'source_backed_native_window' and evidence['kind'] != 'captured_stream_events':
        raise ValueError('source-backed window requires captured stream/event ordering')
    steps = manifest.get('steps')
    if not isinstance(steps, list) or not steps:
        raise ValueError('window needs nonempty occurrence steps')
    nodes = {node.id: node for node in source.nodes}
    by_id, ordered, pending = {}, [], list(steps)
    for step in steps:
        occurrence = step.get('occurrence_id')
        if type(occurrence) is not int or occurrence not in nodes or occurrence in by_id:
            raise ValueError('unknown or duplicate window occurrence')
        node = nodes[occurrence]
        _descriptor(node)
        if step.get('rank') != node.rank or not isinstance(step.get('stream'), str) or not step['stream']:
            raise ValueError('window occurrence rank/stream mismatch')
        deps = step.get('deps')
        if not isinstance(deps, list) or any(type(d) is not int for d in deps) or len(set(deps)) != len(deps):
            raise ValueError('window dependencies must be unique occurrence IDs')
        by_id[occurrence] = step
    if {step['rank'] for step in steps} != set(participants):
        raise ValueError('every process-group rank must participate in window')
    boundaries = manifest.get('live_in_dependencies', [])
    expected_boundaries = sorted({dep for step in steps for dep in nodes[step['occurrence_id']].deps if dep not in by_id})
    if boundaries != expected_boundaries:
        raise ValueError('window live-in dependencies must exactly cover external source predecessors')
    for step in steps:
        for dep in step['deps']:
            if dep not in by_id or by_id[dep]['rank'] != step['rank']:
                raise ValueError('window waits require local existing occurrences; native communication handles peers')
        if any(dep in by_id and nodes[dep].rank != step['rank'] for dep in nodes[step['occurrence_id']].deps):
            raise ValueError('cross-rank source dependencies require dedicated native communication boundary support')
        source_deps = {dep for dep in nodes[step['occurrence_id']].deps if dep in by_id and nodes[dep].rank == step['rank']}
        if not source_deps.issubset(step['deps']):
            raise ValueError('window dropped a source dependency')
    # Stable topological order also preserves launch order of independent steps.
    while pending:
        ready = next((s for s in pending if set(s['deps']) <= {s['occurrence_id'] for s in ordered}), None)
        if ready is None:
            raise ValueError('window dependency cycle')
        ordered.append(ready)
        pending.remove(ready)
    # An explicitly recorded source stream order cannot be reversed by the DAG.
    stream_seen = {}
    for step in steps:
        stream_key = (step['rank'], step['stream'])
        previous = stream_seen.get(stream_key)
        if previous is not None and ordered.index(by_id[previous]) > ordered.index(step):
            raise ValueError('window DAG reverses declared stream launch order')
        stream_seen[stream_key] = step['occurrence_id']
    communication = manifest.get('communication_contract', {})
    if communication.get('kind') not in ('matched_native_collectives', 'provider_owned') or not communication.get('description'):
        raise ValueError('window requires explicit communication coordination contract')
    sequences = {rank: [] for rank in participants}
    unknown = []
    for step in ordered:
        binding = _descriptor(nodes[step['occurrence_id']])
        case = binding['context'].get('case')
        if isinstance(case, dict) and case.get('backend') == 'nccl':
            if case.get('group_size') != len(participants):
                raise ValueError('window native collective subgroup schedule is unsupported')
            sequences[step['rank']].append(case.get('case_id') or digest(case))
        elif binding['implementation'].get('kind') != 'native_aten_overload':
            unknown.append(step['occurrence_id'])
    if any(sequence != sequences[participants[0]] for sequence in sequences.values()):
        raise ValueError('window native collective order or peer signatures differ across ranks')
    if unknown and communication['kind'] != 'provider_owned':
        raise ValueError('custom native window callables require provider-owned communication coordination')
    if manifest['qualification'] == 'source_backed_native_window':
        _ordering_artifact(source, manifest)
    digest(manifest)  # Also rejects non-JSON or NaN context.
    return ordered


def _ordering_artifact(source, manifest):
    """Verify a retained producer ordering assertion, not raw-trace inference."""
    evidence = manifest['ordering_evidence']
    if not isinstance(evidence.get('artifact_path'), str):
        raise ValueError('source-backed window needs retained ordering artifact_path')
    import json
    raw = Path(evidence['artifact_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence['artifact_sha256']:
        raise ValueError('window ordering artifact checksum mismatch')
    artifact = json.loads(raw)
    if (artifact.get('schema'), artifact.get('version')) != ('native-window-ordering-assertion', 1):
        raise ValueError('invalid native ordering assertion schema')
    if artifact.get('source_workload_sha256') != workload_fingerprint(source) or artifact.get('steps') != manifest['steps']:
        raise ValueError('native ordering assertion source or exact step membership mismatch')
    if not artifact.get('producer') or not artifact.get('source_artifacts'):
        raise ValueError('native ordering assertion requires producer and source artifact references')
    if not isinstance(artifact['source_artifacts'], list):
        raise ValueError('ordering source artifact references must be a list')
    for reference in artifact['source_artifacts']:
        if not isinstance(reference, dict) or not reference.get('path') or not reference.get('sha256'):
            raise ValueError('ordering source artifacts require paths and SHA256')
        actual = hashlib.sha256()
        with Path(reference['path']).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                actual.update(chunk)
        if actual.hexdigest() != reference['sha256']:
            raise ValueError('native ordering source artifact checksum mismatch')
    if artifact.get('input_state_policy_sha256') != digest(manifest['input_state_policy']):
        raise ValueError('native ordering assertion input/state policy differs')
    return artifact


def bind_execution_window(source, measurement):
    """Contract a verified closed window into one conservative atomic event.

    One duration owns every participant resource. It cannot substitute a fixture
    window into training, and it never discharges prior completeness/memory gaps.
    """
    if source.manifest.get('memory', {}).get('evidence_binding'):
        raise ValueError('bind windows before memory evidence; source lifetimes will change')
    if not measurement.nodes:
        raise ValueError('window measurement is empty')
    binding = _descriptor(measurement.nodes[0])
    manifest = binding['context'].get('window_manifest', {})
    validate_window_manifest(source, manifest)
    if manifest['qualification'] != 'source_backed_native_window':
        raise ValueError('controlled native windows are measurement-only and cannot bind training costs')
    if workload_fingerprint(execution_window_workload(source, manifest)) != workload_fingerprint(measurement):
        raise ValueError('window measurement source/manifest binding mismatch')
    artifact = _ordering_artifact(source, manifest)
    boundary = artifact.get('boundary_contract', {})
    if boundary.get('entry') != 'all_live_inputs_ready' or boundary.get('exit') != 'all_streams_joined' or boundary.get('external_consumers_wait_for_window_completion') is not True:
        raise ValueError('window binding requires explicit closed entry/exit and external-consumer completion assertions')
    selected = {step['occurrence_id'] for step in manifest['steps']}
    by_id = {node.id:node for node in source.nodes}
    rendezvous = {}
    for node in source.nodes:
        if node.attrs.get('rendezvous') is not None:
            rendezvous.setdefault(node.attrs['rendezvous'], set()).add(node.id)
    if any(members & selected and not members <= selected for members in rendezvous.values()):
        raise ValueError('window binding cannot split a source communication rendezvous')
    incoming = sorted({dep for node_id in selected for dep in by_id[node_id].deps if dep not in selected})
    representative = min(selected)
    placement = source.manifest.get('rank_placement', {})
    resources = set()
    for node_id in selected:
        node = by_id[node_id]
        device = placement.get(str(node.rank), placement.get(node.rank, node.rank))
        resources.add('gpu:'+str(device))
        resources.update(node.attrs.get('resources', []))
    result = deepcopy(source)
    result.nodes = [node for node in result.nodes if node.id not in selected]
    for node in result.nodes:
        node.deps = sorted({representative if dep in selected else dep for dep in node.deps})
    result.nodes.append(Node(representative, manifest['window_id'], min(manifest['participants']),
        deps=incoming, attrs=dict(cost_key=measurement.nodes[0].attrs['cost_key'],
            native_measurement=deepcopy(binding), resources=sorted(resources),
            native_window_members=sorted(selected), native_window_participants=manifest['participants'])))
    result.nodes.sort(key=lambda node:node.id)
    for storage in result.storage:
        if storage.producer in selected:
            storage.producer = representative
        storage.consumers = sorted({representative if node in selected else node for node in storage.consumers})
    result.manifest.setdefault('unresolved', []).append(
        'Atomic native window assumes all participants ready; external interference remains unqualified.')
    memory = result.manifest.setdefault('memory', {})
    memory['storage_evidence'] = 'unverified'
    memory.setdefault('unresolved', []).append(
        'Window contraction conservatively extends internal storage lifetimes; physical memory completeness remains unverified.')
    result.manifest.setdefault('execution_window_bindings', []).append(dict(
        source_workload_sha256=workload_fingerprint(source), window_manifest_sha256=digest(manifest),
        source_occurrences=sorted(selected), representative=representative,
        evidence_class='verified producer ordering and boundary assertion; not automatic raw-trace inference'))
    # Nonconvex membership can introduce an outside -> window -> outside cycle.
    validate(result.nodes, result.storage, result.manifest)
    return result


def execution_window_workload(source, manifest):
    """Make a measurement-only workload; no operator cost substitutions occur."""
    validate_window_manifest(source, manifest)
    implementation = dict(kind='native_execution_window', window_id=manifest['window_id'],
                          provider=manifest['provider'], manifest_sha256=digest(manifest))
    context = dict(hardware_id=source.hardware['id'], cuda_graph=False,
        source_workload_sha256=workload_fingerprint(source), window_manifest=deepcopy(manifest),
        measurement_scope='execution_window', qualification=manifest['qualification'])
    key = native_cost_key(implementation, context)
    nodes = [Node(rank, manifest['window_id'], rank, attrs=dict(cost_key=key,
        native_measurement=dict(kind='executable_region', implementation=deepcopy(implementation),
                                context=deepcopy(context)))) for rank in manifest['participants']]
    return Workload(nodes, [], dict(ranks=manifest['participants'], world_size=len(nodes),
        scope='native execution window measurement', execution_window_source=dict(
            nodes=[asdict(n) for n in source.nodes], storage=[asdict(s) for s in source.storage],
            manifest=deepcopy(source.manifest), hardware=deepcopy(source.hardware)),
        provenance=deepcopy(source.manifest.get('provenance', {})), unresolved=[
            'Window timing qualifies only its exact occurrences, inputs, streams and physical topology.',
            'Window replay is not independent full-training validation or physical memory-fit evidence.']),
        deepcopy(source.hardware))


def _agree(dist, world, value, label, *, errors=False):
    values = [value]
    if world > 1:
        values = [None] * world
        dist.all_gather_object(values, value)
    if (any(values) if errors else any(peer != value for peer in values)):
        raise ValueError('window '+label+' failed on rank(s): '+str(values))


def compose_window(torch, steps, regions, *, before_window=None):
    """Compose verified local native callables; retain returned buffers to boundary.

    The provider owns internal auxiliary streams and must join those before its
    callable returns. This function joins every declared window stream back to
    the runner's timing stream. ``torch`` is injectable for CPU contract tests.
    """
    streams = {name: torch.cuda.Stream() for name in dict.fromkeys(step['stream'] for step in steps)}
    events = {step['occurrence_id']: torch.cuda.Event() for step in steps}
    live_outputs = []
    def reset():
        live_outputs.clear()
        if before_window is not None:
            before_window()
        for step in steps:
            callback = regions[step['occurrence_id']].get('before_sample')
            if callback is not None:
                callback()
    def run():
        current = torch.cuda.current_stream()
        # Timing start and all input resets precede every auxiliary stream.
        for stream in streams.values():
            stream.wait_stream(current)
        for step in steps:
            stream = streams[step['stream']]
            with torch.cuda.stream(stream):
                for dependency in step['deps']:
                    stream.wait_event(events[dependency])
                live_outputs.append(regions[step['occurrence_id']]['run']())
                events[step['occurrence_id']].record(stream)
        for stream in streams.values():
            current.wait_stream(stream)
    return dict(run=run, before_sample=reset, supports_cuda_graph=False,
                stream_semantics='current_stream')


def region_provider(workload, config):
    """Provider adapter for native_runner; existing group, no group creation."""
    import torch
    from .core import Storage
    dist = torch.distributed
    initialized = dist.is_available() and dist.is_initialized()
    world, rank = (dist.get_world_size(), dist.get_rank()) if initialized else (1, 0)
    error, source, manifest, steps, binding = None, None, None, None, None
    try:
        if config.get('cuda_graph', False):
            raise ValueError('window CUDA graph execution is unsupported')
        raw = workload.manifest['execution_window_source']
        source = Workload([Node(**n) for n in raw['nodes']], [Storage(**s) for s in raw['storage']], raw['manifest'], raw['hardware'])
        binding = _descriptor(next(n for n in workload.nodes if n.rank == rank))
        manifest = binding['context']['window_manifest']
        steps = validate_window_manifest(source, manifest)
        expected = execution_window_workload(source, manifest)
        if workload_fingerprint(workload) != workload_fingerprint(expected):
            raise ValueError('window workload differs from canonical source/manifest binding')
        if manifest['participants'] != list(range(world)):
            raise ValueError('window requires matching initialized process group')
        if config.get('physical_topology') != manifest['physical_topology']:
            raise ValueError('window physical topology differs from measurement configuration')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    _agree(dist, world, error, 'preflight', errors=True)
    _agree(dist, world, digest(manifest), 'manifest agreement')
    bundle, regions, error = None, None, None
    local = [step for step in steps if step['rank'] == rank]
    try:
        # Provider construction must be noncollective: local validation/allocation
        # failures are shared before any native communication is allowed to run.
        provider = resolve(manifest['provider'])
        provider_path = inspect.getsourcefile(provider)
        if not provider_path or hashlib.sha256(Path(provider_path).read_bytes()).hexdigest() != manifest['provider_source_sha256']:
            raise ValueError('native window provider source hash mismatch')
        bundle = provider(source, deepcopy(manifest))
        if bundle.get('preparation_is_local') is not True:
            raise ValueError('provider must attest noncollective construction/reset callbacks')
        if bundle.get('communication_contract') != manifest['communication_contract']:
            raise ValueError('provider must attest exact native communication coordination contract')
        if bundle.get('runtime_context') != manifest['runtime_context']:
            raise ValueError('native window provider runtime context mismatch')
        if bundle.get('input_state_policy') != manifest['input_state_policy']:
            raise ValueError('native provider input/state policy differs from window')
        if bundle.get('source_workload_sha256') != manifest['source_workload_sha256']:
            raise ValueError('native provider source digest differs from window')
        if not isinstance(bundle.get('provenance'), dict) or not bundle['provenance']:
            raise ValueError('native window provider must retain provenance')
        regions = {}
        nodes = {n.id:n for n in source.nodes}
        for region in bundle['regions']:
            occurrence = region['occurrence_id']
            if occurrence in regions or occurrence not in {s['occurrence_id'] for s in local}:
                raise ValueError('unknown or duplicate native window provider occurrence')
            descriptor = _descriptor(nodes[occurrence])
            if any(region.get(k) != descriptor[k] for k in ('implementation', 'context')):
                raise ValueError('native window occurrence implementation/context mismatch')
            if not callable(region.get('run')) or region.get('stream_semantics') != 'current_stream':
                raise ValueError('native window callable must complete on current stream')
            if region.get('before_sample') is not None and not callable(region['before_sample']):
                raise ValueError('native window before_sample must be callable')
            regions[occurrence] = region
        if set(regions) != {s['occurrence_id'] for s in local}:
            raise ValueError('native provider must cover exact window membership')
        if bundle.get('before_window') is not None and not callable(bundle['before_window']):
            raise ValueError('before_window must be callable')
        composed = compose_window(torch, local, regions, before_window=bundle.get('before_window'))
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    _agree(dist, world, error, 'native provider contract', errors=True)
    prepare = composed['before_sample']
    def before_sample():
        error = None
        try:
            prepare()
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
        _agree(dist, world, error, 'input preparation', errors=True)
    composed.update(before_sample=before_sample, cost_key=native_cost_key(binding['implementation'], binding['context']),
        implementation=binding['implementation'], context=binding['context'],
        round_id=manifest['window_id'], round_context=dict(participants=manifest['participants'],
            window_manifest_sha256=digest(manifest), measurement_scope='execution_window'))
    return dict(regions=[composed], provenance=dict(native_provider=bundle['provenance'],
        qualification=manifest['qualification'], physical_topology=manifest['physical_topology'],
        ordering_evidence=manifest['ordering_evidence'], measurement_scope='execution_window'))
