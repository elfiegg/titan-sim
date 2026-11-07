"""Lower an extracted native host plan without importing the training framework.

The input is the native_plan.v1 data contract. No schedule is synthesized from
microbatch counts or timestamps. Bindings supply costs and communication facts,
never a replacement schedule. See docs/plan-scheduling.md for modeled semantics.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json

from .core import Node, validate
from .hardware import validate_hardware
from .native_plan import validate_plan
from .prepare import _validate_rendezvous, native_cost_key
from .workload import Workload


_ROLES = {'F': 'fw', 'B': 'full_bw', 'I': 'bw_di', 'W': 'bw_dw',
          'UNSHARD': 'unshard', 'REDUCE_GRAD': 'reduce_grad'}
_JOINT = {'FORWARD_BACKWARD', 'FORWARD_BACKWARD_NOGRADACCUM',
          'FORWARD_BACKWARD_FIRST_WITH_UNSHARD', 'FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD',
          'FULL_FORWARD_BACKWARD'}
_NATIVE_BACKWARD = {'BACKWARD', 'BACKWARD_WITH_REDUCE_GRAD'}
_ROLES.update({t: t for t in _JOINT})
_ROLES.update({t: 'full_bw' for t in _NATIVE_BACKWARD})
_ROLES['BACKWARD_WEIGHT_WITH_REDUCE_GRAD'] = 'bw_dw'
_COMPUTE = {'F', 'B', 'I', 'W'} | _JOINT | _NATIVE_BACKWARD | {'BACKWARD_WEIGHT_WITH_REDUCE_GRAD'}
_P2P = {'SEND_F', 'RECV_F', 'SEND_B', 'RECV_B'}
_WAITS = {'WAIT_SEND_F': 'SEND_F', 'WAIT_SEND_B': 'SEND_B',
          'WAIT_REDUCE_GRAD': 'REDUCE_GRAD'}
_SUPPORTED = set(_ROLES) | _P2P | set(_WAITS) | {'RESHARD'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + ' must be an integer >= ' + str(minimum))
    return value


def _resources(attrs, default):
    value = attrs.setdefault('resources', default)
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError('resources must be a list of nonempty identities')


def _binding(attrs):
    allowed = {'flops', 'bytes', 'legacy_seconds', 'resources', 'cost_key',
               'implementation', 'context', 'kind', 'participants', 'rendezvous',
               'group', 'operation', 'payload_bytes', 'peer',
               'zero_cost_semantic', 'native_wait'}
    if set(attrs) - allowed:
        raise ValueError('unsupported or semantic-overriding binding fields: ' + ', '.join(sorted(set(attrs) - allowed)))
    return attrs


def lower_plan(plan, hardware, config=None):
    """Return a Workload preserving native rank action and graph dependencies.

    Optional ``node_attrs[graph_id][FX_name]`` and
    ``action_attrs[rank][action_index]`` bind analytical quantities, native cost
    contexts, resource identities and communication descriptors. The latter's
    indices count the actual source list, including None entries. Unsupported
    compound actions, incomplete ranks and missing communication facts fail.
    """
    plan, config = deepcopy(plan), deepcopy(config or {})
    validate_hardware(hardware)
    validate_plan(plan)
    if plan.get('capture_status') in ('blocked', 'incomplete'):
        raise ValueError('native capture is blocked or incomplete; new native evidence is required')
    if plan.get('schema_version') != 'titan_sim.native_plan.v1':
        raise ValueError('unsupported native plan schema')
    schedule = plan['schedule']
    if schedule.get('source_attribute') != 'pipeline_order_with_comms':
        raise ValueError('native plan needs pipeline_order_with_comms; compute-only order cannot establish transfers')
    rank_actions = {int(k): v for k, v in schedule['rank_actions'].items()}
    world = config.get('world_size', plan.get('world_size',
                       plan.get('provenance', {}).get('world_size', len(rank_actions))))
    _integer(world, 'world_size', 1)
    declared_world = plan.get('world_size', plan.get('provenance', {}).get('world_size'))
    if declared_world is not None and declared_world != world:
        raise ValueError('world_size binding contradicts native plan provenance')
    if set(rank_actions) != set(range(world)):
        raise ValueError('all-rank native plan is incomplete for world_size ' + str(world))
    owners = {int(k): v for k, v in schedule['stage_to_rank'].items()}
    if not owners or set(owners) != set(range(len(owners))):
        raise ValueError('pipeline stage indices must be complete and contiguous')
    if any(type(r) is not int or r not in rank_actions for r in owners.values()):
        raise ValueError('stage ownership references an unknown rank')
    microbatches = _integer(schedule['num_microbatches'], 'num_microbatches', 1)
    graphs, stages = plan.get('graphs', {}), plan.get('stages', {})
    nodes, records, action_ends, action_starts = [], {}, {}, {}
    graph_calls = []
    gaps = set(plan.get('unresolved', []))
    if schedule.get('rank_namespace') == 'pipeline_group':
        gaps.add('Native ranks are pipeline-group ranks; global DP/TP/CP/EP mesh placement and all-global-rank completeness are unverified.')
    gaps.add('Native host plan models device dependencies; CPU dispatch, data loading and host synchronization durations are not captured.')
    used_actions, used_nodes = set(), set()
    placement = deepcopy(config.get('rank_placement', plan.get('rank_placement', {})))
    def device(rank):
        return str(placement.get(str(rank), placement.get(rank, rank)))
    def emit(name, rank, deps=(), attrs=None, kind='compute'):
        n = Node(len(nodes), name, rank, kind, sorted(set(deps)), deepcopy(attrs or {}))
        nodes.append(n)
        return n.id
    def semantic(name, rank, deps=(), attrs=None):
        return emit(name, rank, deps, dict(attrs or {}, zero_cost_semantic=True, resources=[]))
    def bind_action(rank, index):
        attrs = config.get('action_attrs', {}).get(str(rank), {}).get(str(index), {})
        if attrs:
            used_actions.add((str(rank), str(index)))
        return _binding(deepcopy(attrs))
    def base(rank, index, action):
        return dict(plan_action_id=f'{rank}:{index}', stage_index=action['stage_index'],
                    microbatch_index=action.get('microbatch_index'),
                    computation_type=action['computation_type'])
    def cost_attrs(attrs, identity):
        # Every bound key incorporates immutable native context. An upstream
        # supplied key is part of that context, never a way to bypass it.
        source_key = attrs.pop('cost_key', None)
        implementation = attrs.pop('implementation', {'kind': 'native_fx_instruction',
                                                       'target': identity.get('target')})
        context = deepcopy(attrs.pop('context', {}))
        context['native_plan'] = identity
        context['resources'] = deepcopy(attrs.get('resources', []))
        if source_key:
            context['source_cost_key'] = source_key
        attrs.update(implementation=implementation, context=context,
                     cost_key=native_cost_key(implementation, context))
        return attrs
    def expand(rank, index, action, deps, bound):
        stage, role = action['stage_index'], _ROLES[action['computation_type']]
        stage_info = stages.get(str(stage), {})
        graph_roles = stage_info.get('graphs', {})
        graph_id = graph_roles.get(action['computation_type']) or graph_roles.get(role)
        if (graph_id is None and action['computation_type'] == 'FULL_FORWARD_BACKWARD'
                and not stage_info.get('metadata', {}).get('call_specs')):
            # Native GraphTrainerJointStageGraphs.forward_backward executes its
            # traced.gm directly; scheduled call_specs must never use this fallback.
            graph_id = graph_roles.get('joint')
        if graph_id is None:
            raise ValueError(f'missing native {role} graph for stage {stage}')
        if graph_id not in graphs:
            raise ValueError('unknown native graph: ' + str(graph_id))
        graph = graphs[graph_id]
        if graph.get('stage_index', stage) != stage:
            raise ValueError('graph stage ownership mismatch')
        if bound:
            raise ValueError('compute action bindings are unsupported; bind native graph nodes instead')
        mapping, computed = {}, []
        stream_tail = None
        for fx in graph['nodes']:
            name, op = fx['name'], fx['op']
            if name in mapping:
                raise ValueError('duplicate FX node name')
            inputs = fx.get('inputs', [])
            if any(i not in mapping for i in inputs):
                raise ValueError('FX graph input missing or not topologically ordered')
            parents = [mapping[i] for i in inputs] + list(deps)
            attrs = base(rank, index, action)
            attrs.update(plan_graph_id=graph_id, plan_graph_node=name)
            node_bound = _binding(deepcopy(config.get('node_attrs', {}).get(graph_id, {}).get(name, {})))
            if node_bound:
                used_nodes.add((graph_id, name))
            if op in ('placeholder', 'get_attr', 'output'):
                if node_bound:
                    raise ValueError('semantic graph nodes cannot receive cost bindings')
                nid = semantic(f'{graph_id}/{name}', rank, parents, attrs)
            elif op in ('call_function', 'call_method', 'call_module'):
                attrs.update(node_bound)
                kind = attrs.pop('kind', 'compute')
                if kind not in ('compute', 'collective'):
                    raise ValueError('FX graph node kind requires compute or collective')
                target = str(fx.get('target', ''))
                if attrs.get('native_wait') and not attrs.get('zero_cost_semantic'):
                    raise ValueError('native_wait requires an explicit semantic wait node')
                if kind == 'compute' and not attrs.get('native_wait') and any(x in target for x in ('c10d', '_c10d', 'all_reduce', 'all_gather', 'reduce_scatter', 'all_to_all')):
                    raise ValueError('distributed graph operation requires explicit collective/wait semantics: ' + target)
                if kind == 'collective':
                    if isinstance(attrs.get('rendezvous'), str):
                        try:
                            attrs['rendezvous'] = attrs['rendezvous'].format(microbatch=action.get('microbatch_index'), stage=stage, action=index)
                        except (KeyError, ValueError) as exc:
                            raise ValueError('unsupported collective rendezvous template') from exc
                    _collective(attrs, rank)
                if attrs.get('zero_cost_semantic'):
                    if not attrs.pop('native_wait', False) or kind != 'compute' or 'wait' not in target.lower():
                        raise ValueError('only an explicitly bound native_wait may be semantic inside an FX graph')
                    _resources(attrs, [])
                else:
                    identity = dict(graph_id=graph_id, graph_sha256=_digest(graph), node=fx,
                                    provenance=plan.get('provenance', {}),
                                    stage_index=stage, rank=rank,
                                    communication={k: attrs[k] for k in ('participants', 'group', 'operation', 'payload_bytes') if k in attrs})
                    _resources(attrs, [f'gpu:{device(rank)}'] if kind == 'compute' else [f'collective:{device(rank)}'])
                    cost_attrs(attrs, identity)
                    gaps.add('FX instruction cost identities do not establish selected compiled kernels or fusion; native implementation matching remains required.')
                if stream_tail is not None:
                    parents.append(stream_tail)
                nid = emit(f'{graph_id}/{name}', rank, parents, attrs, kind)
                if kind == 'compute':
                    stream_tail = nid
                computed.append(nid)
            else:
                raise ValueError('unsupported FX node operation: ' + str(op))
            mapping[name] = nid
        if not mapping:
            raise ValueError('empty native graph')
        completion = semantic(f'{graph_id}/complete', rank, list(mapping.values()), base(rank, index, action))
        graph_calls.append(dict(id=f'{rank}:{index}', graph_id=graph_id, rank=rank,
                                stage_index=stage, microbatch_index=action.get('microbatch_index'),
                                action_type=action['computation_type'], node_ids=mapping,
                                completion_node_id=completion))
        return completion

    # Source positions are retained, but empty slots are scheduling annotations,
    # not measured delays and not barriers across ranks.
    for rank, actions in sorted(rank_actions.items()):
        if not isinstance(actions, list):
            raise ValueError('rank actions must be lists')
        for index, action in enumerate(actions):
            if action is None:
                continue
            stage = _integer(action.get('stage_index'), 'stage_index')
            if owners.get(stage) != rank:
                raise ValueError('action violates stage ownership')
            typ, mb = action.get('computation_type'), action.get('microbatch_index')
            if action.get('sub_actions') is not None or typ not in _SUPPORTED:
                raise ValueError('unsupported native schedule action: ' + str(typ))
            if typ in _COMPUTE | _P2P | {'WAIT_SEND_F', 'WAIT_SEND_B'}:
                if type(mb) is not int or not 0 <= mb < microbatches:
                    raise ValueError('invalid action microbatch')
            elif mb is not None:
                raise ValueError('stage lifecycle action must not identify a microbatch')
            key = (stage, typ, mb)
            if typ not in ('UNSHARD', 'RESHARD', 'REDUCE_GRAD', 'WAIT_REDUCE_GRAD'):
                if key in records:
                    raise ValueError('duplicate native action: ' + str(key))
                records[key] = (rank, index)
                aliases = ('F', 'B') if typ in _JOINT else ('B',) if typ in _NATIVE_BACKWARD else ('W',) if typ == 'BACKWARD_WEIGHT_WITH_REDUCE_GRAD' else ()
                if typ in _JOINT and len(owners) != 1:
                    raise ValueError('joint forward/backward action is supported only for one native pipeline stage')
                for alias in aliases:
                    alias_key = (stage, alias, mb)
                    if alias_key in records:
                        raise ValueError('duplicate native compute action for stage/microbatch')
                    records[alias_key] = (rank, index)
    for stage in owners:
        for mb in range(microbatches):
            if (stage, 'F', mb) not in records:
                raise ValueError('missing forward for stage/microbatch')
            has_b = (stage, 'B', mb) in records
            has_i, has_w = (stage, 'I', mb) in records, (stage, 'W', mb) in records
            if has_b == (has_i or has_w) or has_i != has_w:
                raise ValueError('each stage/microbatch requires B or both I and W')

    transfers, pending_edges = defaultdict(list), []
    for rank, actions in sorted(rank_actions.items()):
        device_tail = None
        issue_tail = None
        unshard, last_stage, reshard, pending = {}, {}, {}, {}
        backward_done = defaultdict(set)
        lifecycle_stages = {a['stage_index'] for a in actions if a is not None and a['computation_type'] == 'UNSHARD'}
        for index, action in enumerate(actions):
            if action is None:
                continue
            stage, typ, mb = action['stage_index'], action['computation_type'], action.get('microbatch_index')
            attrs, bound = base(rank, index, action), bind_action(rank, index)
            # Dispatch order is represented by zero-duration issue nodes, separate
            # from asynchronous completion. Stream waits gate subsequent work.
            issue = semantic(f'issue:{rank}:{index}', rank,
                             [] if issue_tail is None else [issue_tail])
            issue_tail = issue
            action_starts[(rank, index)] = issue
            deps = [issue] + ([] if device_tail is None else [device_tail])
            if typ in _COMPUTE:
                if stage in lifecycle_stages and stage not in unshard and typ not in _JOINT:
                    raise ValueError('compute requires active native UNSHARD for its stage')
                if stage in unshard:
                    deps.append(unshard[stage])
                end = expand(rank, index, action, deps, bound)
                device_tail = end
                last_stage[stage] = end
                if typ in {'B', 'W', 'BACKWARD_WEIGHT_WITH_REDUCE_GRAD'} | _NATIVE_BACKWARD | _JOINT:
                    backward_done[stage].add(mb)
                if typ == 'F' and stage > 0:
                    source = (stage - 1, 'F', mb) if owners[stage-1] == rank else (stage, 'RECV_F', mb)
                    pending_edges.append((issue, source))
                if typ in {'B', 'I'} | _NATIVE_BACKWARD:
                    pending_edges.append((issue, (stage, 'F', mb)))
                    if stage < len(owners)-1:
                        source = ((stage+1, 'B' if (stage+1, 'B', mb) in records else 'I', mb)
                                  if owners[stage+1] == rank else (stage, 'RECV_B', mb))
                        pending_edges.append((issue, source))
                if typ in ('W', 'BACKWARD_WEIGHT_WITH_REDUCE_GRAD'):
                    pending_edges.append((issue, (stage, 'I', mb)))
            elif typ in _P2P:
                forward = typ.endswith('_F')
                send = typ.startswith('SEND')
                source_stage = stage if send else stage - 1 if forward else stage + 1
                dest_stage = stage + 1 if forward and send else stage - 1 if send else stage
                if source_stage not in owners or dest_stage not in owners or owners[source_stage] == owners[dest_stage]:
                    raise ValueError('invalid pipeline transfer boundary')
                peer = owners[dest_stage if send else source_stage]
                if 'peer' in bound and bound['peer'] != peer:
                    raise ValueError('communication binding contradicts stage peer')
                native_size = _pipeline_payload(plan, source_stage, forward, records, mb)
                if native_size is not None:
                    if 'payload_bytes' in bound and bound['payload_bytes'] != native_size:
                        raise ValueError('pipeline payload binding contradicts native graph tensor metadata')
                    bound['payload_bytes'] = native_size
                if 'payload_bytes' not in bound:
                    raise ValueError(f'missing native pipeline payload_bytes for rank {rank} action {index}; no transfer size is fabricated')
                _integer(bound['payload_bytes'], 'payload_bytes')
                if bound.get('zero_cost_semantic') or bound.get('native_wait') or 'kind' in bound:
                    raise ValueError('pipeline transfers cannot override native action semantics')
                attrs.update(bound)
                identity = dict(direction='forward' if forward else 'backward',
                                source_stage=source_stage, dest_stage=dest_stage, microbatch=mb)
                rendezvous = 'native-pp:' + _digest(identity)
                if 'rendezvous' in attrs and attrs['rendezvous'] != rendezvous:
                    raise ValueError('pipeline rendezvous is determined by stage and microbatch')
                attrs.update(peer=peer, rendezvous=rendezvous, operation='pipeline_transfer')
                _resources(attrs, [f'pp:{device(rank)}'])
                cost_attrs(attrs, dict(communication=identity, payload_bytes=attrs['payload_bytes'],
                                      rank=rank, peer=peer, provenance=plan.get('provenance', {})))
                end = emit(f'{stage}{typ}{mb}', rank, [issue], attrs, 'send' if send else 'recv')
                transfers[rendezvous].append(end)
                if send:
                    producer = 'F' if forward else 'B' if (stage, 'B', mb) in records else 'I'
                    pending_edges.append((end, (stage, producer, mb)))
                    pending[(typ, stage, mb)] = end
            elif typ == 'UNSHARD':
                if stage in unshard:
                    raise ValueError('UNSHARD without prior RESHARD')
                parents = [issue] + ([reshard[stage]] if stage in reshard else [])
                end = expand(rank, index, action, parents, bound)
                unshard[stage] = end
            elif typ == 'REDUCE_GRAD':
                if ('REDUCE_GRAD', stage, None) in pending:
                    raise ValueError('repeated reduction before wait')
                if len(backward_done[stage]) != microbatches:
                    raise ValueError('gradient reduction before all native stage backwards')
                end = expand(rank, index, action, [issue, last_stage[stage]], bound)
                pending[('REDUCE_GRAD', stage, None)] = end
                # Native finalization owns the parameter transition; it cannot be
                # inferred from a reduction graph's output sizes alone.
            elif typ in _WAITS:
                if bound:
                    raise ValueError('wait actions cannot receive cost bindings')
                key = (_WAITS[typ], stage, mb)
                if key not in pending:
                    raise ValueError('native wait lacks a previously issued matching action')
                end = semantic(f'{stage}{typ}{mb}', rank, deps + [pending.pop(key)], attrs)
                device_tail = end
                issue_tail = end
            else:  # RESHARD: native lifecycle marker, no invented transfer cost.
                if bound:
                    raise ValueError('RESHARD cost must be captured as a native graph, not fabricated')
                if stage not in unshard or stage not in last_stage:
                    raise ValueError('RESHARD without a used UNSHARD')
                end = semantic(f'{stage}RESHARD', rank, deps + [unshard.pop(stage), last_stage[stage]], attrs)
                reshard[stage] = end
                device_tail = end
            action_ends[(rank, index)] = end
        if pending:
            gaps.add('Native plan leaves asynchronous sends or reductions without explicit completion waits.')
    for node_id, source in pending_edges:
        if source not in records:
            raise ValueError('native plan missing required producer/receive: ' + str(source))
        ref = records[source]
        if nodes[node_id].kind == 'send':
            send_index = int(nodes[node_id].attrs['plan_action_id'].split(':')[1])
            if ref[0] != nodes[node_id].rank or ref[1] >= send_index:
                raise ValueError('native send appears before its producer')
        # Edges may span ranks. No wall-clock comparison is used.
        nodes[node_id].deps.append(action_ends[ref])
        nodes[node_id].deps = sorted(set(nodes[node_id].deps))
    for members in transfers.values():
        if sorted(nodes[i].kind for i in members) != ['recv', 'send']:
            raise ValueError('native plan has an unmatched pipeline transfer')
    for rank, bindings in config.get('action_attrs', {}).items():
        for index in bindings:
            if (rank, index) not in used_actions:
                raise ValueError('unused action binding: ' + rank + ':' + index)
    for graph_id, bindings in config.get('node_attrs', {}).items():
        for name in bindings:
            if (graph_id, name) not in used_nodes:
                raise ValueError('unused graph node binding: ' + graph_id + '/' + name)
    manifest = dict(ranks=sorted(rank_actions), world_size=world,
                    provenance=deepcopy(plan.get('provenance', {})),
                    unresolved=sorted(gaps),
                    useful_tokens=config.get('useful_tokens', plan.get('useful_tokens', 0)),
                    native_plan=dict(schema_version=plan['schema_version'], plan_sha256=_digest(plan),
                                     schedule_class=schedule.get('class'),
                                     action_nodes={f'{r}:{i}': n for (r, i), n in action_ends.items()},
                                     graph_invocations=graph_calls,
                                     resource_model='exclusive identities; unbound compute uses gpu:<device>, pipeline transfers pp:<device>, collectives collective:<device>'))
    if placement:
        manifest['rank_placement'] = placement
    workload = Workload(nodes, [], manifest, deepcopy(hardware))
    validate(nodes, [], manifest)
    _validate_rendezvous(nodes)
    return workload


def _collective(attrs, rank):
    participants = attrs.get('participants')
    if (not isinstance(participants, list) or not participants or
            any(type(r) is not int or r < 0 for r in participants) or
            len(set(participants)) != len(participants) or rank not in participants):
        raise ValueError('collective requires exact participants including its rank')
    for key in ('rendezvous', 'group', 'operation'):
        if not isinstance(attrs.get(key), str) or not attrs[key]:
            raise ValueError('collective requires explicit ' + key)
    _integer(attrs.get('payload_bytes'), 'collective payload_bytes')


def _pipeline_payload(plan, stage, forward, records, microbatch):
    """Transport bytes from the native SplitStageGraphs flat calling convention."""
    info = plan.get('stages', {}).get(str(stage), {})
    meta = info.get('metadata', {})
    role = 'fw' if forward else 'full_bw' if (stage, 'B', microbatch) in records else 'bw_di'
    graph = plan.get('graphs', {}).get(info.get('graphs', {}).get(role), {})
    count = meta.get('num_user_outputs' if forward else 'num_input_grad_values')
    offset = 0 if forward or role == 'bw_di' else meta.get('num_param_grad_values')
    if type(count) is not int or count <= 0 or type(offset) is not int or offset < 0:
        return None
    outputs = graph.get('outputs', [])[offset:offset+count]
    if len(outputs) != count:
        return None
    by_name = {n['name']: n for n in graph.get('nodes', [])}
    total = 0
    for name in outputs:
        leaves = by_name.get(name, {}).get('outputs', [])
        # Flat graph values must be single tensors. Opaque tuples/non-tensors
        # require the actual runtime transport metadata instead of a guess.
        if len(leaves) != 1 or type(leaves[0].get('logical_bytes')) is not int:
            return None
        total += leaves[0]['logical_bytes']
    return total
