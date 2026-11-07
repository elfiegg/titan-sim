"""Logical backing-storage liveness from native FX plans and their schedule.

No model constants or activation-checkpointing policy are implemented here. The
native runtime owns graph transformations and the calling convention. Fake FX
storage is evidence for logical aliasing, never a physical allocator measurement.
"""
from __future__ import annotations

from .core import Storage
from .memory import COVERAGE_CATEGORIES


def _leaves(value):
    """Accept a tensor leaf, a list of leaves, or a serialized value tree."""
    if isinstance(value, dict):
        if 'storage_bytes' in value or 'storage_id' in value:
            yield value
        else:
            for child in value.values():
                yield from _leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _leaves(child)


def _state_slots(values):
    """Restore native flat-value positions from extractor tensor-leaf paths."""
    if not isinstance(values, list):
        return []
    if values and all(isinstance(v, dict) and v.get('path') and
                      isinstance(v['path'][0], int) for v in values):
        slots = [[] for _ in range(max(v['path'][0] for v in values) + 1)]
        for value in values:
            slots[value['path'][0]].append(value)
        return slots
    return values


def derive_plan_memory(plan, invocations, nodes):
    """Return storage, memory contract and gaps for expanded graph invocations.

    An invocation supplies id, graph_id, rank, stage_index, microbatch_index,
    node_ids (FX name -> workload node ID), and optionally completion_node_ids.
    Completion overrides model asynchronous device completion instead of host
    dispatch. Explicit input_bindings map placeholder names to earlier values:
    {"invocation": id, "node": fx_name}; tensor aliases are still checked locally.
    Graphless RS/RESHARD invocations release runtime ownership, never device uses.
    """
    by_node = {node.id: node for node in nodes}
    records = {}
    gaps = list(dict.fromkeys(plan.get('unresolved', [])))
    graph_values = {}
    stage_state = {}
    unsharded = {}
    saved = {}
    saved_weight = {}
    gradients = {}
    reduced = {}
    held_outputs = []
    state_identities = {}
    constant_identities = {}
    cpu_storage = set()

    def gap(message):
        if message not in gaps:
            gaps.append(message)

    def touch(storage_ids, event):
        if event is None:
            return
        if event not in by_node:
            raise ValueError('unknown memory completion event: ' + str(event))
        for sid in storage_ids:
            if sid in records and event not in records[sid].consumers:
                records[sid].consumers.append(event)

    def allocate(leaf, sid, rank, producer, *, persistent=False, category='tensor'):
        device = leaf.get('device')
        if device is not None and str(device).split(':')[0] == 'cpu':
            cpu_storage.add(sid)
            return sid
        if device is None:
            gap('native graph tensor device metadata missing; device residency is unverified')
        size = leaf.get('storage_bytes')
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            gap('native graph has unknown or symbolic backing-storage size')
            return None
        if leaf.get('storage_id') is None:
            gap('native graph tensor lacks backing-storage identity; aliases are incomplete')
        if sid in records:
            record = records[sid]
            if record.size != size or record.rank != rank:
                gap('native alias metadata has incompatible storage sizes or ranks: ' + sid)
                return None
            record.persistent |= persistent
        else:
            records[sid] = Storage(sid, rank, size, producer, [], persistent, category)
        # A produced, otherwise unused value dies at producer completion.
        touch([sid], producer)
        return sid

    def stage_rank(stage_index, state):
        rank = plan.get('schedule', {}).get('stage_to_rank', {}).get(str(stage_index))
        if rank is None:
            rank = state.get('rank')
        if rank is None:
            rank = next((i['rank'] for i in invocations
                         if str(i.get('stage_index')) == str(stage_index)), None)
        return rank

    for stage_index, stage in plan.get('stages', {}).items():
        rank = stage_rank(stage_index, stage)
        groups = {}
        for group, values in stage.get('state', {}).items():
            groups[group] = []
            # State groups are flat native graph values; None keeps its slot.
            for index, value in enumerate(_state_slots(values)):
                ids = []
                for leaf_index, leaf in enumerate(_leaves(value)):
                    if rank is None:
                        gap('native persistent state has no rank placement')
                        continue
                    identity = leaf.get('storage_id')
                    key = (rank, identity) if identity is not None else (rank, stage_index, group, index, leaf_index)
                    sid = state_identities.setdefault(key, f'plan:state:{rank}:{stage_index}:{group}:{index}:{leaf_index}')
                    category = 'parameter' if group == 'sharded_param_values' else ('buffer' if group == 'buffer_values' else 'optimizer_state' if 'optimizer' in group else 'gradient')
                    sid = allocate(leaf, sid, rank, None, persistent=True, category=category)
                    if sid is not None:
                        ids.append(sid)
                groups[group].append(ids)
        stage_state[str(stage_index)] = groups
        if not groups.get('sharded_param_values'):
            gap('native persistent parameter state is not fully bound for stage ' + str(stage_index))

    for inv in invocations:
        action_id = str(inv['id'])
        rank = inv['rank']
        stage_index = str(inv.get('stage_index'))
        action_type = str(inv.get('action_type', inv.get('computation_type', '')))
        stage_key = (rank, stage_index)
        mb_key = (*stage_key, inv.get('microbatch_index'))
        if action_type in ('RS', 'RESHARD'):
            # Host ownership and outstanding device uses are separate. Union
            # this release marker with every already registered completion.
            for ids in unsharded.get(stage_key, []):
                for event in inv.get('action_node_ids', []):
                    touch(ids, event)
            unsharded.pop(stage_key, None)
        graph_id = inv.get('graph_id')
        if graph_id is None:
            continue
        graph = plan.get('graphs', {}).get(graph_id)
        if graph is None:
            gap('memory invocation references missing native graph: ' + str(graph_id))
            continue
        role = graph.get('role', '')
        stage = plan.get('stages', {}).get(stage_index, {})
        meta = stage.get('metadata', {})
        state = stage_state.get(stage_index, {})
        graph_nodes = graph.get('nodes', [])
        placeholders = [n for n in graph_nodes if n.get('op') == 'placeholder']
        names = [n['name'] for n in placeholders]
        bindings = {}
        params = state.get('sharded_param_values', [])
        buffers = state.get('buffer_values', [])
        outputs = []

        def bind_names(input_names, values):
            for name, value in zip(input_names, values):
                bindings[name] = value
            if len(input_names) != len(values):
                gap('native calling-convention value count mismatch: ' + action_id)

        def bind_forward(call_meta):
            num_params = call_meta.get('num_fw_param_inputs', call_meta.get('num_param_inputs', 0))
            input_names = call_meta.get('fwd_input_names', call_meta.get('input_names', names))
            param_values = unsharded.get(stage_key, params)
            if num_params:
                bind_names(input_names[:num_params], param_values[:num_params])
            flat_indices = call_meta.get('fwd_flat_input_indices', call_meta.get('flat_input_indices', []))
            num_sharded = meta.get('num_sharded_param_values', len(params))
            for name, index in zip(input_names[num_params:], flat_indices):
                if index < num_sharded:
                    if index < len(params):
                        bindings[name] = params[index]
                elif index - num_sharded < len(buffers):
                    bindings[name] = buffers[index - num_sharded]

        if role == 'unshard':
            indices = meta.get('unshard_flat_param_indices', [])
            bind_names(names, [params[i] if 0 <= i < len(params) else [] for i in indices])
        elif role == 'fw':
            bind_forward(meta)
        elif role in ('full_bw', 'bw_di'):
            partition = meta.get('partition', {})
            saved_names = partition.get('saved_for_backward_names', [])
            saved_values = saved.get(mb_key, [])
            if saved_names:
                bind_names(saved_names, saved_values)
            elif meta.get('num_saved_for_backward', 0):
                gap('native backward saved-value bindings are missing: ' + action_id)
        elif role == 'bw_dw':
            previous = saved_weight.get(mb_key, [])
            bind_names(names[:len(previous)], previous)
            if not previous:
                gap('native weight-backward saved values are missing: ' + action_id)
        elif role == 'reduce_grad':
            grad_names = meta.get('bw_no_fsdp_output_names', [])[:meta.get('num_param_grad_values', 0)]
            by_name = dict(zip(grad_names, gradients.get(stage_key, [])))
            for name in meta.get('reduce_grad_input_names', []):
                if name in by_name:
                    bindings[name] = by_name[name]
                else:
                    gap('native gradient-reduction input is unbound: ' + action_id + ':' + name)
        elif role in meta.get('call_specs', {}):
            call_spec = meta['call_specs'][role]
            bind_forward(call_spec)
            indices = call_spec.get('grad_accumulator_input_indices', [])
            current = gradients.get(stage_key, [])
            if indices:
                bind_names(names[-len(indices):],
                           [current[i] if i < len(current) else [] for i in indices])
        elif role == 'joint':
            # extract_module_state and GraphRuntime._populate_stage_states
            # both flatten named parameters (including aliases), then buffers.
            traced = meta.get('traced', {})
            static_count = traced.get('num_static_inputs')
            graph_state = traced.get('graph_state', {})
            if (static_count == len(params) + len(buffers) and
                    not graph_state.get('mappings') and
                    not traced.get('num_optimizer_state_inputs', 0)):
                bind_names(names[:static_count], params + buffers)
            else:
                gap('native joint traced static-state bindings are incomplete: ' + action_id)
        else:
            gap('native graph calling convention is not fully bound for role: ' + str(role))

        # Native accumulation inputs are an appended prefix-index selection.
        accumulator_prefix = 'full_bw' if role == 'full_bw' else 'bw_dw' if role == 'bw_dw' else None
        if accumulator_prefix:
            indices = meta.get(accumulator_prefix + '_grad_accumulator_indices', [])
            current = gradients.get(stage_key)
            initial = state.get(accumulator_prefix + '_initial_grad_accumulators', [])
            values = ([current[i] if i < len(current) else [] for i in indices]
                      if current is not None else initial)
            if indices:
                bind_names(names[-len(indices):], values)

        for name, source in inv.get('input_bindings', {}).items():
            source_key = (str(source['invocation']), source['node'])
            if source_key not in graph_values:
                raise ValueError('input binding references an unavailable graph value: ' + str(source_key))
            bindings[name] = graph_values[source_key]

        local_aliases = {}
        values = {}
        for node in graph_nodes:
            name = node['name']
            issue = inv.get('node_ids', {}).get(name)
            completion = inv.get('completion_node_ids', {}).get(name, issue)
            if issue is not None and issue not in by_node:
                raise ValueError('unknown memory issue event: ' + str(issue))
            if completion is None and node.get('op') not in ('placeholder', 'output'):
                gap('native graph node lacks a completion event: ' + str(graph_id) + ':' + name)
            for dependency in node.get('inputs', []):
                touch(values.get(dependency, []), completion)
            ids = []
            leaves = list(_leaves(node.get('outputs', [])))
            bound = bindings.get(name)
            if bound is not None and len(bound) != len(leaves):
                gap('native input tensor-leaf binding mismatch: ' + action_id + ':' + name)
            for index, leaf in enumerate(leaves):
                identity = leaf.get('storage_id')
                alias_key = identity if identity is not None else (name, index)
                if bound is not None and index < len(bound):
                    sid = bound[index]
                    old = local_aliases.get(alias_key)
                    if old is not None and old != sid:
                        gap('native trace aliases disagree with runtime input bindings: ' + action_id)
                    local_aliases[alias_key] = sid
                    if sid in records and leaf.get('storage_bytes') != records[sid].size:
                        gap('native bound input storage extent differs from state: ' + action_id + ':' + name)
                elif alias_key in local_aliases:
                    sid = local_aliases[alias_key]
                else:
                    sid = f'plan:{rank}:{action_id}:{graph_id}:{name}:{index}'
                    if node.get('op') == 'get_attr':
                        key = (rank, graph_id, alias_key)
                        sid = constant_identities.setdefault(key, sid)
                    producer = None if node.get('op') in ('placeholder', 'get_attr') else issue
                    sid = allocate(leaf, sid, rank, producer,
                                   persistent=node.get('op') == 'get_attr')
                    if sid is not None:
                        local_aliases[alias_key] = sid
                    if node.get('op') == 'placeholder' and bound is None:
                        gap('native graph input ownership is unbound: ' + str(graph_id) + ':' + name)
                if sid is not None:
                    ids.append(sid)
                    touch([sid], completion)
            values[name] = ids
            graph_values[(action_id, name)] = ids
        for reference in graph.get('outputs', []):
            # Constants preserve output slots. A repeated node preserves aliases.
            name = reference.get('node') if isinstance(reference, dict) else reference
            outputs.append(values.get(name, []) if isinstance(name, str) else [])
        if role == 'unshard':
            unsharded[stage_key] = outputs
        elif role == 'fw':
            start = meta.get('num_user_outputs', 0)
            count = meta.get('num_saved_for_backward', 0)
            saved[mb_key] = outputs[start:start + count]
            held_outputs.extend(outputs[:start])
        elif role == 'bw_di':
            count = meta.get('num_input_grad_values', 0)
            saved_weight[mb_key] = outputs[count:]
            held_outputs.extend(outputs[:count])
            saved.pop(mb_key, None)
        elif role in ('full_bw', 'bw_dw'):
            count = meta.get('num_param_grad_values', len(outputs))
            gradients[stage_key] = outputs[:count]
            held_outputs.extend(outputs[count:])
            saved.pop(mb_key, None)
            saved_weight.pop(mb_key, None)
        elif role == 'reduce_grad':
            reduced[stage_key] = outputs
        elif role in meta.get('call_specs', {}):
            count = meta.get('num_param_grad_values', 0)
            gradients[stage_key] = outputs[1:1 + count]
            held_outputs.extend(outputs[:1])
            extra = meta.get('num_unsharded_param_values', 0)
            if extra and len(outputs) == 1 + count + extra:
                unsharded[stage_key] = outputs[1 + count:]
        elif role == 'joint':
            # Plain native joint graph convention is (loss, *parameter_grads).
            gradients[stage_key] = outputs[1:]
            held_outputs.extend(outputs[:1])
        else:
            held_outputs.extend(outputs)

    # Runtime-owned caches which survive the supplied schedule retain ownership.
    # Cached forward user outputs have upstream transport/loss-list ownership;
    # without release evidence retaining them to step end is an explicit bound.
    for values_list in [*unsharded.values(), *saved.values(), *saved_weight.values(),
                        *gradients.values(), *reduced.values(), held_outputs]:
        for ids in values_list:
            for sid in ids:
                if sid in records:
                    records[sid].persistent = True
    if saved or saved_weight:
        gap('native schedule leaves saved backward values live at its end')
    if held_outputs:
        gap('native graph-output transport/cache release is conservatively retained to schedule end')
    gap('native FX fake storage describes logical aliases, not compiled allocation/reuse or physical storage lifetimes')
    gap('native optimizer state and step allocations require complete native state/optimizer graph evidence')
    gap('native asynchronous workspace, external allocations, communication buffers, graph pools, allocator slack, initialization and checkpoint-loading coverage are unverified')
    memory = {
        'scope': 'schedule',
        'storage_evidence': 'unverified',
        'coverage': dict.fromkeys(COVERAGE_CATEGORIES, False),
        'unresolved': list(gaps),
        'allocations': [],
        'lifecycle_peaks': [],
        'logical_storage_source': 'native_fx_backing_storage_and_runtime_call_bindings',
        'host_storage_records_excluded': len(cpu_storage),
    }
    return {'storage': list(records.values()), 'memory': memory, 'unresolved': gaps}


def lower_graph_memory(workload, plan):
    """Attach logical memory to a workload produced by native plan lowering."""
    invocations = {}
    for node in workload.nodes:
        attrs = node.attrs
        action_id = attrs.get('plan_action_id')
        if action_id is None:
            continue
        graph_id = attrs.get('plan_graph_id')
        key = (str(action_id), graph_id)
        inv = invocations.setdefault(key, {
            'id': str(action_id), 'graph_id': graph_id, 'rank': node.rank,
            'stage_index': attrs.get('stage_index'),
            'microbatch_index': attrs.get('microbatch_index'),
            'action_type': attrs.get('computation_type', ''), 'node_ids': {},
            'completion_node_ids': {}, 'action_node_ids': [],
        })
        inv['action_node_ids'].append(node.id)
        if attrs.get('plan_graph_node') is not None:
            name = attrs['plan_graph_node']
            inv['node_ids'][name] = node.id
            inv['completion_node_ids'][name] = attrs.get('completion_node_id', node.id)
    result = derive_plan_memory(plan, list(invocations.values()), workload.nodes)
    if workload.storage:
        raise ValueError('native plan memory requires a workload without existing storage records')
    workload.storage = result['storage']
    workload.manifest['memory'] = result['memory']
    existing = workload.manifest.setdefault('unresolved', [])
    for gap in result['unresolved']:
        if gap not in existing:
            existing.append(gap)
    return workload
