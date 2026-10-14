"""Explicit-topology analytical communication scenarios.

A logical collective is priced ONCE across all participants. Phase traffic is
aggregated on shared link resources, then latency plus bottleneck serialization
is charged. A scheduler may reserve network_resources for the whole operation;
this conservative contention approximation is deliberately not NCCL simulation.
"""
from collections import defaultdict, deque
from .events import Event, nonnegative

_ALIASES = {'allreduce': 'all_reduce', 'allgather': 'all_gather',
            'reducescatter': 'reduce_scatter', 'alltoall': 'all_to_all',
            'send': 'p2p', 'recv': 'p2p', 'broadcast': 'unsupported'}


def _positive(value, label):
    value = nonnegative(value, label)
    if not value:
        raise ValueError(f'{label} must be positive')
    return value


def _topology(hardware):
    topology = hardware.get('network_topology')
    if not isinstance(topology, dict) or not isinstance(topology.get('links'), list):
        raise ValueError('topology-aware network timing requires network_topology.links')
    links, adjacency, resources = {}, defaultdict(list), {}
    for raw in topology['links']:
        if not isinstance(raw, dict):
            raise ValueError('network link must be an object')
        identifier = raw.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in links:
            raise ValueError('network link id must be unique and nonempty')
        src, dst = str(raw['src']), str(raw['dst'])
        if src == dst:
            raise ValueError('network links cannot be self loops')
        bandwidth = _positive(raw['bandwidth_bytes_s'], 'link bandwidth_bytes_s')
        latency = nonnegative(raw.get('latency_us', 0), 'link latency_us')
        resource = str(raw.get('resource', identifier))
        resource_bandwidth = _positive(raw.get('resource_bandwidth_bytes_s', bandwidth), 'resource bandwidth')
        if resource in resources and resources[resource] != resource_bandwidth:
            raise ValueError('shared network resources must specify identical resource_bandwidth_bytes_s')
        resources[resource] = resource_bandwidth
        link = {'id': identifier, 'src': src, 'dst': dst, 'bandwidth_bytes_s': bandwidth,
                'latency_us': latency, 'resource': resource, 'bidirectional': raw.get('bidirectional', False)}
        links[identifier] = link
        adjacency[src].append((dst, identifier))
        if link['bidirectional']:
            # A bidirectional link shares capacity by default (half duplex).
            # Full duplex requires two explicit directed links/resources.
            adjacency[dst].append((src, identifier))
    return topology, links, adjacency, resources


def _route(src, dst, topology, links, adjacency):
    src, dst = str(src), str(dst)
    if src == dst:
        return []
    routes = topology.get('routes', {})
    if not isinstance(routes, dict):
        raise ValueError('network routes must map "src->dst" to link ID lists')
    explicit = routes.get(f'{src}->{dst}')
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError('explicit route must be a link ID list')
        node = src
        path = []
        for identifier in explicit:
            if identifier not in links:
                raise ValueError(f'unknown route link {identifier}')
            link = links[identifier]
            if node == link['src']:
                node = link['dst']
            elif link['bidirectional'] and node == link['dst']:
                node = link['src']
            else:
                raise ValueError(f'noncontiguous route {src}->{dst}')
            path.append(link)
        if node != dst:
            raise ValueError(f'route does not reach {dst}')
        return path
    if topology.get('require_explicit_routes', False):
        raise ValueError(f'unknown route {src}->{dst}; explicit route required')
    queue, visited = deque([(src, [])]), {src}
    while queue:
        node, path = queue.popleft()
        for neighbor, identifier in sorted(adjacency[node]):
            if neighbor == dst:
                return path + [links[identifier]]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [links[identifier]]))
    raise ValueError(f'unknown route {src}->{dst}; topology is disconnected')


def estimate_network(event, hardware):
    """Return cost and global resources for one logical communication event.

    metadata: collective, participants (ordered), message_bytes, algorithm;
    p2p uses src/dst. message_bytes is the full all-reduce tensor, final
    all-gather tensor, full reduce-scatter input or each rank's all-to-all input.
    All-to-all input_splits is an NxN matrix of byte payloads (or element counts
    with explicit split_element_bytes), including self entries which stay local.
    """
    event = Event.from_dict(event) if isinstance(event, dict) else event
    if event.kind != 'collective':
        raise ValueError('network estimate requires collective event')
    metadata = event.metadata
    if ('participants' in metadata and 'participant_ranks' in metadata
            and metadata['participants'] != metadata['participant_ranks']):
        raise ValueError('participants and participant_ranks must agree in rank order')
    operation = metadata.get('collective', metadata.get('op', event.name))
    operation = _ALIASES.get(operation, operation)
    if operation not in ('all_reduce', 'all_gather', 'reduce_scatter', 'all_to_all', 'p2p', 'barrier'):
        raise ValueError(f'unsupported collective {operation!r}')
    participants = metadata.get('participants', metadata.get('participant_ranks'))
    if operation == 'p2p':
        if 'src' not in metadata or 'dst' not in metadata:
            raise ValueError('p2p requires src and dst')
        endpoints = [metadata['src'], metadata['dst']]
        if participants is not None and (len(participants) != 2 or set(participants) != set(endpoints)):
            raise ValueError('p2p participants must match src and dst')
        participants = endpoints
    if not isinstance(participants, (list, tuple)) or not participants:
        raise ValueError('collective requires explicit participants')
    if any(isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in participants):
        raise ValueError('participants must be nonnegative rank integers')
    if len(set(participants)) != len(participants):
        raise ValueError('collective participants must be distinct')
    payload = nonnegative(metadata.get('message_bytes', 0 if operation == 'barrier' else None), 'message_bytes')
    if event.flops:
        raise ValueError('collective arithmetic must be represented as separate compute work')
    n = len(participants)
    default_algorithm = 'pairwise' if operation == 'all_to_all' else 'direct' if operation == 'p2p' else 'ring'
    algorithm = metadata.get('algorithm', default_algorithm)
    if algorithm != default_algorithm:
        raise ValueError(f'{operation} supports only explicit {default_algorithm} algorithm scenario')
    topology, links, adjacency, resources = _topology(hardware)
    rank_nodes = topology.get('rank_nodes', {})
    def node(rank):
        return rank_nodes.get(str(rank), rank_nodes.get(rank, rank))
    phases = []
    if operation == 'p2p':
        transfers = [[(participants[0], participants[1], payload)]]
    elif operation in ('all_reduce', 'all_gather', 'reduce_scatter', 'barrier'):
        steps = (n - 1) * (2 if operation == 'all_reduce' else 1)
        chunk = payload / n
        transfers = [[(p, participants[(i + 1) % n], chunk) for i, p in enumerate(participants)] for _ in range(steps)]
    else:
        split = metadata.get('input_splits')
        if split is not None:
            if not isinstance(split, list) or len(split) != n or any(not isinstance(row, list) or len(row) != n for row in split):
                raise ValueError('input_splits must be an NxN matrix')
            element_bytes = _positive(metadata.get('split_element_bytes', 1), 'split_element_bytes')
            split = [[nonnegative(v, 'input_splits') * element_bytes for v in row] for row in split]
            output = metadata.get('output_splits')
            if output is not None:
                if not isinstance(output, list) or len(output) != n or any(not isinstance(row, list) or len(row) != n for row in output):
                    raise ValueError('output_splits must be an NxN matrix')
                if any(nonnegative(output[j][i], 'output_splits') * element_bytes != split[i][j] for i in range(n) for j in range(n)):
                    raise ValueError('all_to_all input/output split payload mismatch')
        transfers = [[(p, participants[(i + offset) % n], split[i][(i + offset) % n] if split else payload / n)
                      for i, p in enumerate(participants)] for offset in range(1, n)]
    all_resources, total_traffic = set(), 0.0
    for index, phase in enumerate(transfers):
        loads, link_loads, max_latency, records = defaultdict(float), defaultdict(float), 0.0, []
        for src, dst, size in phase:
            if src != dst and str(node(src)) == str(node(dst)):
                raise ValueError('distinct ranks require distinct topology endpoints; intra-node traffic must have an explicit route')
            path = _route(node(src), node(dst), topology, links, adjacency)
            max_latency = max(max_latency, sum(link['latency_us'] for link in path))
            for link in path:
                loads[link['resource']] += size
                link_loads[link['id']] += size
                all_resources.add('network:' + link['resource'])
            total_traffic += size
            records.append({'src': src, 'dst': dst, 'bytes': size, 'route': [link['id'] for link in path]})
        serialization = max([0.0] + [size / resources[resource] * 1e6 for resource, size in loads.items()]
                            + [size / links[identifier]['bandwidth_bytes_s'] * 1e6 for identifier, size in link_loads.items()])
        phases.append({'phase': index, 'transfers': records, 'resource_bytes': dict(loads),
                       'latency_us': max_latency, 'serialization_us': serialization,
                       'duration_us': max_latency + serialization})
    duration = sum(phase['duration_us'] for phase in phases)
    return {'duration_us': duration, 'source': 'network_analytical', 'bound': 'network',
            'compute_us': 0.0, 'memory_us': duration, 'traffic_bytes': total_traffic,
            'network_resources': sorted(all_resources), 'network_phases': phases,
            'network_participants': list(participants), 'network_algorithm': algorithm,
            'network_operation': operation,
            'provenance': 'explicit-topology latency + phase link-load/bandwidth analytical scenario',
            'assumptions': ['One logical collective counted once across all participants.',
                            'Ring rank order follows participants; pairwise all-to-all uses cyclic destination offsets.',
                            'Routes use explicit paths or deterministic minimum-hop routing; no adaptive routing.',
                            'Per-phase pipelined links, summed hop latencies and shared-resource serialization.',
                            'Scheduler reserves all network resources for the full operation; conservative inter-operation contention.',
                            'Bidirectional links share capacity; full duplex requires separate directed resources.',
                            'No NCCL protocol, packet overhead, GPU reduction kernel or memory-bandwidth interference model.']}
