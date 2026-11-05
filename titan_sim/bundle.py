"""Official Chakra protobuf transport plus explicit prototype sidecars.

No Torch/TorchTitan imports. Cross-rank dependencies are a namespaced extension;
ordinary Chakra data_deps remain local to each rank's trace. This consumer knows
the extension; compatibility with an arbitrary Chakra simulator is not implied.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
import et_def_pb2 as pb
from .core import Node, Storage


TYPES = {'compute': pb.COMP_NODE, 'send': pb.COMM_SEND_NODE,
         'recv': pb.COMM_RECV_NODE, 'collective': pb.COMM_COLL_NODE}


def encode(stream, msg):
    data = msg.SerializeToString()
    size = len(data)
    while size >= 128:
        stream.write(bytes([(size & 127) | 128]))
        size >>= 7
    stream.write(bytes([size]))
    stream.write(data)


def decode(stream, cls):
    size, shift = 0, 0
    while True:
        b = stream.read(1)
        if not b:
            if shift:
                raise ValueError('truncated Chakra frame')
            return None
        size |= (b[0] & 127) << shift
        if b[0] < 128:
            break
        shift += 7
        if shift > 35:
            raise ValueError('oversized Chakra frame')
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('truncated Chakra message')
    msg = cls()
    msg.ParseFromString(data)
    return msg


def write_bundle(path, nodes, storage, manifest, hardware):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ranks = defaultdict(list)
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        ranks[n.rank].append(n)
    files = []
    for rank, rank_nodes in sorted(ranks.items()):
        name = f'rank-{rank:04d}.et'
        files.append(name)
        with (path/name).open('wb') as f:
            meta = pb.GlobalMetadata(version='1.0.0')
            meta.attr.add(name='prototype.bundle_version', string_val='1')
            meta.attr.add(name='rank', int64_val=rank)
            encode(f, meta)
            for n in rank_nodes:
                p = pb.Node(id=n.id, name=n.name, type=TYPES[n.kind])
                p.data_deps.extend(d for d in n.deps if by_id[d].rank == rank)
                attrs = dict(n.attrs)
                attrs['cross_rank_deps'] = [d for d in n.deps if by_id[d].rank != rank]
                attrs['kind'] = n.kind
                p.attr.add(name='prototype.semantics', string_val=json.dumps(attrs,separators=(',',':')))
                if n.kind in ['send','recv','collective']:
                    p.attr.add(name='comm_size', int64_val=n.attrs['payload_bytes'])
                    p.attr.add(name='pg_name', string_val=n.attrs.get('group', n.attrs['rendezvous']))
                    if n.kind == 'collective':
                        p.attr.add(name='comm_type', int64_val=n.attrs.get('comm_type',pb.ALL_REDUCE))
                encode(f, p)
    (path/'storage.json').write_text(json.dumps([asdict(s) for s in storage],separators=(',',':')))
    (path/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
    manifest = dict(manifest, bundle_version=1, traces=files)
    manifest['sha256'] = {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                          for name in files + ['storage.json', 'hardware.json']}
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def read_bundle(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['bundle_version'] != 1:
        raise ValueError('unsupported bundle schema')
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != expected:
            raise ValueError('bundle checksum mismatch: ' + name)
    nodes = []
    for name in manifest['traces']:
        with (path/name).open('rb') as f:
            meta = decode(f, pb.GlobalMetadata)
            rank = next(a.int64_val for a in meta.attr if a.name=='rank')
            while (p := decode(f, pb.Node)) is not None:
                a = json.loads(next(a.string_val for a in p.attr if a.name=='prototype.semantics'))
                kind = a.pop('kind')
                if TYPES[kind] != p.type:
                    raise ValueError('Chakra type disagrees with semantics')
                deps = list(p.data_deps) + a.pop('cross_rank_deps')
                nodes.append(Node(p.id,p.name,rank,kind,deps,a))
    storage = [Storage(**s) for s in json.loads((path/'storage.json').read_text())]
    return nodes, storage, manifest, json.loads((path/'hardware.json').read_text())
