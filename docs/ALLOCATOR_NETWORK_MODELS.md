# Allocator and topology scenarios

These modules consume observed event structure and explicit hardware assumptions. Neither implementation fits a supplied end-to-end GPU runtime or memory total. Both expose their assumptions and missing coverage.

## Allocator replay

`replay_allocator(runtime_result, hardware, config=None, events=None)` consumes `storage_lifetimes` from the discrete-event runtime. `reclaim_us` already includes the last asynchronous use of every storage generation, so releasing a Python reference does not make an in-flight allocation immediately reusable. Alias references share one generation; reusing a storage identifier after its final release creates another generation.

Configuration fields:

| Field | Default | Meaning |
|---|---|---|
| `policy` | `caching` | `ideal` tracks exact live bytes without cached reserve; `caching` uses segments and best-fit blocks |
| `alignment_bytes` | 512 | Round every caching request and segment to this alignment |
| `segment_bytes` | 2097152 | Minimum segment size; larger requests create larger segments |
| `bin_sizes_bytes` | `[]` | Ascending, alignment-multiple request rounding bins; requests above the largest bin use alignment |
| `split` | true | Split a larger free block; otherwise the full block becomes allocated |
| `coalesce` | true | Merge adjacent free blocks within a segment |
| `release_cached_on_oom` | true | Retry capacity after releasing completely free default-pool segments |
| `external_memory_bytes` | 0 | Fixed external usage, either scalar per rank or a rank-to-byte mapping |

Allocation metadata `graph_pool` (or `allocator_pool`) selects an isolated pool. Private pools retain their segments after reclamation and cannot be scavenged by the default pool. `external_memory: true` marks observed storage whose capacity usage belongs outside the caching allocator. Pass original events to preserve allocation metadata; direct lifetime inputs can instead contain their own `metadata` object.

Samples separate requested bytes, occupied allocator blocks, reserved segments, inactive bytes, inactive split bytes, external bytes, and total device use. The OOM record includes the failed request, rounded request, required segment, largest free block, and total free pool bytes. A successful replay reports `fits_recorded_allocations_only` and explicitly leaves `full_target_fit_proven` false; coverage counts unclassified lifetimes. Hidden allocations prevent interpreting this result as a full-model fit proof. A nonzero scalar external reservation with no recorded ranks applies to rank zero. A request can fail even when live tensors and total free space appear to fit: free bytes may occupy separate blocks or private pools. Replay stops the affected rank at its first failed request; later runtime results are counterfactual for that rank.

The policy is an approximate allocator experiment, not a reproduction of PyTorch's CUDA caching allocator, CUDA async pools, or driver placement. It does not model same-stream early reuse, stream-specific arenas, graph destruction, expandable segments, hidden workspaces, or unrecorded library/driver buffers. Allocation-before-free timestamp ties intentionally match the runtime's conservative ordering. Without a representative host submission schedule, memory peaks remain a scheduling scenario.

## Network cost

`estimate_network(event, hardware)` returns the cost of one logical collective, a phase breakdown, and globally named `network_resources`. Rank participant events must rendezvous into one logical transfer before this function is used. The distributed lowering module performs that conversion. Charging each rank's copy of a full collective would double-count traffic.

A minimal topology is:

```json
{
  "network_topology": {
    "links": [
      {"id": "0-to-1", "src": 0, "dst": 1, "bandwidth_bytes_s": 100000000000, "latency_us": 2},
      {"id": "1-to-0", "src": 1, "dst": 0, "bandwidth_bytes_s": 100000000000, "latency_us": 2}
    ]
  }
}
```

Links are directed. `bidirectional: true` shares one resource in both directions; model full duplex with separate directed resources. Several links may share a channel using `resource` and the same `resource_bandwidth_bytes_s`. `rank_nodes` optionally maps rank IDs to topology node IDs. Switch nodes may be strings. Distinct communicating ranks require distinct endpoint nodes; intra-node GPU traffic must have an explicit link route rather than being erased by mapping ranks to one host node. `routes` optionally maps `"src->dst"` to ordered link-ID lists. Otherwise the model uses deterministic minimum-hop routing. Set `require_explicit_routes: true` to prohibit inferred routes. Disconnected or invalid routes fail instead of falling back to a scalar bandwidth.

Collective metadata contains `collective`, ordered `participants`, and `message_bytes`:

| Operation | Supported algorithm | Payload definition |
|---|---|---|
| `all_reduce` | ring | Full tensor bytes; two sequences of N−1 phases |
| `all_gather` | ring | Final gathered tensor bytes; N−1 phases |
| `reduce_scatter` | ring | Full input tensor bytes; N−1 phases |
| `all_to_all` | pairwise | Per-rank input bytes, uniformly split unless split matrices are supplied |
| `p2p` | direct | Message bytes; explicit `src` and `dst` |
| `barrier` | ring | Zero payload, N−1 latency phases |

Variable all-to-all accepts NxN `input_splits` and optional transposed `output_splits`, measured in bytes. Explicit `split_element_bytes` permits element-count matrices. Self transfers remain local. Distributed participant lowering accepts per-rank byte-count vectors and validates every sender/receiver pair.

Each phase aggregates payload on every routed link and shared channel, computes the largest traffic/bandwidth serialization time, then adds the largest path latency. The scheduler can reserve all returned resources for the full operation, conservatively serializing overlapping operations on shared links while allowing disjoint routes to overlap. The model does not simulate NCCL protocol selection, adaptive routing, packets, launch overhead, GPU reduction kernels, or memory bandwidth interference. Hardware bandwidth and latency should come from independent calibration or documented assumptions; the resulting prediction is a scenario, not a measured runtime.
