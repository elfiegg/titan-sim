# Recipe-prescreen examples

These are the earlier prototype experiments, not simulator modes or automatic
support for arbitrary TorchTitan recipes. Run them from the repository checkout;
they are not part of the installed simulator wheel. Their output workloads use the
same direct interface consumed by the generic simulator.

| Example | What actually runs |
|---|---|
| `dense` | Small CPU FakeTensor forward/backward/SGD and explicitly intercepted fake all-reduce: 102 nodes, two logical ranks |
| `hybridep` | Four source-extracted native custom-op fake functions; shape propagation, no native GPU buffer initialization/transfers |
| `kimi` | Generated analytical PP8/DP32/EP32 stage graph, not a captured native Kimi model |
| `calibration` | Real local CPU two-matmul calibration and independent chain measurement; not GPU evidence |
| `checks` | Synthetic event/storage checks |

```sh
python -m examples.recipe_prescreen.run --example checks --output-dir results/checks
python -m examples.recipe_prescreen.run --example kimi --microbatches 2 --output-dir results/kimi
python -m titan_sim validate results/kimi/kimi/workload.json
python -m titan_sim simulate results/kimi/kimi/workload.json
# Expected failure: generated Kimi coverage remains incomplete.
python -m titan_sim simulate results/kimi/kimi/workload.json --strict
```

Use `--microbatches 128` for the full generated 256-rank scenario (180,736 nodes).
Its costs are historical analytical proxies. Missing native buffers, allocator
behavior, graph pools and verified gradient/workspace lifetimes keep memory fit
inconclusive. It is not a calibrated full-model throughput prediction.

Fake capture requires compatible PyTorch with internal FakeTensor/FakeProcessGroup
APIs. The HybridEP probe additionally needs the pinned public TorchTitan source and
matching hashes in `fixtures/torchtitan-provenance.json`:

```sh
python -m examples.recipe_prescreen.run --example dense --output-dir results/dense
TORCHTITAN_PROTOTYPE_SOURCE=/path/to/pinned/torchtitan \
  python -m examples.recipe_prescreen.run --example hybridep --output-dir results/hybridep
```

`--example all` also attempts native Kimi import/build probes. Full native model
execution was not achieved. Capture inserts conservative sequential dependencies;
it does not establish asynchronous overlap or native workspaces.

## GB200 component benchmark

Inside an authorized allocation, using the intended pinned training runtime:

```sh
CALIBRATION_PHASES=compute CALIBRATION_OUTPUT=results/gb200-compute \
  torchrun --standalone --nproc-per-node=4 --module examples.recipe_prescreen.benchmark
python -m examples.recipe_prescreen.summarize \
  results/gb200-compute results/gb200-compute-summary
python -m titan_sim simulate results/gb200-compute-summary/heldout-cuda_graph/workload.json \
  --calibration results/gb200-compute-summary/rank-calibration.json --strict
```

Use the site's multi-node launcher on 32 appropriately placed GB200 GPUs and set
`CALIBRATION_PHASES=communication` for NCCL collectives/paired transfers and the
native HybridEP probe. `all` runs both groups. Use a **five-minute scheduler limit**;
these Python modules do not allocate resources or enforce it. Each rank flushes
completed samples, so completed configurations can be recovered after a later
failure. The importer requires hardware records from all four or 32 ranks and
excludes incomplete operation configurations. Native HybridEP needs the compatible
DeepEP implementation and dependencies.

The native HybridEP case uses 2,048 tokens/rank, hidden 3,584, 896 experts, top-16
forced round-robin, BF16, padding multiple **32** and capacity 33,152 rows. The fake
probe and legacy fixture use padding 128; capacity/cost claims are not interchangeable.
Placement/topology is part of calibration scope. Same-domain communication does not
calibrate cross-domain communication.

The benchmark cases were manually selected. They do not automatically match a full
compiled model, and batched expert GEMMs are explicitly proxies for grouped/SiTU
implementations. The generic `calibrate --runner` interface does not turn this
fixed-case harness into an automatic workload benchmark generator.

The original completed run measured 39 configurations on four GB200s with PyTorch
`2.14.0a0+b2c75dd062.nv26.09` and CUDA 13.4. Held-out two-GEMM prediction errors were
+3.2% eager and +4.0% graph replay. Private raw logs, device identities and cluster
paths are not published. No EP32 measurement catalog is included; collect matched
data in your environment. Allocator observations are separate from timing and are
not a full-model memory peak.

`--format chakra` is available in this example runner for the earlier optional
transport test, requiring `.[chakra]`. The default uses direct workload JSON.
