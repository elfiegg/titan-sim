# Describe the target hardware

The target describes the physical devices available to a recipe, their placement
and the rates assumed by the timing model. A product specification and a measured
cost catalog are different inputs. Neither a rack memory total nor advertised
FLOP/s establishes achievable per-device memory or training throughput.

Start from a caller-supplied inventory in the prepared runtime. Record device
capacity in integer bytes, physical device IDs, unavailable capacity/headroom,
NVLink domains, scale-out links and process-group placement. Keep the same
hardware identity across capture, preparation and compatible calibration.

For example, the following **synthetic** target shows heterogeneous devices; its
small capacities are test values, not Blackwell or Rubin specifications:

```json
{
  "id": "synthetic-two-device-example",
  "devices": {
    "gpu0": {"capacity_bytes": 100000, "reserve_fraction": 0.1},
    "gpu1": {"capacity_bytes": 200000, "reserve_fraction": 0.15}
  },
  "provenance": "Synthetic contract example only"
}
```

Pair this with `manifest.rank_placement`, such as `{"0":"gpu0","1":"gpu1"}`.
Every rank must map to one physical device. Multiple ranks may share a device;
their simultaneous allocation lifetimes are combined. The [memory model](native-memory.md)
explains the capacity budget, lifecycle and coverage requirements. If the target
has uniform capacity, `capacity_bytes` at the top level is also supported.

## Blackwell and Vera Rubin sources

Versioned source records are included for
[GB200 NVL72](../examples/hardware/gb200-nvl72.spec.json) and
[DGX Vera Rubin NVL72](../examples/hardware/vera-rubin-nvl72.spec.json). They preserve
the scope and units of the user's supplied
[Blackwell technical brief](https://resources.nvidia.com/en-us-blackwell-architecture)
and [Vera Rubin datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-vera-rubin-nvl72-datasheet).
They are specification records, **not ready-to-use simulation targets**.

The source tables describe whole racks and distinguish dense from sparse rates.
The Rubin sheet lists FP8/FP6 and NVFP4 training rates; it does not supply a BF16
rate. Do not infer one from NVFP4 inference. Likewise, do not divide a rounded rack
memory headline by GPU count and claim an exact allocatable device capacity.
Supply actual per-device facts or explicit conditional design assumptions.

## Timing rates and topology

Measured catalogs require exact matching implementation/context keys and hardware
identity. Changing the GPU family or topology does not automatically rescale a
catalog. Re-measure the cases in the matching runtime where hardware is available.

For a specification-only scenario, workloads must carry explicit FLOP/byte or
communication quantities. The analytical cost model consumes
`effective_flops_per_second`, `effective_hbm_bytes_per_second`,
`interconnect_bytes_per_second`, `pp_bytes_per_second`,
`comm_latency_seconds` and optional `launch_seconds` as relevant. These are
effective scenario rates, not automatically the published peaks. Their precision,
dense/sparse convention, efficiency factors and topology scope belong in
`assumptions` and provenance. Missing required rates are errors, not zero work.

`titan_sim.hardware.analytical_target(specification, config)` validates supplied
capacity/rates and retains the specification and assumptions. It never populates
missing performance rates from a product name. The result is labeled
`analytical_assumptions`. Analytical scenarios cannot pass timing `--strict`.

The event model schedules declared exclusive resources and communication
rendezvous. Explicit resource IDs can serialize operations sharing a modeled link;
they do not implement continuous bandwidth sharing or runtime NCCL algorithm
selection. Default compute resources follow physical placement. Custom stream
resources allow overlap only under the producer's explicit assumptions. Record
unmodeled contention, routing imbalance and topology effects as gaps.
