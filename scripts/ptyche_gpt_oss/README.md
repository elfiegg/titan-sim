# Reproducing the Ptyche GPT-OSS Allocation Experiment

These are the scripts used for the October 14, 2026 experiment, not a portable
cluster launcher. They assume four GB200s and 144 CPU cores per node, account
`core_dlfw_ci`, the `backfill` partition, and the cached public NVIDIA PyTorch
26.09 aarch64 image named in the Slurm scripts. Change the explicit experiment
and image paths for a new campaign. Existing results should not be overwritten.

The experiment directory is mounted as `/workspace` inside the container.
Stage the following source directories there:

| Directory | Public source revision |
|---|---|
| `titan-sim` | This repository, including `titan_sim` and `scripts` |
| `Megatron-Bridge` | `fcbb6031103d0ca845c1a54d4fee55ecfcca17b6` |
| `Megatron-LM` | `d0b3b775498a8ae6870075f1a5fc32bc260aa1b3` |
| `DeepEP` | `34152ae28f80bcc3ee38d7a12cb2ad87cfd4ea72`, with recursive submodules initialized |

Copy this directory's shell/Python/Slurm files to the experiment root.
`bootstrap.sbatch` creates a private environment over the container's packages.
`build-deepep.sbatch` installs the required Python overlay, builds the pinned
extension for SM100, checks imports and constructs both model configurations.
It preserves the installed GPU runtime and records the package versions.
The native extension adds optional metadata logging in HybridEP's external
allocator; it does not modify its allocation sizes or dispatcher algorithms.

Submit the setup jobs sequentially and require successful completion before
training. Use Slurm `afterok` dependencies if submitting them together.

```sh
titan_setup_job=$(sbatch --parsable bootstrap.sbatch)
titan_build_job=$(sbatch --parsable --dependency=afterok:$titan_setup_job build-deepep.sbatch)
sbatch --dependency=afterok:$titan_build_job profile.sbatch
```

The last job defaults to two nodes/eight GPUs and runs six steps for each of
MXFP8-control, MXFP8-traced, NVFP4-control and NVFP4-traced. A one-node case uses:

```sh
sbatch --nodes=1 --time=00:10:00 profile.sbatch
```

World size and global batch are derived from node count. TP=1, PP=1, CP=2,
EP=4, sequence length 8192 and microbatch 1 stay fixed. The program checks
model identity, parameter count, BF16 parameter storage and distributed Adam's
storage policy. Detailed snapshots default to ranks 0 and 1. Change `RANKS` in
`profile-node.sh` for targeted captures of other ranks. Every rank records phase
counters. A failed precision/mode is recorded and the launcher attempts the
other cases before returning a failure status.

The four-GPU campaign was job `3047840`; the eight-GPU campaign was `3047728`.
Both completed every case. Their raw artifacts are under:

```text
/lustre/fsw/core_dlfw_ci/elfieg/gptoss-memory-20261014/results/profile-<job-id>/
```

An earlier job stopped before forward on missing mock-dataset cache files.
The final launcher disables the recipe's dataset cache fast paths for mock
inputs. It does not disable HybridEP, quantized kernels or CUDA graphs.

Export a bounded subset only after all cases have completed:

```sh
python scripts/export_allocation_evidence.py RUN_DIRECTORY --output evidence.tar.gz
```

The exporter keeps selected rank-0 snapshots, all-rank counters and native
HybridEP allocation metadata. It marks omitted snapshots in the exported phase
records. Analyze each traced case with the dependency-free CPU analyzer:

```sh
python scripts/analyze_allocation_snapshots.py CASE_DIRECTORY --ranks 0 --output attribution.json
```

Only load this experiment's trusted snapshots: pickle can execute code. Raw
snapshots contain allocator metadata and stacks, not tensor values, and are
excluded from git. Keep the original captures for deeper transient-allocation
analysis; the exported subset is not a continuous trace.

The comparison script expects attribution JSON files named
`fp8_mx-4gpu-attribution.json`, `nvfp4-4gpu-attribution.json`,
`fp8_mx-8gpu-attribution.json`, and `nvfp4-8gpu-attribution.json` in its output
directory. It verifies completion and configuration, recomputes CPU state,
compares the frozen four-GPU component calibration with eight-GPU observations,
and creates a phase plot. Plot generation requires Matplotlib.

```sh
python scripts/compare_gpt_oss_allocation_runs.py \
  --four-gpu FOUR_GPU_EXPORTED_ROOT --eight-gpu EIGHT_GPU_EXPORTED_ROOT \
  --output docs/validation/gpt_oss_probe
```
