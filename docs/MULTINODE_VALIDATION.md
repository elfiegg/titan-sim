# Running the selected validation

`scripts/validate_full.sh` runs one `torchrun` agent per **already allocated**
node. It does not submit a job. All nodes must see the same mounted workspace,
source checkout, prepared Python environment and output directory. Invoke it
once per node with a distinct `NODE_RANK`; the launcher must terminate the whole
allocation if an agent fails. A scheduler dependency on a successful debug job
is appropriate, but this helper does not establish that dependency itself.

The selected full run is DeepSeek-V3 671B, 256 GB200 GPUs, PP=1, EP=64, TP=1,
automatic FSDP and upstream `FullAC`. The shared recipe retains the upstream
sequence length, token budget, local compilation and model code unless explicit
sequence/token overrides are supplied. Ten steps is the minimum, not evidence
that compilation or training will finish within a particular allocation limit.

## Prepare and pass the reduced checks

Use TorchTitan commit `607852350b41b4b280d8b8de6340f4847ed71c5e` and the selected
NVIDIA 26.09 runtime. `scripts/validate_container.sh` prepares the constrained
Python environment and runs container unit tests, the upstream FullAC effect
test, CPU metadata probes, and ten-step single-GPU and EP4 debug experiments.
CPU probe failures are saved separately; a successful GPU debug run does not
establish successful CPU projection.

Inside the selected container on an allocated four-GPU node:

```bash
export TITAN_WORKSPACE=/mounted/experiment
export TITAN_VALIDATION_OUTPUT="$TITAN_WORKSPACE/results/debug-run-001"
bash "$TITAN_WORKSPACE/titan-sim/scripts/validate_container.sh"
```

The full helper can also repeat the EP4 stage with its own per-rank recording:

```bash
export TITAN_PYTHON="$TITAN_WORKSPACE/venv-6078523/bin/python"
export TITAN_VALIDATION_OUTPUT="$TITAN_WORKSPACE/results/debug-repeat-001"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29571 NODE_RANK=0
export NNODES=1 NPROC_PER_NODE=4 TITAN_FLAVOR=debugmodel TITAN_EP_DEGREE=4
export TITAN_SEQ_LEN=128 TITAN_TOKENS=1024 TITAN_STEPS=10
bash "$TITAN_WORKSPACE/titan-sim/scripts/validate_full.sh"
```

## Launch the full run

The allocation launcher must start this command on every node, inside the same
prepared image. Set `MASTER_ADDR` to a reachable address of node zero,
`MASTER_PORT` to the same available port, and `NODE_RANK` to 0 through 63. These
are static rendezvous parameters; do not use `--standalone` for this run.

```bash
export TITAN_WORKSPACE=/mounted/experiment
export TITAN_PYTHON="$TITAN_WORKSPACE/venv-6078523/bin/python"
export TITAN_VALIDATION_OUTPUT="$TITAN_WORKSPACE/results/full-671b-run-001"
# Optional provenance label for the cached image actually used by the launcher:
export TITAN_CONTAINER_IMAGE=/mounted/images/pytorch-26.09.sqsh
export NNODES=64 NPROC_PER_NODE=4 TITAN_FLAVOR=671B TITAN_EP_DEGREE=64
export TITAN_STEPS=10 TITAN_OBSERVE_RANKS=all
export TITAN_HISTORY_RANKS=0 TITAN_HISTORY_STEP=3 TITAN_HISTORY_ENTRIES=250000
export TITAN_DUMP_HISTORY=0
unset TITAN_SEQ_LEN TITAN_TOKENS
# MASTER_ADDR, MASTER_PORT and NODE_RANK are supplied by your allocation launcher.
bash "$TITAN_WORKSPACE/titan-sim/scripts/validate_full.sh"
```

Use a new output directory for each run. The helper rejects an existing node's
stage file rather than overwriting its evidence. `TITAN_CHECKOUT` and
`SIM_CHECKOUT` may override the default workspace subdirectories. No dependency
installation is performed during the full launch.

Every node checks the exact TorchTitan revision, unchanged tracked model sources,
PyTorch `2.14.0a0+b2c75dd062.nv26.09`, CUDA 13.4, NCCL 2.31.2, visible GPU count and
GB200 device identity before rendezvous. It records the expected image digest
from `profiles/versions.json` separately from the supplied image label. **A
cached squashfs label and matching runtime versions do not prove its original
OCI digest.** The manifest explicitly records `image_digest_verified=false`;
the helper does not claim byte-for-byte image verification.

## Records and bounded allocator history

- `nodes/<node-rank>/environment.json`: runtime versions, device capacities,
  revision, image provenance and selected observation/history ranks.
- `nodes/<node-rank>/stage.txt`, `exit-code.txt`, `training.log`: agent status.
- `gpu/resolved-config.json`: actual shared recipe configuration.
- `gpu/observed-rank<rank>.json`: independent phase/module samples, per-step
  allocator peaks, reserved memory and run status.
- `gpu/allocator-events-rank<rank>-step<step>.json`: bounded allocator history
  only for explicitly selected history ranks.

All ranks are observed by default so rank imbalance is visible. To reduce hook
and output overhead, set `TITAN_OBSERVE_RANKS` to a comma-separated list, for
example `0,63,64,127,128,191,192,255`; this no longer establishes the worst rank
among all 256. `TITAN_HISTORY_RANKS` must be a subset of observed ranks. Set it
to `none` to disable history. History defaults to rank zero at step three;
`TITAN_HISTORY_STEP=1` instead records the first optimizer step. Optional full
pickle snapshots require `TITAN_DUMP_HISTORY=1` and remain limited to the
selected history ranks, including failure snapshots. The internal worker
passes history step zero to other ranks; it does not secretly record history
on every observed rank.

## Compare independently generated results

A successful GPU run is still only an observation. The CPU projection must
independently complete with matching revision, PyTorch version, world size,
rank, dimensions and policy/routing fingerprint. Blocked or partial metadata
captures cannot be used for accuracy claims. Do not alter metadata to make the
comparison gate pass. In particular, an assumed routing scenario is not the
observed natural routing policy, and a post-initialization projection does not
validate initialization timing or startup memory.

The comparison tool accepts exact named phase identities or an explicit
correspondence. Current capture names and GPU boundary names generally differ,
so inspect the artifacts and supply actual event/sample identifiers:

```json
[
  {
    "label": "Verified corresponding forward/backward boundary",
    "mode": "boundary",
    "prediction": {"event_id": "REPLACE_WITH_ACTUAL_EVENT_ID"},
    "observation": {"sample_index": 0}
  }
]
```

The sample index above is a placeholder; replace it with the verified matching
row. Never align CPU event numbers to GPU wall times. Then run:

```bash
python -m titan_sim.compare projection/simulation.json \
  results/gpu/observed-rank0.json --correspondence correspondence.json \
  --output comparison-rank0 --plot
```

This writes JSON/CSV and, with optional matplotlib installed, a standalone SVG.
It reports logical tensor storage and observed allocator allocated memory as
**different metrics**. Their signed difference is not prediction error.
`sampled_peak` correspondence compares selected sample maxima; the tool does
not relabel the GPU's cumulative peak counter as a phase peak or interpolate
unrelated coordinates. Hidden workspaces, allocator reserve, asynchronous
reclamation, sampling gaps, instrumentation overhead and unobserved ranks
remain explicit limitations. Completion and independent matching evidence are
required before reporting a full-model result.
