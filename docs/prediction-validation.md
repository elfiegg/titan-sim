# Compare a frozen prediction with a fresh control

Prediction accuracy requires a separate native control after the prediction is
fixed. `prediction_validation` freezes simulation output and identity hashes,
then invokes an explicit trusted control runner. The runner receives the exact
case context and its configuration, without prediction values or fitted costs.

The bundled comparison does not construct or qualify every native training
runner. Use a control adapter in the original prepared runtime, preserving the
submitted recipe and timing boundaries. GPU allocation stays with the caller.
Imported observations can be compared, but their independence is not established
by a checksum or timestamp.

## Freeze the intended scope

The workload producer must record `manifest.validation_context`, and the caller
must supply the identical object. It contains these fields:

```python
context = {
    "recipe": {"resolved_config_sha256": "retained-native-config-digest"},
    "runtime": {"source": "pinned-source", "image": "pinned-image"},
    "hardware": workload.hardware,
    "placement": workload.manifest["rank_placement"],
    "scope": "training_update",
    "useful_tokens": workload.manifest["useful_tokens"],
    "timing_metric": "completed_scope_wall_seconds",
}
```

`scope` is `training_update`, `native_window` or `native_region`. The timing metric
is `completed_scope_wall_seconds` or `completed_scope_device_seconds`; the model
and native control must cover the same boundaries. Producer recipe/runtime
identity remains an attestation tied to the complete workload hash, not an
automatic interpretation of arbitrary recipe modules.

A full training update additionally requires `manifest.prediction_scope` to be
`training_update`, every target rank, positive unique-token count, complete
capture statuses, no declared execution gaps and exact calibrated costs. It can
still have inconclusive physical memory. A partial native region or window must
not be labeled a full training update to obtain a throughput comparison.

```sh
python -m titan_sim freeze-prediction prediction-workload.json \
  --calibration costs.json --context validation-context.json \
  --output frozen-prediction.json
```

The artifact records the complete source workload fingerprint, catalog hash,
context, prediction and checksum. Retain the original workload and catalog too.
Changing the frozen output invalidates its checksum. Freeze failures identify
missing prediction coverage; running another control cannot fix those omissions.

## Execute the independent control

```sh
python -m titan_sim validate-prediction frozen-prediction.json \
  --runner my_project.control:run --config native-control.json \
  --output comparison.json
```

`my_project.control:run` is a caller-provided trusted adapter, not a bundled
module. It receives `(context, config)` and performs a fresh native run. Use one
CLI coordinator; the runner owns any authorized distributed launch and result
aggregation. It returns:

```python
{
    "context": context,
    "measurement_kind": "fresh_native_control",
    "run_id": "separate-native-control-id",
    "evidence": {"raw_measurements": "SHA256 of retained measurements"},
    "samples_seconds": measured_completed_scope_durations,
    "memory": {
        "physical-device-id": {
            "metric": "physical_device_used_bytes",
            "scope": "recipe",
            "peak_bytes": measured_peak,
        }
    },
}
```

Memory is optional. Each physical device key must match the prediction. Raw
allocator measurements can use `process_allocated_bytes` or
`process_reserved_bytes`; these remain incomparable with physical-memory bounds.
Physical-device comparison also requires the prediction's memory scope to match.
The control producer must establish device attribution and lifecycle coverage;
another process's usage or a sampled interval cannot silently stand in for the
recipe's whole lifecycle.

The runner is trusted to execute rather than return saved calibration samples.
The comparator rejects known prediction/catalog digests used as control evidence
and mismatched context, but does not authenticate arbitrary artifact contents.
`compare_control` also supports imported controls and labels their weaker
execution-order evidence explicitly.

## Interpret the result

Timing error is `(predicted - observed_median) / observed_median`. The report
preserves sample count, minimum and maximum; these are not statistical confidence
bounds. Throughput uses the same unique-token count for prediction and control.

Memory comparisons identify observed upper-bound violations. A false fit means
an unconditional fit prediction was followed by comparable observed usage above
the device budget. Missing or incompatible memory evidence leaves this unknown,
not false. Remaining prediction gaps are retained. Passing a component comparison
qualifies that component's scope; full-model accuracy remains unvalidated until
the complete full-update prediction and separate matching control are available.
