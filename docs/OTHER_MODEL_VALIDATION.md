# Additional model and configuration validation

**Follow-up:** The [GPT-OSS CPU study](GPT_OSS_CPU_VALIDATION.md) now resolves the
synthetic vocabulary and historical storage defaults, matches all five runs'
parameter counts, and compares independent CPU state projections with their
GPU counters. The candidate-only statements below describe this earlier audit;
full training-peak coverage remains incomplete.

## Abstract

V3's close agreement is specific to a calibrated model and recipe. Additional logs support the same fitting method on **DeepSeek-V3 16B**, but do not support transferring the 671B coefficients unchanged. After separate 16B calibration, the largest error on batches excluded from fitting is **2.20%**. Reusing the old 671B factor with 16B state and activation dimensions underpredicts by up to **34.9%**. This distinction matters: a reusable fitting method is not a universal memory predictor.

The archive also contains two useful **15-layer V4 hybrid** runs with different sequence packing and global batch sizes. The previous parser missed their legacy allocator format. Transferring reservation ratios between these configurations yields at most **0.56%** error across 16 observed ranks and two reported iterations, conditional on the destination run's measured current allocation. This is not an independent prediction of total V4 memory.

## Evidence coverage

The updated archive census rechecks all 2,613 manifest records and 2,598 unique stored files. Supporting the legacy `memory (MB) | allocated:` format raises memory candidates from 240 to **245**. Five previously missed files include the two V4 long-context logs and three additional V3-like Megatron logs. The latter report 61 layers, hidden size 7,168, and 256 experts; they are not additional V4 configurations. The original census remains preserved, with the extension recorded separately.

| Evidence group | Verified configuration or status | Use in this analysis |
|---|---|---|
| V3 671B | Previously calibrated 256 GB300, PP1/TP1/EP64 recipe | Original coefficients remain frozen |
| V3 16B | Ten logs; 15,706,484,224 parameters; 27 layers; hidden size 2,048; 64 experts; 8 GB300 GPUs | New model-specific fit and excluded-batch checks |
| V4 Pro, jobs 2565249 / 2568526 | Previously checked 61-layer, 256-GB300 recipe with matching reviewed configuration fields | Repeat-run reservation variability; not a new configuration |
| V4 hybrid, jobs 2378960 / 2379453 | 15 layers, hidden size 7,168, 384 experts, top-k 6; world size 64; TP1/PP1/CP16/EP64 | Different packing/global-batch configurations; conditional allocator check |
| V4 hybrid, job 2203614 | 5 layers, 24 experts, world size 4; TP1/PP1/CP1/EP4; sequence 4,096; microbatch 1; global batch 16 | One memory record followed by MXFP8 tensor-view failures; excluded from steady-state validation |
| GPT-OSS-20B directory group | Five logs show 24 layers, hidden size 2,880, 32 experts, top-k 4, sequence 8,192, and FP8 or FP4 compute | Further architecture candidates; no full-model accuracy claim |

V4 hybrid identity is established by the runtime `experimental_attention_variant = dsv4_hybrid` setting, not the filename. GPU model and complete software identity are not established for these additional V4 logs; world size 64 must not be relabeled “64 GB300s” or “64 nodes.” The GPT-OSS family label comes from the archive path. Its dimensions and precision are verified in configuration sections, but full identity, omitted defaults, and a matched CPU execution adapter are not yet established.

## V3 16B configuration and method

The 16B runs use TP1/PP1/CP1/EP8, FSDP shard degree 8, sequence length 4,096, one microbatch per step, FullAC, compiled transformer blocks and loss, and AdamW. Each log records 100 configured steps and training completion. Local batches are 1, 2, 8, 32, and 64, separately for MXFP8 and mixed NVFP4/MXFP8 compute.

All ten logs identify TorchTitan revision `791f2415fa73a0baf07b776bbf6fb8ee6a9e81a9`. Its historical model factory confirms 27 layers and hidden size 2,048; its logger again prints peak reserved bytes converted to GiB. Source hashes, configuration line anchors, and log hashes are preserved in the results. The archived directory happens to contain “megatron” in its path, but runtime messages identify TorchTitan; the audit follows runtime evidence.

The experiment uses the existing piecewise-affine CPU estimator without changing its mathematical form. Fit batches **1, 8, 64** and evaluate **2, 32**. These values were inspected during this investigation, so this is retrospective exclusion from fitting, not blind validation. The new profiles live with the experiment artifacts and do not replace the 671B profiles.

| Compute path | Excluded batch | GPU peak-reserved log (GiB) | Separately calibrated CPU estimate (GiB) | Error |
|---|---:|---:|---:|---:|
| MXFP8 | 2 | 34.49 | 35.15 | +1.92% |
| MXFP8 | 32 | 86.52 | 86.70 | +0.21% |
| Mixed NVFP4/MXFP8 | 2 | 34.19 | 34.94 | +2.20% |
| Mixed NVFP4/MXFP8 | 32 | 81.60 | 81.86 | +0.32% |

For a deliberately unsupported transfer check, retain the original 671B runtime factor while changing the state proxy to `15,706,484,224 / 8 × 16 bytes` and the activation proxy to `27 × batch × 4096 × 2048 × 2 bytes`. At batch 64, that predicts only **92.73 GiB versus 142.46 GiB** for MXFP8 and **87.27 GiB versus 132.79 GiB** for mixed compute. This diagnostic is explicitly outside the profile's supported domain; the normal scoped estimator rejects such a request. State assumptions, expert placement, intermediates, kernel choice, and allocator behavior do not follow a universal model-size multiplier.

The evidence therefore supports model-specific calibration, with separate evaluation data. It does not establish that CPU metadata execution independently predicts a new model within 2.2%.

## V4 configurations and results

Both additional 15-layer runs use BF16 parameters, MXFP8 compute, hidden size 7,168, 128 attention heads, 384 experts, router top-k 6, one MTP layer, selective recomputation, and the Transformer Engine CUDA-graph mode. Both configure a sequence capacity of 65,536 and microbatch size 1. Within the explicitly reviewed fields, the differences are:

| Setting | Job 2378960 | Job 2379453 |
|---|---:|---:|
| Mock sequence length | 65,534, configured min/max | 16,384, configured min/max |
| Packing scenario | One approximately 64K sequence | Four approximately 16K sequences per 64K capacity |
| Global batch | 256 | 1,024 |
| Logged training iterations | Through 15 | Through 15 |
| Allocator observations | Ranks 0–15, after iterations 1 and 2 | Ranks 0–15, after iterations 1 and 2 |

The packing descriptions follow the dataset configuration and common capacity; these logs are not a tensor-by-tensor packed-layout trace. Global batch is expressed in samples, so quadrupling the number of shorter samples must not be interpreted as quadrupling concurrent activation storage.

The CPU check freezes each rank/iteration's reservation-to-current-allocation ratio from job 2378960. It multiplies that ratio by job 2379453's **measured current allocation**, then compares with its reported peak reservation. In these records, current and peak reserved counters coincide.

| Rank 0 observation | Current allocated, both runs | Peak allocated, both runs | CPU conditional reservation | Target reservation | Error |
|---|---:|---:|---:|---:|---:|
| After iteration 1 | 136,968.70 | 161,621.95 | 170,000 | 169,580 | +0.25% |
| After iteration 2 | 141,268.27 | 174,485.53 | 201,686 | 201,406 | +0.14% |

Numbers retain the logger's printed **MB** units. Its decimal/binary divisor and peak-reset policy remain unverified; no silent conversion to MiB or GiB is made. Ratios cancel a common divisor, assuming consistent logger behavior between the two runs.

Across all **32 paired observations**, maximum absolute reservation error is **0.5554%**. All paired current-allocation values are identical at printed precision. This suggests that changing the represented sequence packing under the same configured capacity did not change the observed live footprint at those reporting points. It does not prove identical allocation trajectories or graph-pool peaks later in training. Sixteen observed ranks cover only part of the 64-rank job. Two early reporting windows are not steady-state coverage, and 32 observations are not 32 independent experiments.

The 5-layer V4 run adds a distinct failure case. It reports rank-0 current allocation 70,230.98, peak allocation 71,996.81, and reservation 76,482.00 in printed MB after iteration 1. Later records show `ValueError` failures when flattening MXFP8 tensors. It can support startup/failure analysis, but treating its memory as a completed-training peak would be misleading. No new full-V4-Pro configuration with usable comparison counters was established in this audit.

## Other architectures and the next validation boundary

The five GPT-OSS-directory logs contain usable allocator records and reach iterations 50 or 100. Their runtime configuration summaries show different CP, TP, EP, microbatch, and precision settings. An important parser constraint is that model configuration and checkpoint configuration can both contain `tensor_model_parallel_size` with different values. The audit now keeps section ownership rather than selecting the last occurrence. Omitted defaults remain unknown, and an FP4 run's `fp8_recipe = tensorwise` is not mistaken for proof of FP8 compute: its model section explicitly has `fp4 = e2m1`.

These are promising candidates for cross-architecture validation after identifying the complete model and execution recipe and adding matched CPU coverage. Applying a V3 factor directly would test an unsupported extrapolation, not validate the simulator. A strong next test freezes the prediction before exposing GPU results for a new model or a fresh configuration; fitting a new curve to that run would instead be another calibration exercise.

## Implementation and reproduction

The parser now recognizes both modern named counters and legacy allocated/reserved counters, preserves physical source lines and explicit ranks across concatenated records, and retains unverified unit labels. The existing allocator-ratio code accepts either format without interpreting printed MB as bytes. The archive scanner also recognizes the legacy format, so those logs are no longer silently omitted.

```sh
python scripts/validate_other_models.py /path/to/expanded-archive \
  --torchtitan-root /path/to/torchtitan --refresh-census
python -m unittest discover -s tests
```

The checkout's object database must contain the historical 16B revision; its active checkout need not change. [Machine-readable results](validation/other_models/results.json), [the census extension](validation/other_models/census_extension.json), and the two 16B profile files in that directory preserve the observations and assumptions. New tests check legacy counter parsing, section boundaries, rank/window/unit alignment, fitting/evaluation separation, and the recorded numerical comparisons. The suite ran 257 tests: **248 passed, 9 skipped**. No new GPU job was launched.
