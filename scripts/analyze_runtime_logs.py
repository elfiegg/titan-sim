#!/usr/bin/env python3
"""Reproduce sanitized observations and conditional budgets for five audited runs.

Requires matplotlib for static PNG/SVG figures. The private source logs are not
included. Give the extracted log-bundle root and the additional four-file log
directory explicitly. Embedded source hashes bind manual configuration evidence
to the exact reviewed inputs; a different input fails instead of reusing stale
line annotations. No source lines, endpoints or absolute paths are exported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from titan_sim.log_observations import parse_log, summarize, assess_cross_validation
from titan_sim.memory_budget import budget_scenarios

# Sanitized manually reviewed configuration evidence, bound to source hashes.
AUDIT = json.loads(r'''
{
  "runs": {
    "titan_v3_nvfp4_batch16": {
      "primary_source": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
      "configuration_evidence": [
        {
          "field": "model",
          "value": "DeepSeek-V3 671B",
          "total_parameter_elements": 671026404352,
          "dense_parameter_elements": 14456871936,
          "expert_parameter_elements_logged_as_sparse": 656569532416,
          "active_parameter_elements": 37552282624,
          "caution": "Model label and count do not establish every architectural dimension. Sparse here is the logged MoE parameter category, not evidence of structured sparse GEMM throughput.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 23
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 26
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 27
            }
          ]
        },
        {
          "field": "hardware",
          "gpu_name": "NVIDIA GB300",
          "reported_capacity_gib": 276.62,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 25
            }
          ],
          "caution": "Rounded capacity and rank0 identity; no all-rank hardware inventory or exact byte count supplied. This is not GB200."
        },
        {
          "field": "mesh",
          "pp": 1,
          "dp_replicate": 1,
          "dp_shard": 256,
          "cp": 1,
          "tp": 1,
          "ep": 64,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 20
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 21
            }
          ]
        },
        {
          "field": "batch_and_sequence",
          "local_batch_size": 16,
          "global_batch_size": 4096,
          "gradient_accumulation_steps": 1,
          "sequence_length": 4096,
          "configured_steps": 50,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 42
            }
          ]
        },
        {
          "field": "checkpointing_and_compile",
          "activation_checkpointing": "FullAC",
          "fsdp_applied": true,
          "compile_loss": true,
          "compile_each_transformer_block": true,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 28
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 29
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 30
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 31
            }
          ]
        },
        {
          "field": "optimizer",
          "name": "AdamW",
          "fused": true,
          "foreach": false,
          "learning_rate": 0.00022,
          "betas": [
            0.9,
            0.95
          ],
          "eps": 1e-08,
          "weight_decay": 0.1,
          "logged_parameter_tensor_count": 967,
          "caution": "967 params refers to optimizer tensor entries, not the model scalar parameter count. Optimizer-state dtype is not logged.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 35
            }
          ]
        },
        {
          "field": "scheduler_adjustment",
          "warmup_requested": 2000,
          "warmup_adjusted": 50,
          "decay_requested": 40,
          "decay_adjusted": 0,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 36
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 37
            }
          ]
        },
        {
          "field": "hybrid_expert_parallel_buffer",
          "hidden_dim": 7168,
          "max_tokens_per_rank": 65536,
          "num_local_experts": 4,
          "ep_size": 64,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 44
            }
          ]
        },
        {
          "field": "precision_nvfp4_arm",
          "observed_conversions": [
            "NVFP4Linear",
            "dynamic NVFP4 GroupedExperts grouped_mm",
            "MXFP8Linear"
          ],
          "kernel_preference": "cutedsl",
          "resolved_backend": "cutedsl",
          "caution": "Mixed NVFP4/MXFP8 conversion is observed; exact module exclusion/selection and weights/gradients/accumulators/optimizer-state dtypes remain unknown. Do not call the whole training state FP4.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 12
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 13
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 14
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 15
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 16
            }
          ]
        },
        {
          "field": "runtime_package_version",
          "package": "torchtitan",
          "value": "0.2.2+gita615ec2",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 10
            }
          ],
          "caution": "Abbreviated package git suffix only; full immutable commit and dirty-tree state absent."
        },
        {
          "field": "mfu_denominator_as_logged",
          "flops_per_second": 2500000000000000.0,
          "reported_mfu": "N/A",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 32
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 45
            }
          ],
          "caution": "This is an application denominator, not independent verification of GB300 precision-specific hardware ceilings."
        },
        {
          "field": "dataset_and_initial_load_warning",
          "dataset_label": "c4_test",
          "initial_load_warning": "initial_load_model_only=True has no effect without an initial_load_path",
          "caution": "No evidence of a pretrained checkpoint being loaded; initialization seed and exact dataset provenance were not established from the supplied excerpts.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 39
            },
            {
              "file": "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt",
              "line": 41
            }
          ]
        }
      ],
      "reported_only_evidence": [
        {
          "field": "force_load_balance",
          "value": true,
          "evidence": [
            {
              "file": "rspring/RESULTS.md",
              "line": 134
            },
            {
              "file": "rspring/RESULTS.md",
              "line": 135
            },
            {
              "file": "rspring/RESULTS.md",
              "line": 136
            }
          ]
        },
        {
          "field": "torchtitan_ref",
          "value": "nvfp4_kp_cutedsl_default @ a615ec2d",
          "evidence": [
            {
              "file": "rspring/RESULTS.md",
              "line": 128
            }
          ]
        },
        {
          "field": "torchao_ref",
          "value": "nvfp4_moe_cutedsl @ 14555766",
          "evidence": [
            {
              "file": "rspring/RESULTS.md",
              "line": 129
            }
          ]
        }
      ],
      "completion": "Last retained step 40; no Training completed record. Exact last executed step unknown.",
      "memory_interpretation": "Sparse rank0 logged training memory. Allocated/reserved/peak API and reset window are unverified. Flat rounded samples do not establish flat live memory.",
      "context": {
        "model": "DeepSeek-V3 671B",
        "framework": "TorchTitan",
        "revision": null,
        "torch_version": null,
        "hardware": "NVIDIA GB300",
        "world_size": 256,
        "tp": 1,
        "pp": 1,
        "ep": 64,
        "cp": 1,
        "dp_shard": 256,
        "dp_replicate": 1,
        "sequence_length": 4096,
        "tokens_per_rank": 65536,
        "precision": "mixed NVFP4/MXFP8 compute",
        "checkpoint_policy": "FullAC",
        "compile_policy": "loss and each TransformerBlock",
        "global_batch": 4096,
        "microbatches": 1,
        "memory_metric": "training_memory_unspecified"
      }
    },
    "titan_v3_mxfp8_batch16": {
      "primary_source": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
      "configuration_evidence": [
        {
          "field": "model",
          "value": "DeepSeek-V3 671B",
          "total_parameter_elements": 671026404352,
          "dense_parameter_elements": 14456871936,
          "expert_parameter_elements_logged_as_sparse": 656569532416,
          "active_parameter_elements": 37552282624,
          "caution": "Model label and count do not establish every architectural dimension. Sparse here is the logged MoE parameter category, not evidence of structured sparse GEMM throughput.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 20
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 23
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 24
            }
          ]
        },
        {
          "field": "hardware",
          "gpu_name": "NVIDIA GB300",
          "reported_capacity_gib": 276.62,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 22
            }
          ],
          "caution": "Rounded capacity and rank0 identity; no all-rank hardware inventory or exact byte count supplied. This is not GB200."
        },
        {
          "field": "mesh",
          "pp": 1,
          "dp_replicate": 1,
          "dp_shard": 256,
          "cp": 1,
          "tp": 1,
          "ep": 64,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 17
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 18
            }
          ]
        },
        {
          "field": "batch_and_sequence",
          "local_batch_size": 16,
          "global_batch_size": 4096,
          "gradient_accumulation_steps": 1,
          "sequence_length": 4096,
          "configured_steps": 50,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 39
            }
          ]
        },
        {
          "field": "checkpointing_and_compile",
          "activation_checkpointing": "FullAC",
          "fsdp_applied": true,
          "compile_loss": true,
          "compile_each_transformer_block": true,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 25
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 26
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 27
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 28
            }
          ]
        },
        {
          "field": "optimizer",
          "name": "AdamW",
          "fused": true,
          "foreach": false,
          "learning_rate": 0.00022,
          "betas": [
            0.9,
            0.95
          ],
          "eps": 1e-08,
          "weight_decay": 0.1,
          "logged_parameter_tensor_count": 967,
          "caution": "967 params refers to optimizer tensor entries, not the model scalar parameter count. Optimizer-state dtype is not logged.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 32
            }
          ]
        },
        {
          "field": "scheduler_adjustment",
          "warmup_requested": 2000,
          "warmup_adjusted": 50,
          "decay_requested": 40,
          "decay_adjusted": 0,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 33
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 34
            }
          ]
        },
        {
          "field": "hybrid_expert_parallel_buffer",
          "hidden_dim": 7168,
          "max_tokens_per_rank": 65536,
          "num_local_experts": 4,
          "ep_size": 64,
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 41
            }
          ]
        },
        {
          "field": "precision_mxfp8_arm",
          "observed_conversions": [
            "MXFP8Linear",
            "dynamic mxfp8_rceil GroupedExperts grouped_mm"
          ],
          "caution": "Conversion messages establish quantized compute paths, not universal FP8 state storage.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 12
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 13
            }
          ]
        },
        {
          "field": "runtime_package_version",
          "package": "torchtitan",
          "value": "0.2.2+gita615ec2",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 10
            }
          ],
          "caution": "Abbreviated package git suffix only; full immutable commit and dirty-tree state absent."
        },
        {
          "field": "mfu_denominator_as_logged",
          "flops_per_second": 2500000000000000.0,
          "reported_mfu": "N/A",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 29
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 42
            }
          ],
          "caution": "This is an application denominator, not independent verification of GB300 precision-specific hardware ceilings."
        },
        {
          "field": "dataset_and_initial_load_warning",
          "dataset_label": "c4_test",
          "initial_load_warning": "initial_load_model_only=True has no effect without an initial_load_path",
          "caution": "No evidence of a pretrained checkpoint being loaded; initialization seed and exact dataset provenance were not established from the supplied excerpts.",
          "evidence": [
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 36
            },
            {
              "file": "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt",
              "line": 38
            }
          ]
        }
      ],
      "reported_only_evidence": [
        {
          "field": "force_load_balance",
          "value": true,
          "evidence": [
            {
              "file": "rspring/RESULTS.md",
              "line": 134
            },
            {
              "file": "rspring/RESULTS.md",
              "line": 135
            },
            {
              "file": "rspring/RESULTS.md",
              "line": 136
            }
          ]
        },
        {
          "field": "torchtitan_ref",
          "value": "nvfp4_kp_cutedsl_default @ a615ec2d",
          "evidence": [
            {
              "file": "rspring/RESULTS.md",
              "line": 128
            }
          ]
        },
        {
          "field": "torchao_ref",
          "value": "nvfp4_moe_cutedsl @ 14555766",
          "evidence": [
            {
              "file": "rspring/RESULTS.md",
              "line": 129
            }
          ]
        }
      ],
      "completion": "Step 50 and Training completed observed.",
      "memory_interpretation": "Sparse rank0 logged training memory. Allocated/reserved/peak API and reset window are unverified. Flat rounded samples do not establish flat live memory.",
      "context": {
        "model": "DeepSeek-V3 671B",
        "framework": "TorchTitan",
        "revision": null,
        "torch_version": null,
        "hardware": "NVIDIA GB300",
        "world_size": 256,
        "tp": 1,
        "pp": 1,
        "ep": 64,
        "cp": 1,
        "dp_shard": 256,
        "dp_replicate": 1,
        "sequence_length": 4096,
        "tokens_per_rank": 65536,
        "precision": "MXFP8 compute",
        "checkpoint_policy": "FullAC",
        "compile_policy": "loss and each TransformerBlock",
        "global_batch": 4096,
        "microbatches": 1,
        "memory_metric": "training_memory_unspecified"
      }
    },
    "megatron_v3": {
      "primary_source": "chasse/slurm-3061209.out",
      "configuration_evidence": {
        "identity": {
          "value": "DeepSeek-V3 671B workload through Megatron-Bridge/Megatron-Core, with Transformer Engine; not TorchTitan",
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                3152,
                3184,
                3247,
                3249,
                3252,
                10877
              ]
            }
          ]
        },
        "hardware": {
          "value": {
            "direct_device": "NVIDIA GB200",
            "devices_in_nvidia_smi": 4,
            "capacity_per_device_MiB": 189471,
            "world_size": 256
          },
          "classification": "direct_hardware_observation_and_runtime_launch",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                212,
                213,
                216,
                220,
                224,
                1700
              ]
            }
          ],
          "qualification": "SYSJSON says GB300/Theia and contradicts direct nvidia-smi. Prefer direct device evidence; do not propagate SYSJSON hardware/software assumptions as verified runtime facts."
        },
        "parallelism": {
          "value": {
            "TP": 2,
            "PP": 4,
            "virtual_PP": 4,
            "EP": 32,
            "expert_TP": 1,
            "CP": 1,
            "sequence_parallel": true
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                3203,
                3204,
                3260,
                3265,
                3266,
                3269,
                6120
              ]
            }
          ]
        },
        "derived_parallelism": {
          "value": {
            "dense_DP": 32,
            "expert_DP": 2,
            "microbatches_per_step": 480
          },
          "classification": "derived_and_runtime_cross_checked",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                1700,
                3203,
                3204,
                3260,
                3266,
                3446
              ]
            }
          ],
          "qualification": "Dense DP=256/(TP2*PP4*CP1)=32. Expert DP=256/(expertTP1*PP4*EP32)=2. Dataset data_parallel_size=1 at3327 is a dataset config field, not the distributed mesh degree."
        },
        "workload": {
          "value": {
            "sequence_length": 4096,
            "micro_batch_samples": 1,
            "global_batch_samples": 15360,
            "global_tokens_per_step": 62914560,
            "train_iters_configured": 100,
            "dataset_mock": false
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                3286,
                3291,
                3293,
                3338,
                3347
              ]
            }
          ],
          "qualification": "Global token count is derived as15360*4096. It is about3.75 times selected TorchTitan target step tokens, with different context and accumulation."
        },
        "architecture": {
          "value": {
            "layers": 61,
            "hidden_size": 7168,
            "routed_experts": 256,
            "router_topk": 8,
            "mtp_layers": 1,
            "dispatcher": "flex/hybridep"
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                3214,
                3224,
                3242,
                3245,
                3247,
                3249,
                3252
              ]
            }
          ],
          "qualification": "MTP layer is an additional mismatch to selected TorchTitan recipe; headline671B identity does not prove identical parameter census."
        },
        "precision": {
          "value": {
            "parameters": "BF16 configuration with FP8 parameter representation/gather enabled",
            "compute_recipe": "MXFP8/e4m3; FP8 dot-product attention enabled",
            "main_grads": "BF16",
            "main_params_config": "FP32",
            "adam_moments": "BF16/BF16",
            "precision_aware_optimizer": true,
            "store_param_remainders": true
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                3158,
                3159,
                3162,
                3166,
                3169,
                3177,
                3206,
                3207,
                3208,
                3209,
                5829
              ]
            }
          ],
          "qualification": "Do not infer persistent4-byte master allocation from main_params_dtype alone: precision-aware/remainder storage and reuse flags alter representation."
        },
        "optimizer_and_sharding": {
          "value": {
            "optimizer": "Adam with decoupled weight decay",
            "distributed_optimizer": true,
            "use_megatron_fsdp": false,
            "use_custom_fsdp": false,
            "gradient_reduce_in_fp32": false,
            "overlap_param_gather": true,
            "overlap_grad_reduce": true
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                5390,
                5829
              ]
            }
          ],
          "qualification": "Printed data_parallel_sharding_strategy=optim_grads_params is not evidence of TorchTitan FSDP; enabled implementation is Megatron distributed optimizer."
        },
        "checkpointing_and_graphs": {
          "value": {
            "recompute": "selective",
            "modules": [
              "mlp"
            ],
            "cuda_graphs": "local full_iteration",
            "optimizer_cuda_graph": true
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                3164,
                3198,
                3199,
                3263,
                3264
              ]
            }
          ],
          "qualification": "The run label full_cg denotes full CUDA graphs, not full activation checkpointing."
        },
        "run_status": {
          "value": {
            "logged_run_stop": "success",
            "samples_at_stop": 737280,
            "steps_derived": 48,
            "shell_exit": 0,
            "compliance_check": "SUCCESS"
          },
          "classification": "logged_execution_result",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                10602,
                10604,
                10864,
                10878
              ]
            }
          ],
          "qualification": "Finite train/eval loss values appear, unlike V4 NaN gradient logs. No independent numerical validation; compliance success is the benchmark checker output."
        },
        "memory": {
          "available": "Prelaunch nvidia-smi occupancy/capacity only; no training allocator allocated/reserved/peak series identified in either supplied Chasse log.",
          "sources": [
            {
              "file": "chasse/slurm-3061209.out",
              "lines": [
                203,
                213,
                217,
                221,
                225
              ]
            }
          ],
          "not_available": [
            "per-rank training memory",
            "peak-reset window",
            "allocation lifetime trace",
            "parameter/optimizer/gradient/activation memory partition"
          ],
          "eligibility": "Not a quantitative training-memory reference; prelaunch0MiB is not a training baseline or peak."
        },
        "target_mismatches": [
          "Megatron-Bridge versus TorchTitan",
          "TP2 PP4 EP32 versusTP1 PP1 EP64",
          "selectiveMLP recompute versusFullAC",
          "MXFP8 and BF16 optimizer moments/grad buffers versusBF16 compute and FP32 state",
          "sequence4096 versus16384",
          "globalbatch/480microbatch accumulation differs",
          "full CUDA graphs versusdisabled",
          "MTP enabled",
          "distributed optimizer and hybridEP/paged stash differ"
        ]
      },
      "context": {
        "model": "DeepSeek-V3 671B",
        "framework": "Megatron-Bridge/Megatron-Core",
        "revision": null,
        "torch_version": null,
        "hardware": "NVIDIA GB200",
        "world_size": 256,
        "tp": 2,
        "pp": 4,
        "ep": 32,
        "cp": 1,
        "expert_tp": 1,
        "sequence_length": 4096,
        "tokens_per_rank": 4096,
        "precision": "MXFP8 compute with precision-aware optimizer",
        "gradient_dtype": "bfloat16",
        "optimizer_state_dtype": "bfloat16 moments",
        "checkpoint_policy": "selective recompute",
        "compile_policy": "full iteration CUDA graphs",
        "global_batch": 15360,
        "microbatches": 480,
        "memory_metric": null
      }
    },
    "megatron_v4": {
      "primary_source": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
      "configuration_evidence": {
        "identity": {
          "value": {
            "manifest_family": "DeepSeek-V4",
            "manifest_size": "1t",
            "runtime_reported_parameter_billions": 1637.42,
            "dense_embedding_billions": 64.66,
            "routed_expert_billions": 1572.76,
            "attention_variant": "dsv4_hybrid",
            "hyper_connections": true
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "rmukundan/llmb-config_2568526.yaml",
              "lines": [
                14,
                15
              ]
            },
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                22713,
                22714,
                64263,
                64264,
                64286
              ]
            }
          ],
          "qualification": "Actual model config and parameter report establish a V4-family workload, not V3.1t is a coarse manifest label; reported parameter count is a runtime calculation, not independently recounted storage."
        },
        "framework": {
          "value": "Megatron-Bridge / Megatron-Core / Transformer Engine; not TorchTitan",
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                22657,
                27869,
                28615
              ]
            },
            {
              "file": "rmukundan/llmb-config_2568526.yaml",
              "lines": [
                8
              ]
            }
          ]
        },
        "hardware": {
          "value": {
            "manifest_gpu": "GB300",
            "runtime_profiler_family": "NVIDIA GB11x",
            "world_size_supported": 256,
            "node_process_layout": "64 nodes*4 processes supported by rank0..255 and launch/profile naming"
          },
          "classification": "manifest_and_runtime_corroboration",
          "sources": [
            {
              "file": "rmukundan/llmb-config_2568526.yaml",
              "lines": [
                17,
                18
              ]
            },
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                1,
                2,
                3,
                4,
                64000,
                64255
              ]
            },
            {
              "file": "rmukundan/sbatch_coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526.out",
              "lines": [
                19,
                56
              ]
            }
          ],
          "qualification": "No direct nvidia-smi model/capacity observation supplied for this run; exact device identity and capacity not independently verified. Manifest cluster_name differs from launched node naming; do not infer hardware solely from directory or manifest."
        },
        "parallelism": {
          "value": {
            "PP": 4,
            "virtual_PP": 4,
            "EP": 64,
            "expert_TP": 1,
            "TP": 1,
            "CP": 1,
            "dense_DP": 64,
            "expert_DP": 1,
            "microbatches_per_step": 64
          },
          "classification": "logged_and_derived",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                22715,
                22716,
                22774,
                22785,
                22961,
                23004
              ]
            }
          ],
          "qualification": "PP/VPP/EP/expertTP explicit model fields; TP1 appears in tokenizer config and no nondefault modelTP is printed. DP64 is implied by globalbatch4096/(microbatch1*64microbatches); withworld256/PP4 this supportsTP1/CP1. CP1 is derived, not separately printed model-field evidence. Dataset data_parallel_size=1 is not runtimeDP."
        },
        "workload": {
          "value": {
            "sequence_length": 4096,
            "micro_batch_samples": 1,
            "global_batch_samples": 4096,
            "global_tokens_per_step": 16777216,
            "train_iters": 50,
            "mock_dataset": true
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                22801,
                22806,
                22808,
                22855,
                22863,
                22864,
                28648
              ]
            }
          ],
          "qualification": "Global token count equals selected TorchTitan target by arithmetic only; model/context/schedule/precision differ."
        },
        "architecture": {
          "value": {
            "layers": 61,
            "hidden_size": 7168,
            "routed_experts": 384,
            "router_topk": 6,
            "mtp_layers": 1,
            "force_load_balancing": true,
            "quantization_padding": true,
            "dispatcher": "flex/hybridep"
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                22727,
                22736,
                22749,
                22751,
                22752,
                22754,
                22757,
                22759,
                22761,
                22764
              ]
            }
          ]
        },
        "precision_and_optimizer": {
          "value": {
            "params": "BF16 config,FP8 parameter/gather enabled",
            "compute_recipe": "MXFP8/e4m3",
            "main_grads": "BF16 default; explicit FP32 exceptions in mHC gradient buckets",
            "main_params_config": "FP32",
            "moments": "BF16/BF16",
            "optimizer": "Adam with decoupled weight decay",
            "precision_aware": true,
            "store_param_remainders": true,
            "distributed_optimizer": true,
            "optimizer_cpu_offload": false,
            "use_megatron_fsdp": false,
            "use_custom_fsdp": false,
            "check_for_nan_in_grad": false
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                22664,
                22665,
                22668,
                22671,
                22673,
                22720,
                22721,
                22722,
                27869,
                28615
              ]
            },
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                25874,
                25875,
                25876,
                25877,
                25878,
                25879
              ]
            }
          ],
          "qualification": "Do not translate dtype fields into a complete persistent-state byte count without enabled precision-aware storage implementation."
        },
        "checkpointing_graphs_offload": {
          "value": {
            "recompute": "selective",
            "recompute_modules": [
              "mla_up_proj",
              "mhc"
            ],
            "cuda_graph_impl": "full_iteration",
            "activation_offloading": true,
            "offload_modules": [
              "core_attn",
              "attn_proj"
            ],
            "max_inflight_offloads": 2,
            "moe_paged_stash": true
          },
          "classification": "logged_configuration",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                22705,
                22718,
                22719,
                22744,
                22767,
                22778,
                22779
              ]
            }
          ],
          "qualification": "Different from FullAC, and attention activation CPU offload materially changes GPU residency. CUDA graph capture occurs atiteration4 after quoted allocator samples."
        },
        "numerical_status": {
          "value": {
            "iterations_logged": 50,
            "grad_norm_nan_iterations": 49,
            "first_nan_iteration": 2,
            "last_nan_iteration": 50,
            "logged_nan_iteration_counter": 0,
            "job_exit_code": 0
          },
          "classification": "logged_execution_result",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                64266,
                64282,
                64292,
                65154
              ]
            },
            {
              "file": "rmukundan/sbatch_coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526.out",
              "lines": [
                67
              ]
            }
          ],
          "qualification": "First iteration gradnorm5.979; iterations2..50 gradnormNaN. Zero nan counter/exit0 does not establish valid numerical training. Later output omits loss fields; do not call loss finite/valid across the run."
        },
        "memory_qualification": {
          "unit": "Logger labels GB/gigabytes. Byte divisor not present in supplied source; retain native numbers, do not silently convert decimalGB versusGiB.",
          "window": "Afteriterations1and2 only. No peak-reset code or reset timestamp supplied; max counters have unknown accumulation window. They are not proven per-step peaks.",
          "rank_coverage": [
            0,
            64,
            128,
            192
          ],
          "coverage_warning": "4/256ranks only; largest observed rank is not demonstrated global worst rank. Later full-iteration graph residency is not covered by these summaries.",
          "metric_warning": "Allocated,active,reserved,max_allocated,max_active,max_reserved differ. Maxima need not coincide; no semantic allocation categories can be recovered by subtraction.",
          "sources": [
            {
              "file": "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out",
              "lines": [
                64265,
                64267,
                64272,
                64273,
                64280,
                64281,
                64284,
                64290,
                64293
              ]
            }
          ]
        },
        "target_mismatches": [
          "V4 architecture and1637.42B printed params versusV3/671B",
          "Megatron versusTorchTitan",
          "GB300-labelled versusGB200target",
          "PP4 versusPP1",
          "selective recompute plusactivationoffload versusFullAC",
          "MXFP8/BF16states versusBF16compute/FP32states",
          "MTP andforcedloadbalancing enabled",
          "full CUDA graphs versusdisabled",
          "context4096 and64microbatches versuscontext16384 and1microbatch",
          "numerically invalid gradient norms"
        ]
      },
      "context": {
        "model": "DeepSeek-V4 runtime estimate 1637.42B",
        "framework": "Megatron-Bridge/Megatron-Core",
        "revision": null,
        "torch_version": null,
        "hardware": null,
        "world_size": 256,
        "tp": 1,
        "pp": 4,
        "ep": 64,
        "cp": 1,
        "expert_tp": 1,
        "sequence_length": 4096,
        "tokens_per_rank": 4096,
        "precision": "MXFP8 compute with precision-aware optimizer",
        "gradient_dtype": "bfloat16 with float32 mHC exceptions",
        "optimizer_state_dtype": "bfloat16 moments",
        "checkpoint_policy": "selective recompute",
        "compile_policy": "full iteration CUDA graphs",
        "global_batch": 4096,
        "microbatches": 64,
        "memory_metric": "allocator counters, unverified unit divisor/reset window"
      }
    },
    "titan_v3_expert_mxfp8_batch8": {
      "primary_source": "additional/deepseek_v3_671b.out",
      "configuration_evidence": {
        "framework": {
          "value": "TorchTitan 0.2.0; exact revision unavailable",
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                1
              ]
            }
          ]
        },
        "model": {
          "value": {
            "model": "DeepSeek-V3 671B",
            "parameter_elements": 671026404352
          },
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                34405
              ]
            }
          ]
        },
        "hardware": {
          "value": {
            "gpu": "NVIDIA GB200",
            "capacity_gib_printed": 184.0
          },
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                28131
              ]
            }
          ],
          "qualification": "Rounded display, not exact byte capacity or all-device census."
        },
        "mesh": {
          "value": {
            "pp": 1,
            "tp": 1,
            "ep": 64,
            "dp_shard": 256,
            "dp_replicate": 1,
            "cp": 1,
            "expert_tp": 1
          },
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                227
              ]
            }
          ]
        },
        "precision": {
          "value": "MoE modules matching ['experts'] converted to TE grouped GEMM mxfp8",
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                28111
              ]
            }
          ],
          "qualification": "Expert compute conversion does not establish parameter, gradient or optimizer storage precision."
        },
        "attention": {
          "value": "sdpa",
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                1825
              ]
            }
          ]
        },
        "checkpointing": {
          "value": "FullAC",
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                61174
              ]
            }
          ]
        },
        "compile": {
          "value": "Each TransformerBlock compiled",
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                65395
              ]
            }
          ]
        },
        "workload": {
          "value": {
            "local_batch": 8,
            "global_batch": 2048,
            "gradient_accumulation": 1,
            "sequence_length": 4096,
            "steps": 50
          },
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                133127
              ]
            }
          ],
          "qualification": "Runtime values override the supplied TOML."
        },
        "memory": {
          "value": {
            "initialization_gib": 9.85,
            "step1_gib_range": [
              101.83,
              101.84
            ],
            "later_gib_range": [
              102.2,
              102.21
            ]
          },
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                133122,
                135425
              ]
            }
          ],
          "qualification": "256 unranked fan-out samples per logged step; do not assign ranks by ordering or claim all-rank identity."
        },
        "completion": {
          "value": "Training completed",
          "classification": "runtime_log_observed",
          "evidence": [
            {
              "file": "additional/deepseek_v3_671b.out",
              "lines": [
                139772
              ]
            }
          ]
        }
      },
      "context": {
        "model": "DeepSeek-V3 671B",
        "framework": "TorchTitan",
        "revision": null,
        "torch_version": null,
        "hardware": "NVIDIA GB200",
        "world_size": 256,
        "tp": 1,
        "pp": 1,
        "ep": 64,
        "cp": 1,
        "dp_shard": 256,
        "dp_replicate": 1,
        "expert_tp": 1,
        "sequence_length": 4096,
        "tokens_per_rank": 32768,
        "precision": "expert-only MXFP8 compute",
        "attention": "sdpa",
        "checkpoint_policy": "FullAC",
        "compile_policy": "each TransformerBlock",
        "global_batch": 2048,
        "microbatches": 1,
        "memory_metric": "training_memory_unspecified"
      }
    }
  },
  "reviewed_sources": {
    "additional/deepseek_v3_671b.out": {
      "sha256": "6f9354483ad56a3b9a95f4e801c18f0b3f624ac04a9d42570bc3064f475a9057"
    },
    "chasse/slurm-3061209.out": {
      "sha256": "6e26e504d829400c64bfae7a03da6ec04fdc38d0550def3fee53cca191ae4119"
    },
    "rmukundan/llmb-config_2568526.yaml": {
      "sha256": "32c491ec0fc8afb51a4ed4ed59c8787dba0dc1ee892543ba0c20135bb7bd05f8"
    },
    "rmukundan/log-coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526_0.out": {
      "sha256": "c3aa50e8098d33bc73009c0df76c8eca4caf7ebb22ff9e6445db4eeefdba1bd6"
    },
    "rmukundan/sbatch_coreai_dlalgo_llm-llm.pretrain_deepseek_v4_pro_fp8_mx_gpus256_gb300_2568526.out": {
      "sha256": "6686cff7f2fcc5298ac3caf1f8eb0668d1b46d8b1ef1591a9fd9ec42d17b9c06"
    },
    "rspring/RESULTS.md": {
      "sha256": "16548d291f3b7441db79465fa5816eed5cedcb1e3a700cdf56304edc05431bf3"
    },
    "rspring/lyris_expB-rung5-mxfp8-64n-408471707.txt": {
      "sha256": "d37c4bc1f5da809cf6a61c103782fa9e0da4bf62c45898379ecbac88647dd1b5"
    },
    "rspring/lyris_expB-rung5-nvfp4-64n-408471710.txt": {
      "sha256": "132f699bf12a67fa041c6e4b0f5843c357da72f2ef0637fb6bf711c85898d1cf"
    }
  }
}
''')

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def target_context():
    profile = json.loads((REPO / 'profiles/deepseek-v3-256gb200.json').read_text())
    defaults = profile['upstream_defaults_preserved']
    versions = json.loads((REPO / 'profiles/versions.json').read_text())
    return dict(model='DeepSeek-V3 671B', framework='TorchTitan',
                revision=profile['torchtitan_commit'], hardware='NVIDIA GB200',
                world_size=profile['world_size'], tp=1, pp=1, ep=64, cp=1,
                dp_shard=256, dp_replicate=1, expert_tp=1,
                sequence_length=defaults['context_length'],
                tokens_per_rank=defaults['tokens_per_microbatch_per_dp_rank'],
                precision='BF16 compute', parameter_dtype=defaults['master_parameters'],
                gradient_dtype=defaults['gradient_reduction'], optimizer_state_dtype='float32',
                optimizer='AdamW', attention=defaults['attention'],
                checkpoint_policy='FullAC', compile_policy='gated_rmsnorm, loss, swiglu',
                global_batch=1024, microbatches=1, pipeline_schedule='PP1',
                memory_metric='logical_storage', routing_policy=None,
                torch_version=versions['torch_version'], memory_unit_basis='bytes',
                offload_policy=None)


def figures(output, documents, budgets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'titan-sim-runtime-log-audit-v1'})

    def save(fig, name):
        fig.savefig(output / (name + '.png'), dpi=180, facecolor='white')
        fig.savefig(output / (name + '.svg'), facecolor='white', metadata={'Date': None})
        svg = output / (name + '.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.7))
    arms = [('titan_v3_expert_mxfp8_batch8', '#2463a5', 'GB200 / expert MXFP8 / batch 8'),
            ('titan_v3_mxfp8_batch16', '#27826c', 'GB300 / MXFP8 / batch 16'),
            ('titan_v3_nvfp4_batch16', '#b86c29', 'GB300 / mixed NVFP4–MXFP8 / batch 16')]
    for key, color, label in arms:
        rows = documents[key]['samples']
        training = [r for r in rows if r['metric'] == 'training_memory_unspecified']
        initial = [r for r in rows if r['metric'] == 'model_initialization_memory_unspecified']
        if any(r['unit'] != 'GiB' for r in training + initial):
            raise ValueError('V3 plot requires explicit GiB observations')
        ax.scatter([r['step'] for r in training], [r['value'] for r in training],
                   s=34, color=color, label=label, alpha=.8, zorder=3)
        ax.scatter([0 for r in initial], [r['value'] for r in initial],
                   marker='x', s=45, color=color, zorder=3)
    ax.set(title='DeepSeek-V3: sparse TorchTitan memory observations',
           xlabel='Logged training step (0 = model initialization, ×)',
           ylabel='Printed memory (GiB); counter semantics unverified', ylim=(0, 190))
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='lower right', frameon=True)
    fig.text(.10, .065, 'Points are log samples, not an allocation timeline. No interpolation or all-rank peak inference.\n'
             'GB200 batch 8: rank identity absent (256 samples per logged step); GB300 excerpts: rank 0.\n'
             'Megatron V3: no training allocator series supplied. Its partial stash geometries remain separate observations.',
             fontsize=9, va='bottom')
    fig.subplots_adjust(bottom=.23, top=.91, left=.10, right=.98)
    save(fig, 'v3_observations')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    metrics = [('mem-allocated-gigabytes', 'Allocated', '#2463a5', 'o'),
               ('mem-reserved-gigabytes', 'Reserved', '#b86c29', 's'),
               ('mem-max-allocated-gigabytes', 'Max allocated', '#27826c', '^'),
               ('mem-max-reserved-gigabytes', 'Max reserved', '#995eaa', 'x')]
    samples = documents['megatron_v4']['samples']
    for ax, rank in zip(axes.flat, (0, 64, 128, 192)):
        for metric, label, color, marker in metrics:
            rows = [r for r in samples if r['rank'] == rank and r['metric'] == metric]
            if any(r['unit'] != 'GB' or r['bytes'] is not None for r in rows):
                raise ValueError('V4 plot must preserve native GB with unknown divisor')
            ax.scatter([r['step'] for r in rows], [r['value'] for r in rows],
                       label=label, color=color, marker=marker, s=52, alpha=.85)
        ax.set(title=f'Explicit rank {rank}', xticks=[1, 2], xlim=(.8, 2.2), ylim=(110, 285))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('DeepSeek-V4: observed allocator counters after iterations 1 and 2', y=.97)
    fig.supylabel('Native printed “GB” / “gigabytes” — byte divisor unknown', x=.02)
    fig.supxlabel('After logged iteration; max-counter reset window unverified', y=.14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .065), ncol=4, frameon=False)
    fig.text(.10, .025, 'Only 4 of 256 ranks observed; no global peak claim. Maxima may occur at different times.\n'
             'Gradient norms are NaN at iterations 2–50; exit code 0 does not establish numerically valid training.', fontsize=9)
    fig.subplots_adjust(left=.10, right=.98, top=.90, bottom=.22, hspace=.24)
    save(fig, 'v4_observations')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)
    colors = {'parameters': '#2463a5', 'optimizer': '#5b97c8', 'activations': '#d59b42', 'gradients': '#27826c'}
    for ax, (batch, document) in zip(axes, budgets.items()):
        rows = document['ledger']['per_rank']['0']['phases']
        bottom = [0.] * len(rows)
        for category, color in colors.items():
            values = [row['by_category'][category] / 2**30 for row in rows]
            ax.bar(range(len(rows)), values, bottom=bottom, color=color, label=category.replace('_', ' '), width=.7)
            bottom = [a+b for a,b in zip(bottom, values)]
        for scenario, color in zip(document['capacity_scenarios'], ('#666666', '#a7668c')):
            cap = scenario['memory_capacity_bytes'] / 2**30
            ax.axhline(cap, linestyle='--', linewidth=1.2, color=color)
            ax.text(4.4, cap+3, f'{cap:.2f} GiB capacity scenario', ha='right', color=color, fontsize=8)
        ax.set(title=f'Batch {batch}: declared rank 0 components', xticks=range(len(rows)),
               xticklabels=[r['phase'].replace('_', '\n') for r in rows], ylim=(0, 310))
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Specified component bytes / 2³⁰ (GiB)')
    fig.suptitle('Conditional memory budgets — incomplete coverage, not full-run predictions', y=.96)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .12), ncol=4, frameon=False)
    fig.text(.08, .035, 'Assumptions: evenly sharded FP32 parameters / Adam moments / gradients; BF16 checkpoint boundaries.\n'
             'Missing gathered parameters, MoE/native buffers, quantization, workspaces and allocator coverage remain unknown.\n'
             'Capacity lines use rounded log displays. Being below a line does not establish fit; no observed bytes were fitted.', fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.27, top=.88, wspace=.12)
    save(fig, 'conditional_budgets')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-root', type=Path, required=True, help='Extracted bundle containing rspring/, chasse/, rmukundan/')
    parser.add_argument('--additional-log-dir', type=Path, required=True, help='Directory containing the additional deepseek_v3_671b.out')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    resolved = {}
    # Check all source files before emitting anything. Their reviewed annotations
    # are versioned here; inputs with different contents require a new audit.
    for source, expected in AUDIT['reviewed_sources'].items():
        path = (args.additional_log_dir / source.removeprefix('additional/')
                if source.startswith('additional/') else args.log_root / source)
        if not path.is_file():
            parser.error('Missing audited source: ' + source)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            parser.error('Audited source hash mismatch: ' + source)
        resolved[source] = path
    documents, runs = {}, {}
    target = target_context()
    for key, reviewed in AUDIT['runs'].items():
        source = reviewed['primary_source']
        document = parse_log(resolved[source], source_id=source)
        document['summary'] = summarize(document)
        documents[key] = document
        runs[key] = {**reviewed, 'source_sha256': document['sha256'],
                     'cross_validation_against_selected_target': assess_cross_validation(target, reviewed['context'])}
        if runs[key]['cross_validation_against_selected_target']['eligible']:
            raise ValueError('Audited historical run unexpectedly eligible for selected-target comparison')
    profiles = [json.loads((REPO / 'profiles/memory' / name).read_text()) for name in
                ('gb200-observed-1098308.json', 'gb300-observed-rspring.json')]
    budgets = {}
    for batch in (8, 16):
        source = f'examples/memory/deepseek-v3-batch{batch}-conditional.json'
        spec = json.loads((REPO / source).read_text())
        budgets[batch] = budget_scenarios(spec, profiles)
        budgets[batch]['input_spec'] = spec
        budgets[batch]['input_spec_source'] = source
        budgets[batch]['input_spec_sha256'] = hashlib.sha256((REPO / source).read_bytes()).hexdigest()
        if budgets[batch]['ledger']['coverage_complete']:
            raise ValueError('Partial audited scenario unexpectedly claims complete memory coverage')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'run_audit.json', {
        'schema_version': 1, 'source': 'manually_audited_runtime_configurations',
        'reviewed_sources': AUDIT['reviewed_sources'], 'selected_target': target, 'runs': runs,
        'limitations': ['Source hashes bind line evidence to private inputs; raw logs are not published.',
                        'Observed settings, reported-only statements and derived assumptions have distinct labels.',
                        'None of these runs is a matched validation of the selected 6078523 recipe.',
                        'Unknown state precision, revisions, counter semantics and rank identities remain unknown.']})
    write_json(args.output / 'observations.json', {
        'schema_version': 1, 'source': 'training_log_observation_collection', 'runs': documents,
        'limitations': ['No interpolation between sparse samples; no wall-clock memory trajectory reconstructed.',
                        'Observed stash payloads are partial geometry records, not independent predictions or simultaneous peaks.',
                        'Unknown ranks are retained; unlabelled launcher IDs are not treated as GPU ranks.',
                        'Megatron GB counters retain unknown byte divisor and peak-reset windows.']})
    for batch, document in budgets.items():
        write_json(args.output / f'budget-batch{batch}.json', document)
    with tempfile.TemporaryDirectory(prefix='titan-sim-mpl-') as cache:
        previous_cache = os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR'] = previous_cache or cache
        try:
            figures(args.output, documents, budgets)
        finally:
            if previous_cache is None:
                os.environ.pop('MPLCONFIGDIR', None)
    print(json.dumps({'runs': len(runs), 'samples_by_run': {k: len(v['samples']) for k,v in documents.items()},
                      'all_target_comparisons_blocked': True, 'budgets_complete': False}))


if __name__ == '__main__':
    main()
