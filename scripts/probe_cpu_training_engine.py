#!/usr/bin/env python3
"""Opt-in, subprocess-only original TrainingEngine CPU semantic diagnostic.

This explicitly replaces CUDA placement and Inductor execution. It retains the
original TrainingEngine, FSDP, FullAC, local compile region selection and optimizer
path. A fake process group executes one logical rank, not every distributed rank.
No result establishes CUDA numerical, allocator, compiler or timing fidelity.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch


@contextlib.contextmanager
def _host_controls(engine_module, *, seed):
    """Record hardware controls; execute actual deterministic host topology."""
    import torch
    from torch.utils._python_dispatch import _disable_current_modes
    from torchtitan.distributed.parallelism_context import ParallelismContext

    report = []
    original_mesh = ParallelismContext.build_mesh
    original_recipe = engine_module.load_validation_recipe
    original_set = torch._C._set_fp32_precision_setter
    original_get = torch._C._get_fp32_precision_getter
    precision = {}
    # An opaque host token only supports compiler save/restore bookkeeping.
    # It is not a Philox state and cannot supply numerical CUDA RNG behavior.
    rng_token = torch.empty(0, dtype=torch.uint8, device="cpu")

    def build_mesh(context):
        report.append(dict(control="ParallelismContext.build_mesh",
                           execution="original host topology outside dispatch modes",
                           excluded_storage="deterministic CPU rank-map tensors"))
        with _disable_current_modes():
            return original_mesh(context)

    def recipe(**kwargs):
        config = original_recipe(**kwargs)
        if seed is not None:
            report.append(dict(control="scenario.debug.seed", original=config.debug.seed,
                               selected=seed, numerical_cuda_rng_supported=False))
            config.debug.seed = seed
        return config

    def set_precision(backend, operation, value):
        if (backend, operation, value) == ("cuda", "matmul", "bfx9"):
            precision[(backend, operation)] = value
            report.append(dict(control="cuda.matmul.fp32_precision", requested=value,
                               native_setter_executed=False,
                               numerical_precision_supported=False))
            return
        return original_set(backend, operation, value)

    def get_precision(backend, operation):
        if (backend, operation) in precision:
            return precision[(backend, operation)]
        return original_get(backend, operation)

    def get_rng(device="cuda"):
        report.append(dict(control="cuda.get_rng_state", execution="opaque host token",
                           numerical_cuda_rng_supported=False))
        with _disable_current_modes():
            return rng_token.clone()

    def set_rng(state, device="cuda"):
        if state.device.type != "cpu" or state.dtype != torch.uint8 or state.numel() != 0:
            raise NotImplementedError("Only the explicit virtual CUDA RNG token can be restored")
        report.append(dict(control="cuda.set_rng_state", execution="opaque host token restored",
                           numerical_cuda_rng_supported=False))

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(ParallelismContext, "build_mesh", build_mesh))
        stack.enter_context(patch.object(engine_module, "load_validation_recipe", recipe))
        stack.enter_context(patch.object(torch._C, "_set_fp32_precision_setter", set_precision))
        stack.enter_context(patch.object(torch._C, "_get_fp32_precision_getter", get_precision))
        stack.enter_context(patch.object(torch.cuda, "get_rng_state", get_rng))
        stack.enter_context(patch.object(torch.cuda, "set_rng_state", set_rng))
        yield report


def _run_child(args):
    if os.environ.get("TITAN_SIM_ENGINE_DIAGNOSTIC_CHILD") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run through the isolated hidden-GPU diagnostic wrapper")
    import torch
    import torch._dynamo
    from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
    from titan_sim import capture as capture_module
    from titan_sim.attention_metadata import flex_attention_metadata_scope
    from titan_sim.compile_capture import AOTGraphCapture
    from titan_sim.cpu_metadata import cpu_metadata_execution
    from titan_sim.recipes import import_torchtitan

    # Cache native function identities before FakeCudaRuntime installs aliases.
    TorchInGraphFunctionVariable._get_handlers()
    import_torchtitan(args.torchtitan_root)
    engine_module = importlib.import_module("titan_sim.project_engine")
    capture = AOTGraphCapture(source="cpu_projection")

    def capture_factory(**kwargs):
        if kwargs != {"source": "cpu_projection"}:
            raise NotImplementedError("Unexpected original engine capture constructor")
        return capture

    with contextlib.ExitStack() as stack:
        controls = stack.enter_context(_host_controls(engine_module, seed=args.seed))
        stack.enter_context(patch.object(capture_module, "TensorCapture", capture_factory))
        cpu = stack.enter_context(cpu_metadata_execution())
        attention = stack.enter_context(flex_attention_metadata_scope())
        # Import and adapt Flex before overriding subsequent local compile calls.
        stack.enter_context(capture.compile_scope())
        document = engine_module.project_engine(
            torchtitan_root=args.torchtitan_root,
            output_dir=args.output.parent / (args.output.stem + "-engine"),
            world_size=args.world_size, expert_parallel_degree=args.expert_parallel_degree,
            flavor=args.flavor, seq_len=args.seq_len, tokens=args.tokens, steps=args.steps,
            routing_scenario=args.routing_scenario,
            capture_scope="post_initialization", probe=False,
        )
        document["metadata"].update(
            diagnostic="original_training_engine_cpu_aot_semantics",
            production_training_capture=False, cuda_memory_equivalent=False,
            all_ranks_executed=False, captured_logical_rank=0,
            native_cuda_autograd=False, numerical_values_validated=False,
            cpu_diagnostic_contract=cpu, attention_metadata_contract=attention,
            control_adaptations=controls,
            native_cuda_probe="not used: explicit CPU FakeTensor autograd selected",
            scenario=dict(world_size=args.world_size, expert_parallel_degree=args.expert_parallel_degree,
                          seq_len=args.seq_len, tokens=args.tokens, steps=args.steps,
                          seed=args.seed, routing=args.routing_scenario),
        )
    return document


def summarize(document):
    """Public-safe evidence without local paths, tracebacks or model dumps."""
    metadata = document["metadata"]
    compilation = metadata.get("aot_graph_capture", {})
    policy = metadata.get("execution_policy", {})
    controls = metadata.get("control_adaptations", [])
    return dict(
        status=metadata["status"], diagnostic="original_training_engine_cpu_aot_semantics",
        production_training_capture=False, cuda_memory_equivalent=False,
        all_ranks_executed=False, captured_logical_rank=0,
        torch_version=metadata.get("torch_version"), torchtitan_commit=metadata.get("torchtitan_commit"),
        scenario=metadata.get("scenario"), resolved_topology=metadata.get("resolved_topology"),
        model_parameter_count=metadata.get("model_parameter_count"),
        events=len(document.get("events", [])), storages=len(document.get("storages", [])),
        events_by_phase=dict(Counter(e["phase"] for e in document.get("events", []))),
        collective_operations=dict(Counter(e["name"] for e in document.get("events", []) if e["kind"] == "collective")),
        aot_graph_kinds=dict(Counter(g["kind"] for g in compilation.get("graphs", []))),
        aot_execution_kinds=dict(Counter(e["graph_kind"] for e in compilation.get("executions", []))),
        original_compile_policy=policy.get("compile"), selected_backend=compilation.get("backend"),
        original_activation_checkpoint=policy.get("activation_checkpoint"),
        original_optimizer=policy.get("optim", {}).get("optimizer"),
        original_training_precision=policy.get("training"),
        control_adaptations=list({json.dumps(row, sort_keys=True): row for row in controls}.values()),
        error_type=metadata.get("error_type"),
        limitations=[
            "One logical rank and fake process group; other ranks, collective data and rank imbalance are unvalidated.",
            "CPU placement, metadata Flex and AOTAutograd ATen execution replace CUDA/Inductor behavior.",
            "CUDA numerical values, BF16x9 arithmetic, RNG sequences, stream/allocator/workspace semantics are unvalidated.",
            "Random parameter initialization is excluded; mask callback tensor read dependencies remain opaque.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-diagnostic", action="store_true", help="Required explicit fidelity opt-in")
    parser.add_argument("--torchtitan-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--expert-parallel-degree", type=int, default=1)
    parser.add_argument("--flavor", choices=["debugmodel", "671B"], default="debugmodel")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Explicit host seed scenario; original default may read unknown fake values")
    parser.add_argument("--routing-scenario", choices=["strict", "balanced"], default="strict")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.cpu_diagnostic:
        parser.error("--cpu-diagnostic is required; this is not a CUDA/Inductor training backend")
    if not 1 <= args.timeout <= 60:
        parser.error("timeout must be between 1 and 60 seconds")
    if args.output.exists():
        parser.error("output already exists; use a new path to preserve prior evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.child:
        try:
            document = _run_child(args)
        except Exception as error:
            document = dict(schema_version=1, source="cpu_projection", events=[], storages=[],
                            metadata=dict(status="blocked", error_type=type(error).__name__,
                                          error=str(error), traceback=traceback.format_exc()))
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        raise SystemExit(0 if document["metadata"]["status"] == "complete" else 2)

    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", TITAN_SIM_ENGINE_DIAGNOSTIC_CHILD="1")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")]
    )
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"]
    try:
        result = subprocess.run(command, env=environment, capture_output=True, timeout=args.timeout)
        process = dict(returncode=result.returncode, timed_out=False, cpu_only=True)
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        process = dict(returncode=None, timed_out=True, cpu_only=True)
        stdout, stderr = error.stdout or b"", error.stderr or b""
    args.output.with_suffix(".stdout").write_bytes(stdout)
    args.output.with_suffix(".stderr").write_bytes(stderr)
    args.output.with_suffix(".process.json").write_text(json.dumps(process, indent=2) + "\n")
    if args.output.exists():
        summary = summarize(json.loads(args.output.read_text()))
        summary["trace_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    else:
        summary = dict(status="blocked", reason="Child exited before writing a trace",
                       production_training_capture=False, all_ranks_executed=False)
    summary["process"] = process
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["status"] == "complete" and process["returncode"] == 0 else 2)


if __name__ == "__main__":
    main()
