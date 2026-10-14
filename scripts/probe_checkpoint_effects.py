#!/usr/bin/env python3
"""Independently probe checkpoint preservation of registered ordered effects.

Runs on CPU with only PyTorch imports. The first policy is the unmodified policy
used by pinned TorchTitan FullAC: always PREFER_RECOMPUTE. A separate MUST_SAVE
control isolates the checkpoint caching contract; it does not patch TorchTitan
or change any training run. Exit 2 means expected effect preservation failed,
while JSON still records the completed diagnostic and its control.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform


def run_probe():
    import torch
    from torch._higher_order_ops.effects import has_effects
    import torch.utils.checkpoint as checkpoint_module
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

    count = 0

    @torch.library.custom_op("titan_checkpoint_probe::effect_identity", mutates_args=())
    def effect_identity(x: torch.Tensor) -> torch.Tensor:
        nonlocal count
        count += 1
        return x.clone()

    @effect_identity.register_fake
    def fake(x):
        return torch.empty_like(x)

    effect_identity.register_autograd(lambda ctx, grad: grad)
    effect_identity.register_effect(torch.library.EffectType.ORDERED)
    operation = torch.ops.titan_checkpoint_probe.effect_identity.default
    results = []
    for save_effect in (False, True):
        count = 0
        visits = []
        def policy(ctx, op, *args, **kwargs):
            if op == operation:
                visits.append(dict(is_recompute=ctx.is_recompute, effect_registered=has_effects(op)))
            if save_effect and op == operation:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        iterations = []
        for iteration in range(2):
            x = torch.randn(8, requires_grad=True, device="cpu")
            output = checkpoint(lambda value: effect_identity(value).sin(), x,
                                use_reentrant=False,
                                context_fn=lambda: create_selective_checkpoint_contexts(policy))
            calls_after_forward = count
            output.sum().backward()
            forward_matches = torch.allclose(output, x.sin())
            gradients_match = torch.allclose(x.grad, x.cos())
            iterations.append(dict(iteration=iteration, calls_after_forward=calls_after_forward,
                                   calls_after_backward=count, expected_calls_after_backward=iteration+1,
                                   forward_matches=forward_matches, gradients_match=gradients_match))
        results.append(dict(policy="explicit_must_save_control" if save_effect else "original_prefer_recompute",
                            iterations=iterations, effect_policy_visits=visits,
                            preserves_registered_effect=all(row["calls_after_backward"] == row["expected_calls_after_backward"]
                                                            for row in iterations)))
    checkpoint_source = Path(inspect.getfile(checkpoint_module))
    return dict(schema_version=1, source="independent_cpu_runtime_diagnostic", status="diagnostic_complete",
                torch_version=torch.__version__, torch_git_version=torch.version.git_version,
                python_version=platform.python_version(), platform=platform.system(), device="cpu",
                torchtitan_imported=False, simulator_imported=False, upstream_modified=False,
                torchtitan_policy_reference="607852350b41b4b280d8b8de6340f4847ed71c5e: torchtitan/distributed/activation_checkpoint.py:_full_ac_policy",
                expected_behavior="One execution of a registered ordered effect per original forward, none during recomputation",
                effect_registered=has_effects(operation), results=results,
                original_policy_meets_expected_behavior=results[0]["preserves_registered_effect"],
                explicit_save_control_meets_expected_behavior=results[1]["preserves_registered_effect"],
                checkpoint_source=dict(module="torch.utils.checkpoint",
                                       sha256=hashlib.sha256(checkpoint_source.read_bytes()).hexdigest()),
                limitations=["CPU custom-operator diagnostic, not a full model or GPU validation.",
                             "Correct gradients in this identity example do not establish correctness of other stateful operations.",
                             "The explicit-save control is not applied to the pinned training recipe."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not report["original_policy_meets_expected_behavior"] or not report["explicit_save_control_meets_expected_behavior"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
