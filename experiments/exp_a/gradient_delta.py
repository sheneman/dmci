############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# gradient_delta.py: Gradient delta analysis: DMCI vs. direct compilation. For each program P1-P6, at N random parameter settings,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Gradient delta analysis: DMCI vs. direct compilation.

For each program P1-P6, at N random parameter settings, compile via both
direct compilation and DMCI (compiled self-hosted interpreter), compute
gradients via autograd, and measure:
  - Relative gradient error:  ||grad_DMCI - grad_direct||_2 / ||grad_direct||_2
  - Gradient cosine similarity

This directly addresses the reviewer question: do gradients (not just
losses) match between the two compilation paths?

Usage:
    python -m experiments.exp_a.gradient_delta [--n-samples 50] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .programs import ProgramSpec, ALL_PROGRAMS, _all_input_names


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compile_graph(source: str, input_names: list[str]):
    """Compile a Scheme source string to a compute graph."""
    inputs = {n: None for n in input_names}
    return compile_program(source, inputs=inputs, prelude=True)


def _make_random_params(
    param_names: list[str], seed: int
) -> dict[str, nn.Parameter]:
    """Generate random parameter values for one trial."""
    torch.manual_seed(seed)
    return {
        name: nn.Parameter(torch.randn(1).squeeze() * 2.0)
        for name in param_names
    }


def _build_tagged_inputs(
    params: dict[str, nn.Parameter],
    x_val: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build tagged-value inputs for the evaluator."""
    inputs = {"x": make_float(x_val)}
    for name, param in params.items():
        inputs[name] = make_float(param)
    return inputs


def _forward_sum(graph, params, x_values):
    """Forward pass: sum of squared outputs over x_values.

    Uses output^2 as a simple differentiable scalar loss so we can compare
    gradients between compilation paths without needing a target function.
    Both paths should produce identical outputs for identical parameters,
    so their gradients of this loss should match exactly.
    """
    total = torch.tensor(0.0)
    for x_val in x_values:
        x_t = torch.tensor(x_val, dtype=torch.float32)
        inputs = _build_tagged_inputs(params, x_t)
        result = evaluate(graph, inputs)
        pred = unwrap_number(result)
        total = total + pred ** 2
    return total


def _extract_grad_vector(params: dict[str, nn.Parameter]) -> torch.Tensor:
    """Concatenate all parameter gradients into a single vector."""
    grads = []
    for name in sorted(params.keys()):
        g = params[name].grad
        if g is None:
            grads.append(torch.tensor(0.0))
        else:
            grads.append(g.flatten())
    return torch.cat(grads)


# ── Per-program gradient comparison ─────────────────────────────────────────

def compare_gradients_one_trial(
    spec: ProgramSpec,
    direct_graph,
    dmci_graph,
    seed: int,
    x_values: list[float],
) -> dict:
    """Run one trial: compute gradients for both modes, return metrics."""

    # Same random parameters for both modes
    params_direct = _make_random_params(spec.param_names, seed)
    params_dmci = _make_random_params(spec.param_names, seed)

    # --- Direct compilation ---
    loss_direct = _forward_sum(direct_graph, params_direct, x_values)
    loss_direct.backward()
    grad_direct = _extract_grad_vector(params_direct)

    # --- DMCI (compiled interpreter) ---
    loss_dmci = _forward_sum(dmci_graph, params_dmci, x_values)
    loss_dmci.backward()
    grad_dmci = _extract_grad_vector(params_dmci)

    # --- Metrics ---
    diff = grad_dmci - grad_direct
    direct_norm = grad_direct.norm().item()

    if direct_norm < 1e-12:
        # Avoid division by zero; use absolute error instead
        rel_error = diff.norm().item()
    else:
        rel_error = diff.norm().item() / direct_norm

    # Cosine similarity
    if direct_norm < 1e-12 or grad_dmci.norm().item() < 1e-12:
        cos_sim = 1.0 if diff.norm().item() < 1e-12 else 0.0
    else:
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_direct.unsqueeze(0), grad_dmci.unsqueeze(0)
        ).item()

    return {
        "seed": seed,
        "loss_direct": loss_direct.item(),
        "loss_dmci": loss_dmci.item(),
        "grad_direct_norm": direct_norm,
        "grad_dmci_norm": grad_dmci.norm().item(),
        "grad_diff_norm": diff.norm().item(),
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
        "grad_direct": grad_direct.tolist(),
        "grad_dmci": grad_dmci.tolist(),
    }


def run_program(spec: ProgramSpec, n_samples: int, x_values: list[float]):
    """Run gradient comparison for one program across n_samples random seeds."""

    input_names = _all_input_names(spec)

    print(f"\n{'='*60}")
    print(f"Program: {spec.name}")
    print(f"  Source (direct): {spec.direct_source.strip()[:80]}")
    print(f"  Params: {spec.param_names}")
    print(f"  Compiling direct graph...", end=" ", flush=True)

    t0 = time.perf_counter()
    direct_graph = _compile_graph(spec.direct_source, input_names)
    t_direct = time.perf_counter() - t0
    print(f"done ({t_direct:.2f}s)")

    print(f"  Compiling DMCI graph...", end=" ", flush=True)
    t0 = time.perf_counter()
    dmci_graph = _compile_graph(spec.interp_source, input_names)
    t_dmci = time.perf_counter() - t0
    print(f"done ({t_dmci:.2f}s)")

    trials = []
    for i in range(n_samples):
        seed = 10000 + i  # avoid overlap with training seeds
        trial = compare_gradients_one_trial(
            spec, direct_graph, dmci_graph, seed, x_values
        )
        trials.append(trial)

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  Trial {i+1:3d}/{n_samples}: "
                f"rel_err={trial['relative_error']:.2e}  "
                f"cos_sim={trial['cosine_similarity']:.8f}"
            )

    # Aggregate statistics
    rel_errors = [t["relative_error"] for t in trials]
    cos_sims = [t["cosine_similarity"] for t in trials]
    loss_diffs = [
        abs(t["loss_direct"] - t["loss_dmci"]) for t in trials
    ]

    stats = {
        "program": spec.name,
        "n_samples": n_samples,
        "compile_time_direct_s": t_direct,
        "compile_time_dmci_s": t_dmci,
        "relative_error": {
            "mean": sum(rel_errors) / len(rel_errors),
            "max": max(rel_errors),
            "min": min(rel_errors),
            "std": (
                sum((e - sum(rel_errors) / len(rel_errors)) ** 2 for e in rel_errors)
                / len(rel_errors)
            ) ** 0.5,
        },
        "cosine_similarity": {
            "mean": sum(cos_sims) / len(cos_sims),
            "min": min(cos_sims),
            "max": max(cos_sims),
            "std": (
                sum((c - sum(cos_sims) / len(cos_sims)) ** 2 for c in cos_sims)
                / len(cos_sims)
            ) ** 0.5,
        },
        "loss_diff_abs": {
            "mean": sum(loss_diffs) / len(loss_diffs),
            "max": max(loss_diffs),
        },
        "trials": trials,
    }

    return stats


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(all_stats: list[dict]):
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print("GRADIENT DELTA SUMMARY: DMCI vs. Direct Compilation")
    print("=" * 90)

    header = (
        f"{'Program':<22s}  "
        f"{'RelErr Mean':>12s}  {'RelErr Max':>12s}  {'RelErr Std':>12s}  "
        f"{'CosSim Mean':>12s}  {'CosSim Min':>12s}"
    )
    print(header)
    print("-" * 90)

    for stats in all_stats:
        re = stats["relative_error"]
        cs = stats["cosine_similarity"]
        row = (
            f"{stats['program']:<22s}  "
            f"{re['mean']:>12.2e}  {re['max']:>12.2e}  {re['std']:>12.2e}  "
            f"{cs['mean']:>12.8f}  {cs['min']:>12.8f}"
        )
        print(row)

    print("-" * 90)

    # Overall summary
    all_re = [s["relative_error"]["mean"] for s in all_stats]
    all_cs = [s["cosine_similarity"]["mean"] for s in all_stats]
    overall_max_re = max(s["relative_error"]["max"] for s in all_stats)

    print(f"\nOverall mean relative error:   {sum(all_re)/len(all_re):.2e}")
    print(f"Overall max relative error:    {overall_max_re:.2e}")
    print(f"Overall mean cosine similarity: {sum(all_cs)/len(all_cs):.8f}")

    # Verdict
    if overall_max_re < 1e-4:
        print("\nVERDICT: Gradients match to high precision (< 1e-4 relative error).")
        print("DMCI and direct compilation produce numerically equivalent gradients.")
    elif overall_max_re < 1e-2:
        print("\nVERDICT: Gradients match within floating-point tolerance (< 1e-2).")
    else:
        print("\nWARNING: Non-trivial gradient differences detected.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure gradient deltas between DMCI and direct compilation"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of random parameter samples per program (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/exp_a/results",
        help="Directory for output JSON (default: experiments/exp_a/results)"
    )
    args = parser.parse_args()

    # Increase recursion limit for DMCI with recursive programs
    sys.setrecursionlimit(5000)

    x_values = [0.5, 1.0, 1.5, 2.0, 2.5]

    print("Gradient Delta Analysis: DMCI vs. Direct Compilation")
    print(f"  n_samples = {args.n_samples}")
    print(f"  x_values  = {x_values}")
    print(f"  output    = {args.output_dir}/gradient_delta.json")
    print(f"  programs  = {[p.name for p in ALL_PROGRAMS]}")

    all_stats = []
    t_total_start = time.perf_counter()

    for spec in ALL_PROGRAMS:
        stats = run_program(spec, args.n_samples, x_values)
        all_stats.append(stats)

    t_total = time.perf_counter() - t_total_start

    print_summary(all_stats)
    print(f"\nTotal wall time: {t_total:.1f}s")

    # Save JSON results (without per-trial raw gradient vectors to keep size reasonable)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "gradient_delta.json")

    # Create a compact version without the full gradient vectors
    compact_stats = []
    for stats in all_stats:
        compact = dict(stats)
        compact["trials"] = [
            {k: v for k, v in t.items() if k not in ("grad_direct", "grad_dmci")}
            for t in stats["trials"]
        ]
        compact_stats.append(compact)

    output = {
        "experiment": "gradient_delta",
        "description": "Gradient comparison between DMCI and direct compilation",
        "n_samples": args.n_samples,
        "x_values": x_values,
        "total_wall_time_s": t_total,
        "programs": compact_stats,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
