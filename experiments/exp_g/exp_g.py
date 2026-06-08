############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# exp_g.py: Experiment G: Runtime Compositional Scientific Modeling. Demonstrates DMCI's unique capability: symbolic...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment G: Runtime Compositional Scientific Modeling.

Demonstrates DMCI's unique capability: symbolic modules are strings,
composition is string manipulation, and the composed program flows through
the same compiled interpreter. Modules can be hot-swapped at runtime with
zero engineering overhead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import (
    PROBLEMS, N_PROBLEMS, MODULE_BY_NAME, MODULE_LIBRARY,
    ExpGConfig, DEFAULT, TestProblem,
)
from .modules import (
    instantiate_module, build_composition, composition_label,
)


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


# -- Helpers -----------------------------------------------------------------

def _make_env(names: list[str]) -> str:
    pairs = " ".join(f"(cons '{n} {n})" for n in names)
    return f"(list {pairs})"


def build_dmci_source(expression: str, param_names: list[str]) -> str:
    all_names = ["x"] + param_names
    env = _make_env(all_names)
    return EVALUATOR_SOURCE + f"\n(scheme-eval '{expression} {env})\n"


def generate_data(problem: TestProblem, cfg: ExpGConfig, seed: int):
    torch.manual_seed(seed)
    xs = torch.linspace(problem.x_range[0], problem.x_range[1],
                        cfg.n_data_points)
    ys_clean = torch.tensor(
        [problem.ground_truth_fn(x.item()) for x in xs])
    ys = ys_clean + cfg.noise_std * torch.randn_like(ys_clean)
    return xs, ys


# -- DMCI fitting ------------------------------------------------------------

@dataclass
class FitResult:
    label: str
    expression: str
    param_names: list[str]
    fitted_values: dict[str, float]
    final_mse: float
    n_epochs: int
    t_compile: float
    t_train: float
    converged: bool


def fit_expression(label: str, expression: str, param_names: list[str],
                   xs, ys, cfg: ExpGConfig, seed: int) -> FitResult:
    t0 = time.perf_counter()
    source = build_dmci_source(expression, param_names)
    all_inputs = {n: None for n in ["x"] + param_names}
    graph = compile_program(source, inputs=all_inputs, prelude=True)
    t_compile = time.perf_counter() - t0

    torch.manual_seed(seed + 2000)
    params = {
        name: nn.Parameter(
            torch.tensor(1.0) + 0.5 * torch.randn(1).squeeze())
        for name in param_names
    }
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)

    t_train_start = time.perf_counter()
    best_loss = float("inf")
    best_params = {n: p.item() for n, p in params.items()}
    patience = 0
    # Batched data: ONE interpreter walk per epoch over all points (v1.1.7 batched DMCI),
    # not len(xs) sequential walks. Drivers are fixed across epochs, so tag them once.
    x_tagged = make_float(xs if isinstance(xs, torch.Tensor) else torch.tensor(xs))
    y_batch = ys if isinstance(ys, torch.Tensor) else torch.stack(
        [y if isinstance(y, torch.Tensor) else torch.tensor(float(y)) for y in ys])
    n_data = len(xs)

    for epoch in range(cfg.max_epochs):
        tagged = {"x": x_tagged}
        for n, p in params.items():
            tagged[n] = make_float(p)
        try:
            preds = unwrap_number(evaluate(graph, tagged))
            total_loss = ((preds - y_batch) ** 2).sum()
        except Exception:
            total_loss = torch.tensor(float(1e6 * n_data))

        mse = total_loss / n_data
        loss_val = mse.item()
        # Snapshot params behind THIS loss before stepping, so fitted_values match
        # the reported best final_mse.
        current_snapshot = {n: p.item() for n, p in params.items()}

        optimizer.zero_grad()
        if torch.isfinite(total_loss):
            try:
                total_loss.backward()
                # Skip the step on any non-finite gradient — clip_grad_norm_ turns
                # an inf grad (pow(0, b<0), div-by-zero) into nan and permanently
                # poisons Adam.
                if all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in params.values()):
                    torch.nn.utils.clip_grad_norm_(list(params.values()), 10.0)
                    optimizer.step()
            except Exception:
                pass

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = current_snapshot
            patience = 0
        else:
            patience += 1

        if best_loss < cfg.convergence_threshold:
            break
        if patience > cfg.early_stop_patience:
            break

    t_train = time.perf_counter() - t_train_start

    return FitResult(
        label=label,
        expression=expression,
        param_names=param_names,
        fitted_values=best_params,
        final_mse=best_loss,
        n_epochs=epoch + 1,
        t_compile=t_compile,
        t_train=t_train,
        converged=best_loss < cfg.convergence_threshold,
    )


# -- Problem runner -----------------------------------------------------------

def wrong_compositions(problem: TestProblem) -> list[tuple[str, str, str]]:
    op, m1, m2 = problem.correct_composition
    wrongs = []
    if op == "sum":
        wrongs.append(("product", m1, m2))
        wrongs.append(("chain", m1, m2))
    elif op == "product":
        wrongs.append(("sum", m1, m2))
        wrongs.append(("chain", m1, m2))
    elif op == "chain":
        wrongs.append(("sum", m1, m2))
        wrongs.append(("chain", m2, m1))
    return wrongs


def pick_swap_module(original: str) -> str:
    alternatives = {
        "exponential_decay": "gaussian",
        "oscillation": "polynomial2",
        "polynomial2": "power_law",
        "sigmoid": "gaussian",
        "power_law": "exponential_decay",
        "gaussian": "sigmoid",
    }
    return alternatives.get(original, "power_law")


def run_problem(problem_idx: int, seed: int,
                cfg: ExpGConfig = DEFAULT) -> dict:
    problem = PROBLEMS[problem_idx]
    xs, ys = generate_data(problem, cfg, seed)

    print(f"\n{'='*60}")
    print(f"Problem: {problem.name} | Seed: {seed}")
    print(f"Target:  {problem.description}")
    print(f"{'='*60}")

    results = {"problem": problem.name, "seed": seed, "config": asdict(cfg)}
    all_fits = []

    # Phase 1: Individual modules
    op, m1_name, m2_name = problem.correct_composition
    for mod_name in [m1_name, m2_name]:
        mod = MODULE_BY_NAME[mod_name]
        expr, params = instantiate_module(mod, prefix="")
        label = f"individual_{mod_name}"
        print(f"\n  [{label}] {expr}")
        fit = fit_expression(label, expr, params, xs, ys, cfg, seed)
        print(f"    MSE={fit.final_mse:.6f} | "
              f"compile={fit.t_compile*1000:.1f}ms | "
              f"train={fit.t_train:.1f}s")
        all_fits.append(asdict(fit))

    # Phase 2: Correct composition
    expr, params = build_composition(op, m1_name, m2_name)
    label = f"correct_{composition_label(op, m1_name, m2_name)}"
    print(f"\n  [{label}] {expr[:80]}...")
    fit = fit_expression(label, expr, params, xs, ys, cfg, seed)
    print(f"    MSE={fit.final_mse:.6f} | "
          f"compile={fit.t_compile*1000:.1f}ms | "
          f"train={fit.t_train:.1f}s")
    all_fits.append(asdict(fit))

    # Phase 3: Wrong compositions
    for wop, wm1, wm2 in wrong_compositions(problem):
        expr, params = build_composition(wop, wm1, wm2)
        label = f"wrong_{composition_label(wop, wm1, wm2)}"
        print(f"\n  [{label}] {expr[:80]}...")
        fit = fit_expression(label, expr, params, xs, ys, cfg, seed)
        print(f"    MSE={fit.final_mse:.6f} | "
              f"compile={fit.t_compile*1000:.1f}ms | "
              f"train={fit.t_train:.1f}s")
        all_fits.append(asdict(fit))

    # Phase 4: Hot-swap demonstration
    print(f"\n  --- Hot-swap demo ---")
    swap_target = m2_name
    swap_alt = pick_swap_module(swap_target)

    # Original composition
    expr_orig, params_orig = build_composition(op, m1_name, m2_name)
    t0 = time.perf_counter()
    source_orig = build_dmci_source(expr_orig, params_orig)
    inputs_orig = {n: None for n in ["x"] + params_orig}
    _ = compile_program(source_orig, inputs=inputs_orig, prelude=True)
    t_compile_orig = time.perf_counter() - t0

    # Swapped composition
    expr_swap, params_swap = build_composition(op, m1_name, swap_alt)
    t0 = time.perf_counter()
    source_swap = build_dmci_source(expr_swap, params_swap)
    inputs_swap = {n: None for n in ["x"] + params_swap}
    _ = compile_program(source_swap, inputs=inputs_swap, prelude=True)
    t_compile_swap = time.perf_counter() - t0

    # Swap back
    t0 = time.perf_counter()
    _ = compile_program(source_orig, inputs=inputs_orig, prelude=True)
    t_compile_back = time.perf_counter() - t0

    swap_label = composition_label(op, m1_name, swap_alt)
    print(f"  Swap {m2_name} -> {swap_alt}")
    print(f"  Compile times: orig={t_compile_orig*1000:.1f}ms, "
          f"swap={t_compile_swap*1000:.1f}ms, "
          f"back={t_compile_back*1000:.1f}ms")

    # Fit the swapped composition
    fit_swap = fit_expression(
        f"hotswap_{swap_label}", expr_swap, params_swap,
        xs, ys, cfg, seed)
    print(f"  Swapped MSE={fit_swap.final_mse:.6f} | "
          f"train={fit_swap.t_train:.1f}s")
    all_fits.append(asdict(fit_swap))

    hot_swap = {
        "original_module": m2_name,
        "swapped_module": swap_alt,
        "compile_times_ms": [
            t_compile_orig * 1000,
            t_compile_swap * 1000,
            t_compile_back * 1000,
        ],
        "mean_compile_ms": (
            t_compile_orig + t_compile_swap + t_compile_back) / 3 * 1000,
        "swapped_mse": fit_swap.final_mse,
    }

    results["fits"] = all_fits
    results["hot_swap"] = hot_swap
    results["total_unique_programs"] = len(all_fits)

    return results


def save_result(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result['problem']}_seed{result['seed']:02d}"
    path = output_dir / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment G")
    parser.add_argument("--problem", type=int, required=True,
                        help=f"Problem index 0-{N_PROBLEMS - 1}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/exp_g/results"))
    args = parser.parse_args()

    if args.problem < 0 or args.problem >= N_PROBLEMS:
        print(f"Invalid problem {args.problem}")
        sys.exit(1)

    sys.setrecursionlimit(5000)
    result = run_problem(args.problem, args.seed)
    save_result(result, args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {result['problem']} seed={result['seed']}")
    print(f"  Unique programs tested: {result['total_unique_programs']}")
    for fit in result["fits"]:
        print(f"  {fit['label']:40s} MSE={fit['final_mse']:.6f} "
              f"compile={fit['t_compile']*1000:.1f}ms "
              f"train={fit['t_train']:.1f}s")
    hs = result["hot_swap"]
    print(f"  Hot-swap compile: {hs['mean_compile_ms']:.1f}ms avg")


if __name__ == "__main__":
    main()
