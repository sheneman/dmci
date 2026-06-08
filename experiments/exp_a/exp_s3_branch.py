#!/usr/bin/env python3
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# exp_s3_branch.py: S3: Branch-dependent constant experiment. Addresses reviewer concern about programs where learnable constants...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""S3: Branch-dependent constant experiment.

Addresses reviewer concern about programs where learnable constants affect
conditional branches.  The program under test is:

    (if (< x alpha) (* beta x) (* gamma x))

with learnable constants alpha, beta, gamma (true values: 1.0, 2.0, 0.5).
The comparison is (< x alpha) so that the program matches target_fn (beta*x
for x < alpha, else gamma*x); at the true values it reproduces the target
exactly, making alpha=1 the true loss minimum. alpha still appears only in the
branch condition, so it receives zero gradient and cannot be learned --- which
is the phenomenon this experiment characterizes.

Two experiment parts:
  1. Standard training (direct / DMCI / hand-coded) over 10 seeds.
  2. Basin-of-attraction study: sweep initial alpha while fixing beta, gamma.

Usage:
    python3 -u -m experiments.exp_a.exp_s3_branch --mode train --method direct
    python3 -u -m experiments.exp_a.exp_s3_branch --mode train --method dmci
    python3 -u -m experiments.exp_a.exp_s3_branch --mode train --method handcoded
    python3 -u -m experiments.exp_a.exp_s3_branch --mode basin
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()

TRUE_ALPHA = 1.0
TRUE_BETA = 2.0
TRUE_GAMMA = 0.5


def target_fn(x: float, alpha: float = TRUE_ALPHA,
              beta: float = TRUE_BETA, gamma: float = TRUE_GAMMA) -> float:
    """Piecewise-linear target: 2x for x<1, 0.5x for x>=1."""
    return beta * x if x < alpha else gamma * x


# ---------------------------------------------------------------------------
# Source strings
# ---------------------------------------------------------------------------

DIRECT_SOURCE = "(if (< x alpha) (* beta x) (* gamma x))"

_ENV_PAIRS = "(cons 'x x) (cons 'alpha alpha) (cons 'beta beta) (cons 'gamma gamma)"
DMCI_SOURCE = (
    EVALUATOR_SOURCE
    + f"\n(scheme-eval '(if (< x alpha) (* beta x) (* gamma x))"
    + f" (list {_ENV_PAIRS}))\n"
)

PARAM_NAMES = ["alpha", "beta", "gamma"]
INPUT_NAMES = ["x"]
ALL_NAMES = INPUT_NAMES + PARAM_NAMES

TARGET_VALUES = {"alpha": TRUE_ALPHA, "beta": TRUE_BETA, "gamma": TRUE_GAMMA}
INIT_VALUES = {"alpha": 0.0, "beta": 1.0, "gamma": 1.0}

# Hand-coded AST for the hand-coded interpreter
HC_EXPR = ["if", ["<", "x", "alpha"],
           ["*", "beta", "x"],
           ["*", "gamma", "x"]]


# ---------------------------------------------------------------------------
# Graph cache
# ---------------------------------------------------------------------------

_GRAPH_CACHE: dict[str, object] = {}


def _get_graph(key: str, source: str):
    if key not in _GRAPH_CACHE:
        inputs = {n: None for n in ALL_NAMES}
        _GRAPH_CACHE[key] = compile_program(source, inputs=inputs, prelude=True)
    return _GRAPH_CACHE[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(seed: int, init_override: dict[str, float] | None = None
                 ) -> dict[str, nn.Parameter]:
    """Create learnable parameters with small random perturbation."""
    torch.manual_seed(seed)
    init = init_override if init_override is not None else INIT_VALUES
    return {
        name: nn.Parameter(
            torch.tensor(init[name])
            + 0.3 * max(abs(init[name]), 0.1) * torch.randn(1).squeeze()
        )
        for name in PARAM_NAMES
    }


def _generate_data(n_points: int = 20, x_lo: float = 0.1, x_hi: float = 3.0):
    xs = torch.linspace(x_lo, x_hi, n_points)
    ys = torch.tensor([target_fn(x.item()) for x in xs])
    return xs, ys


def _build_inputs(params: dict[str, nn.Parameter],
                  x_val: torch.Tensor) -> dict:
    inputs = {"x": make_float(x_val)}
    for name, param in params.items():
        inputs[name] = make_float(param)
    return inputs


def _compute_loss(graph, params, xs, ys):
    total = torch.tensor(0.0)
    for x_val, y_val in zip(xs, ys):
        inputs = _build_inputs(params, x_val)
        result = evaluate(graph, inputs)
        pred = unwrap_number(result)
        total = total + (pred - y_val) ** 2
    return total


def _grad_norm(params: dict[str, nn.Parameter]) -> float:
    grads = [p.grad.flatten() for p in params.values() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat(grads).norm().item()


def _param_errors(params: dict[str, nn.Parameter]) -> dict[str, float]:
    return {name: abs(params[name].item() - TARGET_VALUES[name])
            for name in PARAM_NAMES}


# ---------------------------------------------------------------------------
# Hand-coded interpreter (reused from baselines.py pattern)
# ---------------------------------------------------------------------------

class _HandCodedInterpreter:
    """Minimal tree-walking evaluator for the S3 branch program."""

    def eval_expr(self, expr, env: dict[str, object]) -> torch.Tensor:
        if isinstance(expr, (int, float)):
            return torch.tensor(float(expr))
        if isinstance(expr, str):
            return env[expr]
        if not isinstance(expr, list) or len(expr) == 0:
            return torch.tensor(0.0)

        head = expr[0]

        if head == "if":
            test = self.eval_expr(expr[1], env)
            test_val = test.item() if isinstance(test, torch.Tensor) else test
            if test_val != 0.0 and test_val is not False:
                return self.eval_expr(expr[2], env)
            return self.eval_expr(expr[3], env)

        if head in ("+", "-", "*", "/"):
            a = self.eval_expr(expr[1], env)
            b = self.eval_expr(expr[2], env)
            if head == "+":
                return a + b
            if head == "-":
                return a - b
            if head == "*":
                return a * b
            return a / b

        if head in ("=", "<", ">", "<=", ">="):
            a = self.eval_expr(expr[1], env)
            b = self.eval_expr(expr[2], env)
            a_v = a.item() if isinstance(a, torch.Tensor) else float(a)
            b_v = b.item() if isinstance(b, torch.Tensor) else float(b)
            if head == "<":
                return torch.tensor(1.0) if a_v < b_v else torch.tensor(0.0)
            if head == "=":
                return torch.tensor(1.0) if a_v == b_v else torch.tensor(0.0)
            if head == ">":
                return torch.tensor(1.0) if a_v > b_v else torch.tensor(0.0)
            if head == "<=":
                return torch.tensor(1.0) if a_v <= b_v else torch.tensor(0.0)
            return torch.tensor(1.0) if a_v >= b_v else torch.tensor(0.0)

        return torch.tensor(0.0)


_HC = _HandCodedInterpreter()


def _hc_compute_loss(params, xs, ys):
    total = torch.tensor(0.0)
    env: dict[str, object] = {}
    for x_val, y_val in zip(xs, ys):
        env["x"] = x_val
        for name, param in params.items():
            env[name] = param
        pred = _HC.eval_expr(HC_EXPR, env)
        total = total + (pred - y_val) ** 2
    return total


# ---------------------------------------------------------------------------
# Part 1: Standard training loop
# ---------------------------------------------------------------------------

def run_train(method: str, n_seeds: int, output_dir: str,
              max_epochs: int = 3000, lr: float = 0.01,
              convergence_threshold: float = 1e-6):
    """Train S3 branch program with the specified method over multiple seeds."""

    results = []
    xs, ys = _generate_data(n_points=20, x_lo=0.1, x_hi=3.0)

    # Compile graph once (outside seed loop)
    if method == "direct":
        graph = _get_graph("s3_direct", DIRECT_SOURCE)
    elif method == "dmci":
        graph = _get_graph("s3_dmci", DMCI_SOURCE)
    elif method == "handcoded":
        graph = None  # no compiled graph needed
    else:
        raise ValueError(f"Unknown method: {method}")

    for seed in range(n_seeds):
        print(f"[S3/{method}] seed={seed}", file=sys.stderr)
        params = _make_params(seed)
        optimizer = torch.optim.Adam(list(params.values()), lr=lr)

        loss_history = []
        param_history = {n: [] for n in PARAM_NAMES}
        grad_norm_history = []
        wall_time_history = []
        conv_epoch = None
        t_start = time.perf_counter()

        for epoch in range(max_epochs):
            t0 = time.perf_counter()

            if method == "handcoded":
                loss = _hc_compute_loss(params, xs, ys)
            else:
                loss = _compute_loss(graph, params, xs, ys)

            optimizer.zero_grad()
            loss.backward()
            gn = _grad_norm(params)
            optimizer.step()
            wt = time.perf_counter() - t0

            loss_val = loss.item()
            loss_history.append(loss_val)
            grad_norm_history.append(gn)
            wall_time_history.append(wt)
            for name in PARAM_NAMES:
                param_history[name].append(params[name].item())

            # Early stopping: converge then run 50 more epochs
            if conv_epoch is None and loss_val < convergence_threshold:
                conv_epoch = epoch
            elif conv_epoch is not None and epoch >= conv_epoch + 50:
                break

            # Progress logging
            if epoch % 500 == 0 or epoch == max_epochs - 1:
                pstr = " ".join(f"{n}={params[n].item():.4f}"
                                for n in PARAM_NAMES)
                print(f"  epoch {epoch:5d}  loss={loss_val:.8f}  {pstr}",
                      file=sys.stderr)

        total_wt = time.perf_counter() - t_start
        final_errors = _param_errors(params)

        result = {
            "method": method,
            "seed": seed,
            "converged": conv_epoch is not None,
            "convergence_epoch": conv_epoch,
            "final_loss": loss_history[-1],
            "final_params": {n: params[n].item() for n in PARAM_NAMES},
            "final_param_errors": final_errors,
            "total_wall_time": total_wt,
            "n_epochs": len(loss_history),
            "loss_history": loss_history,
            "param_history": param_history,
            "grad_norm_history": grad_norm_history,
        }
        results.append(result)

        status = "CONV" if conv_epoch is not None else "----"
        print(f"  => {status}  final_loss={loss_history[-1]:.8f}  "
              f"errors={final_errors}  time={total_wt:.1f}s",
              file=sys.stderr)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"s3_branch_train_{method}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}", file=sys.stderr)

    # Save summary (without per-epoch histories)
    summary = []
    for r in results:
        s = {k: v for k, v in r.items()
             if k not in ("loss_history", "param_history", "grad_norm_history")}
        summary.append(s)
    summary_path = os.path.join(output_dir, f"s3_branch_train_{method}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Part 2: Basin-of-attraction study
# ---------------------------------------------------------------------------

def run_basin(output_dir: str, method: str = "direct",
              n_alpha_points: int = 21, alpha_lo: float = -2.0,
              alpha_hi: float = 4.0, max_epochs: int = 1000,
              lr: float = 0.01):
    """Sweep initial alpha to characterize the basin of attraction at alpha=1.0.

    Fixes beta=2.0, gamma=0.5 (at their true values).  Only alpha is learned.
    """
    print(f"[S3/basin] {n_alpha_points} initial alpha values in "
          f"[{alpha_lo}, {alpha_hi}]", file=sys.stderr)

    # Compile graph once
    if method == "direct":
        graph = _get_graph("s3_direct", DIRECT_SOURCE)
    elif method == "dmci":
        graph = _get_graph("s3_dmci", DMCI_SOURCE)
    else:
        graph = None  # handcoded

    xs, ys = _generate_data(n_points=20, x_lo=0.1, x_hi=3.0)
    alpha_inits = torch.linspace(alpha_lo, alpha_hi, n_alpha_points).tolist()

    results = []
    for i, alpha0 in enumerate(alpha_inits):
        # Only alpha is learnable; beta and gamma are fixed at true values
        alpha_param = nn.Parameter(torch.tensor(alpha0))
        beta_param = nn.Parameter(torch.tensor(TRUE_BETA))
        gamma_param = nn.Parameter(torch.tensor(TRUE_GAMMA))
        params = {"alpha": alpha_param, "beta": beta_param, "gamma": gamma_param}

        optimizer = torch.optim.Adam([alpha_param], lr=lr)
        loss_curve = []

        for epoch in range(max_epochs):
            if method == "handcoded":
                loss = _hc_compute_loss(params, xs, ys)
            else:
                loss = _compute_loss(graph, params, xs, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())

        result = {
            "initial_alpha": alpha0,
            "final_alpha": alpha_param.item(),
            "final_loss": loss_curve[-1],
            "loss_curve": loss_curve,
        }
        results.append(result)

        print(f"  [{i+1}/{n_alpha_points}] alpha0={alpha0:+.3f} -> "
              f"alpha_final={alpha_param.item():.4f}  "
              f"loss={loss_curve[-1]:.8f}", file=sys.stderr)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "s3_branch_basin.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}", file=sys.stderr)

    # Save summary (without per-epoch loss curves)
    summary = [{k: v for k, v in r.items() if k != "loss_curve"}
               for r in results]
    summary_path = os.path.join(output_dir, "s3_branch_basin_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="S3: Branch-dependent constant experiment")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "basin"],
                        help="Experiment mode: 'train' or 'basin'")
    parser.add_argument("--method", type=str, default="direct",
                        choices=["direct", "dmci", "handcoded"],
                        help="Optimization method (for train mode, also used "
                             "for basin graph compilation)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (train mode only)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max epochs (default: 3000 for train, "
                             "1000 for basin)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Adam learning rate")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/exp_a/results",
                        help="Output directory for results")
    args = parser.parse_args()

    sys.setrecursionlimit(5000)  # needed for DMCI

    if args.mode == "train":
        max_epochs = args.max_epochs if args.max_epochs is not None else 3000
        run_train(
            method=args.method,
            n_seeds=args.seeds,
            output_dir=args.output_dir,
            max_epochs=max_epochs,
            lr=args.lr,
        )
    elif args.mode == "basin":
        max_epochs = args.max_epochs if args.max_epochs is not None else 1000
        run_basin(
            output_dir=args.output_dir,
            method=args.method,
            max_epochs=max_epochs,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
