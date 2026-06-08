############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# exp_f.py: Experiment F: LLM-in-the-Loop Scientific Model Discovery. Demonstrates DMCI's unique capability: any...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment F: LLM-in-the-Loop Scientific Model Discovery.

Demonstrates DMCI's unique capability: any LLM-generated Scheme program is
automatically differentiable through the compiled interpreter. An iterative
loop of LLM proposal -> DMCI fitting -> residual analysis -> LLM refinement
converges to the correct model without manual implementation or per-program
verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import TARGETS, N_TARGETS, ExpFConfig, DEFAULT
from .llm_client import (
    call_llm, extract_scheme, format_data_for_llm,
    make_initial_prompt, make_refinement_prompt, make_retry_prompt,
)


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


# -- Data generation ---------------------------------------------------------

def generate_data(target, cfg: ExpFConfig, seed: int):
    torch.manual_seed(seed)
    xs = torch.linspace(cfg.x_lo, cfg.x_hi, cfg.n_data_points)
    ys_clean = torch.tensor(
        [target.ground_truth_fn(x.item()) for x in xs])
    ys = ys_clean + cfg.noise_std * torch.randn_like(ys_clean)
    return xs, ys


# -- Residual analysis -------------------------------------------------------

def analyze_residuals(xs, ys, predictions) -> dict:
    residuals = (ys - predictions).numpy()
    x_np = xs.numpy()
    n = len(residuals)

    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals))
    max_abs = float(np.max(np.abs(residuals)))

    if std_r > 1e-10:
        slope = float(np.polyfit(x_np, residuals, 1)[0])
    else:
        slope = 0.0

    centered = residuals - mean_r
    if np.std(centered) > 1e-10:
        ac1 = float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
    else:
        ac1 = 0.0

    fft_vals = np.fft.rfft(residuals)
    magnitudes = np.abs(fft_vals[1:])
    if len(magnitudes) > 0 and np.max(magnitudes) > std_r * 0.5:
        peak_idx = int(np.argmax(magnitudes))
        x_span = float(x_np[-1] - x_np[0])
        dx = x_span / n
        freq = (peak_idx + 1) / (n * dx)
    else:
        freq = 0.0

    parts = []
    if abs(slope) > 0.05 * std_r:
        parts.append(
            f"linear trend in residuals (slope={slope:.4f})")
    if abs(ac1) > 0.5:
        parts.append(
            f"strong autocorrelation (lag-1 r={ac1:.3f}) suggesting "
            f"systematic structure")
    if freq > 0:
        parts.append(
            f"dominant oscillatory component at frequency ~{freq:.2f} Hz")
    if not parts:
        parts.append("residuals appear random (no systematic pattern)")
    pattern = "; ".join(parts)

    return {
        "residual_mean": mean_r,
        "residual_std": std_r,
        "max_abs_residual": max_abs,
        "trend_slope": slope,
        "autocorrelation_lag1": ac1,
        "dominant_frequency": freq,
        "systematic_pattern": pattern,
    }


def format_residual_summary(ra: dict) -> str:
    return (
        f"  Mean residual: {ra['residual_mean']:.4f}\n"
        f"  Residual std:  {ra['residual_std']:.4f}\n"
        f"  Max |residual|: {ra['max_abs_residual']:.4f}\n"
        f"  Trend slope:   {ra['trend_slope']:.4f}\n"
        f"  Autocorrelation (lag-1): {ra['autocorrelation_lag1']:.3f}\n"
        f"  Dominant frequency: {ra['dominant_frequency']:.2f} Hz\n"
        f"  Pattern: {ra['systematic_pattern']}"
    )


# -- DMCI source construction ------------------------------------------------

def _make_env(names: list[str]) -> str:
    pairs = " ".join(f"(cons '{n} {n})" for n in names)
    return f"(list {pairs})"


def build_dmci_source(expression: str, used_params: list[str]) -> str:
    all_names = ["x"] + used_params
    env = _make_env(all_names)
    return EVALUATOR_SOURCE + f"\n(scheme-eval '{expression} {env})\n"


def detect_used_params(expression: str, param_names: list[str]) -> list[str]:
    import re
    used = []
    for p in param_names:
        if re.search(rf'\b{p}\b', expression):
            used.append(p)
    return used


# -- Validation --------------------------------------------------------------

# Operator arities for the supported primitives. scheme-eval reads only the
# first two args of +,-,*,/ and pow, so a form like (* a b c) does NOT error —
# it silently computes a*b and wastes the whole fit. We reject wrong arity here.
_OP_ARITY = {
    "+": 2, "-": 2, "*": 2, "/": 2, "pow": 2,
    "exp": 1, "log": 1, "sin": 1, "cos": 1, "sqrt": 1, "abs": 1,
}


def _parse_sexp(s: str):
    """Parse a Scheme expression into nested lists/atoms; raise ValueError if malformed."""
    tokens = s.replace("(", " ( ").replace(")", " ) ").split()
    if not tokens:
        raise ValueError("empty expression")
    pos = 0

    def parse_expr():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of expression")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst = []
            while pos < len(tokens) and tokens[pos] != ")":
                lst.append(parse_expr())
            if pos >= len(tokens):
                raise ValueError("unbalanced parentheses")
            pos += 1  # consume ')'
            return lst
        if tok == ")":
            raise ValueError("unexpected ')'")
        return tok

    tree = parse_expr()
    if pos != len(tokens):
        raise ValueError("trailing tokens after expression")
    return tree


def _check_arity(tree) -> str:
    """Return '' if every operator call has the right arity, else an error message."""
    if not isinstance(tree, list):
        return ""
    if not tree:
        return "empty application ()"
    head, operands = tree[0], tree[1:]
    if isinstance(head, str) and head in _OP_ARITY:
        want = _OP_ARITY[head]
        if len(operands) != want:
            return (f"operator '{head}' expects {want} args, "
                    f"got {len(operands)}")
    for sub in operands:
        err = _check_arity(sub)
        if err:
            return err
    return ""


def validate_expression(expression: str, param_names: list[str],
                        cfg: ExpFConfig) -> tuple[bool, str]:
    try:
        # Structural checks first: malformed parens or wrong operator arity.
        tree = _parse_sexp(expression)
        arity_err = _check_arity(tree)
        if arity_err:
            return False, arity_err

        used = detect_used_params(expression, param_names)
        if not used:
            return False, "No learnable parameters found in expression"
        source = build_dmci_source(expression, used)
        all_inputs = {n: None for n in ["x"] + used}
        graph = compile_program(source, inputs=all_inputs, prelude=True)

        # Evaluate across the actual data domain, not just x=1: pow/division can
        # be finite at x=1 yet blow up at the x_lo=0 endpoint.
        test_xs = [cfg.x_lo, 0.5 * (cfg.x_lo + cfg.x_hi), cfg.x_hi, 1.0]
        for xv in test_xs:
            tagged = {"x": make_float(torch.tensor(float(xv)))}
            for p in used:
                tagged[p] = make_float(torch.tensor(1.0))
            val = unwrap_number(evaluate(graph, tagged))
            if torch.isnan(val) or torch.isinf(val):
                return False, f"Expression produces NaN/Inf at x={xv:g}"
        return True, ""
    except Exception as e:
        return False, str(e)


# -- DMCI fitting -------------------------------------------------------------

@dataclass
class FitResult:
    expression: str
    used_params: list[str]
    fitted_values: dict[str, float]
    final_mse: float
    n_epochs: int
    t_compile: float
    t_train: float
    converged: bool
    loss_history: list[float]


def fit_parameters(expression: str, param_names: list[str],
                   xs, ys, cfg: ExpFConfig, seed: int) -> FitResult:
    used = detect_used_params(expression, param_names)

    t0 = time.perf_counter()
    source = build_dmci_source(expression, used)
    all_inputs = {n: None for n in ["x"] + used}
    graph = compile_program(source, inputs=all_inputs, prelude=True)
    t_compile = time.perf_counter() - t0

    torch.manual_seed(seed + 1000)
    params = {
        name: nn.Parameter(
            torch.tensor(1.0) + 0.5 * torch.randn(1).squeeze())
        for name in used
    }
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)

    t_train_start = time.perf_counter()
    loss_history = []
    best_loss = float("inf")
    best_params = {n: p.item() for n, p in params.items()}
    patience = 0

    for epoch in range(cfg.max_epochs):
        total_loss = torch.tensor(0.0)
        for i in range(len(xs)):
            tagged = {"x": make_float(xs[i])}
            for n, p in params.items():
                tagged[n] = make_float(p)
            try:
                result = evaluate(graph, tagged)
                pred = unwrap_number(result)
                total_loss = total_loss + (pred - ys[i]) ** 2
            except Exception:
                total_loss = total_loss + torch.tensor(1e6)

        mse = total_loss / len(xs)
        loss_val = mse.item()
        # Snapshot the params that produced THIS loss, before the optimizer step,
        # so reported fitted_values match the reported best final_mse.
        current_snapshot = {n: p.item() for n, p in params.items()}

        optimizer.zero_grad()
        if torch.isfinite(total_loss):
            try:
                total_loss.backward()
                # Skip the step on any non-finite gradient. A single inf/nan grad
                # (pow(0, b<0), div-by-zero) is otherwise turned into nan by
                # clip_grad_norm_ and permanently poisons Adam's moment buffers.
                if all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in params.values()):
                    torch.nn.utils.clip_grad_norm_(list(params.values()), 10.0)
                    optimizer.step()
            except Exception:
                pass

        loss_history.append(loss_val)

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
        expression=expression,
        used_params=used,
        fitted_values=best_params,
        final_mse=best_loss,
        n_epochs=len(loss_history),
        t_compile=t_compile,
        t_train=t_train,
        converged=best_loss < cfg.mse_threshold,
        loss_history=loss_history[::max(1, len(loss_history) // 50)],
    )


def get_predictions(expression: str, used_params: list[str],
                    fitted_values: dict[str, float], xs) -> torch.Tensor:
    source = build_dmci_source(expression, used_params)
    all_inputs = {n: None for n in ["x"] + used_params}
    graph = compile_program(source, inputs=all_inputs, prelude=True)

    preds = []
    with torch.no_grad():
        for i in range(len(xs)):
            tagged = {"x": make_float(xs[i])}
            for n in used_params:
                tagged[n] = make_float(torch.tensor(fitted_values[n]))
            result = evaluate(graph, tagged)
            preds.append(unwrap_number(result).item())
    return torch.tensor(preds)


# -- Discovery loop -----------------------------------------------------------

@dataclass
class IterationResult:
    iteration: int
    expression: str
    valid: bool
    error: str
    fit: FitResult | None
    residual_analysis: dict | None


@dataclass
class DiscoveryResult:
    target_name: str
    seed: int
    iterations: list[dict]
    n_iterations: int
    converged: bool
    final_mse: float
    total_compile_s: float
    total_train_s: float
    total_wall_s: float
    config: dict


def run_discovery(target_idx: int, seed: int,
                  cfg: ExpFConfig = DEFAULT) -> DiscoveryResult:
    target = TARGETS[target_idx]
    xs, ys = generate_data(target, cfg, seed)

    print(f"\n{'='*60}")
    print(f"Target: {target.name} | Seed: {seed}")
    print(f"{'='*60}")

    t_wall_start = time.perf_counter()
    iterations = []
    prev_fit = None
    prev_ra = None

    for iteration in range(cfg.max_iterations):
        print(f"\n--- Iteration {iteration} ---")

        # Build prompt
        if iteration == 0 or prev_fit is None:
            prompt = make_initial_prompt(xs, ys)
        else:
            prompt = make_refinement_prompt(
                prev_fit.expression, prev_fit.final_mse,
                prev_fit.fitted_values,
                format_residual_summary(prev_ra))

        # Call LLM
        raw = call_llm(prompt, target.name, seed, iteration,
                       temperature=cfg.llm_temperature,
                       max_tokens=cfg.llm_max_tokens)
        expression = extract_scheme(raw)
        print(f"  LLM proposed: {expression}")

        # Validate (with retries)
        valid, error = validate_expression(expression, cfg.param_names, cfg)
        retries = 0
        while not valid and retries < cfg.llm_max_retries:
            retries += 1
            print(f"  Invalid ({error}), retry {retries}...")
            retry_raw = call_llm(
                make_retry_prompt(expression, error),
                target.name, seed, iteration * 100 + retries,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens)
            expression = extract_scheme(retry_raw)
            print(f"  Retry proposed: {expression}")
            valid, error = validate_expression(
                expression, cfg.param_names, cfg)

        if not valid:
            print(f"  FAILED: {error}")
            iterations.append(asdict(IterationResult(
                iteration=iteration, expression=expression,
                valid=False, error=error, fit=None,
                residual_analysis=None)))
            continue

        # Fit parameters via DMCI
        print(f"  Fitting parameters...")
        fit = fit_parameters(expression, cfg.param_names, xs, ys, cfg, seed)
        print(f"  MSE={fit.final_mse:.6f} | "
              f"compile={fit.t_compile*1000:.1f}ms | "
              f"train={fit.t_train:.1f}s | "
              f"epochs={fit.n_epochs}")
        print(f"  Params: {fit.fitted_values}")

        # Residual analysis
        predictions = get_predictions(
            expression, fit.used_params, fit.fitted_values, xs)
        ra = analyze_residuals(xs, ys, predictions)
        print(f"  Residuals: {ra['systematic_pattern']}")

        iterations.append(asdict(IterationResult(
            iteration=iteration, expression=expression,
            valid=True, error="",
            fit=asdict(fit), residual_analysis=ra)))

        prev_fit = fit
        prev_ra = ra

        if fit.final_mse < cfg.mse_threshold:
            print(f"  CONVERGED at iteration {iteration}")
            break

    t_wall = time.perf_counter() - t_wall_start

    valid_iters = [it for it in iterations if it.get("valid", False)]
    total_compile = sum(
        it["fit"]["t_compile"] for it in valid_iters if it.get("fit"))
    total_train = sum(
        it["fit"]["t_train"] for it in valid_iters if it.get("fit"))
    final_mse = (valid_iters[-1]["fit"]["final_mse"]
                 if valid_iters and valid_iters[-1].get("fit")
                 else float("inf"))
    converged = final_mse < cfg.mse_threshold

    return DiscoveryResult(
        target_name=target.name,
        seed=seed,
        iterations=iterations,
        n_iterations=len(iterations),
        converged=converged,
        final_mse=final_mse,
        total_compile_s=total_compile,
        total_train_s=total_train,
        total_wall_s=t_wall,
        config=asdict(cfg),
    )


def save_result(result: DiscoveryResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result.target_name}_seed{result.seed:02d}"
    path = output_dir / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment F")
    parser.add_argument("--target", type=int, required=True,
                        help=f"Target index 0-{N_TARGETS - 1}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/exp_f/results"))
    args = parser.parse_args()

    if args.target < 0 or args.target >= N_TARGETS:
        print(f"Invalid target {args.target}, must be 0-{N_TARGETS - 1}")
        sys.exit(1)

    sys.setrecursionlimit(5000)
    result = run_discovery(args.target, args.seed)
    save_result(result, args.output_dir)

    print(f"\nSummary: {result.target_name} seed={result.seed}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Converged:  {result.converged}")
    print(f"  Final MSE:  {result.final_mse:.6f}")
    print(f"  Compile:    {result.total_compile_s*1000:.1f}ms total")
    print(f"  Train:      {result.total_train_s:.1f}s total")
    print(f"  Wall:       {result.total_wall_s:.1f}s total")


if __name__ == "__main__":
    main()
