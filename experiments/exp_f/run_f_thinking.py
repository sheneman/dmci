############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_f_thinking.py: Experiment F, thinking-mode re-run (concurrent, multi-provider). Differences from the original `exp_f`: -...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment F, thinking-mode re-run (concurrent, multi-provider).

Differences from the original `exp_f`:
  - reasoning/thinking ENABLED (original used `/nothink`); see `llm_providers`;
  - configurable model/provider — qwen/qwen3.6-27b (MindRouter) and gpt-5.5 (OpenAI);
  - `max_completion_tokens=32768` so the model has room to reason;
  - the 12 (target × seed) discovery runs execute CONCURRENTLY (8 threads; LLM calls overlap,
    DMCI work serialized under a lock);
  - the per-iteration DMCI fit is BATCHED (one interpreter walk over all 64 points per epoch,
    bit-identical to the sequential fit, seconds instead of minutes);
  - the discovery loop keeps a BEST-of-iterations checkpoint and reports that (so a run that
    finds a good model mid-loop and then drifts is scored by its best, not its last).

Each discovery run is otherwise the original loop: LLM proposes a Scheme model from the data,
DMCI compiles + fits its parameters, residuals are fed back, repeat up to `max_iterations`.

Run on HPC:  python -m experiments.exp_f.run_f_thinking --specs qwen27b_think gpt55_think
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import TARGETS, N_TARGETS, ExpFConfig, DEFAULT
from .exp_f import (
    FitResult, IterationResult, DiscoveryResult, generate_data,
    validate_expression, get_predictions, analyze_residuals,
    format_residual_summary, build_dmci_source, detect_used_params)
from .llm_client import (
    make_initial_prompt, make_refinement_prompt, make_retry_prompt,
    extract_scheme, SYSTEM_PROMPT)
from .llm_providers import LLMSpec, SPECS, complete

# The 12 discovery runs share this process via threads (so the slow LLM calls overlap). DMCI
# compile/evaluate has process-global state (symbol table) and isn't safe to run concurrently,
# so all DMCI work is serialized under this lock — cheap, since the LLM round-trips (minutes,
# outside the lock) dominate while the DMCI fits (seconds) do not.
_DMCI_LOCK = threading.Lock()


def _fit_batched(expression, param_names, xs, ys, cfg, seed) -> FitResult:
    """Batched DMCI fit: one interpreter walk over all points per epoch (bit-identical to the
    sequential per-point loop in exp_f.fit_parameters, but vectorized)."""
    used = detect_used_params(expression, param_names)
    t0 = time.perf_counter()
    source = build_dmci_source(expression, used)
    graph = compile_program(source, inputs={n: None for n in ["x"] + used}, prelude=True)
    t_compile = time.perf_counter() - t0

    torch.manual_seed(seed + 1000)
    params = {n: nn.Parameter(torch.tensor(1.0) + 0.5 * torch.randn(1).squeeze())
              for n in used}
    opt = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    x_tagged = make_float(xs)
    n = len(xs)

    t_train = time.perf_counter()
    best_loss = float("inf")
    best_params = {k: p.item() for k, p in params.items()}
    patience = 0
    epoch = 0
    for epoch in range(cfg.max_epochs):
        tagged = {"x": x_tagged}
        for k, p in params.items():
            tagged[k] = make_float(p)
        try:
            preds = unwrap_number(evaluate(graph, tagged))
            total = ((preds - ys) ** 2).sum()
        except Exception:
            total = torch.tensor(float(1e6 * n))
        loss_val = (total / n).item()
        snap = {k: p.item() for k, p in params.items()}
        opt.zero_grad()
        if torch.isfinite(total):
            try:
                total.backward()
                if all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in params.values()):
                    torch.nn.utils.clip_grad_norm_(list(params.values()), 10.0)
                    opt.step()
            except Exception:
                pass
        if loss_val < best_loss:
            best_loss, best_params, patience = loss_val, snap, 0
        else:
            patience += 1
        if best_loss < cfg.convergence_threshold or patience > cfg.early_stop_patience:
            break

    return FitResult(
        expression=expression, used_params=used, fitted_values=best_params,
        final_mse=best_loss, n_epochs=epoch + 1, t_compile=t_compile,
        t_train=time.perf_counter() - t_train, converged=best_loss < cfg.mse_threshold,
        loss_history=[])


def _fit_multistart(expression, param_names, xs, ys, cfg, seed, n_starts=48) -> FitResult:
    """Exp-I-style fitter: scipy L-BFGS-B with exact DMCI gradients (torch.autograd.grad),
    from many BROADLY-spread initializations. The original Adam fit inits all params near 1.0
    and so never explores a sine frequency like d~=3 (it locks into a wrong-frequency basin);
    broad multi-start covers those basins. Params optimized in raw (linear) space — F's
    amplitudes can be negative, so no log-reparam (unlike Exp I's all-positive case)."""
    import numpy as np
    from scipy.optimize import minimize

    used = detect_used_params(expression, param_names)
    t0 = time.perf_counter()
    source = build_dmci_source(expression, used)
    graph = compile_program(source, inputs={n: None for n in ["x"] + used}, prelude=True)
    t_compile = time.perf_counter() - t0
    x_tagged = make_float(xs)
    k = len(used)

    def loss_grad(theta_np):
        raw = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
        tagged = {"x": x_tagged}
        for i, nm in enumerate(used):
            tagged[nm] = make_float(raw[i])
        try:
            preds = unwrap_number(evaluate(graph, tagged))
            loss = ((preds - ys) ** 2).mean()
            g = torch.autograd.grad(loss, raw)[0]
            return float(loss), g.detach().numpy().astype(np.float64)
        except Exception:
            return 1e6, np.zeros(k, dtype=np.float64)

    rng = np.random.default_rng(seed)
    t_train = time.perf_counter()
    best_loss, best = float("inf"), None
    for _s in range(n_starts):
        # broad init spans small amplitudes AND large frequencies (the F3 failure mode)
        x0 = rng.uniform(-6.0, 6.0, size=k)
        try:
            r = minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                         options={"maxiter": 300})
            if r.fun < best_loss:
                best_loss, best = float(r.fun), r.x
        except Exception:
            pass

    fitted = ({nm: float(best[i]) for i, nm in enumerate(used)} if best is not None
              else {nm: 0.0 for nm in used})
    return FitResult(
        expression=expression, used_params=used, fitted_values=fitted,
        final_mse=best_loss, n_epochs=n_starts, t_compile=t_compile,
        t_train=time.perf_counter() - t_train, converged=best_loss < cfg.mse_threshold,
        loss_history=[])


def _fit_portfolio(expression, param_names, xs, ys, cfg, seed) -> FitResult:
    """Parameterized multi-optimizer portfolio (Adam -> multi-start L-BFGS -> DE, cheap-first
    with held-out selection). final_mse is the winner's HELD-OUT mse; the per-solver breakdown
    and full-data mse are stashed in loss_history for diagnostics."""
    from .portfolio import fit_portfolio
    r = fit_portfolio(expression, param_names, xs, ys, threshold=cfg.mse_threshold, seed=seed)
    return FitResult(
        expression=expression, used_params=r.used_params, fitted_values=r.fitted_values,
        final_mse=r.val_mse, n_epochs=0, t_compile=r.t_compile, t_train=r.t_fit,
        converged=r.converged,
        loss_history=[{"winner": r.winner, "full_mse": r.full_mse,
                       "train_mse": r.train_mse, "per_solver": r.per_solver}])


_FITTERS = {"adam": _fit_batched, "multistart": _fit_multistart, "portfolio": _fit_portfolio}


def _think_prompt(p: str, spec: LLMSpec) -> str:
    # thinking is enabled via the API param, so drop the qwen `/nothink` text switch
    return p.replace(" /nothink", "") if spec.thinking else p


def discover(target_idx: int, seed: int, spec: LLMSpec, cfg: ExpFConfig,
             fitter: str = "adam") -> dict:
    sys.setrecursionlimit(5000)
    fit_fn = _FITTERS[fitter]
    target = TARGETS[target_idx]
    xs, ys = generate_data(target, cfg, seed)

    iterations: list[dict] = []
    prev_fit = prev_ra = None
    best_fit = None
    best_iter = -1
    t0 = time.perf_counter()

    for iteration in range(cfg.max_iterations):
        if iteration == 0 or prev_fit is None:
            prompt = make_initial_prompt(xs, ys)
        else:
            prompt = make_refinement_prompt(
                prev_fit.expression, prev_fit.final_mse, prev_fit.fitted_values,
                format_residual_summary(prev_ra))
        raw = complete(spec, SYSTEM_PROMPT, _think_prompt(prompt, spec))  # LLM call: no lock
        expr = extract_scheme(raw)

        with _DMCI_LOCK:
            valid, error = validate_expression(expr, cfg.param_names, cfg)
        retries = 0
        while not valid and retries < cfg.llm_max_retries:
            retries += 1
            raw = complete(spec, SYSTEM_PROMPT,
                           _think_prompt(make_retry_prompt(expr, error), spec))
            expr = extract_scheme(raw)
            with _DMCI_LOCK:
                valid, error = validate_expression(expr, cfg.param_names, cfg)

        if not valid:
            iterations.append(asdict(IterationResult(
                iteration=iteration, expression=expr, valid=False, error=error,
                fit=None, residual_analysis=None)))
            continue

        with _DMCI_LOCK:                              # all DMCI compile/eval serialized
            fit = fit_fn(expr, cfg.param_names, xs, ys, cfg, seed)
            preds = get_predictions(expr, fit.used_params, fit.fitted_values, xs)
        ra = analyze_residuals(xs, ys, preds)
        iterations.append(asdict(IterationResult(
            iteration=iteration, expression=expr, valid=True, error="",
            fit=asdict(fit), residual_analysis=ra)))
        prev_fit, prev_ra = fit, ra
        if best_fit is None or fit.final_mse < best_fit.final_mse:
            best_fit, best_iter = fit, iteration
        if fit.final_mse < cfg.mse_threshold:
            break

    valid_iters = [it for it in iterations if it.get("valid")]
    total_compile = sum(it["fit"]["t_compile"] for it in valid_iters if it.get("fit"))
    total_train = sum(it["fit"]["t_train"] for it in valid_iters if it.get("fit"))
    final_mse = best_fit.final_mse if best_fit else float("inf")  # BEST-of-iterations
    converged = final_mse < cfg.mse_threshold

    res = asdict(DiscoveryResult(
        target_name=target.name, seed=seed, iterations=iterations,
        n_iterations=len(iterations), converged=converged, final_mse=final_mse,
        total_compile_s=total_compile, total_train_s=total_train,
        total_wall_s=time.perf_counter() - t0, config=asdict(cfg)))
    res["best_iteration"] = best_iter
    res["best_expression"] = best_fit.expression if best_fit else None
    res["llm_label"], res["llm_model"], res["thinking"] = spec.label, spec.model, spec.thinking
    res["fitter"] = fitter
    return res


def _worker(args):
    target_idx, seed, spec, cfg, fitter = args
    return discover(target_idx, seed, spec, cfg, fitter)


def run_spec(spec: LLMSpec, cfg: ExpFConfig, workers: int, output_root: Path,
             fitter: str = "adam", targets=None):
    suffix = "" if fitter == "adam" else f"_{fitter}"
    out_dir = output_root / f"results_{spec.label}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tlist = targets if targets is not None else range(N_TARGETS)
    jobs = [(t, s) for t in tlist for s in range(cfg.n_seeds)]
    print(f"\n=== {spec.label} ({spec.model}, thinking={spec.thinking}, fitter={fitter}) — "
          f"{len(jobs)} runs, {workers}-way concurrent ===", flush=True)

    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, (t, s, spec, cfg, fitter)): (t, s) for t, s in jobs}
        for fut in as_completed(futs):
            t, s = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [{spec.label}] target {t} seed {s} FAILED: {e}", flush=True)
                continue
            (out_dir / f"{res['target_name']}_seed{s:02d}.json").write_text(
                json.dumps(res, indent=2, default=str))
            print(f"  [{spec.label}] {res['target_name']:24s} seed{s} | "
                  f"conv={str(res['converged']):5s} mse={res['final_mse']:.2e} "
                  f"bestIter={res['best_iteration']} iters={res['n_iterations']} "
                  f"wall={res['total_wall_s']:.0f}s", flush=True)
            results.append(res)

    nconv = sum(r["converged"] for r in results)
    print(f"=== {spec.label}: CONVERGED {nconv}/{len(results)} "
          f"in {time.perf_counter()-t0:.0f}s wall ===", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description="Exp F thinking-mode re-run")
    ap.add_argument("--specs", nargs="+", default=["qwen27b_think"],
                    choices=list(SPECS.keys()))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fitter", choices=list(_FITTERS), default="adam")
    ap.add_argument("--targets", type=int, nargs="+", default=None,
                    help="target indices to run (default all)")
    ap.add_argument("--output-root", type=Path, default=Path("experiments/exp_f"))
    args = ap.parse_args()
    for name in args.specs:
        run_spec(SPECS[name], DEFAULT, args.workers, args.output_root,
                 fitter=args.fitter, targets=args.targets)


if __name__ == "__main__":
    main()
