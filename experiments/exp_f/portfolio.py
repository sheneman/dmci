############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# portfolio.py: A parameterized multi-optimizer portfolio for fitting an arbitrary DMCI-compiled program. Motivation (from...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""A parameterized multi-optimizer portfolio for fitting an arbitrary DMCI-compiled program.

Motivation (from Experiments F and I): different program structures induce different loss
landscapes, and no single optimizer dominates. Smooth structures (F1/F4) fit instantly with
Adam; a sine-frequency structure (F3) is multimodal and needs broad multi-start L-BFGS; rugged
or high-dimensional landscapes (Exp I, 18-24 params) favor a derivative-free global method. Since
an LLM emits the structure at runtime, you cannot hand-pick the solver — so run a *portfolio* and
keep the best.

Design:
  - compile the DMCI graph ONCE per structure; all solvers share it (the structure is the cost);
  - CHEAP-FIRST CASCADE with early exit: ordered solvers, stop when one clears threshold, so easy
    structures pay almost nothing and only hard ones escalate;
  - HELD-OUT SELECTION: split train/val, score every solver on validation, pick the winner by VAL
    loss and judge convergence on val (the Exp I lesson — picking by training loss selects the
    most-overfit solver);
  - each solver is a parameterized `Solver` (init range, restarts, lr/iters, optional log-reparam
    for all-positive params), so the portfolio is fully configurable;
  - DIAGNOSTIC: returns per-solver train/val loss and the winner, distinguishing a hard landscape
    (some solver succeeds) from a wrong/unidentifiable structure (all fail).

Reusable beyond F: the only DMCI-specific piece is `_compiled_loss` (compile + batched evaluate);
the solvers operate on plain (loss, grad) closures.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

import numpy as np
import torch

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number
from .exp_f import build_dmci_source, detect_used_params


@dataclass
class Solver:
    name: str
    kind: str                       # "adam" | "lbfgs" | "de"
    restarts: int = 1               # adam/lbfgs: number of random starts
    init_low: float = -6.0          # uniform init / DE box lower bound
    init_high: float = 6.0          # uniform init / DE box upper bound
    reparam_positive: bool = False  # optimize in log-space (params constrained > 0); Exp-I style
    # adam
    lr: float = 0.05
    epochs: int = 1500
    patience: int = 150
    # lbfgs
    maxiter: int = 300
    # differential evolution
    de_maxiter: int = 120
    de_popsize: int = 15


# GIL NOTE: the solvers are CPU-bound (DMCI eval is Python-level and holds the GIL), so running
# them in THREADS interleaves rather than parallelizes — concurrent wall-time ~= sum, not max.
# (Thread concurrency pays off for the I/O-bound LLM calls in the discovery loop, not here.) So
# the default portfolio is the two fast, gradient-based solvers that cover F's landscapes: local
# Adam (smooth: F1/F4) + broad multi-start L-BFGS (multimodal: F3 frequency). Differential
# evolution is EXPENSIVE (thousands of evals/fit) and only earns its cost on rugged / high-dim
# landscapes (e.g. Exp I, 18-24 params) — it's opt-in via DEEP_PORTFOLIO, not the default.
DEFAULT_PORTFOLIO = [
    Solver("adam_local", "adam", restarts=4, init_low=-2.0, init_high=2.0, lr=0.05, epochs=1500),
    Solver("lbfgs_ms", "lbfgs", restarts=48, init_low=-6.0, init_high=6.0, maxiter=300),
]

DE_SOLVER = Solver("de", "de", init_low=-8.0, init_high=8.0, de_maxiter=120, de_popsize=15)
DEEP_PORTFOLIO = DEFAULT_PORTFOLIO + [DE_SOLVER]   # add DE for rugged / high-dim problems


@dataclass
class SolverResult:
    name: str
    train_mse: float
    val_mse: float
    theta: list
    seconds: float


@dataclass
class PortfolioResult:
    expression: str
    used_params: list
    fitted_values: dict
    val_mse: float            # held-out MSE of the winner (the selection/convergence metric)
    train_mse: float
    full_mse: float           # MSE over all data (comparable to the single-Adam fitter)
    converged: bool           # val_mse < threshold
    winner: str
    t_compile: float
    t_fit: float
    per_solver: list          # list of SolverResult dicts (diagnostic)


def _compiled_loss(expression, used):
    """Compile the structure once; return (graph, eval_fn) where eval_fn(x_tagged, y, raw_tensor)
    -> torch scalar MSE. `raw` is the parameter tensor (already de-reparameterized by the caller)."""
    source = build_dmci_source(expression, used)
    graph = compile_program(source, inputs={n: None for n in ["x"] + used}, prelude=True)

    def eval_mse(x_tagged, y, raw):
        tagged = {"x": x_tagged}
        for i, nm in enumerate(used):
            tagged[nm] = make_float(raw[i])
        preds = unwrap_number(evaluate(graph, tagged))
        return ((preds - y) ** 2).mean()

    return graph, eval_mse


def _transform(raw, positive):
    return torch.exp(raw) if positive else raw


# --- individual solvers: each returns (best_train_mse, best_theta_np) -------

def _solve_adam(eval_mse, xtr, ytr, k, sv, rng):
    best_loss, best = float("inf"), None
    for _r in range(sv.restarts):
        z = torch.tensor(rng.uniform(sv.init_low, sv.init_high, size=k),
                         dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([z], lr=sv.lr)
        run_best, run_theta, patience = float("inf"), z.detach().clone(), 0
        for _e in range(sv.epochs):
            loss = eval_mse(xtr, ytr, _transform(z, sv.reparam_positive))
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                if z.grad is not None and torch.isfinite(z.grad).all():
                    torch.nn.utils.clip_grad_norm_([z], 10.0)
                    opt.step()
            lv = float(loss)
            if lv < run_best - 1e-9:
                run_best, run_theta, patience = lv, z.detach().clone(), 0
            else:
                patience += 1
            if patience > sv.patience:
                break
        if run_best < best_loss:
            best_loss = run_best
            best = _transform(run_theta, sv.reparam_positive).numpy().astype(np.float64)
    return best_loss, best


def _solve_lbfgs(eval_mse, xtr, ytr, k, sv, rng):
    from scipy.optimize import minimize

    def lg(theta_np):
        z = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
        loss = eval_mse(xtr, ytr, _transform(z, sv.reparam_positive))
        if not torch.isfinite(loss):
            return 1e6, np.zeros(k, dtype=np.float64)
        g = torch.autograd.grad(loss, z)[0]
        return float(loss.detach()), g.detach().numpy().astype(np.float64)

    best_loss, best = float("inf"), None
    for _r in range(sv.restarts):
        x0 = rng.uniform(sv.init_low, sv.init_high, size=k)
        try:
            r = minimize(lg, x0, jac=True, method="L-BFGS-B", options={"maxiter": sv.maxiter})
            if r.fun < best_loss:
                z = torch.tensor(r.x, dtype=torch.float32)
                best_loss = float(r.fun)
                best = _transform(z, sv.reparam_positive).numpy().astype(np.float64)
        except Exception:
            pass
    return best_loss, best


def _solve_de(eval_mse, xtr, ytr, k, sv, rng):
    from scipy.optimize import differential_evolution

    def f(theta_np):
        with torch.no_grad():
            z = torch.tensor(theta_np, dtype=torch.float32)
            loss = eval_mse(xtr, ytr, _transform(z, sv.reparam_positive))
            return float(loss) if torch.isfinite(loss) else 1e6

    bounds = [(sv.init_low, sv.init_high)] * k
    try:
        r = differential_evolution(f, bounds, maxiter=sv.de_maxiter, popsize=sv.de_popsize,
                                   seed=int(rng.integers(1 << 31)), polish=True, tol=1e-10)
        z = torch.tensor(r.x, dtype=torch.float32)
        return float(r.fun), _transform(z, sv.reparam_positive).numpy().astype(np.float64)
    except Exception:
        return float("inf"), None


_SOLVERS = {"adam": _solve_adam, "lbfgs": _solve_lbfgs, "de": _solve_de}


def fit_portfolio(expression, param_names, xs, ys, *, threshold=1e-3, seed=0,
                  solvers=None, val_frac=0.3, concurrent=False) -> PortfolioResult:
    """Fit `expression` with a parameterized solver portfolio; select the winner on held-out MSE.

    concurrent=False (DEFAULT): cheap-first cascade — run solvers in order, early-exit as soon as
                       one's VALIDATION mse < threshold. Easy structures stop after cheap Adam;
                       only hard ones (e.g. F3 frequency) pay the expensive multi-start L-BFGS.
    concurrent=True  : run ALL solvers at once (ThreadPoolExecutor) and pick the best by held-out.

    MEASURED: concurrent is the WRONG default here. The solvers are CPU/GIL-bound (DMCI eval holds
    the GIL), so threads interleave rather than parallelize (wall ~= sum, not max) AND you pay the
    expensive solver on EVERY structure instead of only the ones that need it. The cascade gives the
    same best-solver robustness at a fraction of the cost. (Thread concurrency pays off for the
    I/O-bound LLM calls in the discovery loop, not for these solvers.) The graph is compiled ONCE
    either way; solver runs only read it (forward + autograd)."""
    solvers = solvers or DEFAULT_PORTFOLIO
    used = detect_used_params(expression, param_names)
    k = len(used)
    rng = np.random.default_rng(seed)

    # deterministic train/val split (random permutation so val spans the whole domain)
    n = len(xs)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    t0 = time.perf_counter()
    graph, eval_mse = _compiled_loss(expression, used)
    t_compile = time.perf_counter() - t0

    xtr, ytr = make_float(xs[tr_idx]), ys[tr_idx]
    xva, yva = make_float(xs[val_idx]), ys[val_idx]
    xfull = make_float(xs)

    def val_of(theta):
        if theta is None:
            return float("inf")
        with torch.no_grad():
            return float(eval_mse(xva, yva, torch.tensor(theta, dtype=torch.float32)))

    def full_of(theta):
        with torch.no_grad():
            return float(eval_mse(xfull, ys, torch.tensor(theta, dtype=torch.float32)))

    def run_solver(sv):
        s0 = time.perf_counter()
        # each solver gets its own rng stream (deterministic, but decorrelated by name)
        sv_rng = np.random.default_rng(seed + hash(sv.name) % 10_000)
        train_mse, theta = _SOLVERS[sv.kind](eval_mse, xtr, ytr, k, sv, sv_rng)
        return SolverResult(sv.name, train_mse, val_of(theta),
                            (theta.tolist() if theta is not None else []),
                            time.perf_counter() - s0)

    t_fit0 = time.perf_counter()
    if concurrent:
        # run every solver at once; pick best by held-out (no early exit)
        with ThreadPoolExecutor(max_workers=len(solvers)) as ex:
            per_solver = list(ex.map(run_solver, solvers))
    else:
        # cheap-first cascade: stop as soon as a solver clears the held-out threshold
        per_solver = []
        for sv in solvers:
            rec = run_solver(sv)
            per_solver.append(rec)
            if rec.val_mse < threshold:
                break
    best = min(per_solver, key=lambda r: r.val_mse) if per_solver else None
    t_fit = time.perf_counter() - t_fit0

    theta = np.array(best.theta) if best and best.theta else None
    fitted = {nm: float(theta[i]) for i, nm in enumerate(used)} if theta is not None else {}
    return PortfolioResult(
        expression=expression, used_params=used, fitted_values=fitted,
        val_mse=best.val_mse if best else float("inf"),
        train_mse=best.train_mse if best else float("inf"),
        full_mse=full_of(theta) if theta is not None else float("inf"),
        converged=(best.val_mse < threshold) if best else False,
        winner=best.name if best else "none",
        t_compile=t_compile, t_fit=t_fit,
        per_solver=[asdict(r) for r in per_solver])
