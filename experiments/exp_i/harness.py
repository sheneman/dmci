############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# harness.py: Fit harness for Experiment I: multi-driver data, the DMCI/direct fit loop (with the just-fixed finite-guard +...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Fit harness for Experiment I: multi-driver data, the DMCI/direct fit loop
(with the just-fixed finite-guard + best-param checkpointing), and the black-box
(differential evolution) baseline.

The fit loop is the SAME non-batched scalar double loop as exp_c/exp_f:
    graph = compile_program(...)  ONCE
    for epoch: for datapoint: evaluate(graph, ...); loss.backward()
generalised from a single driver `x` to an arbitrary driver schema (Q, T, psi).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import ExpIConfig, DEFAULT
from .models import ModelSpec


@dataclass
class FitResult:
    method: str
    model_name: str
    seed: int
    converged: bool
    best_mse: float
    fitted_values: dict[str, float]
    param_rel_error: dict[str, float]
    n_epochs: int
    t_fit_s: float
    nan_stall: bool                  # did a non-finite loss/grad ever force a skip?
    loss_history: list[float] = field(default_factory=list)


_GRAPH_CACHE: dict[str, object] = {}


# --- Data -------------------------------------------------------------------

def generate_data(model: ModelSpec, cfg: ExpIConfig, seed: int):
    """Latin-hypercube-ish sample of the driver space (deterministic per seed)."""
    g = torch.Generator().manual_seed(seed)
    n = cfg.n_data_points
    xs_list = []
    cols = {}
    for d in model.input_names:
        lo, hi = model.driver_ranges[d]
        # stratified: one sample per equal-width bin, shuffled — covers the range
        edges = torch.linspace(lo, hi, n + 1)
        u = torch.rand(n, generator=g)
        samples = edges[:-1] + u * (edges[1:] - edges[:-1])
        perm = torch.randperm(n, generator=g)
        cols[d] = samples[perm]
    for i in range(n):
        xs_list.append({d: cols[d][i].item() for d in model.input_names})

    ys = []
    noise = torch.zeros(n)
    if cfg.noise_std > 0:
        noise = cfg.noise_std * torch.randn(n, generator=g)
    for i, xd in enumerate(xs_list):
        y = model.ground_truth(**xd, **model.target_values)
        ys.append(torch.tensor(float(y) + noise[i].item()))
    return xs_list, ys


def _make_params(model: ModelSpec, seed: int) -> dict[str, nn.Parameter]:
    torch.manual_seed(seed)
    out = {}
    for name in model.param_names:
        tgt = model.target_values[name]
        out[name] = nn.Parameter(
            torch.tensor(float(tgt))
            + 0.3 * max(abs(tgt), 0.1) * torch.randn(1).squeeze())
    return out


def _rel_error(params, model) -> dict[str, float]:
    out = {}
    for name in model.param_names:
        tgt = model.target_values[name]
        denom = abs(tgt) if abs(tgt) > 1e-9 else 1.0
        out[name] = abs(_as_float(params[name]) - tgt) / denom
    return out


def _as_float(p) -> float:
    return p.item() if isinstance(p, torch.Tensor) else float(p)


# --- Unified prediction (handles tagged DMCI graphs and untagged direct ones) ---

def _predict(graph, input_vals: dict) -> torch.Tensor:
    """One forward pass, returning a 0-dim tensor with autograd intact.

    Tagged graphs (DMCI / recursive direct) take 14-dim tagged inputs and return a
    TaggedValue. Untagged graphs (pure-arithmetic direct) collapse a 0-dim result to
    a detached Python float in engine.evaluate (engine.py:537-538) -- so we feed
    shape-[1] inputs to keep dim()==1 (no .item() collapse, gradients preserved) and
    reshape back to a scalar."""
    if graph.uses_tagged_values:
        tagged = {n: make_float(_to_tensor(v)) for n, v in input_vals.items()}
        return unwrap_number(evaluate(graph, tagged))
    bare = {n: _to_tensor(v).reshape(1) for n, v in input_vals.items()}
    out = evaluate(graph, bare)
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out, dtype=torch.float32)
    return out.reshape(())


def _to_tensor(v):
    return v if isinstance(v, torch.Tensor) else torch.tensor(float(v))


def _predict_batched(graph, driver_batch: dict, params: dict) -> torch.Tensor:
    """One batched forward pass over all N data points -> [N] predictions.

    Replaces N sequential interpreter walks with a single walk: structural values
    (env spine, AST, counters) stay scalar and data-independent; only the [N] numeric
    leaves carry the batch dimension and arithmetic broadcasts. Works for both the
    heap-backed DMCI graph and the heap-free direct graph (each driver is an [N] vec,
    each param a shared scalar that broadcasts)."""
    if graph.uses_tagged_values:
        binp = {d: make_float(v) for d, v in driver_batch.items()}
        for n, p in params.items():
            binp[n] = make_float(p if isinstance(p, torch.Tensor)
                                 else torch.tensor(float(p)))
        return unwrap_number(evaluate(graph, binp))
    binp = {d: _to_tensor(v) for d, v in driver_batch.items()}
    for n, p in params.items():
        binp[n] = _to_tensor(p)
    out = evaluate(graph, binp)
    return out if isinstance(out, torch.Tensor) else torch.as_tensor(out, dtype=torch.float32)


# --- DMCI / direct-compile fit loop -----------------------------------------

def _compile(model: ModelSpec, method: str):
    key = f"{method}_{model.name}"
    if key not in _GRAPH_CACHE:
        all_inputs = {n: None for n in model.input_names + model.param_names}
        if method == "dmci":
            _GRAPH_CACHE[key] = compile_program(
                model.interp_source, inputs=all_inputs, prelude=True)
        else:  # direct
            _GRAPH_CACHE[key] = compile_program(
                model.direct_source, inputs=all_inputs, prelude=False)
    return _GRAPH_CACHE[key]


def _train_compiled(method: str, model: ModelSpec, cfg: ExpIConfig,
                    seed: int, time_budget: float | None = None) -> FitResult:
    graph = _compile(model, method)
    params = _make_params(model, seed)
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    xs_list, ys = generate_data(model, cfg, seed)
    # Batched data: ONE interpreter walk per epoch over all N points (not N walks),
    # via the batched DMCI path. Built once outside the epoch loop.
    driver_batch = {d: torch.tensor([xd[d] for xd in xs_list], dtype=torch.float32)
                    for d in model.input_names}
    y_batch = torch.stack([y if isinstance(y, torch.Tensor)
                           else torch.tensor(float(y)) for y in ys])
    n_data = len(xs_list)

    best_mse = float("inf")
    best_params = {n: p.item() for n, p in params.items()}
    patience = 0
    nan_stall = False
    loss_history: list[float] = []

    t0 = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        try:
            preds = _predict_batched(graph, driver_batch, params)
            total_loss = ((preds - y_batch) ** 2).sum()
        except Exception:
            total_loss = torch.tensor(float(1e6 * n_data))

        mse = total_loss / n_data
        loss_val = mse.item()
        current_snapshot = {n: p.item() for n, p in params.items()}

        optimizer.zero_grad()
        if torch.isfinite(total_loss):
            try:
                total_loss.backward()
                if all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in params.values()):
                    torch.nn.utils.clip_grad_norm_(
                        list(params.values()), cfg.grad_clip)
                    optimizer.step()
                else:
                    nan_stall = True
            except Exception:
                nan_stall = True
        else:
            nan_stall = True

        loss_history.append(loss_val)
        if loss_val < best_mse:
            best_mse = loss_val
            best_params = current_snapshot
            patience = 0
        else:
            patience += 1
        if best_mse < cfg.convergence_threshold:
            break
        if patience > cfg.early_stop_patience:
            break
        if time_budget is not None and (time.perf_counter() - t0) > time_budget:
            break

    t_fit = time.perf_counter() - t0
    return FitResult(
        method=method, model_name=model.name, seed=seed,
        converged=best_mse < cfg.convergence_threshold,
        best_mse=best_mse, fitted_values=best_params,
        param_rel_error={n: abs(best_params[n] - model.target_values[n])
                         / (abs(model.target_values[n]) or 1.0)
                         for n in model.param_names},
        n_epochs=len(loss_history), t_fit_s=t_fit, nan_stall=nan_stall,
        loss_history=loss_history[::max(1, len(loss_history) // 50)],
    )


def run_dmci(model: ModelSpec, cfg: ExpIConfig = DEFAULT, seed: int = 0,
             time_budget: float | None = None):
    return _train_compiled("dmci", model, cfg, seed, time_budget)


def run_direct(model: ModelSpec, cfg: ExpIConfig = DEFAULT, seed: int = 0,
               time_budget: float | None = None):
    return _train_compiled("direct", model, cfg, seed, time_budget)


# --- Black-box baseline: scipy differential evolution -----------------------

def run_diffevo(model: ModelSpec, cfg: ExpIConfig = DEFAULT, seed: int = 0,
                time_budget: float | None = None):
    """Per-structure black-box calibration. The forward model is the Python
    ground-truth form (a HUMAN had to transcribe it — part of the engineering
    cost). Bounds are the pre-registered domain-plausible PARAM_BOUNDS.
    ``time_budget`` (seconds) stops DE early for an equal-wall-clock comparison."""
    from scipy.optimize import differential_evolution

    xs_list, ys = generate_data(model, cfg, seed)
    ys_np = np.array([y.item() for y in ys])
    names = model.param_names
    bounds = [model.param_bounds[n] for n in names]

    def objective(theta):
        pdict = dict(zip(names, theta))
        preds = np.array([model.ground_truth(**xd, **pdict) for xd in xs_list])
        if not np.all(np.isfinite(preds)):
            return 1e9
        return float(np.mean((preds - ys_np) ** 2))

    t0 = time.perf_counter()

    def _cb(*_a):  # stop at the wall-clock budget (scipy-version-agnostic signature)
        return time_budget is not None and (time.perf_counter() - t0) > time_budget

    res = differential_evolution(
        objective, bounds, maxiter=cfg.de_maxiter, popsize=cfg.de_popsize,
        seed=seed, polish=True, tol=1e-10, callback=_cb)
    t_fit = time.perf_counter() - t0

    fitted = dict(zip(names, res.x.tolist()))
    return FitResult(
        method="diffevo", model_name=model.name, seed=seed,
        converged=res.fun < cfg.convergence_threshold,
        best_mse=float(res.fun), fitted_values=fitted,
        param_rel_error={n: abs(fitted[n] - model.target_values[n])
                         / (abs(model.target_values[n]) or 1.0) for n in names},
        n_epochs=int(res.nit), t_fit_s=t_fit, nan_stall=False,
    )


# --- Predictive (held-out) MSE: the PRIMARY metric --------------------------
# (converged != recovered: Exp C shows MSE-converged fits still have 24-61%
# param error, so we lead with held-out predictive MSE, not param recovery.)

def heldout_mse(model: ModelSpec, fitted_values: dict[str, float],
                cfg: ExpIConfig, seed: int) -> float:
    """Predictive MSE on a fresh driver sample the fit never saw."""
    xs_list, ys = generate_data(model, cfg, seed + 10_000)
    err = 0.0
    for xd, y_val in zip(xs_list, ys):
        pred = model.ground_truth(**xd, **fitted_values)
        err += (pred - y_val.item()) ** 2
    return err / len(xs_list)


# --- Reparameterized, multi-start L-BFGS (the Exp-I-v2 gradient fitter) ------
# Diagnosis (docs/): the composite-GPP loss is ill-conditioned (multiplicative
# light*temp*water coupling => ~2700x gradient spread) and multi-basin (PFT symmetry).
# Single-start Adam (run_dmci) stalls and loses to differential evolution. Fix, confirmed
# to beat DE by ~2 orders of magnitude on the landscape:
#   (1) log-reparameterize positive params  -> conditioning + positivity (no pow/div blowups)
#   (2) a curvature-aware optimizer (L-BFGS) -> handles the off-diagonal coupling Adam can't
#   (3) multi-start                          -> escapes local basins DE's global search finds
# Each restart is cheap because the per-step forward is ONE batched DMCI walk (v1.1.7).

_POSITIVE = {"alpha", "Amax", "w", "s"}     # log-reparameterized; Topt/psi50 stay raw


def _suffix(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def _random_init(model: ModelSpec, seed: int, start: int) -> dict:
    """A random start drawn uniformly from the (target-free) parameter bounds."""
    g = torch.Generator().manual_seed(10_000 * seed + start)
    out = {}
    for n in model.param_names:
        lo, hi = model.param_bounds[n]
        out[n] = float(lo + (hi - lo) * torch.rand((), generator=g))
    return out


def _to_raw(init_vals: dict) -> dict:
    """Optimizer variables: log-space for positive params, identity otherwise."""
    raw = {}
    for n, v in init_vals.items():
        if _suffix(n) in _POSITIVE:
            raw[n] = torch.tensor(math.log(max(float(v), 1e-6)), requires_grad=True)
        else:
            raw[n] = torch.tensor(float(v), requires_grad=True)
    return raw


def _to_physical(raw: dict) -> dict:
    return {n: (torch.exp(p) if _suffix(n) in _POSITIVE else p) for n, p in raw.items()}


def run_lbfgs_multistart(model: ModelSpec, cfg: ExpIConfig = DEFAULT, seed: int = 0,
                         method: str = "dmci", n_starts: int = 32,
                         max_iter: int = 200, time_budget: float | None = None):
    """Reparam + L-BFGS + multi-start fit through the (batched) DMCI/direct graph.

    Runs up to ``n_starts`` restarts (or until ``time_budget`` seconds), each a strong-Wolfe
    L-BFGS optimization in log-reparameterized coordinates over a single batched forward
    walk; keeps the best. Returns the best FitResult."""
    graph = _compile(model, method)
    xs_list, ys = generate_data(model, cfg, seed)
    driver_batch = {d: torch.tensor([xd[d] for xd in xs_list], dtype=torch.float32)
                    for d in model.input_names}
    y_batch = torch.stack([y if isinstance(y, torch.Tensor) else torch.tensor(float(y))
                           for y in ys])

    best_mse = float("inf")
    best_params = {n: model.target_values[n] for n in model.param_names}
    nan_stall = False
    t0 = time.perf_counter()
    starts_done = 0
    for s in range(n_starts):
        if time_budget is not None and (time.perf_counter() - t0) > time_budget:
            break
        raw = _to_raw(_random_init(model, seed, s))
        opt = torch.optim.LBFGS(list(raw.values()), max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            preds = _predict_batched(graph, driver_batch, _to_physical(raw))
            loss = ((preds - y_batch) ** 2).mean()
            if torch.isfinite(loss):
                loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception:
            nan_stall = True
        with torch.no_grad():
            preds = _predict_batched(graph, driver_batch, _to_physical(raw))
            mse = ((preds - y_batch) ** 2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_params = {n: float(v) for n, v in _to_physical(raw).items()}
        starts_done += 1

    t_fit = time.perf_counter() - t0
    return FitResult(
        method=f"{method}_lbfgs_ms", model_name=model.name, seed=seed,
        converged=best_mse < cfg.convergence_threshold, best_mse=best_mse,
        fitted_values=best_params,
        param_rel_error={n: abs(best_params[n] - model.target_values[n])
                         / (abs(model.target_values[n]) or 1.0)
                         for n in model.param_names},
        n_epochs=starts_done, t_fit_s=t_fit, nan_stall=nan_stall)


# --- Real-data validation: AmeriFlux US-Ha1 loader stub (pillar 1) ----------

AMERIFLUX_NOTE = """\
Real-data validation (pillar 1) — AmeriFlux BASE, site US-Ha1 (Harvard Forest
EMS Tower), half-hourly. Fit the light response of midday growing-season GPP:
  driver  Q   <- PPFD_IN  (or PAR = 2.04 * SW_IN if PPFD missing)
  driver  T   <- TA
  driver  psi <- proxy from SWC / VPD (state the substitution; psi not measured)
  target  GPP <- GPP_PI_F  (or partition NEE_PI_F into GPP for defensibility)
Acquire via the AmeriFlux data portal (CC-BY-4.0; requires registration + a DOI),
place the BASE CSV at experiments/exp_i/data/US-Ha1_BASE.csv, then load_ameriflux()
parses it. This is a single-site / single-PFT calibration, NOT data assimilation.
"""


def load_ameriflux(path: str | None = None):
    """Load + filter US-Ha1 BASE data. Stub: raises with acquisition instructions
    until the CSV is present (no network access on compute nodes / locally)."""
    from pathlib import Path
    p = Path(path) if path else (Path(__file__).parent / "data" / "US-Ha1_BASE.csv")
    if not p.exists():
        raise FileNotFoundError(
            f"AmeriFlux US-Ha1 BASE file not found at {p}.\n" + AMERIFLUX_NOTE)
    raise NotImplementedError(
        "CSV present — implement column mapping (PPFD_IN/TA/SWC -> Q/T/psi, "
        "GPP_PI_F -> y) + midday growing-season filtering here once the data "
        "schema is confirmed against the downloaded file.")
