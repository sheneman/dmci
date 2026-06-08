############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# score.py: Fit-early / forecast-late scoring through DMCI. The calibration (the load-bearing, novel capability) runs...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Fit-early / forecast-late scoring through DMCI.

The calibration (the load-bearing, novel capability) runs through the compiled interpreter:
calibrate() does Adam on the structure's unconstrained raw leaves to minimize the Gaussian NLL
that the DMCI rollout computes over the EARLY cycles [0, ksplit). The forecast is then the
fitted structure's extrapolation over the held-out tail [ksplit, T); because each structure is
an algebraic function of the cycle index, the predicted curve is read out in closed form
(curves.predict_curve) from the DMCI-fitted parameters -- and gradient_health() separately
verifies that the DMCI forward path (run_predict) reproduces that closed form, so no fidelity
is lost.
"""

from __future__ import annotations

import numpy as np
import torch

import math

from experiments.exp_fluzoo.programs import parse_program, run_predict, run_nll_batched
from experiments.exp_fluzoo.paramspec import constrain, make_raw
from experiments.exp_fluzoo.calibrate import calibrate

from .config import BCFG, N_SERIES, T_CYCLES
from .curves import predict_curve
from .structures import STRUCTURES

_PROG_CACHE: dict[str, object] = {}


def _prog(name: str):
    p = _PROG_CACHE.get(name)
    if p is None:
        p = parse_program(STRUCTURES[name], name=name)
        _PROG_CACHE[name] = p
    return p


def _constrained_floats(prog, raw) -> dict:
    con = constrain(prog.specs, raw)
    return {k: float(v.detach()) for k, v in con.items()}


def fit_structure(name: str, obs_cell: torch.Tensor, ksplit: int, cfg=BCFG,
                  multistart: bool = True) -> dict:
    """Fit structure `name` to the early cycles of one cell; return fit + constrained params.

    obs_cell is `[T, 1]`. Calibrates on obs_cell[:ksplit] through DMCI.
    """
    prog = _prog(name)
    obs_early = obs_cell[:ksplit]
    seeds = cfg.seeds if multistart else (0,)
    best = None
    for s in seeds:
        r = calibrate(prog, obs_early, cfg=cfg, seed=s)
        if best is None or r.nll < best.nll:
            best = r
    return {"raw": best.raw, "nll": best.nll, "nan_stall": best.nan_stall,
            "params": _constrained_floats(prog, best.raw)}


def score_cell(name_fit: str, obs_cell: torch.Tensor, ksplit: int, cfg=BCFG) -> dict:
    """Fit `name_fit` to early cycles, score in-sample + held-out RMSE on this cell."""
    T = int(obs_cell.shape[0])
    fit = fit_structure(name_fit, obs_cell, ksplit, cfg=cfg)
    k = np.arange(T)
    pred = predict_curve(name_fit, fit["params"], k)
    obs = obs_cell[:, 0].numpy().astype(np.float64)
    err = pred - obs
    insample = float(np.sqrt(np.mean(err[:ksplit] ** 2)))
    holdout = float(np.sqrt(np.mean(err[ksplit:] ** 2)))
    return {"train_nll": fit["nll"], "insample_rmse": insample,
            "holdout_rmse": holdout, "nan_stall": fit["nan_stall"],
            "params": fit["params"], "pred": pred, "obs": obs}


def _make_raw_batched(prog, n: int, seed: int) -> dict:
    """Per-cell batched raw leaves `[n]` (one fit per cell, all in one interpreter walk)."""
    g = torch.Generator().manual_seed(seed)
    scalar = make_raw(prog.specs, seed=seed)
    return {nm: (scalar[nm].detach().reshape(1).repeat(n)
                 + 0.02 * torch.randn(n, generator=g)).requires_grad_(True)
            for nm in prog.param_names}


def score_pooled(name_fit: str, obs_batch: torch.Tensor, ksplit: int, cfg=BCFG,
                 iters: int = 60, lr: float = 0.1) -> dict:
    """Fit `name_fit` to every cell at once (batched), score per-cell held-out RMSE.

    obs_batch is `[N, T, 1]`. Each cell gets its OWN parameters (raw leaves batched over N);
    a single batched interpreter walk per iter folds all N cells -- ~constant wall-time in N.
    Returns per-cell in-sample / held-out RMSE arrays.
    """
    prog = _prog(name_fit)
    N, T = int(obs_batch.shape[0]), int(obs_batch.shape[1])
    early = obs_batch[:, :ksplit, :]
    raw = _make_raw_batched(prog, N, seed=0)
    leaves = list(raw.values())
    opt = torch.optim.Adam(leaves, lr=lr)
    prev, best, best_snap, nan_stall = math.inf, math.inf, None, False
    for _ in range(iters):
        opt.zero_grad()
        nll = run_nll_batched(prog, raw, early, cfg=cfg, grad=True).sum()
        v = float(nll.detach())
        if not math.isfinite(v):
            nan_stall = True
            break
        nll.backward()
        if any((g.grad is None) or (not torch.isfinite(g.grad).all()) for g in leaves):
            nan_stall = True
            break
        torch.nn.utils.clip_grad_norm_(leaves, cfg.grad_clip)
        opt.step()
        if v < best:
            best, best_snap = v, {n: raw[n].detach().clone() for n in prog.param_names}
        if abs(prev - v) < cfg.conv_tol:
            break
        prev = v
    if best_snap is not None:
        raw = {n: best_snap[n].requires_grad_(True) for n in prog.param_names}

    con = constrain(prog.specs, raw)
    k = np.arange(T)
    obs = obs_batch[:, :, 0].numpy().astype(np.float64)
    insample, holdout = [], []
    for i in range(N):
        p_i = {nm: float(con[nm][i].detach()) for nm in prog.param_names}
        pred = predict_curve(name_fit, p_i, k)
        err = pred - obs[i]
        insample.append(float(np.sqrt(np.mean(err[:ksplit] ** 2))))
        holdout.append(float(np.sqrt(np.mean(err[ksplit:] ** 2))))
    return {"train_nll": best, "nan_stall": nan_stall,
            "insample_rmse": insample, "holdout_rmse": holdout}


def dmci_predict_check(name: str, params_raw, horizons, cfg=BCFG) -> list[tuple[int, float, float]]:
    """Verify the DMCI forward path (run_predict) matches the closed-form curve.

    Returns [(week, dmci_pred, closed_form_pred)] so gradient_health can assert agreement.
    """
    prog = _prog(name)
    con = constrain(prog.specs, params_raw)
    cf_params = {k: float(v.detach()) for k, v in con.items()}
    out = []
    for w in horizons:
        dmci = float(run_predict(prog, params_raw, int(w), N_SERIES, cfg=cfg)[0])
        cf = float(predict_curve(name, cf_params, np.array([w]))[0])
        out.append((int(w), dmci, cf))
    return out
