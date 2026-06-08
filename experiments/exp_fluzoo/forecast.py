############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# forecast.py: Held-out forecast skill via filter-then-forecast (the selection metric). For a program whose STRUCTURAL...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Held-out forecast skill via filter-then-forecast (the selection metric).

For a program whose STRUCTURAL parameters were fit on the training seasons, score true
1- to 4-week-ahead skill on each held-out test season:

    for each forecast origin t within the season:
        refit ONLY the initial-condition parameters on the observed weeks 0..t-1
        (structural parameters frozen)
        for each horizon h: roll the autonomous model to week t+h and read its
        predicted %ILI; compare to the held-out observation.

The model never sees obs at or beyond the origin. Skill is reported per horizon as RMSE,
MAE, and correlation over all (origin, region) pairs, alongside national peak-week and
peak-intensity error. Pure scoring on top of the DMCI fold -- no per-program numpy twin.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import DEFAULT
from .calibrate import calibrate
from .programs import FluProgram, run_predict

# Parameters treated as initial conditions (refit per forecast origin). Heuristic by name;
# the structural parameters (rates, seasonal, reporting) are everything else.
_IC_PREFIXES = ("i0", "e0", "a0", "h0", "init", "i_0", "e_0", "seed")


def ic_param_names(prog: FluProgram) -> list[str]:
    return [s.name for s in prog.specs
            if any(s.name.lower().startswith(p) for p in _IC_PREFIXES)]


def season_matrix(data: dict, season: int) -> np.ndarray:
    """The [weeks, R] wILI matrix for one season (row 0 = epiweek w40)."""
    m = data["seasons"] == int(season)
    return data["wili"][m]


def forecast_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """RMSE / MAE / correlation over flattened (origin, region) prediction pairs."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    if pred.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "corr": float("nan"), "n": 0}
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    if pred.std() > 1e-12 and true.std() > 1e-12:
        corr = float(np.corrcoef(pred, true)[0, 1])
    else:
        corr = float("nan")
    return {"rmse": rmse, "mae": mae, "corr": corr, "n": int(pred.size)}


def origins(season_len: int, horizons, stride: int) -> list[int]:
    """Forecast origins = last-observed week indices o (room left for the longest horizon).
    Observed weeks are 0..o; horizon h forecasts week o+h."""
    hi = season_len - max(horizons)
    return list(range(8, max(9, hi), stride))


def _batched_ic_refit(prog: FluProgram, raw_struct: dict, obs_batch: torch.Tensor,
                      ic_names: list[str], refit_iters: int, refit_lr: float,
                      n_seasons: int, cfg) -> dict:
    """Re-estimate per-season initial conditions in ONE batched walk; structural params frozen.

    The IC parameters become [n_seasons] leaves (one per season), structural params stay shared
    scalars; the summed batched NLL gives per-season IC gradients. Returns a raw dict whose ICs
    are batched so the subsequent prediction is also batched across seasons.
    """
    from .programs import run_nll_batched
    raw = {}
    for n in prog.param_names:
        v = raw_struct[n].detach().reshape(())
        raw[n] = (v.expand(n_seasons).clone().requires_grad_(True) if n in ic_names
                  else v.clone())
    leaves = [raw[n] for n in ic_names]
    if not leaves or refit_iters <= 0:
        return raw
    opt = torch.optim.Adam(leaves, lr=refit_lr)
    for _ in range(refit_iters):
        opt.zero_grad()
        nll = run_nll_batched(prog, raw, obs_batch, cfg=cfg, grad=True).sum()
        if not torch.isfinite(nll).all():
            break
        nll.backward()
        torch.nn.utils.clip_grad_norm_(leaves, cfg.grad_clip)
        opt.step()
    return raw


def filter_then_forecast(prog: FluProgram, raw_struct: dict, data: dict, cfg=DEFAULT, *,
                         test_seasons=None, origin_stride: int = 8,
                         refit_iters: int = 5, refit_lr: float = 0.08) -> dict:
    """Held-out 1..4-week forecast skill, batched across the held-out seasons.

    At each forecast origin the held-out seasons share the horizon length T, so the per-season
    initial-condition refit and the multi-horizon predictions are done in single batched
    interpreter walks (~n_seasons-fold fewer walks than the per-season loop).
    """
    from .programs import run_predict_batched
    test_seasons = cfg.test_seasons if test_seasons is None else test_seasons
    R = cfg.n_regions
    ic = ic_param_names(prog)

    mats = [season_matrix(data, s) for s in test_seasons]
    mats = [m for m in mats if m.shape[0] >= 12]
    if not mats:
        out = {f"h{h}": forecast_metrics([], []) for h in cfg.horizons}
        out["mean_rmse"] = float("nan")
        return out
    Lmin = min(m.shape[0] for m in mats)
    n_seasons = len(mats)
    S = np.stack([m[:Lmin] for m in mats], axis=0).astype(np.float32)   # [n_seasons, Lmin, R]
    S_t = torch.tensor(S)

    bucket = {h: {"pred": [], "true": []} for h in cfg.horizons}
    nat_curves = np.full((n_seasons, Lmin), np.nan)

    for o in origins(Lmin, cfg.horizons, origin_stride):
        raw_t = _batched_ic_refit(prog, raw_struct, S_t[:, :o + 1, :], ic,
                                  refit_iters, refit_lr, n_seasons, cfg)
        for h in cfg.horizons:
            w = o + h
            if w >= Lmin:
                continue
            yhat = run_predict_batched(prog, raw_t, w, R, n_seasons, cfg=cfg).numpy()  # [n_seasons,R]
            for si in range(n_seasons):
                bucket[h]["pred"].append(yhat[si])
                bucket[h]["true"].append(S[si, w])
                if h == 1:
                    nat_curves[si, w] = yhat[si, 0]

    metrics = {f"h{h}": forecast_metrics(bucket[h]["pred"], bucket[h]["true"])
               for h in cfg.horizons}
    # national peak-week / peak-intensity error, averaged over held-out seasons
    pw_e, pi_e = [], []
    for si in range(n_seasons):
        c = nat_curves[si]
        valid = ~np.isnan(c)
        if valid.sum() > 4:
            pw = int(np.nanargmax(np.where(valid, c, -np.inf)))
            pw_e.append(abs(pw - int(np.argmax(S[si, :, 0]))))
            pi_e.append(abs(float(c[pw]) - float(S[si, :, 0].max())))
    if pw_e:
        metrics["peak"] = {"week_mae": float(np.mean(pw_e)),
                           "intensity_mae": float(np.mean(pi_e)), "n_seasons": len(pw_e)}
    metrics["mean_rmse"] = float(np.nanmean([metrics[f"h{h}"]["rmse"] for h in cfg.horizons]))
    return metrics
