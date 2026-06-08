############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# runner.py: Per-run driver: fit a LIM operator by ONE solver, score it, assemble the record. ``run_single`` is the atomic...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Per-run driver: fit a LIM operator by ONE solver, score it, assemble the record.

``run_single`` is the atomic unit of the experiment: pick a solver from ``baselines`` (Adam
with exact DMCI gradients -- the PRIMARY -- or L-BFGS multi-start, or batched differential
evolution), fit the shared LIM Kalman-NLL objective on the TRAIN window
(``cfg.T_train``), then score the FITTED (F, Q, R) with ``forecast`` -- held-out forecast
skill (ACC/RMSE per lead vs persistence / damped-persistence / the Green-function reference
operator) plus the LIM operator diagnostics (ENSO mode period/decay, eigen-timescales,
spectral radius, optimal growth). Everything is collapsed into ONE flat JSON-serialisable
record + a per-iteration CSV (iter, nll, grad_norm, wall_time) so the aggregator can build
the manuscript tables.

The DMCI graph for ``(structure, D, T)`` is compiled ONCE inside ``models._get_graph`` and
reused across solvers and seeds (the compile cache lives in ``models``). The interpreter is
ONLY used for the MLE gradient; everything in ``forecast`` is pure numpy/scipy on the fitted
float matrices. dynamax / the numpy twin are correctness oracles; the Green-function operator
is the SCIENTIFIC reference, never a speed contest -- the headline is CAPABILITY (a real-data
dynamical-systems MLE expressed as a program and optimised by exact DMCI gradients).

Public API
----------
``run_single(method, D, structure, seed, cfg, output_dir, obs, eofs, pc_std, lat, lon) -> dict``
    Run the chosen solver, score it, write ``results/<tag>.json`` + ``results/<tag>.csv``,
    and return the per-run record. ``tag = {method}_{structure}_D{D}_seed{seed}``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DEFAULT, ExpLimConfig
from . import params, baselines, forecast


# Solver dispatch: method name -> baselines callable. All share the result-dict schema.
_SOLVERS = {
    "dmci_adam": baselines.run_dmci_adam,
    "lbfgs_multistart": baselines.run_lbfgs_multistart,
    "diffevo_batched": baselines.run_diffevo_batched,
}

# Forecast leads (months) -- the ENSO-relevant 3/6/9/12-mo horizons.
_LEADS = (3, 6, 9, 12)


def tag_for(method: str, structure: str, D: int, seed: int) -> str:
    """Canonical per-run tag ``{method}_{structure}_D{D}_seed{seed}`` (file stem)."""
    return f"{method}_{structure}_D{D}_seed{seed}"


# ===========================================================================
# JSON helper: make numpy / torch scalars and arrays serialisable.
# ===========================================================================

def _jsonable(o):
    """Recursively coerce numpy/torch scalars + arrays to plain Python for json.dump."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "detach"):                      # stray torch tensor
        return o.detach().cpu().numpy().tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None                               # NaN/Inf -> null (valid JSON)
    return o


# ===========================================================================
# Forecast-skill flattening: pull per-lead ACC/RMSE into flat dicts.
# ===========================================================================

def _heldout_skill(skill: dict, leads) -> tuple[dict, dict]:
    """Flatten ``forecast.forecast_skill`` into per-lead Nino-3.4 ACC / RMSE dicts.

    Returns ``(heldout_acc, heldout_rmse)`` keyed by lead (months), using the FITTED-LIM
    Nino-3.4 skill (the headline forecast). The full per-baseline / per-PC report is kept
    verbatim under the record's ``forecast`` block; these flats are the at-a-glance metric."""
    acc: dict[int, float] = {}
    rmse: dict[int, float] = {}
    nino = skill.get("nino34", {}) if isinstance(skill, dict) else {}
    for h in leads:
        cell = nino.get(h) or nino.get(str(h))
        if cell and "fitted_lim" in cell:
            acc[h] = float(cell["fitted_lim"]["ACC"])
            rmse[h] = float(cell["fitted_lim"]["RMSE"])
    return acc, rmse


def _eig_timescales(diag: dict) -> list[float]:
    """Per-mode decay timescales (months) from the diagnostics ``modes`` block (sorted desc)."""
    out = []
    for m in diag.get("modes", []):
        d = m.get("decay_months")
        out.append(float(d) if (d is not None and math.isfinite(d)) else float("inf"))
    return out


# ===========================================================================
# Public: one fit + score + record.
# ===========================================================================

def run_single(method: str, D: int, structure: str, seed: int,
               cfg: ExpLimConfig, output_dir, obs, eofs, pc_std, lat, lon,
               *, leads=_LEADS, jitter: bool = False,
               write: bool = True, log_every: Optional[int] = None) -> dict:
    """Fit ONE LIM by ``method``, score forecast skill + diagnostics, return the record.

    Args:
        method: solver key (``dmci_adam`` | ``lbfgs_multistart`` | ``diffevo_batched``).
        D, structure, seed: the run coordinates (D = PC truncation, structure = F-assembly,
            seed = init / multi-start seed).
        cfg: the ExpLimConfig (T_train/T_test, lr, adam_iters, EVAL_KW, ...).
        output_dir: directory to write ``results/<tag>.json`` + ``results/<tag>.csv`` into.
        obs: the FULL ``[T_full, D_max]`` PC tensor (the solver slices ``[:T_train, :D]``;
            forecast slices its own train/test windows). Pass the on-disk pcs once.
        eofs, pc_std, lat, lon: the spatial reconstruction basis (for the Nino-3.4 index).
        leads: forecast leads in months (default 3,6,9,12).
        jitter: PD-escalation lever (S + jitter_eps*I before inv/det); off by default.
        write: write the JSON + CSV artifacts (set False for a dry score).
        log_every: live per-iteration logging cadence threaded to the solver (default
            ``cfg.log_every``); purely diagnostic, never changes the numerics.

    Returns the flat per-run record (see module docstring / README schema).
    """
    if method not in _SOLVERS:
        raise ValueError(f"run_single: unknown method {method!r} "
                         f"(known: {sorted(_SOLVERS)})")
    T = cfg.T_train
    tag = tag_for(method, structure, D, seed)

    # --- 1) FIT: run the chosen solver over the shared DMCI NLL objective (train window). ---
    fit = _SOLVERS[method](D, structure, seed, obs, cfg, T=T, jitter=jitter,
                           tag=tag, log_every=log_every)

    # --- 2) SCORE: held-out forecast skill + LIM operator diagnostics on the FITTED op. ---
    fitted = {
        "F": fit["fitted"]["F"], "Q": fit["fitted"]["Q"], "R": fit["fitted"]["R"],
        "D": D, "structure": structure, "seed": seed,
    }
    try:
        skill = forecast.forecast_skill(fitted, obs, eofs, pc_std, lat, lon,
                                        cfg=cfg, leads=leads)
    except Exception as exc:  # noqa: BLE001  (scoring must never abort the record)
        skill = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        diag = forecast.diagnostics(fitted["F"], dt=1.0)
    except Exception as exc:  # noqa: BLE001
        diag = {"error": f"{type(exc).__name__}: {exc}"}

    heldout_acc, heldout_rmse = _heldout_skill(skill, leads)
    enso = diag.get("enso_mode") or {}

    # --- 3) AIC / BIC from the per-structure parameter budget k and the train-window NLL. ---
    k = params.param_count(D, structure, cfg.lowrank_rank)["total"]
    final_nll = float(fit["final_nll"])
    nll_finite = math.isfinite(final_nll)
    aic = (2 * k + 2 * final_nll) if nll_finite else float("nan")
    bic = (k * math.log(T) + 2 * final_nll) if nll_finite else float("nan")

    # --- 4) flat record (the aggregator reads these keys; nested blocks kept for the paper). ---
    record = {
        "tag": tag,
        "method": method,
        "structure": structure,
        "D": int(D),
        "seed": int(seed),
        "T_train": int(T),
        "T_test": int(cfg.T_test),
        # fit outcome
        "final_nll": final_nll,
        "converged": bool(fit["converged"]),
        "n_iters": int(fit["n_iters"]),
        "wall_time": float(fit["wall_time"]),
        "per_step_ms": float(fit["per_step_ms"]),
        # information criteria
        "k_params": int(k),
        "aic": aic,
        "bic": bic,
        # held-out forecast skill (fitted-LIM Nino-3.4, per lead)
        "heldout_acc": heldout_acc,
        "heldout_rmse": heldout_rmse,
        # LIM operator diagnostics
        "eig_timescales": _eig_timescales(diag),
        "enso_period_mo": enso.get("period_months"),
        "enso_decay_mo": enso.get("decay_months"),
        "rho_F": diag.get("spectral_radius_F"),
        "stable": diag.get("stable"),
        # conditioning at the fitted operator (float64 twin)
        "min_detS": float(fit["min_detS"]),
        "cond_S": float(fit["cond_S"]),
        # solver-specific flags
        "batched": bool(fit.get("batched", False)) if method == "diffevo_batched" else None,
        "nan_stall": bool(fit.get("nan_stall", False)),
        # full nested blocks (kept verbatim for the manuscript; not read by the aggregator)
        "fitted": fit["fitted"],
        "forecast": skill,
        "diagnostics": diag,
    }

    # --- 5) artifacts: per-run JSON + per-iteration CSV. ---
    if write:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.json").write_text(json.dumps(_jsonable(record), indent=2))
        _write_iter_csv(out / f"{tag}.csv", fit)

    return record


def _write_iter_csv(path: Path, fit: dict) -> None:
    """Per-iteration trace CSV: ``iter,nll,grad_norm,wall_time``.

    ``wall_time`` here is the cumulative solver wall-clock linearly apportioned across the
    recorded iterations (the solvers record one NLL per accepted step; the per-step wall is
    not separately timed, so we apportion the total honestly). DiffEvo has no per-step grad
    norm -> that column is blank."""
    nll = fit.get("nll_trace", []) or []
    gnorm = fit.get("grad_norm_trace", []) or []
    n = len(nll)
    total_wall = float(fit.get("wall_time", 0.0))
    rows = ["iter,nll,grad_norm,wall_time"]
    for i in range(n):
        g = gnorm[i] if i < len(gnorm) else ""
        w = total_wall * (i + 1) / n if n else 0.0
        gv = "" if g == "" or (isinstance(g, float) and not math.isfinite(g)) else f"{g:.8e}"
        nv = nll[i]
        nv = "" if (isinstance(nv, float) and not math.isfinite(nv)) else f"{nv:.8e}"
        rows.append(f"{i},{nv},{gv},{w:.6f}")
    path.write_text("\n".join(rows) + "\n")
