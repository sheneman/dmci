############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# runner.py: Atomic unit of the sweep: calibrate one program and score its held-out skill. For one generated program: 1....
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Atomic unit of the sweep: calibrate one program and score its held-out skill.

For one generated program:
  1. STRUCTURAL FIT -- multistart Adam over all parameters, summing the Gaussian NLL across
     the training seasons (shared parameters, each season reset at k=0); keep the best seed.
  2. VAL SKILL -- filter-then-forecast on the validation seasons (used for program SELECTION).
  3. TEST SKILL -- filter-then-forecast on the test seasons (the REPORTED held-out skill).
  4. AIC/BIC from the training NLL and parameter count (for model comparison across the zoo).

Selection is always on held-out validation skill, never training NLL (the exp_i overfit lesson).
Writes a flat results/<name>.json record. No multiprocessing here -- run_all owns the pool.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .config import DEFAULT
from .programs import parse_program
from .paramspec import param_count
from .calibrate import calibrate
from .forecast import filter_then_forecast, season_matrix

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def train_windows(data: dict, cfg=DEFAULT) -> list[torch.Tensor]:
    """Per-season [weeks, R] training matrices (the structural-fit objective summands)."""
    present = set(int(s) for s in data["seasons"])
    out = []
    for s in cfg.train_seasons:
        if s in present:
            m = season_matrix(data, s)
            if m.shape[0] >= 12:
                out.append(torch.tensor(m, dtype=torch.float32))
    return out


def run_single(source: str, name: str, data: dict, cfg=DEFAULT, *, seeds=None,
               output_dir: str | None = None, refit_iters: int = 25,
               origin_stride: int = 3, write: bool = True) -> dict:
    seeds = cfg.seeds if seeds is None else seeds
    prog = parse_program(source, name=name)
    windows = train_windows(data, cfg)

    # 1. multistart structural fit over the training seasons
    best, best_seed = None, seeds[0]
    for s in seeds:
        fit = calibrate(prog, windows, cfg=cfg, seed=s)
        if best is None or fit.nll < best.nll:
            best, best_seed = fit, s
    raw = best.raw

    # 2/3. held-out forecast skill (val selects, test reports)
    val = filter_then_forecast(prog, raw, data, cfg=cfg, test_seasons=cfg.val_seasons,
                               origin_stride=origin_stride, refit_iters=refit_iters)
    test = filter_then_forecast(prog, raw, data, cfg=cfg, test_seasons=cfg.test_seasons,
                                origin_stride=origin_stride, refit_iters=refit_iters)

    # 4. information criteria from the training fit
    k = param_count(prog.specs)
    n_obs = int(sum(int(w.numel()) for w in windows)) or 1
    nll = float(best.nll)
    rec = {
        "name": name,
        "n_params": k,
        "train_nll": nll,
        "best_seed": int(best_seed),
        "converged": bool(best.converged),
        "n_obs": n_obs,
        "aic": 2.0 * k + 2.0 * nll,
        "bic": k * math.log(max(n_obs, 2)) + 2.0 * nll,
        "val_mean_rmse": float(val["mean_rmse"]),
        "test_mean_rmse": float(test["mean_rmse"]),
        "val": val,
        "test": test,
        "fitted_params": {n: float(raw[n]) for n in prog.param_names},
    }
    if write and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"{name}.json").write_text(json.dumps(rec, indent=2, default=float))
    return rec
