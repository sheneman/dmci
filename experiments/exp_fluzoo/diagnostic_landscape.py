############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# diagnostic_landscape.py: Landscape diagnostic: is FluZoo's flat fitness intrinsic, or a short-horizon/metric artifact? Fits three...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Landscape diagnostic: is FluZoo's flat fitness intrinsic, or a short-horizon/metric artifact?

Fits three STRUCTURALLY DISTINCT reference programs (SIR vs SEIR vs SEIRS, regional) and scores
their held-out skill across horizons 1..16 plus peak-week / peak-intensity. The question: does the
SPREAD between the three programs (max-min RMSE) widen at long horizons -- where the autonomous
dynamics unfold and susceptible-depletion / waning / recurrence actually differentiate the
structures -- versus the ~flat 1-4-week regime where simple seasonal shape dominates.

  spread stays small at all horizons  -> flu is intrinsically structure-insensitive (flat is real)
  spread widens at long horizons      -> the landscape is rugged at a harder target; retarget FluZoo

Runs locally (no cluster needed).
"""

import sys
sys.setrecursionlimit(20000)

import dataclasses
import numpy as np
import torch

from .config import DEFAULT
from .data.build_data import load_processed
from .programs import reference_program
from .calibrate import calibrate
from .forecast import filter_then_forecast
from .runner import train_windows

PROGRAMS = ["sir_regional", "seir_regional", "seirs_regional"]
HORIZONS = (1, 2, 4, 8, 12, 16)


def main():
    torch.manual_seed(0)
    data = load_processed()
    cfg = dataclasses.replace(DEFAULT, horizons=HORIZONS, adam_iters=50, seeds=(0,))

    results = {}
    for name in PROGRAMS:
        prog = reference_program(name)
        windows = train_windows(data, cfg)
        fit = calibrate(prog, windows, cfg=cfg, seed=0)
        m = filter_then_forecast(prog, fit.raw, data, cfg=cfg, test_seasons=cfg.val_seasons,
                                 origin_stride=6, refit_iters=6)
        results[name] = m
        print(f"[fit] {name:16s} train_nll={fit.nll:.1f}  mean_rmse={m['mean_rmse']:.4f}", flush=True)

    print("\n=== held-out RMSE by horizon (lower=better) + structural spread ===")
    print(f"{'horizon':>8s} | " + " | ".join(f"{p.replace('_regional',''):>7s}" for p in PROGRAMS)
          + " | spread  (rel%)")
    short_spreads, long_spreads = [], []
    for h in HORIZONS:
        vals = [results[p][f"h{h}"]["rmse"] for p in PROGRAMS]
        spread = max(vals) - min(vals)
        rel = 100 * spread / min(vals) if min(vals) > 0 else float("nan")
        best = PROGRAMS[int(np.argmin(vals))].replace("_regional", "")
        print(f"{('h'+str(h)):>8s} | " + " | ".join(f"{v:7.4f}" for v in vals)
              + f" | {spread:.4f} ({rel:4.0f}%)  best={best}")
        (short_spreads if h <= 4 else long_spreads).append(rel)

    print("\n=== peak-week / peak-intensity error (structure-sensitive) ===")
    for p in PROGRAMS:
        pk = results[p].get("peak", {})
        print(f"  {p:16s} peak_week_mae={pk.get('week_mae')}  peak_intensity_mae={pk.get('intensity_mae')}")

    s_short = float(np.nanmean(short_spreads)) if short_spreads else float("nan")
    s_long = float(np.nanmean(long_spreads)) if long_spreads else float("nan")
    print("\n=== VERDICT ===")
    print(f"mean relative spread  short-horizon (1-4wk): {s_short:.0f}%   "
          f"long-horizon (8-16wk): {s_long:.0f}%")
    if s_long > 2.5 * max(s_short, 1e-6):
        print("-> RUGGED at long horizons: structure differentiates the held-out forecast. "
              "Retarget FluZoo to long-horizon / peak targets.")
    elif s_long > 1.5 * max(s_short, 1e-6):
        print("-> MODESTLY ruggedizes at long horizons; worth pursuing the harder target.")
    else:
        print("-> Still FLAT at long horizons: flu forecasting is intrinsically structure-insensitive "
              "here; the *search* flagship wants a structure-decisive domain.")


if __name__ == "__main__":
    main()
