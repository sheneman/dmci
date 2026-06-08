############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment I configuration: structure-agnostic calibration of runtime-generated ecological models via DMCI....
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment I configuration: structure-agnostic calibration of runtime-generated
ecological models via DMCI.

This is the SCOPED, F/G-tied experiment (NOT a wholesale FATES port). The headline
claim is capability + engineering-cost, never optimizer speed (DMCI is a measured
73-248x slower than direct compilation; we report that openly).

The pilot (run_pilot.py) is a <2h go/no-go gate before committing the full grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- Environmental drivers (FATES leaf-biophysics inputs) -------------------
# Q   = photosynthetically active radiation / PAR  [umol photons m-2 s-1]
# T   = leaf temperature                            [deg C]
# psi = leaf/soil water potential (proxy: -|VPD| or SWC at flux towers) [MPa]
DRIVERS = ["Q", "T", "psi"]

DRIVER_RANGES = {
    "Q": (50.0, 2000.0),
    "T": (5.0, 38.0),
    "psi": (-3.0, -0.1),
}


@dataclass
class ExpIConfig:
    # fit loop (reuses the just-fixed F/G loop semantics)
    n_data_points: int = 24          # capped at <=24: cost is linear in n_data
    max_epochs: int = 2000
    lr: float = 0.02                 # between B/C's 0.05 and F's 0.01
    convergence_threshold: float = 1e-3
    early_stop_patience: int = 200
    grad_clip: float = 10.0
    noise_std: float = 0.0           # pilot = clean synthetic data

    # model structure
    n_pft: int = 2                   # 2 PFTs x 6 params = 12 free params (pilot scale)

    # black-box baseline (scipy.optimize.differential_evolution — CONFIRMED installed;
    # cma/skopt/nevergrad are NOT installed, so DE is the realistic black-box family)
    de_maxiter: int = 200
    de_popsize: int = 15

    # equal-wall-clock budget for the fair comparison (seconds per structure per method)
    # set in the full experiment from the measured DMCI median; pilot just records times
    wall_clock_budget_s: float = 900.0   # ~15 min, ~2x expected DMCI static-fit median

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


DEFAULT = ExpIConfig()


# --- Pilot go/no-go thresholds (from the one-pager) -------------------------
@dataclass
class PilotGate:
    # (1) >= frac_seeds_converge of seeds reach predictive MSE < mse_go
    mse_go: float = 1e-3
    frac_seeds_converge: float = 2 / 3
    # (2) mean DMCI fit wall-clock under this many minutes
    fit_minutes_go: float = 45.0
    # (3) DMCI vs direct-compile predictions agree (sanity that the interpreter path
    #     is numerically correct) within this relative tolerance at the fitted params
    path_agreement_rtol: float = 1e-3
    # (4) no NaN/Inf-induced stalls (the finite-guard must hold)
    # (5) lambdify->JAX needs > 0 per-structure human work on a recursive structure
    # (6) a real AmeriFlux US-Ha1 light-response fit is obtainable
    # (5)/(6) are evaluated qualitatively by run_pilot and the AmeriFlux loader stub.


PILOT_GATE = PilotGate()
