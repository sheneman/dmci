############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# curves.py: Closed-form capacity curves for each structure, in numpy. One source of truth shared by two callers: * synth.py...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Closed-form capacity curves for each structure, in numpy.

One source of truth shared by two callers:
  * synth.py  -- generate mechanism-labeled data from KNOWN ground-truth parameters.
  * score.py  -- read a forecast back from the DMCI-FITTED parameters (the fit itself goes
                 through the interpreter; the predicted curve is then a trivial arithmetic
                 readout, so scoring needs no extra per-horizon interpreter compiles).

These mirror the Scheme models in structures.py exactly; gradient_health() cross-checks that
the DMCI forward path (run_predict) agrees with predict_curve at sample horizons, so the
closed form is verified equivalent to the interpreter, not a separate model.
"""

from __future__ import annotations

import numpy as np


def predict_curve(name: str, p: dict, k: np.ndarray) -> np.ndarray:
    """Predicted capacity at cycles `k` for structure `name` given constrained params `p`."""
    k = np.asarray(k, dtype=np.float64)
    if name == "sqrt_t_SEI":
        return p["q0"] - p["B"] * np.sqrt(k)
    if name == "power_law_kp":
        return p["q0"] - p["B"] * np.power(k, p["p"])
    if name == "stretched_exp":
        return p["qinf"] + (p["q0"] - p["qinf"]) * np.exp(-np.power((k + 1e-6) / p["tau"], p["beta"]))
    if name == "two_reservoir_min":
        li = 1.0 - p["a"] * np.sqrt(k)
        host = 1.0 - p["c"] * k
        return p["q0"] * np.minimum(li, host)
    if name == "sigmoidal_knee":
        sig = 1.0 / (1.0 + np.exp(-(k - p["kn"]) / p["wd"]))
        return p["q0"] - p["m"] * k - p["d"] * k * sig
    raise KeyError(name)


# Ground-truth parameters per structure, tuned on the T_CYCLES=100 grid so every curve falls
# from ~1.0 to ~0.78 by end-of-life and the knee structures knee around cycle ~55.
TRUE_PARAMS: dict[str, dict] = {
    "sqrt_t_SEI":        {"q0": 1.0, "B": 0.0201},
    "power_law_kp":      {"q0": 1.0, "B": 0.00202, "p": 1.0},   # linear-ish wear
    "stretched_exp":     {"q0": 1.0, "qinf": 0.55, "tau": 65.0, "beta": 1.4},
    "two_reservoir_min": {"q0": 1.0, "a": 0.01648, "c": 0.00222},  # knee ~cycle 55
    "sigmoidal_knee":    {"q0": 1.0, "m": 4e-4, "d": 1.83e-3, "kn": 55.0, "wd": 8.0},
}

# Per-parameter multiplicative jitter (cell-to-cell heterogeneity); knee-location params get
# the most jitter so the knee moves cell-to-cell (the realistic, hard-to-forecast case).
JITTER = {
    "B": 0.15, "p": 0.08, "qinf": 0.08, "tau": 0.20, "beta": 0.10,
    "a": 0.15, "c": 0.15, "m": 0.15, "d": 0.15, "kn": 0.10, "wd": 0.20,
}
