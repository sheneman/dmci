############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Config for the battery capacity-fade DMCI de-risk pilot. The pilot reuses FluZoo's structure-agnostic machinery...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Config for the battery capacity-fade DMCI de-risk pilot.

The pilot reuses FluZoo's structure-agnostic machinery (parse_program / paramspec /
calibrate / run_nll / run_predict). We only override the knobs that differ for a per-CYCLE
degradation rollout: a single observed series per cell (n_regions = 1), a shorter Adam
budget, and the same generous interpreter caps (the rollout still conses ~linearly with
interpreted cycles, so the 12M-cell heap matters).

The held-out forecast protocol is fit-early / forecast-late: calibrate a structure's
parameters on cycles [0, KSPLIT) through DMCI, then score its EXTRAPOLATION on the held-out
tail [KSPLIT, T_CYCLES). This is the discriminator -- in-sample every smooth structure fits
an early-life arc, so the signal lives past the split (the FluZoo short-vs-long lesson).
"""

from __future__ import annotations

import dataclasses

from experiments.exp_fluzoo.config import DEFAULT

# A single capacity series per cell -> the observation matrix is obs[T_cycles, 1].
N_SERIES = 1

# Common cycle grid for all synthetic cells. Kept short (the meta-circular interpreter costs
# ~1.7 s per fwd+bwd over a 60-cycle fold, so the fit-window length is the wall-clock lever);
# 100 cycles is enough for a clear knee (~cycle 55) while keeping each eval <= ~2 s.
T_CYCLES = 100

# Two held-out splits, reported separately (the honest nuance):
#   KSPLIT_LATE  -- fit window CONTAINS the knee (~55); knee-capable structures should win.
#   KSPLIT_EARLY -- fit window ends BEFORE the knee; pure extrapolation to an unseen knee
#                   (a known-hard regime; smooth families are still separable by curvature).
KSPLIT_LATE = 70
KSPLIT_EARLY = 45

# Battery-tuned calibration: reuse the validated Adam loop, fewer iters (the rollouts are
# small) at the FluZoo learning rate. Everything else (grad_clip, conv_tol, var floor,
# recursion limit, eval caps, EVAL_KW) is inherited from the FluZoo DEFAULT unchanged.
BCFG = dataclasses.replace(
    DEFAULT,
    n_regions=N_SERIES,
    adam_iters=400,
    lr=0.05,
    seeds=(0, 1, 2),
)

# Gaussian-NLL variance floor literal used inside the Scheme models (float32 safety).
FLOOR = repr(DEFAULT.obs_var_floor)
