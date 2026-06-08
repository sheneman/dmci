############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment LIM-ENSO configuration: single source of truth. This experiment fits a Linear Inverse Model (LIM) /...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment LIM-ENSO configuration: single source of truth.

This experiment fits a Linear Inverse Model (LIM) / linear-Gaussian state-space
model to tropical Indo-Pacific SST EOF/PC time series, with the FULL Kalman-filter
negative log-likelihood folded THROUGH the DMCI meta-circular interpreter so that
gradients w.r.t. the transition operator F and the noise covariances Q, R flow back
from the accumulated NLL through the compiled evaluator (program-as-data). It is the
LIM generalization of the verified det/inv Kalman pilot
(experiments/pilots/kalman_detinv_pilot.py): F=I local-level -> a dense learnable F.

The headline is CAPABILITY (a real-data dynamical-systems MLE expressed as a program
the LLM can emit, optimized by exact DMCI gradients), never optimizer wall-clock. DMCI
is interpreter-bound and run on CPU.

Everything (data build, params, models, reference, gate, run) imports its constants
from here -- change a tolerance or a dimension in ONE place. All numeric work that
touches the interpreter is FLOAT32 (DMCI is float32-native; tagged tensors are hardcoded
torch.float32 in runtime/tagged_value.py). The reference twin runs in BOTH float32
(tight DMCI parity) and float64 (accurate finite-difference) -- see reference.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- Spatial / temporal domain (ERSSTv5; preprocessed by build_data.py) ------
# Tropical Indo-Pacific: the ENSO + Indian-Ocean basin that LIMs are classically
# fit on (Penland & Sardeshmukh 1995). Longitudes are degrees EAST (30 .. 290).
DOMAIN_LAT = (-30.0, 30.0)
DOMAIN_LON = (30.0, 290.0)

# Default analysis period (inclusive month bounds, 'YYYY-MM'). The full PC series in
# pcs.npy spans this; the harness slices a T_train window then a disjoint T_test window.
PERIOD = ("1950-01", "2024-12")

# ERSSTv5 monthly-mean SST source (verified: 152,446,809 bytes, HDF5/netCDF-4).
DATA_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"


@dataclass
class ExpLimConfig:
    """All knobs for the LIM-ENSO experiment. ``DEFAULT`` below is the canonical instance."""

    # --- domain / period (mirrors the module-level constants; kept here so a single
    #     config object fully determines a run and can be serialized into metadata) ---
    domain_lat: tuple[float, float] = DOMAIN_LAT
    domain_lon: tuple[float, float] = DOMAIN_LON
    period: tuple[str, str] = PERIOD
    data_url: str = DATA_URL

    # --- state dimension sweep ---------------------------------------------------
    # We fit at several PC-truncation dimensions D. The SVD/EOF basis is computed ONCE
    # at D_max (nested basis); each D run slices pcs[:, :D]. D_max bounds the heap/cost.
    D_list: list[int] = field(default_factory=lambda: [6, 10, 15, 20])
    D_max: int = 20

    # --- fit / hold-out windows (in months) --------------------------------------
    # pcs.npy holds the full [T_full, D_max] series; the harness fits on the first
    # T_train months and evaluates predictive NLL on the next T_test months.
    T_train: int = 360               # ~30 yr fit window
    T_test: int = 120                # ~10 yr held-out window

    # --- optimizer (Adam on the raw F/Q/R parameters; exact DMCI gradients) -------
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    adam_iters: int = 300
    lr: float = 0.05
    grad_clip: float = 10.0          # guard against the occasional float32 spike
    stability_penalty: float = 0.0   # optional soft spectral-radius(F)<1 penalty weight

    # --- noise-covariance parametrization floors (keep the optimizer in the PD cone)
    q_floor: float = 1e-4            # Q = Lq Lq^T + q_floor*I  (Cholesky-parametrized)
    r_floor: float = 0.1            # R = softplus(r_raw)*I + r_floor*I  (PRIMARY float32 PD lever)

    # --- numerical-stability escalation (NEVER a redesign) -----------------------
    # If covariance loses PD at the real (D,T), gate.py escalates: raise r_floor ->
    # add jitter (S + jitter_eps*I before inv/det) -> float64. jitter is opt-in per run.
    jitter_eps: float = 1e-5

    # --- model-structure variants ------------------------------------------------
    # How F is ASSEMBLED from bound factor tensors (the LLM emits combine-algebra only;
    # never D x D literals). S0 dense is the Phase-1 deliverable; the rest are stubs with
    # defined raw-param shapes so AIC/BIC (which need the param count k) slot in cleanly.
    #   S0 dense        : F = raw.reshape(D, D)
    #   S1 diagonal     : F = diag(raw)
    #   S3 lowrank+diag : F = diag(d) + U V^T   (rank r)
    #   S4 AR2-companion: F = block-companion of AR(2) coefficients
    #   S5 sym/antisym  : F = 0.5(M+M^T) + 0.5(M-M^T)   (re-parametrized full F)
    structures: list[str] = field(default_factory=lambda: ["S0"])
    lowrank_rank: int = 2            # rank r used by S3 (lowrank+diag)

    # --- DMCI evaluation knobs (passed straight to evaluator.evaluate) -----------
    # Generous caps: a T_train-step LIM filter at D_max folds through the interpreter;
    # the loop is trampolined (O(T)) but the per-step heap traffic at D=20 is large.
    EVAL_KW: dict = field(default_factory=lambda: dict(
        max_iter=500000, max_depth=500000, max_heap=20_000_000))

    # --- recursion limit raised BEFORE importing neural_compiler (see run scripts).
    recursion_limit: int = 20000

    # --- live per-iteration logging cadence (solvers print a flushed progress line every
    #     ``log_every`` accepted iterations; purely diagnostic -- NEVER affects the numerics).
    log_every: int = 25


DEFAULT = ExpLimConfig()


# --- Gate thresholds (G1..G6) -----------------------------------------------
@dataclass
class GateThresholds:
    """Tolerances for gate.py, which re-runs the pilot's PD / parity / FD-gradient
    checks at the REAL (D, T) and emits gate_{D}.json {GO: bool, ...}. The MLE only
    runs on GO. These mirror the pilot's bars (float32 autograd vs float64 FD is a
    few-percent regime), tightened/loosened for the higher-dimensional LIM setting.

    (Codes match gate.py exactly. GO = G1..G5 pass on every dataset that ran; G6 never blocks.)
    G1  NLL finite    : accumulated NLL has no NaN/Inf over the T-step filter
    G2  PD            : det(S) > detS_floor at EVERY filter step (covariance stays PD)
    G3  parity        : |NLL_dmci - NLL_ref32| <= parity_rel * max(1, |NLL_ref32|)
    G4  FD-gradient   : ||g_dmci - g_fd64|| / ||g_fd64|| <= fd_rel  (subset of F/Q/R params)
    G5  underflow     : min det(S) comfortably above the float32 denormal floor detS_floor
    G6  batched bind  : [N,D,D] F binding evaluates finite (DiffEvo path) -- NON-BLOCKING
    """

    # G1 forward parity: float32 DMCI vs the float32 numpy twin (IDENTICAL arithmetic order)
    parity_rel: float = 2e-3

    # G2 positive-definiteness: det(S) must stay above the float32 denormal floor with margin.
    # float32 smallest normal ~1.18e-38; we require a 1.2e-38 margin so log(det S) is finite.
    detS_floor: float = 1.2e-38

    # G3 finite-difference gradient agreement (float32 autograd vs float64 central FD).
    fd_rel: float = 3e-2
    fd_eps: float = 1e-3             # central-difference step on raw params (float64 twin)
    fd_n_probe: int = 6             # number of raw-param coordinates probed (random subset)

    # G5 symmetry tolerance on reference covariance matrices (max |M - M^T| / max|M|).
    sym_tol: float = 1e-3

    # spectral radius of F that we WARN above (non-blocking; an explosive F still gates on PD).
    spectral_radius_warn: float = 1.05


GATE = GateThresholds()
