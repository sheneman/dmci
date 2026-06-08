############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# forecast.py: Held-out forecast skill + LIM scientific diagnostics for the fitted operator. This is the SCIENTIFIC pay-off...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Held-out forecast skill + LIM scientific diagnostics for the fitted operator.

This is the SCIENTIFIC pay-off layer of exp_lim_enso: it takes the (F, Q, R) the DMCI
MLE recovered on the train window (run.py / the fit harness emits these), runs a Kalman
analysis over the held-out window with the SAME fitted parameters, makes h-step-ahead
forecasts ``x_hat_{t+h} = F^h x_hat_t``, reconstructs PHYSICAL SST anomalies and the
Nino-3.4 index from the PC/EOF basis, and scores forecast skill (ACC, RMSE) against three
references: persistence, damped persistence, and the classical one-shot Green-function LIM
``G = C(tau) C(0)^-1``. It also runs the standard LIM operator diagnostics on F (continuous
generator ``L = logm(F)/dt``, mode decay timescales / oscillation periods, the ENSO-mode
flag, the tau-test linearity check, and the optimal-growth singular vector of ``G(tau)``).

IMPORTANT: NOTHING here goes through the DMCI interpreter. The interpreter's job was to
produce exact gradients for the MLE; the fitted (F, Q, R) it returned are ordinary float
matrices, so all of the prediction / reconstruction / eigen-analysis below is pure
numpy/scipy. The Green-function ``G`` is fit by MOMENTS on the train window and is the
SCIENTIFIC REFERENCE operator (Penland & Sardeshmukh 1995), NOT an NLL competitor; the
numpy Kalman filter mirrors reference.reference_nll_lim term-for-term.

Public API
----------
``forecast_skill(fitted, pcs, eofs, pc_std, lat, lon, cfg=DEFAULT, leads=(3,6,9,12)) -> dict``
    Structured skill report: per-lead ACC / RMSE of the Nino-3.4 index and of the leading
    PCs for the fitted-LIM forecast and for each baseline, on the held-out window.

``diagnostics(F, dt=1.0) -> dict``
    LIM operator diagnostics on the fitted transition matrix F (one month per step => dt=1).

Conventions taken from data/processed/metadata.json + preprocess_lim.py:
  * ``pcs``    [T_full, D_max] float32 UNIT-VARIANCE PCs (time on axis 0).
  * ``eofs``   [D_max, S]      spatial EOFs in sqrt(cos lat)-WEIGHTED space.
  * ``pc_std`` [D_max]         multiply a normalised PC column by this to get the raw
                               expansion coefficient.
  * ``lat,lon``[S]             per-ocean-column coords (lon in 0..360 East).
  Physical SST-anomaly field at column s for a normalised PC state ``x`` (length D):
      field[s] = ( sum_j  x[j] * pc_std[j] * eofs[j, s] )  /  sqrt(cos(lat[s]))
  i.e. un-normalise the PCs, project through the (weighted) EOFs, then UN-WEIGHT by the
  sqrt(cos-lat) area weight to land in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, ExpLimConfig

# Nino-3.4 box: lat[-5,5], lon[190,240] (=170W..120W in 0..360 East). All columns fall
# inside the tropical Indo-Pacific domain, so this is a plain masked area-weighted mean.
NINO34_LAT = (-5.0, 5.0)
NINO34_LON = (190.0, 240.0)

# Default forecast leads (months).
DEFAULT_LEADS = (3, 6, 9, 12)


# ===========================================================================
# Small numpy helpers (accept torch tensors or numpy arrays everywhere).
# ===========================================================================

def _np(x, dtype=np.float64):
    """Detach a torch tensor / accept a numpy array; return a contiguous float64 array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x), dtype=dtype)


def _unpack_fitted(fitted):
    """Pull F, Q, R (and an optional D) out of the ``fitted`` dict the MLE harness emits.

    ``fitted`` is expected to carry the DECODED matrices under keys 'F','Q','R' (numpy or
    torch). Accepts a few aliases. D is inferred from F unless given. Q/R are optional for
    the diagnostics path but required for the Kalman analysis; we surface a clear error if
    a needed one is missing rather than silently substituting."""
    if not isinstance(fitted, dict):
        raise TypeError("forecast: `fitted` must be a dict carrying the decoded F,Q,R "
                        "(e.g. {'F':..,'Q':..,'R':..,'D':..,'structure':..,'seed':..}).")
    def grab(*keys):
        for k in keys:
            if k in fitted and fitted[k] is not None:
                return _np(fitted[k])
        return None
    F = grab("F", "F_mat", "Fhat")
    if F is None:
        raise KeyError("forecast: `fitted` has no transition matrix under key 'F'.")
    D = int(fitted.get("D", F.shape[-1]))
    F = F.reshape(D, D)
    Q = grab("Q", "Q_mat")
    R = grab("R", "R_mat")
    if Q is not None:
        Q = Q.reshape(D, D)
    if R is not None:
        R = R.reshape(D, D)
    return F, Q, R, D


# ===========================================================================
# Physical reconstruction: PC state -> SST-anomaly field -> Nino-3.4 index.
# ===========================================================================

def _area_weight_cos(lat):
    """Per-column area weight ``w = sqrt(cos(lat))`` (the EOF inner-product weight)."""
    return np.sqrt(np.clip(np.cos(np.deg2rad(_np(lat))), 0.0, None))


def reconstruct_field(x, eofs, pc_std, lat):
    """Reconstruct the PHYSICAL SST-anomaly field [.., S] from normalised PC state ``x``.

    ``x`` may be a single state [D] or a batch of states [N, D]. ``eofs`` is [D_max, S]
    (weighted space), ``pc_std`` is [D_max], ``lat`` is [S]. We slice the basis to D=len(x)
    (the nested-basis convention), un-normalise the PCs, project, then DIVIDE by the
    sqrt(cos-lat) weight to recover physical units. Columns at |lat|>=90 (weight 0) are set
    to 0 (none exist in this tropical domain). Returns float64 [.., S]."""
    x = _np(x)
    eofs = _np(eofs)
    pc_std = _np(pc_std)
    w = _area_weight_cos(lat)                      # [S]
    single = (x.ndim == 1)
    if single:
        x = x[None, :]                              # [1, D]
    D = x.shape[1]
    E = eofs[:D, :]                                 # [D, S] weighted EOFs
    s = pc_std[:D]                                  # [D]
    coeff = x * s[None, :]                          # [N, D] raw expansion coefficients
    field_w = coeff @ E                             # [N, S] WEIGHTED-space field
    safe_w = np.where(w > 0, w, 1.0)
    field = field_w / safe_w[None, :]              # [N, S] physical
    field[:, w <= 0] = 0.0
    return field[0] if single else field


def nino34_index(x, eofs, pc_std, lat, lon):
    """Area-weighted mean SST anomaly over the Nino-3.4 box for normalised PC state ``x``.

    ``x`` may be [D] or [N, D]. Reconstructs the physical field then takes the cos(lat)
    AREA-weighted mean over the columns inside lat[-5,5], lon[190,240]. Returns a float64
    scalar (single state) or [N] vector (batch). The mask is precomputed from lat/lon."""
    lat = _np(lat)
    lon = _np(lon)
    in_box = ((lat >= NINO34_LAT[0]) & (lat <= NINO34_LAT[1]) &
              (lon >= NINO34_LON[0]) & (lon <= NINO34_LON[1]))
    if not in_box.any():
        raise ValueError("nino34_index: no ocean columns fall inside the Nino-3.4 box "
                         f"lat{NINO34_LAT} lon{NINO34_LON} -- check lat/lon arrays.")
    field = reconstruct_field(x, eofs, pc_std, lat)        # [..,S]
    # area (NOT sqrt) weight for a physical area-mean over the box.
    aw = np.clip(np.cos(np.deg2rad(lat[in_box])), 0.0, None)
    aw = aw / aw.sum()
    if field.ndim == 1:
        return float(field[in_box] @ aw)
    return field[:, in_box] @ aw                            # [N]


# ===========================================================================
# Kalman analysis over a window (numpy twin of reference.reference_nll_lim).
# ===========================================================================

def kalman_analysis_states(F, Q, R, ys, *, x0=None, P0=None, return_cov=False):
    """Run the LIM Kalman FILTER over ``ys`` [n, D]; return analysis states ``x_hat`` [n, D].

    Mirrors reference.reference_nll_lim's recursion EXACTLY (predict: xpred=Fx,
    Ppred=F P F^T + Q; update with S=Ppred+R, K=Ppred S^-1, x=xpred+K e, P=(I-K)Ppred),
    but collects the per-step analysis (a-posteriori) state rather than the NLL. ``x0``/``P0``
    let the held-out window be WARM-STARTED from the end of the train window (the physically
    correct way to produce held-out analysis states); default x0=0, P0=I as in the filter.
    Returns ``x_hat`` [n, D] (and, if ``return_cov``, the final (x, P) to chain windows)."""
    F = _np(F); Q = _np(Q); R = _np(R)
    Y = _np(ys)
    D = F.shape[0]
    Y = Y.reshape(-1, D)
    n = Y.shape[0]
    I = np.eye(D)
    x = np.zeros(D) if x0 is None else _np(x0).reshape(D)
    P = np.eye(D) if P0 is None else _np(P0).reshape(D, D)
    out = np.zeros((n, D))
    for k in range(n):
        xpred = F @ x
        Ppred = (F @ P) @ F.T + Q
        e = Y[k] - xpred
        S = Ppred + R
        Sinv = np.linalg.inv(S)
        K = Ppred @ Sinv
        x = xpred + K @ e
        P = (I - K) @ Ppred
        out[k] = x
    if return_cov:
        return out, x, P
    return out


def _matpow(F, h):
    """Integer matrix power ``F^h`` (h>=0; F^0 = I). Plain repeated squaring via np."""
    return np.linalg.matrix_power(_np(F), int(h))


# ===========================================================================
# Skill metrics.
# ===========================================================================

def _acc(pred, truth):
    """Anomaly correlation coefficient (centred Pearson r) between two 1-D series."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    p = pred - pred.mean()
    t = truth - truth.mean()
    denom = np.sqrt((p * p).sum() * (t * t).sum())
    return float((p * t).sum() / denom) if denom > 0 else float("nan")


def _rmse(pred, truth):
    """Root-mean-square error between two series."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _skill_pair(pred, truth):
    return {"ACC": _acc(pred, truth), "RMSE": _rmse(pred, truth), "n": int(len(pred))}


# ===========================================================================
# Green-function LIM (the scientific reference operator): G = C(tau) C(0)^-1.
# ===========================================================================

def green_function_lim(train_pcs, tau=1):
    """Fit the one-shot LIM propagator ``G(tau) = C(tau) C(0)^-1`` by moments on TRAIN PCs.

    ``C(0) = <x_t x_t^T>`` and ``C(tau) = <x_{t+tau} x_t^T>`` are the zero-lag and lag-tau
    covariance matrices estimated on the train window (rows of ``train_pcs`` are time). This
    is the classical Penland-Sardeshmukh LIM operator, fit WITHOUT any likelihood -- it is the
    SCIENTIFIC reference forecaster, not an NLL competitor. Returns ``G`` [D, D] (float64)."""
    X = _np(train_pcs)
    n = X.shape[0]
    if n <= tau + 1:
        raise ValueError(f"green_function_lim: train window n={n} too short for tau={tau}.")
    x0 = X[:n - tau]                       # x_t        [n-tau, D]
    xt = X[tau:]                           # x_{t+tau}  [n-tau, D]
    m = x0.shape[0]
    C0 = (x0.T @ x0) / m                    # <x_t x_t^T>
    Ctau = (xt.T @ x0) / m                 # <x_{t+tau} x_t^T>
    return Ctau @ np.linalg.inv(C0)


# ===========================================================================
# Public: forecast skill.
# ===========================================================================

def forecast_skill(fitted, pcs, eofs, pc_std, lat, lon,
                   cfg: ExpLimConfig = DEFAULT, leads=DEFAULT_LEADS) -> dict:
    """Held-out h-step forecast skill of the fitted LIM vs three baselines.

    Args:
        fitted: dict from the MLE harness carrying the DECODED ``F``,``Q``,``R`` matrices
            (numpy or torch) and optionally ``D``,``structure``,``seed`` (passed through).
        pcs:    [T_full, D_max] normalised PC series (time on axis 0); float32/64 ok.
        eofs:   [D_max, S] weighted spatial EOFs.
        pc_std: [D_max] PC un-normalisation factors.
        lat,lon:[S] per-ocean-column coords (lon 0..360 East).
        cfg:    ExpLimConfig (uses ``T_train`` and ``T_test``).
        leads:  iterable of forecast lead times in MONTHS (default 3,6,9,12).

    Returns a structured dict:
        {
          "meta": {D, T_train, T_test, leads, structure, seed, n_test, ...},
          "nino34": { lead: { "fitted_lim": {ACC,RMSE,n}, "persistence": {...},
                              "damped_persistence": {...}, "green_lim": {...} }, ... },
          "pcs":    { lead: { method: { "per_pc_ACC":[..], "per_pc_RMSE":[..],
                                        "ACC_pc0":.., "RMSE_pc0":.., "mean_ACC":.. } } },
          "series": { lead: { "truth": [...], "fitted_lim": [...], ... } }   # Nino-3.4 traces
        }

    Method:
      * Split: first ``T_train`` months train; the next ``T_test`` (or the remainder, capped)
        held out, WARM-STARTED by chaining the Kalman filter from the train window into the
        held-out window (so held-out analysis states use only past data).
      * For each held-out analysis time t and lead h: fitted forecast ``F^h x_hat_t``;
        persistence ``x_hat_t``; damped persistence ``alpha_h * x_hat_t`` with ``alpha_h`` =
        the per-PC lag-h autocorrelation estimated on TRAIN; Green-function ``G(1)^h x_hat_t``.
      * Truth at the verifying time is the held-out analysis state ``x_hat_{t+h}`` (the
        observed PC field filtered with the same fitted model). Skill is ACC + RMSE of the
        Nino-3.4 index and of the leading PCs, over all verifiable (t, t+h) pairs.
    """
    F, Q, R, D = _unpack_fitted(fitted)
    if Q is None or R is None:
        raise KeyError("forecast_skill: `fitted` must include Q and R for the Kalman "
                       "analysis (the held-out states are filtered with the fitted model).")

    X = _np(pcs)[:, :D]                                     # [T_full, D] normalised PCs
    T_full = X.shape[0]
    T_train = int(cfg.T_train)
    if T_train >= T_full:
        raise ValueError(f"forecast_skill: T_train={T_train} >= available T={T_full}.")
    T_test = min(int(cfg.T_test), T_full - T_train)
    leads = tuple(int(h) for h in leads)
    structure = fitted.get("structure", "S0") if isinstance(fitted, dict) else "S0"
    seed = fitted.get("seed", None) if isinstance(fitted, dict) else None

    train_X = X[:T_train]
    test_X = X[T_train:T_train + T_test]

    # --- 1) Kalman analysis: warm-start the held-out filter from the end of train. ---
    _, x_end, P_end = kalman_analysis_states(F, Q, R, train_X, return_cov=True)
    xhat = kalman_analysis_states(F, Q, R, test_X, x0=x_end, P0=P_end)   # [T_test, D]
    n_test = xhat.shape[0]

    # --- 2) baselines fit on TRAIN ---
    # damped-persistence per-PC lag-h autocorrelation alpha_h (on train PCs).
    def lag_autocorr_train(h):
        a = np.zeros(D)
        for j in range(D):
            c = train_X[:, j] - train_X[:, j].mean()
            v = float((c * c).sum())
            if v > 0 and h < len(c):
                a[j] = float((c[:-h] * c[h:]).sum() / v)
        return a                                            # [D]

    G1 = green_function_lim(train_X, tau=1)                 # Green-function propagator

    # --- 3) precompute forecast operators per lead ---
    Fh = {h: _matpow(F, h) for h in leads}                 # fitted LIM h-step
    Gh = {h: np.linalg.matrix_power(G1, h) for h in leads} # Green-function h-step
    alpha = {h: lag_autocorr_train(h) for h in leads}      # damped-persistence factors

    # --- 4) score every lead ---
    nino_report: dict = {}
    pc_report: dict = {}
    series_report: dict = {}

    # precompute the Nino-3.4 index of every held-out analysis state (the "truth" pool).
    nino_truth_all = nino34_index(xhat, eofs, pc_std, lat, lon)   # [n_test]

    for h in leads:
        m = n_test - h                                      # verifiable (t, t+h) pairs
        if m <= 1:
            continue
        src = xhat[:m]                                      # analysis states x_hat_t  [m, D]
        tgt = xhat[h:h + m]                                 # verifying states x_hat_{t+h} [m, D]

        # PC-space forecasts: [m, D]
        pred_fit = src @ Fh[h].T
        pred_per = src.copy()
        pred_damp = src * alpha[h][None, :]
        pred_grn = src @ Gh[h].T

        # Nino-3.4 truth + forecasts (reconstruct each [m,D] batch to a [m] index series).
        nino_tgt = nino_truth_all[h:h + m]
        nino_methods = {
            "fitted_lim": nino34_index(pred_fit, eofs, pc_std, lat, lon),
            "persistence": nino_truth_all[:m],              # x_hat_t index == persistence
            "damped_persistence": nino34_index(pred_damp, eofs, pc_std, lat, lon),
            "green_lim": nino34_index(pred_grn, eofs, pc_std, lat, lon),
        }
        nino_report[h] = {name: _skill_pair(v, nino_tgt) for name, v in nino_methods.items()}
        series_report[h] = {"truth": nino_tgt.tolist(),
                            **{name: np.asarray(v).ravel().tolist()
                               for name, v in nino_methods.items()}}

        # leading-PC skill (per-PC ACC/RMSE) for each method.
        pc_methods = {"fitted_lim": pred_fit, "persistence": pred_per,
                      "damped_persistence": pred_damp, "green_lim": pred_grn}
        pc_report[h] = {}
        for name, pred in pc_methods.items():
            per_acc = [_acc(pred[:, j], tgt[:, j]) for j in range(D)]
            per_rmse = [_rmse(pred[:, j], tgt[:, j]) for j in range(D)]
            pc_report[h][name] = {
                "per_pc_ACC": per_acc,
                "per_pc_RMSE": per_rmse,
                "ACC_pc0": per_acc[0],
                "RMSE_pc0": per_rmse[0],
                "mean_ACC": float(np.nanmean(per_acc)),
                "mean_RMSE": float(np.nanmean(per_rmse)),
            }

    return {
        "meta": {
            "D": D, "structure": structure, "seed": seed,
            "T_train": T_train, "T_test": T_test, "n_test": int(n_test),
            "leads": list(leads),
            "nino34_box": {"lat": list(NINO34_LAT), "lon": list(NINO34_LON)},
            "green_tau": 1,
        },
        "nino34": nino_report,
        "pcs": pc_report,
        "series": series_report,
    }


# ===========================================================================
# Public: LIM operator diagnostics on the fitted F.
# ===========================================================================

@dataclass
class _Mode:
    """One eigen-mode of the continuous generator L = logm(F)/dt."""
    index: int
    eig_real: float          # Re(lambda) of L  (1/month);  <0 == damped/stable
    eig_imag: float          # Im(lambda) of L  (rad/month)
    decay_months: float      # -1/Re(lambda)    (e-folding decay timescale, months)
    period_months: float     # 2*pi/|Im(lambda)| (oscillation period, months; inf if real)
    growth_F: float          # |eigenvalue of F| (per-step amplification; <1 == stable)


def diagnostics(F, dt: float = 1.0) -> dict:
    """LIM operator diagnostics on the fitted transition matrix ``F`` (dt = months/step).

    Computes:
      * the continuous-time generator ``L = logm(F) / dt`` (matrix logarithm);
      * per-mode DECAY TIMESCALES ``-1/Re(lambda)`` and OSCILLATION PERIODS
        ``2*pi/|Im(lambda)|`` from the eigenvalues of L;
      * an ENSO-MODE FLAG: among OSCILLATORY modes in the canonical ENSO band (period 2-7 yr =
        24-84 mo, e-folding decay 4-30 mo), the LEAST-DAMPED one. NOTE: this is deliberately NOT
        the globally least-damped eigenmode -- that is a near-stationary decadal mode (~20-30 yr)
        which is NOT ENSO. The honest claim is "the least-damped oscillatory mode in the ENSO
        band", never "the least-damped mode is ENSO".
      * the TAU-TEST: compare ``G(2*tau)`` against ``G(tau)^2`` (with ``G(tau)=F`` here, the
        one-step propagator), a sanity check that the dynamics are linear/Markov -- a large
        ``||F^2 - F@F||`` would indicate a non-LIM fit (trivially ~0 for a matrix F, so we
        ALSO report the data-moment version when train PCs are available via forecast_skill);
      * the OPTIMAL-GROWTH singular vector of ``G(tau)=F``: the right singular vector ``v``
        maximising ``||F v|| / ||v||`` (the initial pattern that amplifies most in one step),
        from the SVD of F.

    Returns a JSON-friendly dict. ``dt=1`` => timescales/periods are in MONTHS.
    """
    from scipy.linalg import logm

    Fm = _np(F)
    D = Fm.shape[0]

    # --- continuous generator L = logm(F)/dt (diagnostic only; the mode timescales below come
    #     from log(eig(F)), which is robust to a singular F -- e.g. an AR(2)-companion operator
    #     with a zero trailing coefficient -- so guard the matrix-log and never let it crash). ---
    try:
        Lc = np.asarray(logm(Fm), dtype=np.complex128) / float(dt)
        evL, _ = np.linalg.eig(Lc)
    except Exception:  # noqa: BLE001  (singular/ill-conditioned F -> matrix log undefined)
        Lc = evL = None
    evF = np.linalg.eigvals(Fm)                            # per-step eigenvalues of F

    # pair F-eigenvalues to L-eigenvalues by magnitude/phase: lambda_L = log(lambda_F)/dt.
    # (np.linalg.eig orderings differ; recompute L-eigs directly from F-eigs to be robust.)
    evL_from_F = np.log(evF.astype(np.complex128)) / float(dt)

    modes: list[_Mode] = []
    for i, (lamL, lamF) in enumerate(zip(evL_from_F, evF)):
        re = float(lamL.real)
        im = float(lamL.imag)
        decay = float(-1.0 / re) if re < 0 else float("inf")     # stable -> positive months
        period = float(2.0 * np.pi / abs(im)) if abs(im) > 1e-9 else float("inf")
        modes.append(_Mode(index=i, eig_real=re, eig_imag=im,
                           decay_months=decay, period_months=period,
                           growth_F=float(abs(lamF))))

    # --- ENSO-mode flag: oscillatory mode in the canonical ENSO band (period 2-7 yr = 24-84 mo,
    #     e-folding decay 4-30 mo), pick the LEAST-DAMPED (largest decay timescale) such mode;
    #     else report the least-damped oscillatory mode overall as a fallback (flagged
    #     not-in-band). The band is WIDER than 36-60/6-12 so it robustly captures both the ~4-yr
    #     and the ~2.5-yr ENSO oscillations (the narrow band dropped the genuine 2.6-yr mode and
    #     went False at D=6). It still excludes the decadal/near-stationary low-frequency mode. ---
    ENSO_PERIOD_MO = (24.0, 84.0)      # 2-7 yr
    ENSO_DECAY_MO = (4.0, 30.0)        # several months to ~2.5 yr e-folding
    osc = [m for m in modes if np.isfinite(m.period_months) and m.growth_F < 1.0]
    in_band = [m for m in osc if ENSO_PERIOD_MO[0] <= m.period_months <= ENSO_PERIOD_MO[1]
               and ENSO_DECAY_MO[0] <= m.decay_months <= ENSO_DECAY_MO[1]]
    enso_mode = None
    enso_in_band = False
    if in_band:
        enso = max(in_band, key=lambda m: m.decay_months)
        enso_in_band = True
    elif osc:
        # least-damped oscillatory mode overall (largest decay timescale)
        enso = max(osc, key=lambda m: (m.decay_months if np.isfinite(m.decay_months) else 1e9))
    else:
        enso = None
    if enso is not None:
        enso_mode = {
            "index": enso.index,
            "period_months": enso.period_months,
            "period_years": enso.period_months / 12.0 if np.isfinite(enso.period_months) else None,
            "decay_months": enso.decay_months,
            "growth_F": enso.growth_F,
            "in_enso_band": enso_in_band,
        }

    # --- tau-test (operator form): G(2tau) vs G(tau)^2 with G(tau)=F (one-step propagator).
    #     A pure linear-Markov LIM satisfies G(2tau) = G(tau)^2 EXACTLY at the operator level
    #     (here trivially, since we PROPAGATE with F: F^2 == F@F up to float error). The
    #     scientifically meaningful tau-test compares the DATA-fit Green functions G(2),G(1)^2
    #     -- exposed via forecast_skill's train moments / green_function_lim; we report the
    #     operator residual here as a numerical-consistency check. ---
    F2 = Fm @ Fm
    tau_resid = float(np.linalg.norm(F2 - np.linalg.matrix_power(Fm, 2)) /
                      max(1e-12, np.linalg.norm(F2)))

    # --- optimal growth: leading right singular vector of F (max ||F v||/||v|| = sigma_max). ---
    U, sv, Vt = np.linalg.svd(Fm)
    opt_growth = {
        "max_amplification": float(sv[0]),                 # sigma_max(F) = max ||Fv||/||v||
        "init_singular_vector": Vt[0].real.tolist(),       # v  (the optimal initial pattern)
        "evolved_pattern": (U[:, 0] * sv[0]).real.tolist(),# F v  (the amplified response)
        "singular_values": sv.tolist(),
    }

    spectral_radius = float(np.max(np.abs(evF)))

    return {
        "D": D, "dt_months": float(dt),
        "spectral_radius_F": spectral_radius,
        "stable": bool(spectral_radius < 1.0),
        "generator_L_eigs": [{"real": float(z.real), "imag": float(z.imag)}
                             for z in evL_from_F],
        "modes": [vars(m) for m in
                  sorted(modes, key=lambda m: (m.decay_months
                                               if np.isfinite(m.decay_months) else -1.0),
                         reverse=True)],
        "enso_mode": enso_mode,
        "tau_test": {
            "operator_residual_rel": tau_resid,
            "note": ("operator-level F^2 vs F@F (trivially ~0); the data tau-test compares "
                     "green_function_lim(train,tau=2) against green_function_lim(train,tau=1)^2"),
        },
        "optimal_growth": opt_growth,
    }


def tau_test_data(train_pcs, tau: int = 1) -> dict:
    """Data-moment tau-test: how close is ``G(2tau)`` to ``G(tau)^2`` on the TRAIN window?

    This is the scientifically meaningful linearity/Markov check (Penland & Sardeshmukh): if
    the SST dynamics are well-described by a linear LIM, the lag-2tau Green function should
    equal the squared lag-tau Green function. Returns the relative Frobenius residual plus
    both operators' spectral radii. Optional companion to ``diagnostics`` (which only sees F).
    """
    G_tau = green_function_lim(train_pcs, tau=tau)
    G_2tau = green_function_lim(train_pcs, tau=2 * tau)
    G_tau2 = G_tau @ G_tau
    resid = float(np.linalg.norm(G_2tau - G_tau2) / max(1e-12, np.linalg.norm(G_2tau)))
    return {
        "tau": int(tau),
        "residual_rel": resid,
        "spectral_radius_G_tau": float(np.max(np.abs(np.linalg.eigvals(G_tau)))),
        "spectral_radius_G_2tau": float(np.max(np.abs(np.linalg.eigvals(G_2tau)))),
        "linear_markov_consistent": bool(resid < 0.2),    # loose heuristic threshold
    }
