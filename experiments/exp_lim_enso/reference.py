############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# reference.py: Numpy reference twin of the LIM Kalman-NLL folded through DMCI (models.kalman_lim_src). ``reference_nll_lim``...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Numpy reference twin of the LIM Kalman-NLL folded through DMCI (models.kalman_lim_src).

``reference_nll_lim`` is the ground-truth filter whose ARITHMETIC and ACCUMULATION ORDER
EXACTLY MIRROR the Scheme program so that:
  - the FLOAT32 twin gives tight forward parity vs DMCI (gate G1), and
  - the FLOAT64 twin gives an accurate central finite-difference gradient (gate G3).

It generalizes the pilot's local-level ``reference_nll`` (experiments/pilots/
kalman_detinv_pilot.py) to a full transition operator F:
    xpred = F @ x
    Ppred = F @ P @ F.T + Q
    e     = ys[k] - xpred
    S     = Ppred + R
    Sinv  = inv(S)                     (or inv(S + jitter_eps*I) when jitter=True)
    K     = Ppred @ Sinv
    x     = xpred + K @ e
    P     = (I - K) @ Ppred            (naive update, matching the Scheme)
    L    += log(det S) + e @ Sinv @ e  (det of the JITTERED S when jitter=True)

The op order is chosen to match the Scheme exactly (e.g. ``F @ P @ F.T`` is evaluated as
``(F @ P) @ F.T`` to mirror ``(matmul (matmul F P) (transpose F))``; the Mahalanobis term
is ``e @ (Sinv @ e)`` to mirror ``(dot e (matvec Sinv e))``).

DMCI is float32-native, so pass ``dtype=np.float32`` for parity and ``dtype=np.float64``
for finite differences.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT


def reference_nll_lim(F, Q, R, ys, D, T, *, return_traj: bool = False,
                      jitter: bool = False, jitter_eps: float = DEFAULT.jitter_eps,
                      dtype=np.float64):
    """LIM Kalman-filter negative log-likelihood, numpy twin of the DMCI program.

    Args:
        F: ``[D, D]`` transition matrix (numpy or torch -> cast to ``dtype``).
        Q: ``[D, D]`` process-noise covariance.
        R: ``[D, D]`` observation-noise covariance.
        ys: ``[T, D]`` observation series (rows are y_k; (ref obs k) gathers row k).
        D, T: state dimension and number of steps.
        return_traj: if True, also return the per-step ``det(S)`` list and a PD flag.
        jitter: if True, add ``jitter_eps*I`` to S before inv AND det (matches models.py).
        jitter_eps: the jitter magnitude (config default).
        dtype: ``np.float32`` (tight DMCI parity twin) or ``np.float64`` (accurate FD twin).

    Returns:
        ``L`` (scalar of ``dtype``)  -- the accumulated NLL  (when ``return_traj`` is False), or
        ``(L, dets, pd_ok)`` where ``dets`` is the list of per-step ``det(S)`` and ``pd_ok`` is
        True iff every ``det(S) > 0`` and finite (covariance stayed PD over the whole filter).

    The accumulation order, the naive ``(I - K) Ppred`` update, and the
    ``log(det S) + e^T S^-1 e`` NLL pieces mirror ``models.kalman_lim_src`` term-for-term.
    """
    Fm = np.asarray(_to_numpy(F), dtype=dtype).reshape(D, D)
    Qm = np.asarray(_to_numpy(Q), dtype=dtype).reshape(D, D)
    Rm = np.asarray(_to_numpy(R), dtype=dtype).reshape(D, D)
    Y = np.asarray(_to_numpy(ys), dtype=dtype).reshape(-1, D)[:T]

    I = np.eye(D, dtype=dtype)
    eps = dtype(jitter_eps)

    x = np.zeros(D, dtype=dtype)          # (zeros D) initial state
    P = np.eye(D, dtype=dtype)            # (eye D)  initial covariance
    L = dtype(0.0)
    dets: list = []
    pd_ok = True

    for k in range(T):
        xpred = Fm @ x                                  # (matvec F x)
        Ppred = (Fm @ P) @ Fm.T + Qm                    # (+ (matmul (matmul F P) (transpose F)) Q)
        e = Y[k] - xpred                                # (- y xpred)
        S = Ppred + Rm                                  # (+ Ppred R)
        Sj = S + eps * I if jitter else S               # (+ S (scale eps (eye D))) when jitter
        sign, logdet = np.linalg.slogdet(Sj)            # (logdet Sj)/(logdet S) -- mirrors the
        d = np.linalg.det(Sj)                           # Scheme's slogdet-based logdet term.
        dets.append(d)                                  # keep det for the gate's PD/underflow check
        if not np.isfinite(logdet) or sign <= 0:        # PD test: sign(det) > 0 and logdet finite
            pd_ok = False
        Sinv = np.linalg.inv(Sj)                        # (inv Sj)/(inv S)
        K = Ppred @ Sinv                                # (matmul Ppred Sinv)
        x = xpred + K @ e                               # (+ xpred (matvec Kg e))
        P = (I - K) @ Ppred                             # (matmul (- (eye D) Kg) Ppred)
        maha = e @ (Sinv @ e)                           # (dot e (matvec Sinv e))
        L = L + (logdet + maha)                         # (+ (logdet S) maha)

    return (L, dets, pd_ok) if return_traj else L


def _to_numpy(x):
    """Accept a numpy array or a torch tensor (detached) and return a numpy array."""
    if hasattr(x, "detach"):           # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ===========================================================================
# OPTIONAL: dynamax cross-check stub (NOT wired into the gate).
# ===========================================================================

def reference_nll_dynamax(F, Q, R, ys, D, T):  # pragma: no cover - optional cross-check
    """OPTIONAL cross-check against dynamax's LinearGaussianSSM marginal log-likelihood.

    Left as a clearly-marked stub: dynamax conventions (mean/cov of the prior, the sign and
    normalizing constants of the log-likelihood) differ from our bare ``log(det S) + e^T S^-1 e``
    accumulation, so wiring it requires aligning those constants before any tolerance check.
    Not part of the float32 parity / FD gate -- our numpy twin (above) is the gate reference.
    """
    raise NotImplementedError(
        "dynamax cross-check is an optional, later validation; align the prior and the "
        "log-likelihood normalizing constants before comparing to reference_nll_lim.")
