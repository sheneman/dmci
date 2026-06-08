############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# structures.py: The reference degradation structures as DMCI Scheme programs (the program zoo seed). Each is a two-form FluZoo...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The reference degradation structures as DMCI Scheme programs (the program zoo seed).

Each is a two-form FluZoo artifact: a `(params ...)` schema then a `(loop ...)` model. The
model is a per-CYCLE tail-recursive rollout; the observable is the predicted capacity Q at
cycle k, carried in `yhat` (so the FIT->PREDICT swap forecasts), with a floored-variance
Gaussian NLL of the observed capacity `(ref obs k)` accumulated into `L`.

The five sit on a deliberate smooth->knee axis (NREL/Smith algebraic-life families):
  sqrt_t_SEI         Q = q0 - B*sqrt(k)                       smooth, decelerating, never knees
  power_law_kp       Q = q0 - B*k^p                           one form, p<1/=1/>1 = concave/linear/convex
  stretched_exp      Q = qinf + (q0-qinf)*exp(-(k/tau)^beta)  soft knee, asymptotes to a floor
  two_reservoir_min  Q = q0*min(1-a*sqrt(k), 1-c*k)           SHARP knee (Li/host bottleneck); min non-smooth
  sigmoidal_knee     Q = q0 - m*k - d*k*sigmoid((k-kn)/wd)    smooth knee, sharpness set by wd

Only whitelisted interpreter ops appear (+ - * / = sqrt pow exp log min dot vec ref zeros);
sigmoid is built from exp. Param scales keep the unconstrained raw leaves O(1) (paramspec).
All loop bounds use the FluZoo horizon token NWEEKS (= the cycle count, substituted at compile).
"""

from __future__ import annotations

from .config import FLOOR

# Shared NLL tail: predicted (vec Q) vs observed (ref obs k), floored-variance Gaussian.
_NLL = f"""(ypred (vec Q))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 {FLOOR}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var))))"""


SQRT_T_SEI = f"""
(params
  (q0 positive 1.0)
  (B  positive 0.016 0.05)
  (s2 positive 1e-4 1e-3))

(loop ((k 0) (yhat (zeros 1)) (L 0.0))
  (if (= k NWEEKS)
      L
      (let* ((Q     (- q0 (* B (sqrt k))))
             {_NLL})
        (recur (+ k 1) ypred (+ L nll)))))
""".strip()


POWER_LAW_KP = f"""
(params
  (q0 positive 1.0)
  (B  positive 0.005 0.05)
  (p  positive 0.8 2.0)
  (s2 positive 1e-4 1e-3))

(loop ((k 0) (yhat (zeros 1)) (L 0.0))
  (if (= k NWEEKS)
      L
      (let* ((Q     (- q0 (* B (pow k p))))
             {_NLL})
        (recur (+ k 1) ypred (+ L nll)))))
""".strip()


STRETCHED_EXP = f"""
(params
  (q0   positive 1.0)
  (qinf unit 0.6)
  (tau  positive 90.0 100.0)
  (beta positive 1.0 2.0)
  (s2   positive 1e-4 1e-3))

(loop ((k 0) (yhat (zeros 1)) (L 0.0))
  (if (= k NWEEKS)
      L
      (let* ((Q     (+ qinf (* (- q0 qinf) (exp (- 0.0 (pow (/ (+ k 1e-6) tau) beta))))))
             {_NLL})
        (recur (+ k 1) ypred (+ L nll)))))
""".strip()


TWO_RESERVOIR_MIN = f"""
(params
  (q0 positive 1.0)
  (a  positive 0.012 0.05)
  (c  positive 0.0015 0.01)
  (s2 positive 1e-4 1e-3))

(loop ((k 0) (yhat (zeros 1)) (L 0.0))
  (if (= k NWEEKS)
      L
      (let* ((li   (- 1.0 (* a (sqrt k))))
             (host (- 1.0 (* c k)))
             (Q    (* q0 (min li host)))
             {_NLL})
        (recur (+ k 1) ypred (+ L nll)))))
""".strip()


SIGMOIDAL_KNEE = f"""
(params
  (q0 positive 1.0)
  (m  positive 4e-4 1e-3)
  (d  positive 1e-3 5e-3)
  (kn positive 75.0 100.0)
  (wd positive 10.0 30.0)
  (s2 positive 1e-4 1e-3))

(loop ((k 0) (yhat (zeros 1)) (L 0.0))
  (if (= k NWEEKS)
      L
      (let* ((z    (/ (- k kn) wd))
             (sig  (/ 1.0 (+ 1.0 (exp (- 0.0 z)))))
             (Q    (- (- q0 (* m k)) (* d (* k sig))))
             {_NLL})
        (recur (+ k 1) ypred (+ L nll)))))
""".strip()


STRUCTURES: dict[str, str] = {
    "sqrt_t_SEI": SQRT_T_SEI,
    "power_law_kp": POWER_LAW_KP,
    "stretched_exp": STRETCHED_EXP,
    "two_reservoir_min": TWO_RESERVOIR_MIN,
    "sigmoidal_knee": SIGMOIDAL_KNEE,
}

# Whether the structure can express a knee (used to read the confusion matrix: a smooth
# structure cannot fit knee data, a knee structure CAN fit smooth data by degenerating).
CAN_KNEE = {
    "sqrt_t_SEI": False,
    "power_law_kp": False,    # p>1 accelerates but cannot produce a localized knee
    "stretched_exp": True,    # a soft knee at k~tau
    "two_reservoir_min": True,
    "sigmoidal_knee": True,
}
