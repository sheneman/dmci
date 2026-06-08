# OpenEvolve seed for the battery capacity-fade program search.
# The model you EVOLVE is the Scheme program inside the BATTERY_MODEL string (a compiled
# differentiable interpreter, DMCI, runs it). Edit ONLY the Scheme; keep the triple-quoted string
# and the two-form (params ...) + (loop ... (recur ...)) contract intact. Lower held-out forecast
# RMSE is better. The seed is the simplest SMOOTH structure (sqrt-t SEI growth); it CANNOT express
# a knee, so it will under-forecast cells that knee -- the search must discover better structure.

# EVOLVE-BLOCK-START
BATTERY_MODEL = r"""
(params
  (q0 positive 1.0)
  (B1 positive 0.02 0.05)
  (B2 positive 0.01 0.03)
  (s2 positive 1e-4 1e-3))

(loop ((k 0) (yhat (zeros 1)) (L 0.0))
  (if (= k NWEEKS)
      L
      (let* ((fade1 (* B1 (sqrt (+ k 1e-6))))
             (fade2 (* B2 k))
             (Q (- q0 (max fade1 fade2)))
             (ypred (vec Q))
             (y (ref obs k))
             (resid (- y ypred))
             (var (+ s2 1e-06))
             (nll (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) ypred (+ L nll)))))
"""
# EVOLVE-BLOCK-END


def get_model() -> str:
    return BATTERY_MODEL
