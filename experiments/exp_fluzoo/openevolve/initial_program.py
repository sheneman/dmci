############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# initial_program.py: OpenEvolve seed: the baseline SEIR-style influenza model the structure search starts from
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

# EVOLVE-BLOCK-START
# An influenza forecasting model, written as a Scheme program for the DMCI interpreter.
# OpenEvolve evolves the SCHEME source in the FLU_MODEL string below (diff-based edits).
# Contract: two top-level S-expressions -- a (params (name kind init [scale]) ...) schema and
# a tail-recursive weekly (loop ((k 0) ... (yhat ...) (L 0.0)) (if (= k NWEEKS) L (let* (...)
# (recur ...)))). obs is a [T,11] matrix; (ref obs k) is the week-k 11-vector (national + 10 HHS
# regions); the model predicts an 11-vector yhat of %ILI (proportion) and accumulates a Gaussian
# NLL in L. Seasonal forcing comes from the integer counter k only. Binary-only arithmetic;
# whitelisted ops only. See the evaluator/system prompt for the full op surface and rules.
FLU_MODEL = r"""
(params
  (beta0 positive 1.5)
  (amp   signed-unit 0.2)
  (phase free 0.0)
  (sigma unit 0.5)
  (gamma unit 0.5)
  (rho   unit 0.05 0.2)
  (i0    unit 0.0015 0.01)
  (e0    unit 0.0008 0.01)
  (s2    positive 4e-6))

(loop ((k 0)
       (S (- (- (ones 11) (scale i0 (ones 11))) (scale e0 (ones 11))))
       (E (scale e0 (ones 11)))
       (I (scale i0 (ones 11)))
       (R (zeros 11))
       (yhat (zeros 11))
       (L 0.0))
  (if (= k NWEEKS)
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* 0.12083048667087144 k) phase))))))
             (force (scale beta (* S I)))
             (e2i   (scale sigma E))
             (i2r   (scale gamma I))
             (Snew  (- S force))
             (Enew  (+ E (- force e2i)))
             (Inew  (+ I (- e2i i2r)))
             (Rnew  (+ R i2r))
             (ypred (scale rho I))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 1e-06))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Enew Inew Rnew ypred (+ L nll)))))
"""
# EVOLVE-BLOCK-END


def get_model() -> str:
    """Return the Scheme model source (used by the evaluator)."""
    return FLU_MODEL
