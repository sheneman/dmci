############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# smoke.py: Fast local self-test of the FluZoo inner loop and harness. Proves, without the LLM or real data, that: (A) the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Fast local self-test of the FluZoo inner loop and harness.

Proves, without the LLM or real data, that:
  (A) the DMCI op surface supports the regional vector-state idioms we need
      (Hadamard S.*I over region vectors, (ref obs k) over a [T,11] matrix);
  (B) the validity funnel ACCEPTS the hand-written reference programs and the
      canonical hash collapses structural duplicates;
  (C) the program->params->DMCI fold calibrates: Adam over the unconstrained raw
      leaves reduces a program's Gaussian NLL on synthetic data.

Run locally as a dev sanity check; the real model-zoo sweep runs on HPC (n128).
"""

import sys
sys.setrecursionlimit(20000)

import math
import numpy as np
import torch

from neural_compiler import compile_dmci
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, as_matrix, unwrap_number

from .config import DEFAULT, HORIZON_TOKEN
from .programs import (
    OMEGA, parse_program, reference_program, run_nll, REFERENCE_PROGRAMS,
)
from .paramspec import make_raw, constrain
from .validity import screen, canonical_hash

PASS, FAIL = "PASS", "FAIL"


def probe_hadamard() -> bool:
    """Does (* u v) over two region vectors compute the elementwise product?"""
    g = compile_dmci("(vsum (* (vec 1.0 2.0 3.0) (vec 4.0 5.0 6.0)))")
    out = float(unwrap_number(evaluate(g, {}, **DEFAULT.EVAL_KW)))
    ok = abs(out - 32.0) < 1e-4   # 1*4 + 2*5 + 3*6 = 32 iff Hadamard
    print(f"  [hadamard] (vsum (* u v)) = {out:.4f}  expect 32.0  -> {PASS if ok else FAIL}")
    return ok


#: Regional seasonally-forced SEIR: 11-region vector state, shared seasonal beta,
#: per-region observation via (ref obs k). This is the canonical zoo program shape.
SEIR_REGIONAL = f"""
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
       (L 0.0))
  (if (= k {HORIZON_TOKEN})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
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
             (var   (+ s2 {DEFAULT.obs_var_floor!r}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Enew Inew Rnew (+ L nll)))))
""".strip()


def numpy_seir_ili(T, p):
    """National %ILI twin for synthetic observations."""
    S, E, I, R = 1.0 - p["i0"] - p["e0"], p["e0"], p["i0"], 0.0
    ili = np.empty(T, dtype=np.float64)
    for k in range(T):
        beta = p["beta0"] * (1.0 + p["amp"] * math.cos(OMEGA * k + p["phase"]))
        force = beta * S * I
        e2i, i2r = p["sigma"] * E, p["gamma"] * I
        ili[k] = p["rho"] * I
        S, E, I, R = S - force, E + force - e2i, I + e2i - i2r, R + i2r
    return ili


def synthetic_obs(T, n_regions, seed=0):
    rng = np.random.default_rng(seed)
    truth = dict(beta0=2.0, amp=0.45, phase=0.3, sigma=0.55, gamma=0.85,
                 rho=0.06, i0=1e-3, e0=5e-4)
    base = numpy_seir_ili(T, truth)
    cols = []
    for r in range(n_regions):
        scale = 0.7 + 0.6 * rng.random()          # per-region amplitude
        cols.append(np.clip(scale * base + rng.normal(0, 3e-4, T), 0, None))
    return np.stack(cols, axis=1).astype(np.float32)  # [T, n_regions]


def calibrate(prog, obs, iters=12, lr=0.06):
    raw = make_raw(prog.specs, seed=0)
    opt = torch.optim.Adam(list(raw.values()), lr=lr)
    nll0 = float(run_nll(prog, raw, obs, grad=False))
    for it in range(iters):
        opt.zero_grad()
        nll = run_nll(prog, raw, obs, grad=True)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(list(raw.values()), 10.0)
        opt.step()
    nllf = float(run_nll(prog, raw, obs, grad=False))
    return nll0, nllf


def main():
    torch.manual_seed(0)
    results = {}

    print("[A] op-surface probes")
    results["hadamard"] = probe_hadamard()

    print("[B] validity funnel on reference programs")
    # national references use a single-column obs; regional uses all 11.
    obs_nat = torch.tensor(synthetic_obs(40, 1), dtype=torch.float32)
    obs_reg = torch.tensor(synthetic_obs(40, DEFAULT.n_regions), dtype=torch.float32)
    for name in ("sir_national", "seir_national", "seirs_national"):
        r = screen(REFERENCE_PROGRAMS[name], probe_obs=obs_nat[:12], name=name)
        print(f"  [funnel] {name:16s} stage={r.stage:15s} ok={r.ok}  "
              f"nparams={r.n_params} gradnorm={r.grad_norm:.2e}"
              + ("" if r.ok else f"  <- {r.detail}"))
        results[f"funnel_{name}"] = r.ok
    r_reg = screen(SEIR_REGIONAL, probe_obs=obs_reg[:12], name="seir_regional")
    print(f"  [funnel] {'seir_regional':16s} stage={r_reg.stage:15s} ok={r_reg.ok}  "
          f"nparams={r_reg.n_params} gradnorm={r_reg.grad_norm:.2e}"
          + ("" if r_reg.ok else f"  <- {r_reg.detail}"))
    results["funnel_seir_regional"] = r_reg.ok

    print("[C] canonical (structural) de-duplication")
    a = REFERENCE_PROGRAMS["seir_national"]
    # alpha-rename a bound var + change a constant -> must hash IDENTICAL to `a`.
    b = a.replace("beta0", "b0").replace("1.5", "1.9").replace("force", "foi")
    h_a, h_b = canonical_hash(a), canonical_hash(b)
    h_sir = canonical_hash(REFERENCE_PROGRAMS["sir_national"])
    dedup_ok = (h_a == h_b) and (h_a != h_sir)
    print(f"  [dedup] seir==seir(renamed): {h_a == h_b}   seir!=sir: {h_a != h_sir}"
          f"  -> {PASS if dedup_ok else FAIL}")
    results["dedup"] = dedup_ok

    print("[D] calibration reduces NLL (national SEIR, synthetic data)")
    prog = reference_program("seir_national")
    obs_fit = torch.tensor(synthetic_obs(52, 1, seed=1), dtype=torch.float32)
    nll0, nllf = calibrate(prog, obs_fit, iters=12)
    calib_ok = nllf < nll0
    print(f"  [calib] NLL {nll0:.4f} -> {nllf:.4f}  improved={calib_ok}"
          f"  -> {PASS if calib_ok else FAIL}")
    results["calibrate"] = calib_ok

    ok = all(results.values())
    print("\n=== SMOKE:", PASS if ok else FAIL, "===")
    for k, v in results.items():
        print(f"   {'ok ' if v else 'BAD'} {k}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
