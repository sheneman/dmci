############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# smoke.py: Fast LOCAL smoke test for the LIM-ENSO core (run on the Mac before any HPC submit). Builds a tiny synthetic...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Fast LOCAL smoke test for the LIM-ENSO core (run on the Mac before any HPC submit).

Builds a tiny synthetic stable LIM at D=6, T=40, runs ONE forward + backward DMCI eval
of the Kalman NLL, and asserts:
  - the accumulated NLL is finite,
  - the gradient w.r.t. the F raw parameters is finite AND nonzero (autograd actually
    flowed back through the compiled interpreter),
  - the float32 numpy twin agrees with DMCI to the gate's parity tolerance (sanity that
    reference.py mirrors models.py).
Prints per-step ms. Pure sanity -- the real numerical bars live in gate.py (run on n128).

Usage:  python3 -m experiments.exp_lim_enso.smoke
"""

from __future__ import annotations

import sys
import time

from .config import DEFAULT, GATE

sys.setrecursionlimit(DEFAULT.recursion_limit)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import params, reference  # noqa: E402
from .models import run_kalman_nll  # noqa: E402


def main() -> None:
    D, T, structure = 6, 40, "S0"
    print(f"=== LIM-ENSO smoke: D={D} T={T} structure={structure} (synthetic) ===")

    # --- synthetic stable LIM (near 0.9*I F, small PD Q via Cholesky, r_floor R) ---
    rp = params.init_raw_params(D, structure, seed=0)
    F = params.make_F(rp["F_raw"], D, structure)
    Q = params.make_Q(rp["Lq_raw"], D)
    R = params.make_R(rp["r_raw"], D)
    sr = float(params.spectral_radius(F.detach()))
    print(f"spectral radius(F) = {sr:.4f}  (stable iff < 1)")

    # --- synthetic observation window ---
    rng = np.random.default_rng(0)
    obs = torch.tensor(rng.standard_normal((T, D)).astype(np.float32))

    # --- ONE forward + backward DMCI eval (timed) ---
    t0 = time.perf_counter()
    nll = run_kalman_nll(F, Q, R, obs, D, T, structure, jitter=False, grad=True)
    fwd_ms = 1e3 * (time.perf_counter() - t0) / T
    nll_val = float(nll.detach().reshape(()))
    print(f"forward NLL = {nll_val:.4f}   ({fwd_ms:.2f} ms/step forward)")

    nll.backward()
    g_F = rp["F_raw"].grad
    g_norm = float(g_F.norm()) if g_F is not None else 0.0
    print(f"||dNLL/dF_raw|| = {g_norm:.4f}")

    # --- float32 twin parity (reference.py mirrors models.py) ---
    ref32 = float(np.asarray(
        reference.reference_nll_lim(F, Q, R, obs, D, T, dtype=np.float32)).reshape(()))
    abs_err = abs(nll_val - ref32)
    tol = GATE.parity_rel * max(1.0, abs(ref32))
    print(f"parity: dmci={nll_val:.5f} ref32={ref32:.5f} |Δ|={abs_err:.3e} (tol {tol:.3e})")

    # --- assertions ---
    assert np.isfinite(nll_val), f"NLL not finite: {nll_val}"
    assert g_F is not None and np.isfinite(g_norm) and g_norm > 0, \
        f"F gradient absent/zero/non-finite (norm={g_norm})"
    assert abs_err <= tol, f"parity off: |Δ|={abs_err:.3e} > tol {tol:.3e}"

    print("\nSMOKE PASS: finite NLL, nonzero F grad, float32-twin parity within tol.")


if __name__ == "__main__":
    main()
