############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# flu_seir_pilot.py: De-risk pilot for the LLM + DMCI Flu Model Zoo flagship. The single most important technical unknown (flagged...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""De-risk pilot for the LLM + DMCI Flu Model Zoo flagship.

The single most important technical unknown (flagged by the codebase exploration):
the DMCI interpreter cannot cheaply fold *per-step-varying external forcing*
(heap-mode N x D blowup). But a seasonally-forced epidemic model computes its
time-varying transmission rate *inside* the rollout loop, from the integer week
counter k -- beta(k) = beta0 * (1 + amp * cos(omega*k + phase)) -- with NO
external per-step input bound. If that idiom stays inside the verified op surface
and keeps the heap O(T), the whole Flu Zoo design is sound.

This pilot writes a real discrete-time SEIR flu model as a (loop ... (recur ...))
Scheme program, folds a Gaussian negative log-likelihood of weekly %ILI through
DMCI, and checks the five things that gate the experiment:

  (1) compile_dmci succeeds   -> program is inside the interpreter op surface
  (2) forward is finite       -> the seasonal-forcing-from-counter idiom evaluates
  (3) backward gives finite, NONZERO grads for every parameter
  (4) the heap stays bounded over a 2-season (T=104) horizon (no N x D blowup)
  (5) Adam reduces the NLL    -> the exact DMCI gradient is actually useful

Run locally only as a fast dev sanity check; the real model-zoo calibration runs
on HPC (n128). This is a correctness smoke, not a benchmark.
"""

import sys

# Must be raised BEFORE importing neural_compiler (per the DMCI recursion note).
sys.setrecursionlimit(20000)

import math
import numpy as np
import torch

from neural_compiler import compile_dmci
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, as_matrix, unwrap_number

# 2*pi / 52 weeks-per-year, as a literal (pi is not a named constant in DMCI).
OMEGA = 2.0 * math.pi / 52.0
EVAL_KW = dict(max_iter=500_000, max_depth=500_000, max_heap=4_000_000)

# Free variables (auto-detected by compile_dmci): the 9 learnable params + obs.
# All arithmetic is BINARY. Seasonal beta is built from the integer counter k,
# so there is NO external per-step forcing array -- only the [T,1] obs matrix,
# read one row at a time with (ref obs k), exactly like the Kalman/LIM flagship.
PARAM_NAMES = ["beta0", "amp", "phase", "sigma", "gamma", "rho", "s2", "i0", "e0"]


def flu_seir_src(T: int) -> str:
    """Discrete-time seasonally-forced SEIR with a Gaussian %ILI likelihood."""
    return f"""
(loop ((k 0)
       (S (- (- 1.0 i0) e0))
       (E e0)
       (I i0)
       (R 0.0)
       (L 0.0))
  (if (= k {T})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
             (force (* beta (* S I)))
             (e2i   (* sigma E))
             (i2r   (* gamma I))
             (Snew  (- S force))
             (Enew  (+ E (- force e2i)))
             (Inew  (+ I (- e2i i2r)))
             (Rnew  (+ R i2r))
             (ypred (vec (* rho I)))
             (y     (ref obs k))
             (resid (- y ypred))
             (nll   (+ (/ (dot resid resid) (* 2.0 s2)) (* 0.5 (log s2)))))
        (recur (+ k 1) Snew Enew Inew Rnew (+ L nll)))))
""".strip()


def numpy_seir_ili(T, p):
    """Reference rollout in numpy -- generates synthetic 'observed' %ILI and is
    the term-for-term twin of the Scheme arithmetic/accumulation order."""
    S = 1.0 - p["i0"] - p["e0"]
    E, I, R = p["e0"], p["i0"], 0.0
    ili = np.empty(T, dtype=np.float64)
    for k in range(T):
        beta = p["beta0"] * (1.0 + p["amp"] * math.cos(OMEGA * k + p["phase"]))
        force = beta * S * I
        e2i = p["sigma"] * E
        i2r = p["gamma"] * I
        ili[k] = p["rho"] * I
        S, E, I, R = S - force, E + force - e2i, I + e2i - i2r, R + i2r
    return ili


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T = 104  # two influenza seasons of weekly data

    truth = dict(beta0=2.0, amp=0.45, phase=0.3, sigma=0.55, gamma=0.85,
                 rho=0.06, s2=1e-6, i0=1.0e-3, e0=5.0e-4)

    # Synthetic observed %ILI from the numpy twin + small observation noise.
    ili_clean = numpy_seir_ili(T, truth)
    noise = rng.normal(0.0, 3e-4, size=T)
    obs_np = np.clip(ili_clean + noise, 0.0, None).reshape(T, 1).astype(np.float32)
    obs = as_matrix(torch.tensor(obs_np))

    print(f"[setup] T={T}  obs %ILI range [{obs_np.min():.4e}, {obs_np.max():.4e}]")

    # ---- (1) compile -----------------------------------------------------
    src = flu_seir_src(T)
    print("[gate1] compiling SEIR program through DMCI ...")
    graph = compile_dmci(src)
    print("[gate1] OK  compile_dmci succeeded (program inside the op surface)")

    # Learnable params live in UNCONSTRAINED raw space (the exp_lim_enso
    # params.py pattern): Adam optimises O(1)-scale raw leaves, and the
    # epidemiological constraints (positive rates, reporting/initial-condition
    # fractions in a sane range, amp in (-1,1), positive obs variance) are
    # imposed by smooth transforms whose gradients flow back to the leaves.
    # This is what the 60-line single-lr pilot lacked.
    raw_init = dict(  # chosen so the constrained init sits AWAY from truth
        beta0=1.117, amp=0.203, phase=0.0, sigma=-0.405, gamma=0.405,
        rho=-1.735, s2=-12.43, i0=-1.992, e0=-2.752)
    raw = {n: torch.tensor(float(raw_init[n]), requires_grad=True) for n in PARAM_NAMES}

    sp = torch.nn.functional.softplus

    def constrained(leaves):
        """raw (unconstrained) -> epidemiologically valid params, differentiably."""
        return dict(
            beta0=sp(leaves["beta0"]),                 # > 0   transmission scale
            amp=torch.tanh(leaves["amp"]),             # (-1,1) seasonal amplitude
            phase=leaves["phase"],                     # free  seasonal phase
            sigma=torch.sigmoid(leaves["sigma"]),      # (0,1) E->I weekly rate
            gamma=torch.sigmoid(leaves["gamma"]),      # (0,1) I->R weekly rate
            rho=0.2 * torch.sigmoid(leaves["rho"]),    # (0,0.2) reporting fraction
            i0=0.01 * torch.sigmoid(leaves["i0"]),     # (0,0.01) initial infected
            e0=0.01 * torch.sigmoid(leaves["e0"]),     # (0,0.01) initial exposed
            s2=torch.exp(leaves["s2"]) + 1e-9,         # > 0   obs variance (log-space)
        )

    def forward():
        con = constrained(raw)
        binding = {n: make_float(con[n]) for n in PARAM_NAMES}
        binding["obs"] = obs
        return unwrap_number(evaluate(graph, binding, **EVAL_KW))

    # ---- (2) forward finite ---------------------------------------------
    nll0 = forward()
    finite_fwd = bool(torch.isfinite(nll0).all())
    print(f"[gate2] forward NLL = {float(nll0):.6f}   finite={finite_fwd}")
    assert finite_fwd, "forward produced a non-finite NLL"

    # ---- (3) gradients finite + nonzero for EVERY raw leaf ---------------
    nll0.backward()
    grad_report = {}
    all_ok = True
    for n in PARAM_NAMES:
        g = raw[n].grad
        ok = (g is not None) and bool(torch.isfinite(g).all()) and (float(g.abs()) > 0.0)
        grad_report[n] = (float(g) if g is not None else None, ok)
        all_ok = all_ok and ok
    print("[gate3] per-parameter gradients (value, nonzero&finite):")
    for n in PARAM_NAMES:
        gv, ok = grad_report[n]
        flag = "ok " if ok else "BAD"
        print(f"         {flag} {n:7s} d(NLL)/d(raw)={gv}")
    assert all_ok, "some parameter received a zero or non-finite gradient"

    # ---- (4) heap bounded over the full horizon --------------------------
    # If the seasonal-forcing-from-counter idiom triggered an N x D heap blowup,
    # evaluation would have raised a heap-overflow well under max_heap. Re-run
    # with a deliberately MODEST heap to prove O(T) growth, not O(T*D) or worse.
    modest = dict(max_iter=500_000, max_depth=500_000, max_heap=600_000)
    try:
        con = constrained({n: raw[n].detach() for n in PARAM_NAMES})
        binding = {n: make_float(con[n]) for n in PARAM_NAMES}
        binding["obs"] = obs
        nll_modest = unwrap_number(evaluate(graph, binding, **modest))
        print(f"[gate4] OK  heap bounded: T={T} fold completes under "
              f"max_heap={modest['max_heap']:,} (NLL={float(nll_modest):.6f})")
    except Exception as exc:  # noqa: BLE001
        print(f"[gate4] HEAP CONCERN: {type(exc).__name__}: {exc}")
        raise

    # ---- (5) Adam actually reduces the NLL -------------------------------
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    opt = torch.optim.Adam(list(raw.values()), lr=0.06)
    nll0_val = float(nll0.detach())
    for it in range(n_iters):
        opt.zero_grad()
        nll = forward()
        if not torch.isfinite(nll).all():
            print(f"[gate5] non-finite NLL at iter {it}; stopping", flush=True)
            break
        nll.backward()
        torch.nn.utils.clip_grad_norm_(list(raw.values()), 10.0)
        opt.step()
        if it % 5 == 0 or it == n_iters - 1:
            print(f"[gate5] iter {it:3d}  NLL={float(nll.detach()):.6f}", flush=True)
    nll_final = float(forward().detach())
    improved = nll_final < nll0_val
    fit = constrained(raw)
    print(f"[gate5] NLL {nll0_val:.6f} -> {nll_final:.6f}  improved={improved}")
    print("[gate5] fitted vs truth:")
    for n in PARAM_NAMES:
        print(f"         {n:7s} fit={float(fit[n]):.5f}  truth={truth[n]:.5f}")

    ok = finite_fwd and all_ok and improved
    print("\n=== DE-RISK VERDICT:", "PASS" if ok else "FAIL", "===")
    print("Seasonal-forcing-from-counter SEIR folds through DMCI with exact, "
          "nonzero gradients and a bounded heap." if ok else
          "Investigate before committing to the Flu Zoo design.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
