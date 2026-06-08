############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# kalman_detinv_pilot.py: Flagship de-risk pilot: det/inv gradients + Kalman/SSM NLL numerics through DMCI. The existing suite proves...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Flagship de-risk pilot: det/inv gradients + Kalman/SSM NLL numerics through DMCI.

The existing suite proves det/inv forward + the DIAGONAL gradient case. This pilot
pushes into the flagship's actual demands, which nothing currently covers:

  1. GENERAL (off-diagonal) inverse/det gradients vs a torch reference -- a real
     symmetric-PD S, grads w.r.t. every entry. (Diagonal [[x,0],[0,1]] hides the
     cross terms that the Kalman gain/Mahalanobis term actually exercise.)
  2. Gradient through log det S (the log-likelihood normaliser) and through the
     Mahalanobis term e^T S^-1 e -- the two pieces of the Gaussian NLL.
  3. The full linear-Gaussian Kalman filter FOLDED THROUGH THE DMCI INTERPRETER
     over T steps, reading y_k from a bound obs matrix. Checks the thing I flagged
     as the real flagship risk: with the naive P<-(I-KH)P update (no sqrt/Joseph
     form), does covariance stay PD and the accumulated NLL stay finite over a long
     series, does DMCI match a numpy reference filter, and does dNLL/dq flow and
     match finite-difference?

Run on n128. Prints PASS/FAIL per check, exits nonzero if any fail.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.setrecursionlimit(5000)
# NOTE: DMCI is float32-native (tag/payload tensors are hardcoded torch.float32 in
# tagged_value.py, not configurable). So we run the pilot in float32 to match -- which
# is also the HONEST flagship regime: a long Kalman filter folds through the interpreter
# in float32, exactly where naive covariance updates are most conditioning-fragile.

from neural_compiler.dmci import compile_dmci, as_matrix
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

EVAL_KW = dict(max_iter=500000, max_depth=500000, max_heap=4_000_000)

_results = []
def check(name, cond, detail=""):
    _results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def dmci_scalar(src, feed=None):
    feed = feed or {}
    g = compile_dmci(src)
    tagged = {k: make_float(v if torch.is_tensor(v) else torch.tensor(float(v))) for k, v in feed.items()}
    return unwrap_number(evaluate_g(g, tagged))

def evaluate_g(g, tagged):
    from neural_compiler.evaluator import evaluate
    return evaluate(g, tagged, **EVAL_KW)


# ===========================================================================
print("=== Part 1: general (off-diagonal) det/inv gradients vs torch reference ===")
# S = [[a, b], [b, c]] symmetric PD (a=2, b=0.5, c=1.5 -> det 2.75 > 0)
def torch_S(a, b, c):
    return torch.stack([torch.stack([a, b]), torch.stack([b, c])])

try:
    a0, b0, c0 = 2.0, 0.5, 1.5
    # --- det ---
    a = torch.tensor(a0, requires_grad=True); b = torch.tensor(b0, requires_grad=True); c = torch.tensor(c0, requires_grad=True)
    ref = torch.linalg.det(torch_S(a, b, c)); ref.backward()
    ga, gb, gc = a.grad.item(), b.grad.item(), c.grad.item()

    aa = torch.tensor(a0, requires_grad=True); bb = torch.tensor(b0, requires_grad=True); cc = torch.tensor(c0, requires_grad=True)
    out = dmci_scalar("(det (mat (vec a b) (vec b c)))", {"a": aa, "b": bb, "c": cc})
    check("det off-diagonal forward", abs(out.item() - ref.item()) < 2e-4, f"dmci={out.item():.6f} ref={ref.item():.6f}")
    out.backward()
    check("det off-diagonal grad d/da", abs(aa.grad.item() - ga) < 3e-4, f"dmci={aa.grad.item():.6f} ref={ga:.6f}")
    check("det off-diagonal grad d/db (cross)", abs(bb.grad.item() - gb) < 3e-4, f"dmci={bb.grad.item():.6f} ref={gb:.6f}")
    check("det off-diagonal grad d/dc", abs(cc.grad.item() - gc) < 3e-4, f"dmci={cc.grad.item():.6f} ref={gc:.6f}")

    # --- log det (the NLL normaliser) ---
    a = torch.tensor(a0, requires_grad=True); b = torch.tensor(b0, requires_grad=True); c = torch.tensor(c0, requires_grad=True)
    ref = torch.logdet(torch_S(a, b, c)); ref.backward()
    ga, gb, gc = a.grad.item(), b.grad.item(), c.grad.item()
    aa = torch.tensor(a0, requires_grad=True); bb = torch.tensor(b0, requires_grad=True); cc = torch.tensor(c0, requires_grad=True)
    out = dmci_scalar("(log (det (mat (vec a b) (vec b c))))", {"a": aa, "b": bb, "c": cc})
    check("log det forward", abs(out.item() - ref.item()) < 2e-4, f"dmci={out.item():.6f} ref={ref.item():.6f}")
    out.backward()
    check("log det grad d/da", abs(aa.grad.item() - ga) < 3e-4, f"dmci={aa.grad.item():.6f} ref={ga:.6f}")
    check("log det grad d/db (cross)", abs(bb.grad.item() - gb) < 3e-4, f"dmci={bb.grad.item():.6f} ref={gb:.6f}")

    # --- Mahalanobis e^T S^-1 e, the other NLL piece ---
    e0 = (0.7, -0.4)
    a = torch.tensor(a0, requires_grad=True); b = torch.tensor(b0, requires_grad=True); c = torch.tensor(c0, requires_grad=True)
    e = torch.tensor(e0)
    ref = e @ torch.linalg.inv(torch_S(a, b, c)) @ e; ref.backward()
    ga, gb, gc = a.grad.item(), b.grad.item(), c.grad.item()
    aa = torch.tensor(a0, requires_grad=True); bb = torch.tensor(b0, requires_grad=True); cc = torch.tensor(c0, requires_grad=True)
    src = "(dot (vec 0.7 -0.4) (matvec (inv (mat (vec a b) (vec b c))) (vec 0.7 -0.4)))"
    out = dmci_scalar(src, {"a": aa, "b": bb, "c": cc})
    check("Mahalanobis e^T S^-1 e forward", abs(out.item() - ref.item()) < 2e-4, f"dmci={out.item():.6f} ref={ref.item():.6f}")
    out.backward()
    check("Mahalanobis grad d/da (through inv)", abs(aa.grad.item() - ga) < 6e-4, f"dmci={aa.grad.item():.6f} ref={ga:.6f}")
    check("Mahalanobis grad d/db (cross, through inv)", abs(bb.grad.item() - gb) < 6e-4, f"dmci={bb.grad.item():.6f} ref={gb:.6f}")
except Exception as ex:
    import traceback; traceback.print_exc()
    check("Part 1 (no exception)", False, repr(ex))


# ===========================================================================
print("\n=== Part 2: full Kalman/SSM NLL folded through the DMCI interpreter ===")
# 2D local-level (random-walk) model: x_{k+1}=x_k+w, y_k=x_k+v, w~N(0,Q), v~N(0,R).
# F=H=I. Q=q*I (q learnable), R=r*I. Naive covariance update P<-(I-KH)Ppred.
T = 80
q_true, r_true = 0.05, 0.10
rng = np.random.default_rng(0)
xs = np.zeros((T, 2)); ys = np.zeros((T, 2))
state = np.zeros(2)
for k in range(T):
    state = state + rng.normal(0, np.sqrt(q_true), 2)
    xs[k] = state
    ys[k] = state + rng.normal(0, np.sqrt(r_true), 2)
obs = torch.tensor(ys, dtype=torch.float32)   # DMCI is float32-native; match it

def reference_nll(q, r, return_traj=False, dtype=np.float64):
    """Identical naive Kalman filter in numpy; accumulates sum(logdet S + e^T S^-1 e).
    dtype lets us build BOTH a float32 twin (tight parity vs DMCI) and a float64
    ground-truth (accurate finite-difference)."""
    I2 = np.eye(2, dtype=dtype)
    ysd = ys.astype(dtype)
    x = np.zeros(2, dtype=dtype); P = np.eye(2, dtype=dtype)
    L = 0.0; dets = []; pd_ok = True
    Q = dtype(q) * I2; R = dtype(r) * I2
    for k in range(T):
        xpred = x                      # F=I
        Ppred = P + Q                  # F P F^T + Q
        e = ysd[k] - xpred             # H=I
        S = Ppred + R                  # H Ppred H^T + R
        d = np.linalg.det(S); dets.append(d)
        if d <= 0 or not np.isfinite(d): pd_ok = False
        Sinv = np.linalg.inv(S)
        K = Ppred @ Sinv
        x = xpred + K @ e
        P = (I2 - K) @ Ppred
        L += np.log(d) + e @ Sinv @ e
    return (L, dets, pd_ok) if return_traj else L

# DMCI program: same filter, q,r as scalar inputs, y_k = (ref obs k).
KALMAN_SRC = f"""
(loop ((k 0)
       (x  (vec 0.0 0.0))
       (P  (mat (vec 1.0 0.0) (vec 0.0 1.0)))
       (L  0.0))
  (if (= k {T})
      L
      (let* ((Q     (scale q (eye 2)))
             (R     (scale r (eye 2)))
             (Ppred (+ P Q))
             (y     (ref obs k))
             (e     (- y x))
             (S     (+ Ppred R))
             (Sinv  (inv S))
             (Kg    (matmul Ppred Sinv))
             (xnew  (+ x (matvec Kg e)))
             (Pnew  (matmul (- (eye 2) Kg) Ppred))
             (nll   (+ (log (det S)) (dot e (matvec Sinv e)))))
        (recur (+ k 1) xnew Pnew (+ L nll)))))
"""

try:
    g = compile_dmci(KALMAN_SRC)
    # --- forward parity + finiteness + PD at the true params ---
    qv = torch.tensor(q_true, requires_grad=True); rv = torch.tensor(r_true, requires_grad=True)
    out = unwrap_number(evaluate_g(g, {"q": make_float(qv), "r": make_float(rv), "obs": as_matrix(obs)}))
    ref_L64, dets, pd_ok = reference_nll(q_true, r_true, return_traj=True, dtype=np.float64)
    ref_L32 = reference_nll(q_true, r_true, dtype=np.float32)
    check("Kalman NLL finite over T=80", torch.isfinite(out).all().item(), f"NLL={out.item():.4f}")
    check("covariance stays PD (det S>0 all steps)", pd_ok, f"min det S={min(dets):.4e}")
    check("DMCI NLL == reference filter (float32 twin)", abs(out.item() - ref_L32) < 2e-3 * max(1, abs(ref_L32)),
          f"dmci={out.item():.5f} ref32={ref_L32:.5f} ref64={ref_L64:.5f}")

    # --- gradient dNLL/dq through the whole recursion vs finite difference ---
    out.backward()
    g_dmci_q, g_dmci_r = qv.grad.item(), rv.grad.item()
    eps = 1e-4   # FD on the float64 ground-truth filter
    fd_q = (reference_nll(q_true + eps, r_true) - reference_nll(q_true - eps, r_true)) / (2 * eps)
    fd_r = (reference_nll(q_true, r_true + eps) - reference_nll(q_true, r_true - eps)) / (2 * eps)
    check("dNLL/dq finite", np.isfinite(g_dmci_q), f"{g_dmci_q:.6f}")
    # float32 autograd vs float64 FD: ~few% agreement is the realistic bar in this regime
    check("dNLL/dq == finite-difference", abs(g_dmci_q - fd_q) < 3e-2 * max(1, abs(fd_q)), f"dmci={g_dmci_q:.6f} fd={fd_q:.6f}")
    check("dNLL/dr == finite-difference", abs(g_dmci_r - fd_r) < 3e-2 * max(1, abs(fd_r)), f"dmci={g_dmci_r:.6f} fd={fd_r:.6f}")

    # --- does MLE actually move toward the truth? a short Adam descent on (q,r). ---
    # This is a BONUS sanity check, not part of the numerical gate: the FD-match above
    # already proves the gradient is correct and usable. Each eval folds the 80-step filter
    # through the interpreter (~seconds, unbatched), so keep the loop short for a fast gate.
    MLE_ITERS = 40
    qp = torch.tensor(0.30, requires_grad=True); rp = torch.tensor(0.30, requires_grad=True)
    opt = torch.optim.Adam([qp, rp], lr=0.08)
    L0 = None
    for it in range(MLE_ITERS):
        opt.zero_grad()
        nll = unwrap_number(evaluate_g(g, {"q": make_float(qp), "r": make_float(rp), "obs": as_matrix(obs)}))
        if L0 is None: L0 = nll.item()
        nll.backward()
        # keep variances positive
        opt.step()
        with torch.no_grad():
            qp.clamp_(min=1e-4); rp.clamp_(min=1e-4)
    Lf = nll.item()
    check("MLE reduces NLL", Lf < L0 - 1.0, f"L0={L0:.3f} -> Lf={Lf:.3f}  (q,r)=({qp.item():.4f},{rp.item():.4f})")
    # local-level MLE recovers the q/r RATIO better than absolute values; sanity: both stay
    # finite, positive, and the fit improved. (Tight recovery is an experiment, not a pilot.)
    check("MLE params finite & positive", np.isfinite(qp.item()) and np.isfinite(rp.item()) and qp.item() > 0 and rp.item() > 0,
          f"q={qp.item():.4f} r={rp.item():.4f} (true q={q_true} r={r_true})")
except Exception as ex:
    import traceback; traceback.print_exc()
    check("Part 2 (no exception)", False, repr(ex))


# ===========================================================================
n_pass = sum(1 for _, ok in _results if ok)
n_total = len(_results)
print(f"\n=== PILOT SUMMARY: {n_pass}/{n_total} checks passed ===")
for name, ok in _results:
    if not ok:
        print(f"    FAILED: {name}")
sys.exit(0 if n_pass == n_total else 1)
