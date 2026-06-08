############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# gate.py: Numerical GO/NO-GO gate for the LIM-ENSO experiment. Re-runs the verified det/inv Kalman pilot's PD / parity /...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Numerical GO/NO-GO gate for the LIM-ENSO experiment.

Re-runs the verified det/inv Kalman pilot's PD / parity / finite-difference-gradient
checks (experiments/pilots/kalman_detinv_pilot.py) at the REAL (D, T) the LIM MLE will
run at, for the FULL transition operator F. The MLE in run.py only proceeds on GO; this
is the de-risk gate that proves the float32 DMCI fold is numerically sound BEFORE any
optimization wall-clock is spent.

For each requested D it runs every check on BOTH:
  - a SYNTHETIC stable LIM (F = 0.9*I + small off-diagonal, Q = 0.05-scale PD, R = 0.1*I),
    which is conditioning-benign by construction (always available), and
  - the REAL PC window (obs = pcs[:T, :D]) when data/processed/pcs.npy exists; skipped
    cleanly (recorded, non-blocking) when the data has not been built/rsync'd yet.

Checks (pilot check()/PASS-FAIL style: collect, don't abort; tally; emit JSON):
  G1  NLL finite over T                          (no NaN/Inf in the accumulated NLL)
  G2  covariance PD every step                   (min det S > detS_floor)
  G3  DMCI NLL == reference float32-twin          (|Δ| <= parity_rel * max(1,|ref|))
  G4  gradients (q, r, a few F entries) == central FD on the float64-twin (<= fd_rel)
  G5  det-underflow margin                        (min det S comfortably > detS_floor 1.2e-38)
  G6  batched [N,D,D] F binding works             (eval finite) -- NON-BLOCKING (DiffEvo
                                                   falls back if absent)

GO = G1..G5 all pass (on every dataset that ran). G6 is recorded but never blocks.
Emits experiments/exp_lim_enso/gate_{D}.json. ``main()`` loops D_list (or a single --D),
prints PASS/FAIL lines, and exits 0 iff every requested D is GO.

CPU / interpreter-bound. ``sys.setrecursionlimit`` is raised BEFORE importing
neural_compiler (the LIM filter trampolines O(T) but the compile/eval machinery recurses).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# --- raise recursion limit BEFORE importing neural_compiler (see run scripts / MEMORY) ---
from .config import DEFAULT, GATE, ExpLimConfig, GateThresholds  # config has no heavy import

sys.setrecursionlimit(DEFAULT.recursion_limit)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import params, reference  # noqa: E402
from .models import run_kalman_nll, _get_graph  # noqa: E402


HERE = Path(__file__).parent
PROCESSED_DIR = HERE / "data" / "processed"


# ===========================================================================
# Pilot-style check tally (collect-don't-abort).
# ===========================================================================

class _Tally:
    """Accumulates (name -> bool) check results plus free-form details, pilot-style."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool]] = []
        self.details: dict[str, str] = {}

    def check(self, name: str, cond, detail: str = "") -> bool:
        ok = bool(cond)
        self.results.append((name, ok))
        if detail:
            self.details[name] = detail
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
        return ok

    def passed(self, name: str) -> bool:
        # True only if the named check ran AND passed (missing -> False).
        return any(n == name and ok for n, ok in self.results)


# ===========================================================================
# Synthetic stable LIM (always-available, conditioning-benign reference setup).
# ===========================================================================

def synthetic_lim(D: int):
    """A stable D-dimensional LIM (F = 0.9*I + small off-diagonal, Q ~ 0.05-scale PD, R = 0.1*I).

    Returns leaf raw params ``{F_raw, Lq_raw, r_raw}`` (requires_grad) plus the DECODED
    ``F, Q, R`` float32 matrices. F is contractive (spectral radius ~0.9) so the filter is
    PD-benign; this is the control setup the gate's checks should always pass on -- if they
    fail HERE the interpreter/parametrization is wrong, not the data."""
    g = torch.Generator().manual_seed(0)

    # F = 0.9*I + small off-diagonal (raw layout = flat D*D for S0).
    F0 = 0.9 * torch.eye(D, dtype=torch.float32)
    off = 0.03 * torch.randn(D, D, generator=g, dtype=torch.float32)
    off = off - torch.diag(torch.diag(off))          # keep the strong 0.9 diagonal
    F_raw = (F0 + off).reshape(-1).clone().requires_grad_(True)

    # Q ~ 0.05-scale PD via Cholesky (L diag ~ sqrt(0.05)); R = 0.1*I via the scalar param.
    kQ = D * (D + 1) // 2
    Lq_raw = 0.01 * torch.randn(kQ, generator=g, dtype=torch.float32)
    diag_pos = [i * (i + 1) // 2 + i for i in range(D)]
    Lq_raw[diag_pos] = float(np.sqrt(0.05))
    Lq_raw = Lq_raw.clone().requires_grad_(True)

    # R = softplus(r_raw)*I + r_floor*I; choose r_raw so softplus(r_raw) ~ 0 -> R ~ r_floor*I.
    # With the default r_floor=0.1 this gives R ~ 0.1*I as specified.
    r_raw = torch.tensor(-6.0, dtype=torch.float32).clone().requires_grad_(True)

    F = params.make_F(F_raw, D, "S0")
    Q = params.make_Q(Lq_raw, D)
    R = params.make_R(r_raw, D)
    return {"F_raw": F_raw, "Lq_raw": Lq_raw, "r_raw": r_raw}, F, Q, R


def synthetic_obs(D: int, T: int, F, Q, R, seed: int = 0):
    """Simulate a T-step observation series from the synthetic LIM (x_{k+1}=Fx+w, y=x+v).

    Uses the DECODED (detached) F, Q, R so the data is consistent with the model the gate
    fits. float32 to match the DMCI binding dtype."""
    rng = np.random.default_rng(seed)
    Fn = F.detach().cpu().numpy().astype(np.float64)
    Qn = Q.detach().cpu().numpy().astype(np.float64)
    Rn = R.detach().cpu().numpy().astype(np.float64)
    Lq = np.linalg.cholesky(Qn)
    Lr = np.linalg.cholesky(Rn)
    x = np.zeros(D)
    ys = np.zeros((T, D))
    for k in range(T):
        x = Fn @ x + Lq @ rng.standard_normal(D)
        ys[k] = x + Lr @ rng.standard_normal(D)
    return torch.tensor(ys, dtype=torch.float32)


# ===========================================================================
# The per-dataset check battery (synthetic OR real window).
# ===========================================================================

def _run_checks_on(tally: _Tally, label: str, D: int, T: int, structure: str,
                   raw: dict, F, Q, R, obs, gate: GateThresholds,
                   cfg: ExpLimConfig) -> dict:
    """Run G1..G6 on one (F,Q,R,obs) setup. Returns a per-dataset metrics dict.

    ``label`` is 'synthetic' or 'real'. Check names are suffixed with the label so the
    JSON keeps both runs distinct. ``raw`` holds the leaf raw params so we can do FD on the
    float64 twin AND backward on DMCI. Collect-don't-abort: any Python exception in a block
    is turned into a FAIL of that block's check rather than aborting the whole gate."""
    print(f"\n  --- {label} (D={D}, T={T}, structure={structure}) ---")
    metrics: dict = {"label": label, "min_detS": None, "cond_S": None,
                     "per_step_ms": None, "parity_abs": None, "fd_rel": None}

    # ---- forward + grad DMCI eval (timed) -------------------------------------
    try:
        t0 = time.perf_counter()
        nll = run_kalman_nll(F, Q, R, obs, D, T, structure, jitter=False, grad=True)
        dt = time.perf_counter() - t0
        metrics["per_step_ms"] = 1e3 * dt / max(1, T)
        nll_val = float(nll.detach().reshape(()).item())
        tally.check(f"G1 NLL finite over T [{label}]",
                    bool(torch.isfinite(nll.detach()).all()),
                    f"NLL={nll_val:.4f}  ({metrics['per_step_ms']:.2f} ms/step)")
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        tally.check(f"G1 NLL finite over T [{label}]", False, repr(ex))
        nll = None

    # ---- reference float64 twin: per-step det(S) -> PD, underflow margin, conditioning ----
    try:
        L64, dets, pd_ok = reference.reference_nll_lim(
            F, Q, R, obs, D, T, return_traj=True, dtype=np.float64)
        dets_arr = np.asarray(dets, dtype=np.float64)
        min_detS = float(dets_arr.min()) if dets_arr.size else float("nan")
        metrics["min_detS"] = min_detS

        # conditioning proxy: worst-step condition number of S (float64 twin), recomputed
        # by replaying the filter once more and tracking cond(S). Cheap relative to DMCI.
        cond_S = _max_cond_S(F, Q, R, obs, D, T)
        metrics["cond_S"] = cond_S

        # G2 PD every step: min det S above the float32 denormal floor with margin.
        tally.check(f"G2 covariance PD every step [{label}]",
                    pd_ok and np.isfinite(min_detS) and min_detS > gate.detS_floor,
                    f"min det S={min_detS:.4e}  cond(S)_max={cond_S:.2e}")

        # G5 det-underflow margin: same quantity vs the explicit float32 floor (1.2e-38).
        tally.check(f"G5 det-underflow margin [{label}]",
                    np.isfinite(min_detS) and min_detS > gate.detS_floor,
                    f"min det S={min_detS:.4e} > floor {gate.detS_floor:.2e}")
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        tally.check(f"G2 covariance PD every step [{label}]", False, repr(ex))
        tally.check(f"G5 det-underflow margin [{label}]", False, repr(ex))
        L64 = None

    # ---- G3 forward parity: float32 DMCI vs the float32 numpy twin (identical arith order) --
    try:
        ref32 = float(np.asarray(
            reference.reference_nll_lim(F, Q, R, obs, D, T, dtype=np.float32)).reshape(()))
        dmci_val = float(run_kalman_nll(F, Q, R, obs, D, T, structure,
                                        jitter=False, grad=False).reshape(()).item())
        abs_err = abs(dmci_val - ref32)
        metrics["parity_abs"] = abs_err
        tally.check(f"G3 DMCI NLL == ref float32-twin [{label}]",
                    abs_err <= gate.parity_rel * max(1.0, abs(ref32)),
                    f"dmci={dmci_val:.5f} ref32={ref32:.5f} |Δ|={abs_err:.3e} "
                    f"tol={gate.parity_rel * max(1.0, abs(ref32)):.3e}")
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        tally.check(f"G3 DMCI NLL == ref float32-twin [{label}]", False, repr(ex))

    # ---- G4 gradients (q, r, a few F entries) == central FD on the float64 twin -----------
    try:
        rel = _grad_fd_rel(raw, D, T, structure, obs, gate)
        metrics["fd_rel"] = rel
        tally.check(f"G4 gradient == finite-difference [{label}]",
                    np.isfinite(rel) and rel <= gate.fd_rel,
                    f"||g_dmci - g_fd64|| / ||g_fd64|| = {rel:.3e}  (tol {gate.fd_rel:.1e})")
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        tally.check(f"G4 gradient == finite-difference [{label}]", False, repr(ex))

    # ---- G6 batched [N,D,D] F binding (NON-BLOCKING) -------------------------------------
    try:
        ok6, detail6 = _batched_binding_ok(F, Q, R, obs, D, T, structure, cfg)
        tally.check(f"G6 batched [N,D,D] F binding [{label}]", ok6, detail6)
    except Exception as ex:  # noqa: BLE001
        # Non-blocking by design: record False, DiffEvo will fall back to a Python loop.
        tally.check(f"G6 batched [N,D,D] F binding [{label}]", False,
                    f"(non-blocking) {ex!r}")

    return metrics


# ===========================================================================
# Helpers: conditioning, FD gradient, batched binding.
# ===========================================================================

def _max_cond_S(F, Q, R, obs, D, T) -> float:
    """Worst-step condition number of the innovation covariance S over the float64 filter."""
    Fm = _np(F, D); Qm = _np(Q, D); Rm = _np(R, D)
    Y = _np(obs, D, rows=T)
    I = np.eye(D)
    x = np.zeros(D); P = np.eye(D)
    worst = 0.0
    for k in range(T):
        xpred = Fm @ x
        Ppred = (Fm @ P) @ Fm.T + Qm
        e = Y[k] - xpred
        S = Ppred + Rm
        c = float(np.linalg.cond(S))
        worst = max(worst, c if np.isfinite(c) else float("inf"))
        Sinv = np.linalg.inv(S)
        K = Ppred @ Sinv
        x = xpred + K @ e
        P = (I - K) @ Ppred
    return worst


def _grad_fd_rel(raw: dict, D: int, T: int, structure: str, obs, gate: GateThresholds) -> float:
    """Relative error ||g_dmci - g_fd64|| / ||g_fd64|| over a probe set of raw coordinates.

    Probes q (a Q-Cholesky diagonal entry), r (the scalar R raw), and a few F-raw entries.
    The DMCI gradient is float32 autograd through the interpreter; the reference is central
    finite-difference on the float64 numpy twin (reference_nll_lim) -- exactly the pilot's
    'dNLL/dq == finite-difference' bar generalized to the LIM raw-parameter vector.
    """
    F_raw = raw["F_raw"].detach().clone()
    Lq_raw = raw["Lq_raw"].detach().clone()
    r_raw = raw["r_raw"].detach().clone().reshape(())

    # --- pick the probe coordinates: r, one Q-diagonal, and a few F entries ---
    g = torch.Generator().manual_seed(20240602)
    diag_pos = [i * (i + 1) // 2 + i for i in range(D)]
    probes: list[tuple[str, int]] = [("r", 0), ("Lq", diag_pos[0])]
    nF = F_raw.numel()
    n_extra = max(0, gate.fd_n_probe - len(probes))
    f_idx = torch.randperm(nF, generator=g)[:n_extra].tolist()
    probes += [("F", int(i)) for i in f_idx]

    # --- DMCI float32 autograd gradient at the probe point ---
    F_leaf = F_raw.clone().requires_grad_(True)
    Lq_leaf = Lq_raw.clone().requires_grad_(True)
    r_leaf = r_raw.clone().requires_grad_(True)
    F = params.make_F(F_leaf, D, structure)
    Q = params.make_Q(Lq_leaf, D)
    R = params.make_R(r_leaf, D)
    nll = run_kalman_nll(F, Q, R, obs, D, T, structure, jitter=False, grad=True)
    nll.backward()
    g_dmci = []
    for kind, idx in probes:
        if kind == "r":
            g_dmci.append(float(r_leaf.grad.reshape(())))
        elif kind == "Lq":
            g_dmci.append(float(Lq_leaf.grad[idx]))
        else:
            g_dmci.append(float(F_leaf.grad[idx]))

    # --- central FD on the float64 twin ---
    eps = gate.fd_eps
    obs64 = _np(obs, D, rows=T)

    def nll64(F_v, Lq_v, r_v) -> float:
        Ff = params.make_F(F_v.to(torch.float64), D, structure)
        Qf = params.make_Q(Lq_v.to(torch.float64), D, q_floor=DEFAULT.q_floor)
        Rf = params.make_R(r_v.to(torch.float64), D, r_floor=DEFAULT.r_floor)
        return float(np.asarray(reference.reference_nll_lim(
            Ff, Qf, Rf, obs64, D, T, dtype=np.float64)).reshape(()))

    g_fd = []
    for kind, idx in probes:
        Fp, Fm = F_raw.clone(), F_raw.clone()
        Lp, Lm = Lq_raw.clone(), Lq_raw.clone()
        rp = r_raw.clone(); rm = r_raw.clone()
        if kind == "r":
            rp = rp + eps; rm = rm - eps
        elif kind == "Lq":
            Lp[idx] += eps; Lm[idx] -= eps
        else:
            Fp[idx] += eps; Fm[idx] -= eps
        plus = nll64(Fp, Lp, rp)
        minus = nll64(Fm, Lm, rm)
        g_fd.append((plus - minus) / (2 * eps))

    g_dmci = np.asarray(g_dmci, dtype=np.float64)
    g_fd = np.asarray(g_fd, dtype=np.float64)
    denom = np.linalg.norm(g_fd)
    if denom < 1e-12:
        # near-zero true gradient: fall back to absolute agreement so we don't divide by ~0.
        return float(np.linalg.norm(g_dmci - g_fd))
    return float(np.linalg.norm(g_dmci - g_fd) / denom)


def _batched_binding_ok(F, Q, R, obs, D, T, structure, cfg: ExpLimConfig):
    """Try binding an [N,D,D] F (and Q,R) and evaluating; return (ok, detail).

    DiffEvo wants to evaluate a whole population in one interpreter walk. If the batched
    binding errors, the caller records G6=False (non-blocking) and DiffEvo falls back to a
    per-candidate Python loop -- it does NOT fail the gate."""
    from neural_compiler.dmci import as_matrix
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import unwrap_number

    N = 3
    Fb = F.detach().unsqueeze(0).repeat(N, 1, 1).contiguous()
    Qb = Q.detach().unsqueeze(0).repeat(N, 1, 1).contiguous()
    Rb = R.detach().unsqueeze(0).repeat(N, 1, 1).contiguous()
    obs32 = obs[:T, :D].to(torch.float32).contiguous()
    g = _get_graph(D, T, structure, False)
    with torch.no_grad():
        out = evaluate(g, {"F": as_matrix(Fb), "Q": as_matrix(Qb),
                           "R": as_matrix(Rb), "obs": as_matrix(obs32)}, **cfg.EVAL_KW)
        val = unwrap_number(out)
    finite = bool(torch.isfinite(val).all())
    shape = tuple(val.shape) if hasattr(val, "shape") else ()
    return (finite and shape == (N,)), f"out shape={shape} finite={finite}"


def _np(x, D, rows=None):
    """Detach a torch tensor / numpy array to a float64 numpy array reshaped to [D,D] or [rows,D]."""
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    a = a.astype(np.float64)
    if rows is not None:
        return a.reshape(-1, D)[:rows]
    return a.reshape(D, D)


# ===========================================================================
# Real PC window loader (optional; non-blocking when absent).
# ===========================================================================

def load_real_obs(D: int, T: int):
    """Load pcs[:T, :D] from data/processed/pcs.npy if it exists, else None.

    Returns a float32 [T, D] torch tensor or None (data not built/rsync'd yet -- the gate
    runs on the synthetic LIM only and records the real run as skipped)."""
    p = PROCESSED_DIR / "pcs.npy"
    if not p.exists():
        return None
    pcs = np.load(p)
    if pcs.shape[0] < T or pcs.shape[1] < D:
        print(f"    [skip real] pcs.npy shape {pcs.shape} too small for (T={T}, D={D})")
        return None
    return torch.tensor(pcs[:T, :D].astype(np.float32))


# ===========================================================================
# Public entry point.
# ===========================================================================

def run_gate(D: int, T: int, structure: str = "S0", obs=None,
             cfg: ExpLimConfig = DEFAULT, gate: GateThresholds = GATE) -> dict:
    """Run the full numerical gate at (D, T, structure) and return the verdict dict.

    Runs the G1..G6 battery on a synthetic stable LIM ALWAYS, and on the real PC window
    when available (``obs`` given, else auto-loaded from data/processed/pcs.npy; skipped
    cleanly when absent). GO = G1..G5 all pass on every dataset that ran (G6 never blocks).

    Args:
        D: state dimension (PC truncation).
        T: number of filter steps (observations).
        structure: F-assembly variant (Phase 1: 'S0' dense).
        obs: optional [T,D]-ish observation tensor to use as the REAL window. If None, the
            real window is auto-loaded from pcs.npy (and skipped if that is absent).
        cfg, gate: config + thresholds (defaults from config.py).

    Side effect: writes experiments/exp_lim_enso/gate_{D}.json.
    """
    print(f"\n{'=' * 64}\nLIM-ENSO GATE  D={D}  T={T}  structure={structure}\n{'=' * 64}")
    tally = _Tally()
    per_dataset: dict[str, dict] = {}

    # --- 1) synthetic stable LIM (always available) ---
    raw_s, Fs, Qs, Rs = synthetic_lim(D)
    obs_s = synthetic_obs(D, T, Fs, Qs, Rs, seed=0)
    per_dataset["synthetic"] = _run_checks_on(
        tally, "synthetic", D, T, structure, raw_s, Fs, Qs, Rs, obs_s, gate, cfg)

    # --- 2) real PC window (optional) ---
    real_obs = obs if obs is not None else load_real_obs(D, T)
    real_ran = real_obs is not None
    if real_ran:
        real_obs = real_obs[:T, :D].to(torch.float32).contiguous()
        # Fit the synthetic-init F/Q/R (near 0.9*I, small PD Q, r_floor R) to the REAL data:
        # this is exactly the init the MLE starts from, so the gate proves the very first
        # forward/backward on real data is sound (the MLE then takes over on GO).
        raw_r = params.init_raw_params(D, structure, seed=0)
        Fr = params.make_F(raw_r["F_raw"], D, structure)
        Qr = params.make_Q(raw_r["Lq_raw"], D)
        Rr = params.make_R(raw_r["r_raw"], D)
        per_dataset["real"] = _run_checks_on(
            tally, "real", D, T, structure, raw_r, Fr, Qr, Rr, real_obs, gate, cfg)
    else:
        print("\n  --- real PC window: SKIPPED (data/processed/pcs.npy absent or too small) ---")
        print("      gate runs on the synthetic LIM only; build/rsync data, then re-gate.")

    # --- verdict: GO = G1..G5 pass on every dataset that ran (G6 non-blocking) ---
    datasets = ["synthetic"] + (["real"] if real_ran else [])
    blocking_codes = ["G1", "G2", "G3", "G4", "G5"]
    code_to_name = {
        "G1": "G1 NLL finite over T",
        "G2": "G2 covariance PD every step",
        "G3": "G3 DMCI NLL == ref float32-twin",
        "G4": "G4 gradient == finite-difference",
        "G5": "G5 det-underflow margin",
        "G6": "G6 batched [N,D,D] F binding",
    }
    checks: dict[str, bool] = {}
    blocking_failures: list[str] = []
    for ds in datasets:
        for code in blocking_codes + ["G6"]:
            name = f"{code_to_name[code]} [{ds}]"
            ok = tally.passed(name)
            checks[name] = ok
            if code in blocking_codes and not ok:
                blocking_failures.append(name)

    GO = len(blocking_failures) == 0

    # roll up the worst-case numerics across datasets for the JSON header.
    min_detS = min((m["min_detS"] for m in per_dataset.values()
                    if m.get("min_detS") is not None), default=None)
    cond_S = max((m["cond_S"] for m in per_dataset.values()
                  if m.get("cond_S") is not None), default=None)
    per_step_ms = max((m["per_step_ms"] for m in per_dataset.values()
                       if m.get("per_step_ms") is not None), default=None)

    n_pass = sum(1 for _, ok in tally.results if ok)
    n_total = len(tally.results)
    print(f"\n  {n_pass}/{n_total} checks passed   GO={GO}")
    if blocking_failures:
        print("  blocking failures:", blocking_failures)

    verdict = {
        "D": D, "T": T, "structure": structure,
        "GO": GO,
        "checks": checks,
        "blocking_failures": blocking_failures,
        "min_detS": min_detS,
        "cond_S": cond_S,
        "per_step_ms": per_step_ms,
        "real_window_ran": real_ran,
        "per_dataset": per_dataset,
        "thresholds": {
            "parity_rel": gate.parity_rel, "detS_floor": gate.detS_floor,
            "fd_rel": gate.fd_rel, "fd_eps": gate.fd_eps, "fd_n_probe": gate.fd_n_probe,
        },
        "details": tally.details,
    }
    out_path = HERE / f"gate_{D}.json"
    out_path.write_text(json.dumps(verdict, indent=2, default=str))
    print(f"  wrote {out_path}")
    return verdict


# ===========================================================================
# CLI.
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="LIM-ENSO numerical GO/NO-GO gate")
    ap.add_argument("--D", type=int, default=None,
                    help="state dimension (default: loop config.D_list)")
    ap.add_argument("--T", type=int, default=None,
                    help="filter steps (default: config.T_train)")
    ap.add_argument("--structure", type=str, default="S0",
                    help="F-assembly variant (Phase 1: S0)")
    args = ap.parse_args()

    cfg = DEFAULT
    T = args.T if args.T is not None else cfg.T_train
    D_list = [args.D] if args.D is not None else list(cfg.D_list)

    all_go = True
    summary: list[tuple[int, bool]] = []
    for D in D_list:
        v = run_gate(D, T, structure=args.structure, cfg=cfg)
        all_go = all_go and v["GO"]
        summary.append((D, v["GO"]))

    print("\n" + "=" * 64)
    print("GATE SUMMARY")
    for D, go in summary:
        print(f"  [{'GO  ' if go else 'NO-GO'}] D={D}")
    print(f"  overall: {'GO' if all_go else 'NO-GO'}")
    print("=" * 64)
    sys.exit(0 if all_go else 1)


if __name__ == "__main__":
    main()
