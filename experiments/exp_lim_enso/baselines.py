############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# baselines.py: Optimizer portfolio over the SHARED LIM Kalman-NLL objective (the DMCI fold). Every solver in this module...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Optimizer portfolio over the SHARED LIM Kalman-NLL objective (the DMCI fold).

Every solver in this module minimizes the SAME negative log-likelihood that
``models.run_kalman_nll`` folds through the DMCI meta-circular interpreter, over the
SAME raw (F, Q, R) parametrization (``params.init_raw_params`` ->
``params.make_F``/``make_Q``/``make_R``). The compiled interpreter graph is built ONCE
per ``(structure, D, T)`` inside ``models._get_graph`` and reused across seeds and
solvers -- so the only thing that changes between an Adam step, an L-BFGS restart, and a
DiffEvo generation is the bound F/Q/R/obs tensors. That is the whole point: the portfolio
shares a single program-as-data objective and we compare optimizers, not objectives.

Solvers (all return the SAME result-dict schema; see ``_result``):
  - run_dmci_adam            -- THE PRIMARY solver. Single-start Adam on the raw params
                                with exact DMCI gradients, log/Cholesky-implicit positivity,
                                grad clipping, and a NaN finite-guard.
  - run_lbfgs_multistart     -- reparam-free L-BFGS (strong-Wolfe) with multi-start, reusing
                                the exp_i multistart machinery; keeps the best restart.
  - run_diffevo_batched      -- differential evolution whose population is evaluated in ONE
                                batched interpreter walk (F as as_matrix([N,D,D]); gate G6).
                                Falls back to a correct Python loop (Nx slower) if the
                                batched [N,D,D] eval errors, setting result['batched']=False.

Honest-comparison controls (uniform across solvers):
  - the compiled graph is SHARED (one compile per (structure, D, T));
  - an optional equal wall-clock budget (``time_budget`` seconds) stops any solver early;
  - every result records wall_time, per_step_ms, min_detS, cond_S (worst-step conditioning
    of the innovation covariance at the fitted operator, via the float64 reference twin).

DMCI is interpreter-bound and runs on CPU (see slurm_gate.sh / MEMORY). The headline is
CAPABILITY -- a real-data dynamical-systems MLE expressed as a program and optimized by
exact DMCI gradients -- never optimizer wall-clock. dynamax/the numpy twin are correctness
oracles; the Green-function operator is the scientific reference, never a speed contest.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch

from .config import DEFAULT, ExpLimConfig
from . import params, reference
from .models import run_kalman_nll, _get_graph


_DTYPE = torch.float32


# ===========================================================================
# Shared objective closure (assemble F/Q/R -> DMCI NLL [+ stability penalty]).
# ===========================================================================

def make_objective(D: int, T: int, structure: str, obs: torch.Tensor,
                   cfg: ExpLimConfig = DEFAULT, jitter: bool = False):
    """Build the shared scalar NLL objective over the raw (F, Q, R) parameters.

    Returns a closure ``objective(F_raw, Lq_raw, r_raw, grad=True) -> 0-d Tensor`` that:
      F = params.make_F(F_raw, D, structure)
      Q = params.make_Q(Lq_raw, D, cfg.q_floor)
      R = params.make_R(r_raw, D, cfg.r_floor)
      nll = models.run_kalman_nll(F, Q, R, obs, D, T, structure, jitter, grad)
      + cfg.stability_penalty * params.stability_penalty(F)        (when weight > 0)

    The DMCI graph for ``(structure, D, T, jitter)`` is compiled+cached ONCE (in
    ``models._get_graph``) and reused by every call -- so every solver in this module
    optimizes the IDENTICAL compiled program; only the bound tensors change. The optional
    spectral-radius soft penalty pushes F into the stable disk (default weight 0 -> the
    raw DMCI NLL). The eigvals penalty is only computed when its weight is nonzero, so the
    primary Adam path pays no per-step eig cost unless stability regularization is enabled.

    ``obs`` is sliced/cast to a contiguous ``[T, D]`` float32 tensor ONCE here.
    """
    rank = cfg.lowrank_rank
    obs_t = obs[:T, :D].to(_DTYPE).contiguous()
    w_stab = float(cfg.stability_penalty)
    q_floor = cfg.q_floor
    r_floor = cfg.r_floor

    def objective(F_raw: torch.Tensor, Lq_raw: torch.Tensor, r_raw: torch.Tensor,
                  grad: bool = True) -> torch.Tensor:
        F = params.make_F(F_raw, D, structure, rank)
        Q = params.make_Q(Lq_raw, D, q_floor)
        R = params.make_R(r_raw, D, r_floor)
        # For variants, bind the FACTOR inputs so F is assembled IN-PROGRAM (the LLM's combine-
        # algebra prelude); grad flows through the factors back to F_raw. S0 binds F directly.
        fac = None if structure == "S0" else params.make_F_factors(F_raw, D, structure, rank)
        nll = run_kalman_nll(F, Q, R, obs_t, D, T, structure, jitter=jitter, grad=grad,
                             f_factors=fac)
        nll = nll.reshape(())
        if w_stab > 0.0:
            nll = nll + w_stab * params.stability_penalty(F)
        return nll

    return objective


# ===========================================================================
# Conditioning / PD diagnostics at the FITTED operator (float64 twin).
# ===========================================================================

def _fitted_diagnostics(F: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
                        obs: torch.Tensor, D: int, T: int,
                        jitter: bool = False) -> tuple[float, float]:
    """Worst-step (min det S, max cond S) of the innovation covariance at the fitted (F,Q,R).

    Replays the float64 reference filter once at the fitted operator and tracks per-step
    ``det(S)`` (PD / underflow margin) and ``cond(S)`` (conditioning). Cheap relative to a
    DMCI walk; mirrors gate.py's ``_max_cond_S`` so the numbers are directly comparable to
    the GO/NO-GO gate verdict."""
    Fm = _np(F, D); Qm = _np(Q, D); Rm = _np(R, D)
    # obs is the FULL [T_full, D_max] PC tensor; slice to the (T, D) window. (Previously
    # reshape(-1, D), which mis-grouped a [T_full, D_max] array and RAISED whenever
    # T_full*D_max was not divisible by D -- the D=6/D=15 null-min_detS bug. det S / cond S
    # are observation-independent, so the value is unaffected; the reshape was the only failure.)
    Yfull = np.asarray(_np_arr(obs), dtype=np.float64)
    Y = (Yfull[:, :D] if Yfull.ndim == 2 else Yfull.reshape(-1, D))[:T]
    eps = float(DEFAULT.jitter_eps)
    Im = np.eye(D)
    x = np.zeros(D); P = np.eye(D)
    min_detS = float("inf")
    max_cond = 0.0
    for k in range(T):
        xpred = Fm @ x
        Ppred = (Fm @ P) @ Fm.T + Qm
        e = Y[k] - xpred
        S = Ppred + Rm
        Sj = S + eps * Im if jitter else S
        d = float(np.linalg.det(Sj))
        min_detS = min(min_detS, d if np.isfinite(d) else -float("inf"))
        c = float(np.linalg.cond(Sj))
        max_cond = max(max_cond, c if np.isfinite(c) else float("inf"))
        Sinv = np.linalg.inv(Sj)
        K = Ppred @ Sinv
        x = xpred + K @ e
        P = (Im - K) @ Ppred
    return min_detS, max_cond


def _np(x, D):
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return a.astype(np.float64).reshape(D, D)


def _np_arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


# ===========================================================================
# Result schema (identical across solvers).
# ===========================================================================

def _result(solver: str, D: int, T: int, structure: str, seed: int,
            raw: dict, objective, obs: torch.Tensor, cfg: ExpLimConfig,
            final_nll: float, n_iters: int, converged: bool,
            wall_time: float, nll_trace: list, grad_norm_trace: list,
            *, jitter: bool = False, extra: Optional[dict] = None) -> dict:
    """Assemble the common result dict from the fitted raw params + traces.

    Decodes the FITTED (F, Q, R) to detached numpy, computes the fitted-operator
    conditioning (min det S, max cond S) via the float64 twin, and reports per_step_ms
    (wall_time amortized over the total interpreter steps = n_iters * T). ``raw`` holds the
    final leaf raw params; ``extra`` merges solver-specific keys (e.g. batched flag)."""
    with torch.no_grad():
        F = params.make_F(raw["F_raw"].detach(), D, structure, cfg.lowrank_rank)
        Q = params.make_Q(raw["Lq_raw"].detach(), D, cfg.q_floor)
        R = params.make_R(raw["r_raw"].detach(), D, cfg.r_floor)
    try:
        min_detS, cond_S = _fitted_diagnostics(F, Q, R, obs, D, T, jitter=jitter)
    except Exception:  # noqa: BLE001  (diagnostics must never abort a fit result)
        min_detS, cond_S = float("nan"), float("nan")

    total_steps = max(1, n_iters) * max(1, T)
    out = {
        "solver": solver,
        "D": D, "T": T, "structure": structure, "seed": seed,
        "final_nll": float(final_nll),
        "n_iters": int(n_iters),
        "converged": bool(converged),
        "wall_time": float(wall_time),
        "per_step_ms": 1e3 * float(wall_time) / total_steps,
        "min_detS": float(min_detS),
        "cond_S": float(cond_S),
        "fitted": {
            "F": F.detach().cpu().numpy(),
            "Q": Q.detach().cpu().numpy(),
            "R": R.detach().cpu().numpy(),
        },
        "nll_trace": [float(v) for v in nll_trace],
        "grad_norm_trace": [float(v) for v in grad_norm_trace],
    }
    if extra:
        out.update(extra)
    return out


# ===========================================================================
# 1) PRIMARY solver: single-start Adam with exact DMCI gradients.
# ===========================================================================

def run_dmci_adam(D: int, structure: str, seed: int, obs: torch.Tensor,
                  cfg: ExpLimConfig = DEFAULT, *, T: Optional[int] = None,
                  jitter: bool = False, tol: float = 1e-4,
                  time_budget: Optional[float] = None,
                  tag: Optional[str] = None, log_every: Optional[int] = None) -> dict:
    """Fit the LIM MLE by Adam over the raw (F, Q, R) params -- THE PRIMARY solver.

    Adam (lr=cfg.lr) on the leaf raw parameters from ``params.init_raw_params``; positivity
    of Q/R is implicit in the parametrization (Cholesky Lq, softplus r), so Adam optimizes
    UNCONSTRAINED raw vectors and the decode keeps Q, R in the PD cone by construction.
    Gradients are EXACT DMCI autograd through the compiled interpreter. Per step:
      assemble F/Q/R -> DMCI NLL (+ optional stability penalty) -> backward -> grad-clip
      (cfg.grad_clip) -> Adam step.
    Finite-guard: a non-finite NLL or gradient ABORTS the step (parameters are restored to
    the last finite snapshot and the fit stops) -- never silently steps on NaN. Convergence:
    ``|ΔNLL| < tol`` between consecutive finite steps, or ``cfg.adam_iters`` reached, or the
    optional ``time_budget`` (seconds) elapses.

    Returns the common result dict (see ``_result``) plus ``nll_trace`` / ``grad_norm_trace``.
    """
    if T is None:
        T = cfg.T_train
    if tag is None:
        tag = f"dmci_adam_{structure}_D{D}_seed{seed}"
    if log_every is None:
        log_every = int(getattr(cfg, "log_every", 25))
    raw = params.init_raw_params(D, structure, seed=seed, rank=cfg.lowrank_rank,
                                 q_floor=cfg.q_floor, r_floor=cfg.r_floor)
    leaves = [raw["F_raw"], raw["Lq_raw"], raw["r_raw"]]
    opt = torch.optim.Adam(leaves, lr=cfg.lr)
    objective = make_objective(D, T, structure, obs, cfg, jitter=jitter)

    nll_trace: list = []
    grad_norm_trace: list = []
    best_snapshot = {k: v.detach().clone() for k, v in raw.items()}
    prev_nll = float("inf")
    final_nll = float("inf")
    converged = False
    n_iters = 0
    nan_stall = False

    t0 = time.perf_counter()
    for it in range(cfg.adam_iters):
        opt.zero_grad()
        nll = objective(raw["F_raw"], raw["Lq_raw"], raw["r_raw"], grad=True)
        nll_val = float(nll.detach())

        if not math.isfinite(nll_val):
            nan_stall = True
            break
        try:
            nll.backward()
        except Exception:  # noqa: BLE001
            nan_stall = True
            break
        grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in leaves)
        if not grads_finite:
            nan_stall = True
            break

        gnorm = float(torch.nn.utils.clip_grad_norm_(leaves, cfg.grad_clip))
        opt.step()

        nll_trace.append(nll_val)
        grad_norm_trace.append(gnorm)
        final_nll = nll_val
        n_iters = it + 1
        # last finite snapshot (restore target if a later step blows up)
        best_snapshot = {k: v.detach().clone() for k, v in raw.items()}

        # live progress (diagnostic only; never touches the numerics)
        if log_every and (it % log_every == 0 or it == cfg.adam_iters - 1):
            print(f"[fit {tag}] it={it:>4d} nll={nll_val:.6g} gnorm={gnorm:.4g}",
                  flush=True)

        if abs(prev_nll - nll_val) < tol:
            converged = True
            break
        prev_nll = nll_val
        if time_budget is not None and (time.perf_counter() - t0) > time_budget:
            break

    wall_time = time.perf_counter() - t0
    # restore the last finite snapshot (guards against a final blown-up step)
    for k in raw:
        raw[k] = best_snapshot[k].clone()

    return _result("dmci_adam", D, T, structure, seed, raw, objective, obs, cfg,
                   final_nll=final_nll, n_iters=n_iters, converged=converged,
                   wall_time=wall_time, nll_trace=nll_trace,
                   grad_norm_trace=grad_norm_trace, jitter=jitter,
                   extra={"nan_stall": nan_stall})


# ===========================================================================
# 1b) BATCHED-SHARED-GRAPH Adam: all seeds optimized in ONE interpreter walk.
# ===========================================================================

def _assemble_batched(F_raw: torch.Tensor, Lq_raw: torch.Tensor, r_raw: torch.Tensor,
                      D: int, q_floor: float, r_floor: float):
    """Decode stacked raw leaves ``[N,...]`` into batched ``[N,D,D]`` F, Q, R.

    The batched twin of ``params.make_F``/``make_Q``/``make_R`` for ``structure='S0'``: the
    SAME algebra (S0 dense reshape; Q = L L^T + q_floor I via a tril scatter; R =
    softplus(r)*I + r_floor*I), broadcast over a LEADING batch axis so ``torch.linalg``
    walks the whole batch in one shot. Grad flows back to each stacked leaf; because the
    batch axis is leading and every op is elementwise / batched-matmul, the Jacobian is
    block-diagonal across members (no cross-member mixing). float32 throughout.

    Args:
        F_raw:  ``[N, D*D]`` stacked S0 dense raw operators.
        Lq_raw: ``[N, D*(D+1)/2]`` stacked Cholesky lower-triangle vectors.
        r_raw:  ``[N]`` or ``[N,1]`` stacked scalar observation-noise raw params.
    Returns ``(F, Q, R)`` each ``[N, D, D]`` float32.
    """
    N = F_raw.shape[0]
    F = F_raw.reshape(N, D, D)
    k = D * (D + 1) // 2
    L = torch.zeros(N, D, D, dtype=Lq_raw.dtype, device=Lq_raw.device)
    idx = torch.tril_indices(D, D, device=Lq_raw.device)
    L[:, idx[0], idx[1]] = Lq_raw.reshape(N, k)
    eye = torch.eye(D, dtype=F_raw.dtype, device=F_raw.device)
    Q = L @ L.transpose(-1, -2) + q_floor * eye
    r = torch.nn.functional.softplus(r_raw.reshape(N)) + r_floor
    R = r.reshape(N, 1, 1) * eye
    return F, Q, R


def run_adam_batched_seeds(D: int, structure: str, seeds, obs: torch.Tensor,
                           cfg: ExpLimConfig = DEFAULT, *, T: Optional[int] = None,
                           jitter: bool = False, tol: float = 1e-4,
                           time_budget: Optional[float] = None,
                           tag: Optional[str] = None,
                           log_every: Optional[int] = None) -> dict:
    """Fit several seeds at once: ONE shared-graph interpreter walk over a ``[N,...]`` batch.

    This is the batched twin of ``run_dmci_adam``. Instead of N serial single-start fits, it
    stacks the per-seed raw leaves onto a LEADING ``[N]`` batch axis (``F_raw`` -> ``[N,D*D]``,
    ``Lq_raw`` -> ``[N,k]``, ``r_raw`` -> ``[N]`` from ``params.init_raw_params`` per seed),
    decodes them to batched ``F``/``Q``/``R`` = ``[N,D,D]`` (``_assemble_batched``), and folds
    ALL N negative log-likelihoods through the SAME compiled DMCI graph in ONE walk -- so the
    interpreter dispatch (the dominant cost) is paid once for the whole batch rather than N
    times. The batched walk returns ``NLL`` as ``[N]``; ``NLL.sum().backward()`` yields
    per-member gradients because the Jacobian is BLOCK-DIAGONAL across the batch (every kernel
    is elementwise or batched-matmul over the leading axis, so summing the NLLs does NOT mix
    members -- a member's grad is identical to its solo ``run_dmci_adam`` grad to float32). Adam
    runs on the ``[N,...]`` leaves elementwise, so each member optimizes INDEPENDENTLY with the
    SAME lr / clip schedule as the serial solver.

    Per-member convergence bookkeeping: each member keeps its own NLL trace, grad-norm trace,
    and last-finite raw snapshot; a member is marked converged when its ``|ΔNLL| < tol`` between
    consecutive finite steps (or when it NaN-stalls, frozen at its last finite snapshot). The
    WHOLE batch stops when EVERY member has converged (or NaN-stalled) OR ``cfg.adam_iters`` is
    reached OR the optional ``time_budget`` elapses. NOTE: torch's single fused Adam step over
    the shared ``[N,...]`` leaves cannot early-stop an individual member, so a member that
    converges early keeps being EVALUATED (its grad is masked to zero, so its params no longer
    move) until the slowest member converges. That extra-evaluation-after-local-convergence is
    the documented batching tradeoff: the per-member RESULT is unchanged, only some members are
    walked a few extra (no-op) iterations.

    Numeric behavior matches ``run_dmci_adam`` to float32 tolerance (NOT bit-identical: the
    ``[N,D,D]`` kernels reorder float ops relative to the per-member ``[D,D]`` path).

    Args:
        seeds: iterable of integer init/multistart seeds (defines the batch members + order).
        D, structure: run coordinates (Phase-1 batched path supports ``structure='S0'``).
        obs: FULL ``[T_full, D_max]`` PC tensor; sliced to ``[T, D]`` once.
        cfg: ExpLimConfig (lr, adam_iters, grad_clip, q_floor, r_floor, EVAL_KW, ...).
        T: filter length (default ``cfg.T_train``).
        jitter, tol, time_budget, log_every: as in ``run_dmci_adam``.

    Returns ``dict[seed -> result]`` where each value is the SAME per-result schema
    ``run_dmci_adam`` returns (final_nll, n_iters, converged, wall_time, per_step_ms,
    fitted{F,Q,R np}, nll_trace, grad_norm_trace, min_detS, cond_S, ...), so the runner consumes
    it unchanged. ``wall_time`` is the shared batched wall apportioned EQUALLY across members
    (the batch is fit jointly; no per-member wall is separable), and a ``batched=True`` flag plus
    the shared ``batch_size`` are merged into ``extra``.
    """
    if T is None:
        T = cfg.T_train
    seeds = [int(s) for s in seeds]
    N = len(seeds)
    if N == 0:
        return {}
    if tag is None:
        tag = f"adam_batched_{structure}_D{D}_seedsx{N}"
    if log_every is None:
        log_every = int(getattr(cfg, "log_every", 25))
    if structure != "S0":
        # Only S0's batched assembly twin is authored; non-S0 raw layouts (diag/lowrank)
        # need their own [N,...] decode before batching. Fail loudly rather than mis-assemble.
        raise NotImplementedError(
            f"run_adam_batched_seeds: batched assembly is authored for 'S0' only, got "
            f"{structure!r}. Use run_dmci_adam per seed for non-S0 until the batched twin "
            f"of make_F is wired in _assemble_batched.")

    q_floor = cfg.q_floor
    r_floor = cfg.r_floor
    obs_t = obs[:T, :D].to(_DTYPE).contiguous()
    w_stab = float(cfg.stability_penalty)

    # --- stack per-seed raw leaves onto a leading [N] axis (same inits as run_dmci_adam) ---
    per = [params.init_raw_params(D, structure, seed=s, rank=cfg.lowrank_rank,
                                  q_floor=q_floor, r_floor=r_floor) for s in seeds]
    F_raw = torch.stack([p["F_raw"].detach().reshape(-1) for p in per]).clone().requires_grad_(True)
    Lq_raw = torch.stack([p["Lq_raw"].detach().reshape(-1) for p in per]).clone().requires_grad_(True)
    r_raw = torch.stack([p["r_raw"].detach().reshape(1) for p in per]).clone().requires_grad_(True)
    leaves = [F_raw, Lq_raw, r_raw]
    opt = torch.optim.Adam(leaves, lr=cfg.lr)

    # --- per-member bookkeeping ---
    nll_traces: list[list[float]] = [[] for _ in range(N)]
    grad_traces: list[list[float]] = [[] for _ in range(N)]
    prev_nll = torch.full((N,), float("inf"), dtype=_DTYPE)
    final_nll = torch.full((N,), float("inf"), dtype=_DTYPE)
    converged = [False] * N           # |dNLL|<tol reached
    nan_stall = [False] * N           # member produced a non-finite NLL/grad (frozen)
    active = [True] * N               # still being optimized (not yet converged/stalled)
    n_iters = [0] * N
    # last-finite snapshot per member (restore target if a later step blows up)
    snap_F = F_raw.detach().clone()
    snap_Lq = Lq_raw.detach().clone()
    snap_r = r_raw.detach().clone()

    t0 = time.perf_counter()
    for it in range(cfg.adam_iters):
        opt.zero_grad()
        F, Q, R = _assemble_batched(F_raw, Lq_raw, r_raw, D, q_floor, r_floor)
        nll_b = run_kalman_nll(F, Q, R, obs_t, D, T, structure, jitter=jitter, grad=True)
        nll_b = nll_b.reshape(N)
        if w_stab > 0.0:
            pen = torch.stack([params.stability_penalty(F[i]) for i in range(N)])
            nll_b = nll_b + w_stab * pen
        nll_vals = nll_b.detach().clone()

        # sum over ACTIVE & finite members only -> backward gives each its own grad
        finite = torch.isfinite(nll_vals)
        active_mask = torch.tensor(active, dtype=torch.bool)
        contrib = active_mask & finite
        if contrib.any():
            loss = nll_b[contrib].sum()
            try:
                loss.backward()
                grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in leaves)
            except Exception:  # noqa: BLE001
                grads_ok = False
        else:
            grads_ok = False

        # per-member grad norms (over the member's own leaf rows), pre-clip
        if grads_ok:
            gn = torch.zeros(N, dtype=_DTYPE)
            for p in leaves:
                if p.grad is not None:
                    gp = p.grad.detach().reshape(N, -1)
                    gn = gn + gp.pow(2).sum(dim=1)
            gn = gn.sqrt()
        else:
            gn = torch.full((N,), float("nan"), dtype=_DTYPE)

        # mark NaN-stalled members: active members with a non-finite NLL (or a global grad
        # failure) freeze at their last finite snapshot and stop contributing.
        for i in range(N):
            if not active[i]:
                continue
            if (not bool(finite[i])) or (not grads_ok):
                nan_stall[i] = True
                active[i] = False

        # zero out grads for members that are no longer active (converged/stalled) so the
        # fused Adam step is a no-op for them (their params stop moving).
        if grads_ok:
            for p in leaves:
                if p.grad is not None:
                    mask = torch.tensor(active, dtype=p.grad.dtype).reshape(
                        N, *([1] * (p.grad.dim() - 1)))
                    p.grad.mul_(mask)
            # PER-MEMBER grad clip (matches run_dmci_adam's PER-SEED semantics exactly):
            # each member is clipped to cfg.grad_clip on its OWN gradient norm `gn[i]`, not a
            # single global norm over the whole batch. A global clip couples the members (with
            # huge initial grads ~2e4 >> clip, the combined norm scales every member differently
            # than a per-seed clip would), making the batched trajectory diverge from serial.
            # gn is the pre-clip per-member norm computed above; inactive members have gn=0
            # (grads already masked to 0) so their scale is a no-op.
            clip_scale = (cfg.grad_clip / (gn + 1e-12)).clamp(max=1.0)   # [N]
            for p in leaves:
                if p.grad is not None:
                    p.grad.mul_(clip_scale.reshape(N, *([1] * (p.grad.dim() - 1))))
            opt.step()
            # refresh last-finite snapshots for members that just took a finite step
            with torch.no_grad():
                for i in range(N):
                    if finite[i]:
                        snap_F[i] = F_raw[i].detach().clone()
                        snap_Lq[i] = Lq_raw[i].detach().clone()
                        snap_r[i] = r_raw[i].detach().clone()

        # record traces + convergence for members that produced a finite NLL this step
        for i in range(N):
            v = float(nll_vals[i])
            if not math.isfinite(v):
                continue
            # only log/advance a member while it was active at the START of this step
            # (a member marked converged earlier has already stopped); we detect that via
            # n_iters not advancing once converged.
            if converged[i]:
                continue
            nll_traces[i].append(v)
            grad_traces[i].append(float(gn[i]) if math.isfinite(float(gn[i])) else float("nan"))
            final_nll[i] = v
            n_iters[i] = it + 1
            if abs(float(prev_nll[i]) - v) < tol:
                converged[i] = True
                active[i] = False
            prev_nll[i] = v

        if log_every and (it % log_every == 0 or it == cfg.adam_iters - 1):
            nact = sum(1 for a in active if a)
            print(f"[fit {tag}] it={it:>4d} active={nact}/{N} "
                  f"nll_mean={float(nll_vals[finite].mean()) if finite.any() else float('nan'):.6g}",
                  flush=True)

        if not any(active):                  # every member converged or NaN-stalled
            break
        if time_budget is not None and (time.perf_counter() - t0) > time_budget:
            break

    wall_time = time.perf_counter() - t0
    per_member_wall = wall_time / N        # shared batched wall, apportioned equally

    # restore each member's last-finite snapshot, then build the per-seed result dicts.
    with torch.no_grad():
        F_raw.copy_(snap_F)
        Lq_raw.copy_(snap_Lq)
        r_raw.copy_(snap_r)

    objective = make_objective(D, T, structure, obs, cfg, jitter=jitter)  # for _result decode
    out: dict[int, dict] = {}
    for i, s in enumerate(seeds):
        raw_i = {
            "F_raw": snap_F[i].detach().clone(),
            "Lq_raw": snap_Lq[i].detach().clone(),
            "r_raw": snap_r[i].detach().clone().reshape(()),
        }
        res = _result("dmci_adam", D, T, structure, s, raw_i, objective, obs, cfg,
                      final_nll=float(final_nll[i]), n_iters=int(n_iters[i]),
                      converged=bool(converged[i]), wall_time=per_member_wall,
                      nll_trace=nll_traces[i], grad_norm_trace=grad_traces[i],
                      jitter=jitter,
                      extra={"nan_stall": bool(nan_stall[i]), "batched": True,
                             "batch_size": N})
        out[s] = res
    return out


# ===========================================================================
# 2) L-BFGS multi-start (reuses the exp_i multistart machinery, on the shared graph).
# ===========================================================================

def run_lbfgs_multistart(D: int, structure: str, seed: int, obs: torch.Tensor,
                         cfg: ExpLimConfig = DEFAULT, *, T: Optional[int] = None,
                         n_starts: int = 8, max_iter: int = 50, jitter: bool = False,
                         time_budget: Optional[float] = None,
                         tag: Optional[str] = None, log_every: Optional[int] = None) -> dict:
    """Multi-start L-BFGS (strong-Wolfe) over the SAME shared NLL objective; keep the best.

    Mirrors exp_i.harness.run_lbfgs_multistart: up to ``n_starts`` restarts (or until
    ``time_budget`` seconds), each a curvature-aware L-BFGS optimization over the raw
    (F, Q, R) leaves on a single batched DMCI walk per closure call, keeping the lowest-NLL
    restart. Restart 0 uses the canonical near-stable init (``params.init_raw_params`` at the
    given seed); subsequent restarts perturb that init with seed-derived noise so the
    multi-start explores distinct basins of the (non-convex) likelihood. Each closure
    evaluation reuses the SAME compiled graph as Adam/DiffEvo -- only the optimizer differs.

    Returns the common result dict for the best restart. ``n_iters`` is the number of
    restarts completed; the L-BFGS inner iterations are folded into wall_time / per_step_ms.
    """
    if T is None:
        T = cfg.T_train
    if tag is None:
        tag = f"lbfgs_multistart_{structure}_D{D}_seed{seed}"
    objective = make_objective(D, T, structure, obs, cfg, jitter=jitter)

    best_nll = float("inf")
    best_raw = None
    best_trace: list = []
    best_gnorm: list = []
    nan_stall = False
    starts_done = 0
    # account total L-BFGS inner iterations across restarts for an honest per_step_ms.
    inner_iters_total = 0

    t0 = time.perf_counter()
    for s in range(n_starts):
        if time_budget is not None and (time.perf_counter() - t0) > time_budget:
            break
        raw = _lbfgs_start(D, structure, seed, s, cfg)
        leaves = [raw["F_raw"], raw["Lq_raw"], raw["r_raw"]]
        opt = torch.optim.LBFGS(leaves, max_iter=max_iter,
                                line_search_fn="strong_wolfe")
        trace: list = []
        gnorm_trace: list = []

        def closure():
            opt.zero_grad()
            nll = objective(raw["F_raw"], raw["Lq_raw"], raw["r_raw"], grad=True)
            nll_val = float(nll.detach())
            if math.isfinite(nll_val):
                nll.backward()
                g = 0.0
                for p in leaves:
                    if p.grad is not None:
                        g += float(p.grad.detach().pow(2).sum())
                trace.append(nll_val)
                gnorm_trace.append(float(math.sqrt(g)))
            return nll

        try:
            opt.step(closure)
        except Exception:  # noqa: BLE001
            nan_stall = True
        inner_iters_total += max(1, len(trace))
        starts_done += 1

        with torch.no_grad():
            nll_final = float(objective(raw["F_raw"], raw["Lq_raw"], raw["r_raw"],
                                        grad=False).detach())
        if math.isfinite(nll_final) and nll_final < best_nll:
            best_nll = nll_final
            best_raw = {k: v.detach().clone() for k, v in raw.items()}
            best_trace = trace
            best_gnorm = gnorm_trace

        # live progress: one flushed line per completed restart (diagnostic only).
        gnorm_last = best_gnorm[-1] if best_gnorm else float("nan")
        print(f"[fit {tag}] restart={s:>2d}/{n_starts} it={len(trace):>3d} "
              f"nll={nll_final:.6g} best={best_nll:.6g} gnorm={gnorm_last:.4g}", flush=True)

    wall_time = time.perf_counter() - t0
    if best_raw is None:  # every restart diverged -- fall back to the canonical init
        best_raw = params.init_raw_params(D, structure, seed=seed, rank=cfg.lowrank_rank,
                                          q_floor=cfg.q_floor, r_floor=cfg.r_floor)
        best_raw = {k: v.detach().clone() for k, v in best_raw.items()}

    res = _result("lbfgs_multistart", D, T, structure, seed, best_raw, objective, obs, cfg,
                  final_nll=best_nll, n_iters=starts_done,
                  converged=math.isfinite(best_nll), wall_time=wall_time,
                  nll_trace=best_trace, grad_norm_trace=best_gnorm, jitter=jitter,
                  extra={"n_starts": starts_done, "nan_stall": nan_stall})
    # per_step_ms over ALL inner L-BFGS iterations across restarts (honest amortization).
    res["per_step_ms"] = 1e3 * wall_time / max(1, inner_iters_total * T)
    return res


def _lbfgs_start(D: int, structure: str, seed: int, start: int,
                 cfg: ExpLimConfig) -> dict:
    """A restart point: restart 0 = canonical near-stable init; others perturb it.

    Keeps the optimizer in a sane region (Q/R positivity is parametrization-implicit, so a
    perturbed raw vector is still PD-decodable). The perturbation seed is derived from
    (seed, start) so the multi-start is deterministic per (seed, restart)."""
    raw = params.init_raw_params(D, structure, seed=seed, rank=cfg.lowrank_rank,
                                 q_floor=cfg.q_floor, r_floor=cfg.r_floor)
    if start == 0:
        return raw
    g = torch.Generator().manual_seed(10_000 * int(seed) + int(start))
    out = {}
    for k, v in raw.items():
        v = v.detach()
        scale = 0.1 * max(1.0, float(v.abs().mean()))
        pert = scale * torch.randn(v.shape, generator=g, dtype=v.dtype)
        out[k] = (v + pert).clone().requires_grad_(True)
    return out


# ===========================================================================
# 3) Differential evolution -- batched population in ONE interpreter walk.
# ===========================================================================

def run_diffevo_batched(D: int, structure: str, seed: int, obs: torch.Tensor,
                        cfg: ExpLimConfig = DEFAULT, *, T: Optional[int] = None,
                        pop: int = 24, generations: int = 50,
                        F_mut: float = 0.6, CR: float = 0.9, jitter: bool = False,
                        time_budget: Optional[float] = None,
                        tag: Optional[str] = None, log_every: Optional[int] = None) -> dict:
    """Differential evolution over the raw (F, Q, R) vector, batched through DMCI.

    The whole population (``pop`` raw-param sets) is decoded to an ``[N, D, D]`` F, Q, R and
    bound via ``as_matrix`` so ONE interpreter walk evaluates all N NLLs at once (gate G6 --
    ``torch.linalg`` batches over the leading dim; the scalar ``(ref obs k)`` index and the
    shared observation matrix are preserved). DE mutation / binomial crossover / greedy
    selection are vectorized in torch. This is the BLACK-BOX baseline (no gradients).

    IF the batched ``[N, D, D]`` eval errors (G6 absent on this build), we fall back to a
    per-candidate Python loop (correct, N x slower) and set ``result['batched'] = False`` so
    the wall-time figure stays honest. DiffEvo is opt-in (the runner gates it on a cfg flag).

    Returns the common result dict; ``n_iters`` is the number of generations completed.
    """
    if T is None:
        T = cfg.T_train
    if tag is None:
        tag = f"diffevo_batched_{structure}_D{D}_seed{seed}"
    if log_every is None:
        log_every = int(getattr(cfg, "log_every", 25))
    obs_t = obs[:T, :D].to(_DTYPE).contiguous()
    objective = make_objective(D, T, structure, obs, cfg, jitter=jitter)  # for diagnostics decode

    # --- raw-vector packing: a flat parameter vector per candidate (F_raw|Lq_raw|r_raw) ---
    base = params.init_raw_params(D, structure, seed=seed, rank=cfg.lowrank_rank,
                                  q_floor=cfg.q_floor, r_floor=cfg.r_floor)
    nF = base["F_raw"].numel()
    nLq = base["Lq_raw"].numel()
    nr = base["r_raw"].reshape(-1).numel()
    dim = nF + nLq + nr

    def pack(rd: dict) -> torch.Tensor:
        return torch.cat([rd["F_raw"].detach().reshape(-1),
                          rd["Lq_raw"].detach().reshape(-1),
                          rd["r_raw"].detach().reshape(-1)])

    def unpack(vec: torch.Tensor) -> dict:
        return {"F_raw": vec[:nF], "Lq_raw": vec[nF:nF + nLq],
                "r_raw": vec[nF + nLq:].reshape(())}

    g = torch.Generator().manual_seed(20240602 + int(seed))
    x0 = pack(base)
    # population: base + zero-mean noise (DE explores around the near-stable init).
    P = x0.unsqueeze(0).repeat(pop, 1) + 0.3 * torch.randn(pop, dim, generator=g, dtype=_DTYPE)
    P[0] = x0  # keep the canonical init in the population

    w_stab = float(cfg.stability_penalty)
    rank = cfg.lowrank_rank
    q_floor = cfg.q_floor
    r_floor = cfg.r_floor

    # --- batched population NLL (one interpreter walk); fall back to a Python loop on error ---
    def _decode_batch(pop_mat: torch.Tensor):
        """Decode an [N, dim] raw matrix into batched [N,D,D] F, Q, R (torch.linalg batches)."""
        N = pop_mat.shape[0]
        Fb = torch.stack([params.make_F(pop_mat[i, :nF], D, structure, rank) for i in range(N)])
        Qb = torch.stack([params.make_Q(pop_mat[i, nF:nF + nLq], D, q_floor) for i in range(N)])
        Rb = torch.stack([params.make_R(pop_mat[i, nF + nLq:].reshape(()), D, r_floor)
                          for i in range(N)])
        return Fb, Qb, Rb

    def fitness_batched(pop_mat: torch.Tensor) -> torch.Tensor:
        from neural_compiler.dmci import as_matrix
        from neural_compiler.evaluator import evaluate
        from neural_compiler.runtime.tagged_value import unwrap_number
        from .models import F_FACTOR_INPUTS, _FACTOR_FEATURE_NDIM
        from neural_compiler.runtime.tagged_value import TensorInput
        Fb, Qb, Rb = _decode_batch(pop_mat)
        graph = _get_graph(D, T, structure, jitter)
        binds = {"Q": as_matrix(Qb), "R": as_matrix(Rb), "obs": as_matrix(obs_t)}
        if structure == "S0":
            binds["F"] = as_matrix(Fb)                     # bind assembled F directly (no prelude)
        else:
            # bind the [N,...] factor inputs so the prelude assembles F per candidate in ONE walk
            facb = params.make_F_factors(pop_mat[:, :nF], D, structure, rank)
            for name in F_FACTOR_INPUTS[structure]:
                binds[name] = TensorInput(facb[name].to(_DTYPE), _FACTOR_FEATURE_NDIM[name])
        with torch.no_grad():
            out = evaluate(graph, binds, **cfg.EVAL_KW)
            nll = unwrap_number(out).reshape(-1).to(_DTYPE)
        if w_stab > 0.0:
            with torch.no_grad():
                pen = torch.stack([params.stability_penalty(Fb[i]) for i in range(Fb.shape[0])])
            nll = nll + w_stab * pen
        # non-finite candidates -> +inf so selection rejects them
        return torch.where(torch.isfinite(nll), nll, torch.full_like(nll, float("inf")))

    def fitness_loop(pop_mat: torch.Tensor) -> torch.Tensor:
        N = pop_mat.shape[0]
        vals = torch.empty(N, dtype=_DTYPE)
        for i in range(N):
            rd = unpack(pop_mat[i])
            try:
                v = float(objective(rd["F_raw"], rd["Lq_raw"], rd["r_raw"], grad=False))
            except Exception:  # noqa: BLE001
                v = float("inf")
            vals[i] = v if math.isfinite(v) else float("inf")
        return vals

    # probe the batched path once; fall back to the loop on any error or wrong shape.
    batched = True
    try:
        f0 = fitness_batched(P)
        if f0.shape[0] != pop or not torch.isfinite(f0).any():
            batched = False
    except Exception:  # noqa: BLE001
        batched = False
    fitness = fitness_batched if batched else fitness_loop

    t0 = time.perf_counter()
    fit = f0 if batched else fitness(P)
    best_trace: list = [float(fit.min())]
    gens_done = 0
    for gen in range(generations):
        # --- DE/rand/1/bin: for each i, mutant = a + F*(b - c), a,b,c distinct != i ---
        idx = torch.empty(pop, 3, dtype=torch.long)
        for i in range(pop):
            choices = torch.tensor([j for j in range(pop) if j != i])
            perm = choices[torch.randperm(choices.numel(), generator=g)[:3]]
            idx[i] = perm
        a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
        mutant = a + F_mut * (b - c)
        # binomial crossover
        cross = torch.rand(pop, dim, generator=g) < CR
        jrand = torch.randint(0, dim, (pop,), generator=g)
        cross[torch.arange(pop), jrand] = True
        trial = torch.where(cross, mutant, P)
        # greedy selection
        f_trial = fitness(trial)
        improved = f_trial < fit
        P = torch.where(improved.unsqueeze(1), trial, P)
        fit = torch.where(improved, f_trial, fit)
        best_trace.append(float(fit.min()))
        gens_done = gen + 1
        # live progress: one flushed line every log_every generations (diagnostic only).
        if log_every and (gen % log_every == 0 or gen == generations - 1):
            print(f"[fit {tag}] gen={gen:>4d}/{generations} "
                  f"nll={float(fit.min()):.6g} batched={batched}", flush=True)
        if time_budget is not None and (time.perf_counter() - t0) > time_budget:
            break
    wall_time = time.perf_counter() - t0

    best_i = int(torch.argmin(fit))
    best_nll = float(fit[best_i])
    best_raw = unpack(P[best_i])
    best_raw = {k: v.detach().clone() for k, v in best_raw.items()}

    res = _result("diffevo_batched", D, T, structure, seed, best_raw, objective, obs, cfg,
                  final_nll=best_nll, n_iters=gens_done,
                  converged=math.isfinite(best_nll), wall_time=wall_time,
                  nll_trace=best_trace, grad_norm_trace=[], jitter=jitter,
                  extra={"batched": batched, "pop": pop})
    # per_step_ms amortized over the total population evaluations (each is one T-step walk).
    pop_evals = (gens_done + 1) * pop
    res["per_step_ms"] = 1e3 * wall_time / max(1, pop_evals * T)
    return res
